# syntax=docker/dockerfile:1.4
# --- Build stage ---
FROM rust:1.93-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg-config libssl-dev curl && \
    rm -rf /var/lib/apt/lists/*

# sccache 0.10: content-addressed compiler cache. Adds value over target/ mount alone
# in three scenarios: (1) target/ pruned between builds, (2) parallel branches sharing
# dep code, (3) comment-only source changes that change target/ layer hash but not
# compiled object content. Paths inside docker are stable (/app) so no --remap-path-prefix
# needed — cache keys are deterministic and hit rate is high.
# Download aarch64 or x86_64 musl binary based on build host architecture.
ENV SCCACHE_VERSION=0.15.0
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod +x /usr/local/bin/sccache

ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV SCCACHE_DIR=/root/.cache/sccache
# sccache does not cache incremental build artifacts; disabling avoids wasted work
# writing incremental state that sccache will never read.
ENV CARGO_INCREMENTAL=0

WORKDIR /app

# Layer 1: dep-only build (cached until Cargo.toml / Cargo.lock changes).
# Pre-compiles all crates.io dependencies against a stub binary so that
# source-code changes don't invalidate this layer.
COPY Cargo.toml Cargo.lock ./
# Stubs for all three Cargo targets so the dep-only layer compiles:
# - src/main.rs       — [[bin]] embed-server (default; implicit pre-Phase-2)
# - src/bin/worker.rs — [[bin]] embed-worker (added Wave 1.2)
# - src/lib.rs        — [lib]  embed_server (added Wave 1.2)
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/worker.rs && \
    : > src/lib.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
    cargo build --release --locked --bins && \
    rm -rf src

# Layer 2: real source. Rebuilds only the embed-server crate on code changes.
# Build BOTH binaries (embed-server supervisor + embed-worker child process).
# Binaries must be copied OUT of the cache-mounted target/ dir before the
# RUN ends, or the layer loses them.
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    --mount=type=cache,target=/root/.cache/sccache,sharing=locked \
    touch src/main.rs src/bin/worker.rs src/lib.rs && \
    cargo build --release --locked --bins && \
    mkdir -p /binaries && \
    cp target/release/embed-server /binaries/embed-server && \
    cp target/release/embed-worker /binaries/embed-worker && \
    sccache --show-stats || true

# --- Runtime stage ---
FROM debian:trixie-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ONNX Runtime 1.32.0 (released 2026-11-13), pinned by version and sha256.
# Digests are the ones GitHub publishes for the release assets; a bump must
# update both. `ort` 2.0.0-rc.15 targets ORT 1.31 and negotiates the C API
# version at load time, so a newer dylib is accepted.
ARG TARGETARCH
RUN ORT_VER="1.32.0" && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      ORT_ARCH="aarch64"; \
      ORT_SHA256="81a46251dd90da4122f4b9ceb4433f836fbc79ee1eddffca7c09033af81a1939"; \
    else \
      ORT_ARCH="x64"; \
      ORT_SHA256="bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503"; \
    fi && \
    curl -fL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz" && \
    echo "${ORT_SHA256}  /tmp/ort.tgz" | sha256sum -c - && \
    tar -xzf /tmp/ort.tgz -C /tmp/ && \
    cp /tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}/lib/libonnxruntime.so /usr/lib/ && \
    ldconfig && \
    rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-*

COPY --from=builder /binaries/embed-server /usr/local/bin/embed-server
COPY --from=builder /binaries/embed-worker /usr/local/bin/embed-worker

ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
# Default location for spawned worker processes. Compose can override via
# EMBED_WORKER_BIN / EMBED_WORKER_SOCKET_DIR when EMBED_MULTI_PROCESS=1.
ENV EMBED_WORKER_BIN=/usr/local/bin/embed-worker
ENV EMBED_WORKER_SOCKET_DIR=/tmp/embed-workers

EXPOSE 8082

ENTRYPOINT ["embed-server"]
