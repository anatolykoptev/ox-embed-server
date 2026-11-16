# syntax=docker/dockerfile:1.4
# --- Build stage ---
# rust:1.100.0-slim (multi-arch index). Keep in sync with rust-toolchain.toml;
# scripts/check-toolchain-pin.sh enforces it in CI.
FROM rust:1.100.0-slim@sha256:e9d95b74296f7126a83ae3d041aeaba1ff4c478f13abc5cc034fcf7c131c3d4a AS builder

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
# SHA-256 of the release tarballs (sccache-v${SCCACHE_VERSION}-<arch>-unknown-linux-musl.tar.gz);
# they match the .sha256 files published next to the assets. Bump with SCCACHE_VERSION.
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
      aarch64) SCCACHE_SHA256=3a6a3712b49da3d263bf2d30d702de4302793016019e800bfb81c0c69401d8f8 ;; \
      x86_64)  SCCACHE_SHA256=782d2b5dd7ae0a55ebe368ab258114d0928d019ac2d949ab85d5d02f3926709e ;; \
      *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;; \
    esac && \
    curl -fsSL -o /tmp/sccache.tgz \
      "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" && \
    echo "${SCCACHE_SHA256}  /tmp/sccache.tgz" | sha256sum -c - && \
    tar xzf /tmp/sccache.tgz --strip-components=1 -C /usr/local/bin "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    rm /tmp/sccache.tgz && \
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
# debian:trixie-slim (multi-arch index)
FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2

# Download ONNX Runtime 1.24.3. libonnxruntime.so is loaded by every worker, so
# the tarball is verified against a checked-in SHA-256 (the same digests GitHub
# records for the release assets). Bump the version and both hashes together.
# curl is needed only for this download; it is purged in the same layer so the
# runtime image does not ship it (the healthcheck is `embed-server healthcheck`).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    ORT_VER="1.24.3" && \
    case "$(uname -m)" in \
      aarch64) ORT_ARCH=aarch64; ORT_SHA256=15100fb88b4c692cdd6bf2cca5f4a26a3806cebca8136de6681e2aba4b2ea033 ;; \
      x86_64)  ORT_ARCH=x64;     ORT_SHA256=4c436a280d650f8bf32c921a2bf4de7c42cc32884c51c90e47de991708bbb5a4 ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac && \
    curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz" && \
    echo "${ORT_SHA256}  /tmp/ort.tgz" | sha256sum -c - && \
    tar -xzf /tmp/ort.tgz -C /tmp/ && \
    cp /tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}/lib/libonnxruntime.so /usr/lib/ && \
    ldconfig && \
    rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-* && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /binaries/embed-server /usr/local/bin/embed-server
COPY --from=builder /binaries/embed-worker /usr/local/bin/embed-worker

ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
# Default location for spawned worker processes. Compose can override via
# EMBED_WORKER_BIN / EMBED_WORKER_SOCKET_DIR when EMBED_MULTI_PROCESS=1.
ENV EMBED_WORKER_BIN=/usr/local/bin/embed-worker
ENV EMBED_WORKER_SOCKET_DIR=/tmp/embed-workers

# Run unprivileged. The process writes only to: the worker socket dir (a tmpfs
# `/tmp` in compose) and the ONNX optimised-graph cache (`ONNX_OPT_CACHE_DIR*`,
# a volume in compose). Model mounts are read-only. The cache directory is
# created here, owned by the runtime user, so a fresh named volume mounted over
# it inherits that ownership; an EXISTING volume keeps its owner and must be
# chowned to 10001 once before this image is deployed.
RUN groupadd --system --gid 10001 embed && \
    useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin embed && \
    mkdir -p /var/cache/onnx-opt && chown 10001:10001 /var/cache/onnx-opt
USER 10001:10001

EXPOSE 8082
# `healthcheck` asks the local /ready (every worker up + a real 1-word embed);
# it needs no curl. start-period covers the cold model load (~40 s measured,
# the deploy smoke test allows 150 s).
# --timeout must exceed the healthcheck's own budget: EMBED_READY_PROBE_TIMEOUT_MS
# (default 2 s) + 3 s. Raise it together with that variable beyond ~5 s.
HEALTHCHECK --interval=30s --timeout=8s --start-period=180s --retries=3 \
  CMD ["embed-server", "healthcheck"]
ENTRYPOINT ["embed-server"]
