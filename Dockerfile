# syntax=docker/dockerfile:1.4
# --- Build stage ---
FROM rust:1.93-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: dep-only build (cached until Cargo.toml / Cargo.lock changes).
# Pre-compiles all crates.io dependencies against a stub binary so that
# source-code changes don't invalidate this layer.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    cargo build --release --locked && \
    rm -rf src

# Layer 2: real source. Rebuilds only the embed-server crate on code changes.
# Build BOTH binaries (embed-server supervisor + embed-worker child process).
# Binaries must be copied OUT of the cache-mounted target/ dir before the
# RUN ends, or the layer loses them.
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    touch src/main.rs src/bin/worker.rs && \
    cargo build --release --locked --bins && \
    mkdir -p /binaries && \
    cp target/release/embed-server /binaries/embed-server && \
    cp target/release/embed-worker /binaries/embed-worker

# --- Runtime stage ---
FROM debian:trixie-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime 1.24.3
ARG TARGETARCH
RUN ORT_VER="1.24.3" && \
    if [ "$TARGETARCH" = "arm64" ]; then ORT_ARCH="aarch64"; else ORT_ARCH="x64"; fi && \
    curl -L -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz" && \
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
