# --- Build stage ---
FROM rust:1.87-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependencies: copy manifests and fetch
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Build actual binary
COPY src/ src/
RUN touch src/main.rs && cargo build --release

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

COPY --from=builder /app/target/release/embed-server /usr/local/bin/

ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

EXPOSE 8082

ENTRYPOINT ["embed-server"]
