#!/usr/bin/env bash
# Fetch the pinned ONNX Runtime shared library for the integration tests and
# print its path on stdout. Mirrors the Dockerfile pin (ORT_VER), verified by
# sha256, so the tests run against the same runtime the image ships.
#
#   scripts/fetch-ort.sh [dest-dir]        # default: target/ort
#   export ORT_DYLIB_PATH="$(scripts/fetch-ort.sh)"
#
# Idempotent: a present, already-verified copy is reused.
set -euo pipefail

ORT_VER="1.24.3"
case "$(uname -m)" in
  aarch64 | arm64)
    ORT_ARCH="aarch64"
    ORT_SHA256="15100fb88b4c692cdd6bf2cca5f4a26a3806cebca8136de6681e2aba4b2ea033" ;;
  x86_64 | amd64)
    ORT_ARCH="x64"
    ORT_SHA256="4c436a280d650f8bf32c921a2bf4de7c42cc32884c51c90e47de991708bbb5a4" ;;
  *) echo "fetch-ort: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac

dest="${1:-target/ort}"
mkdir -p "$dest"
lib="$dest/libonnxruntime-${ORT_VER}.so"
if [ ! -s "$lib" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz"
  curl -sSfL --retry 3 -o "$tmp/ort.tgz" "$url" >&2
  echo "${ORT_SHA256}  $tmp/ort.tgz" | sha256sum -c - >&2
  tar -xzf "$tmp/ort.tgz" -C "$tmp"
  # `libonnxruntime.so` is a symlink chain; copy the real file.
  cp -L "$tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}/lib/libonnxruntime.so" "$lib"
fi
echo "$(cd "$dest" && pwd)/$(basename "$lib")"
