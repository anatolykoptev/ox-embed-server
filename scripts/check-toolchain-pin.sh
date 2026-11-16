#!/usr/bin/env bash
# Fail when the Dockerfile builder image and rust-toolchain.toml name different
# Rust versions (or Cargo.toml rust-version). All values are read from the files.
set -euo pipefail
cd "$(dirname "$0")/.."

toolchain=$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' rust-toolchain.toml | head -n1)
docker=$(sed -n 's/^FROM rust:\([^ @]*\)-slim.*/\1/p' Dockerfile | head -n1)
msrv=$(sed -n 's/^rust-version *= *"\([^"]*\)".*/\1/p' Cargo.toml | head -n1)

fail=0
for pair in "rust-toolchain.toml channel=$toolchain" "Dockerfile builder=$docker" "Cargo.toml rust-version=$msrv"; do
  [ -n "${pair#*=}" ] || { echo "toolchain pin: could not read ${pair%% *}" >&2; fail=1; }
done
if [ "$fail" -eq 0 ] && { [ "$toolchain" != "$docker" ] || [ "$toolchain" != "$msrv" ]; }; then
  echo "toolchain pin drift: rust-toolchain.toml=$toolchain Dockerfile=rust:$docker-slim Cargo.toml rust-version=$msrv" >&2
  echo "Update all three to the same version." >&2
  fail=1
fi
[ "$fail" -eq 0 ] && echo "toolchain pin ok: $toolchain"
exit "$fail"
