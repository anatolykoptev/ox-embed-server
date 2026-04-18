#!/usr/bin/env bash
# Export naver/splade-v3-distilbert to ONNX (fill-mask task → sparse lexical expansion).

set -euo pipefail

OUT="${OUT:-/home/krolik/deploy/krolik-server/models/splade-v3-distilbert}"
VENV="${VENV:-/tmp/venv-embed-export}"

mkdir -p "$OUT"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -q --upgrade pip
pip install -q "optimum[onnxruntime]>=1.24" "transformers>=4.45" onnx

optimum-cli export onnx \
  --model naver/splade-v3-distilbert \
  --task fill-mask \
  --opset 17 \
  "$OUT"

echo "--- artifacts ---"
ls -lah "$OUT"
