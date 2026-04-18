#!/usr/bin/env bash
# Export BAAI/bge-reranker-v2-m3 to ONNX + INT8 quantization.
# Output: $OUT/model_quantized.onnx + tokenizer.json
# Usage: bash scripts/export_reranker_bge.sh

set -euo pipefail

OUT="${OUT:-/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3}"
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
  --model BAAI/bge-reranker-v2-m3 \
  --task text-classification \
  --opset 17 \
  "$OUT"

python3 - <<PY
import os
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

out = "$OUT"
q = ORTQuantizer.from_pretrained(out)
# ARM64 uses QDQ int8 dynamic quantization
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
q.quantize(save_dir=out, quantization_config=qconfig)
print("quantized:", os.path.join(out, "model_quantized.onnx"))
PY

echo "--- artifacts ---"
ls -lah "$OUT"/model*.onnx "$OUT"/tokenizer.json "$OUT"/*.json 2>/dev/null || true
