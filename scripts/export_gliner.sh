#!/usr/bin/env bash
# Export urchade/gliner_small-v2.1 to ONNX.
# Output: $OUT/model.onnx + tokenizer.json + gliner_config.json

set -euo pipefail

OUT="${OUT:-./models/gliner-small-v2.1}"
VENV="${VENV:-/tmp/venv-embed-export}"

mkdir -p "$OUT"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -q --upgrade pip
pip install -q "gliner>=0.2.14" onnx onnxruntime "transformers>=4.45"

python3 - <<PY
from gliner import GLiNER
m = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
m.save_pretrained("$OUT", save_onnx=True)
print("saved to", "$OUT")
PY

echo "--- artifacts ---"
ls -lah "$OUT"
