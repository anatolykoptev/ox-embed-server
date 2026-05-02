#!/usr/bin/env bash
# scripts/export_static_modernbert.sh — multi-shape static ONNX export
# for `Alibaba-NLP/gte-reranker-modernbert-base` (ModernBERT cross-encoder
# reranker), one fixed-shape graph per `(batch_size, seq_len)` pair.
#
# Outputs:
#   $OUT/model_quantized_static_b<N>.onnx  (one per requested batch size)
#
# These files are consumed by `embed-server`'s static-shape fast-path
# loader at `src/model_reranker/load.rs::discover_static_shape_files` —
# the loader scans this filename pattern and routes `score_pairs` calls
# whose batch size exactly matches `<N>` through the corresponding
# fixed-shape session pool. See `docs/plans/2026-05-02-multi-shape-static-export.md`
# for the design.
#
# DO NOT RUN THIS ON krolik (the prod box). The export materialises a
# dozen-GB intermediate tree and saturates the model's CPU for tens of
# minutes. Run on a dev box or a one-off Oracle build VM, then rsync
# the resulting `.onnx` files to:
#   krolik:/home/krolik/deploy/krolik-server/models/gte-reranker-modernbert-base/
# and bounce the embed-server container.
#
# Backwards compat: a legacy unsuffixed `model_quantized_static.onnx`
# (PR #27) is still loaded as `b=1`. If you produce `model_quantized_static_b1.onnx`
# alongside it, the explicit-suffix file wins and the loader logs a warn
# about the duplicate. Remove the legacy file once the explicit one is
# in place to silence the warning.
#
# Usage:
#   ./scripts/export_static_modernbert.sh                    # default {1,5}
#   BATCH_SIZES="1 2 5 10" ./scripts/export_static_modernbert.sh
#   OUT=/tmp/out  MODEL_ID=Alibaba-NLP/gte-reranker-modernbert-base \
#     ./scripts/export_static_modernbert.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Alibaba-NLP/gte-reranker-modernbert-base}"
OUT="${OUT:-./gte-reranker-modernbert-base-static}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
# Default policy mirrors `docs/plans/2026-05-02-multi-shape-static-export.md`
# § Shape policy: ship `{b=1, b=5}` to cover the PR #27 fast-path AND
# memdb-go's D7 sub-query hot path. Add `2` and/or `10` only after a
# focused bench shows the per-shape uplift justifies the extra ~510 MiB
# of session-pool RAM and ~250 MiB of disk.
BATCH_SIZES="${BATCH_SIZES:-1 5}"
VENV="${VENV:-/tmp/venv-embed-export-static-modernbert}"

mkdir -p "$OUT"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -q --upgrade pip
# `optimum-cli ... --no-dynamic-axes` is the tool that pre-folds the
# 700+ runtime shape ops into constants (see PR #27 commit msg for the
# node-count delta). transformers >=4.45 added the ModernBERT config;
# onnxruntime is only used by the `onnxruntime.quantization` step
# below for the int8 dynamic-quant pass.
pip install -q "optimum[onnxruntime]>=1.24" "transformers>=4.48" "onnx>=1.17" "onnxruntime>=1.20"

for N in $BATCH_SIZES; do
  echo
  echo "============================================================"
  echo "Exporting static graph: model=$MODEL_ID batch=$N seq=$SEQUENCE_LENGTH"
  echo "============================================================"

  STAGE="$OUT/_stage_b${N}"
  mkdir -p "$STAGE"

  # Phase 1 — fp32 fixed-axes export. `--no-dynamic-axes` plus
  # `--batch_size $N --sequence_length $SEQUENCE_LENGTH` literally
  # bakes `[N, $SEQUENCE_LENGTH]` into the graph as constants. Any
  # downstream session.run() with a different shape will fail; the
  # Rust loader's exact-match routing is what makes this safe.
  optimum-cli export onnx \
    --model "$MODEL_ID" \
    --task text-classification \
    --opset 17 \
    --no-dynamic-axes \
    --batch_size "$N" \
    --sequence_length "$SEQUENCE_LENGTH" \
    "$STAGE"

  # Phase 2 — int8 dynamic quantization, matching the recipe used to
  # produce the existing `model_quantized_static.onnx` shipped with
  # PR #27 (which itself mirrors what
  # `docs/plans/2026-05-01-modernbert-optimization.md` § 3E flagged
  # for follow-up). We use `onnxruntime.quantization.quantize_dynamic`
  # — same pass `optimum[onnxruntime]` invokes for the
  # `--quantization_approach dynamic` shorthand, but called explicitly
  # so we can name the output file.
  python3 - <<PYEOF
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

src = Path("$STAGE/model.onnx")
dst = Path("$OUT/model_quantized_static_b${N}.onnx")
print(f"quantizing {src} -> {dst}")
quantize_dynamic(
    str(src),
    str(dst),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)
print(f"  size: {dst.stat().st_size / (1024*1024):.1f} MiB")
PYEOF

  # Drop the staging tree once the quantized output is in place. Keeps
  # disk usage during multi-shape exports bounded.
  rm -rf "$STAGE"
done

echo
echo "============================================================"
echo "DONE — static-shape ONNX files in $OUT"
echo "============================================================"
ls -lah "$OUT"

# Operator next steps (NOT run by this script):
#   1. rsync the new .onnx files to krolik:
#        rsync -avh "$OUT"/model_quantized_static_b*.onnx \
#          krolik:/home/krolik/deploy/krolik-server/models/gte-reranker-modernbert-base/
#   2. (optional) remove the legacy unsuffixed file once
#      `model_quantized_static_b1.onnx` is present:
#        ssh krolik 'rm /home/krolik/deploy/krolik-server/models/gte-reranker-modernbert-base/model_quantized_static.onnx'
#   3. Recreate the embed-server container so it re-scans the model dir:
#        ssh krolik 'cd ~/deploy/krolik-server && docker compose up -d --no-deps --force-recreate embed-server'
#   4. Confirm via logs:
#        ssh krolik 'docker logs embed-server --since 1m | grep "static-shape ONNX"'
#      Expect one info line per loaded shape:
#        "loading static-shape ONNX fast-path session pool batch=1 path=..."
#        "loading static-shape ONNX fast-path session pool batch=5 path=..."
#   5. Bench BEFORE promising the optimization to memdb-go. PR #27's
#      prod measurement showed only ~1.04× uplift on quantized int8 vs
#      the dynamic pool at batch=1 — most of the static-graph win
#      gets absorbed by the quantize/dequantize node cost. Validate
#      the batch=5 export beats the dynamic pool at the realistic
#      `(batch=5, seq=256)` shape before declaring a win.
