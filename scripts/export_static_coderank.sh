#!/usr/bin/env bash
# scripts/export_static_coderank.sh — multi-shape static ONNX export for
# `nomic-ai/CodeRankEmbed` (the code-index dense embed model go-code cut
# over to in PR #231), one fixed-shape graph per `(batch_size, seq_len)`
# pair.
#
# WHY THIS EXISTS
#   Under sustained go-code autoindex load, code-rank-embed inference is
#   p50 16.7s / p95 28.7s — NOT model compute (solo warm = 0.6s) but
#   per-run BFCArena shrink↔extend thrash on a variable (batch,seq)
#   input shape (investigation
#   reports/embed-server/investigations/2026-06-12-coderank-arena-shrink-load-blowup.md).
#   A fixed-shape graph lets every request hit a session whose arena is
#   already grown to the bucket peak and never re-extends — converting
#   the ~16s arena tax to ~0. See
#   docs/plans/2026-06-12-coderank-static-shape.md for the full design.
#
# Outputs:
#   $OUT/model_quantized_static_b<N>_s<M>.onnx  (one per requested shape)
#   $OUT/model_quantized_static_b<N>_s<M>.onnx.data  (iff external data)
#
# These files are consumed by embed-server's static-shape fast-path
# loader (Phase 2 of the plan — src/model.rs discovery + 2-axis pool).
# The loader scans this filename pattern and routes embed calls whose
# padded (batch,seq) shape exactly matches `(<N>,<M>)` through the
# corresponding fixed-shape session pool.
#
#   Note the 2-axis filename `_b<N>_s<M>` vs the reranker's batch-only
#   `_b<N>` (scripts/export_static_modernbert.sh): the reranker's seq is
#   fixed at max_len=256, so batch is its only free axis. The embed
#   path's seq varies 5→512 and seq is the DOMINANT arena driver
#   (attention/matmul scratch scales with B×heads×S²), so both axes are
#   baked into the filename.
#
# ── DO NOT RUN THIS ON pillow (the prod box) ────────────────────────
#   The export materialises a multi-GB intermediate tree and saturates
#   the CPU for tens of minutes — it would stall the live e5/jina/rerank/
#   splade workers AND go-code + memdb indexing. Run on a dev box or a
#   one-off Oracle build VM, then rsync the resulting `.onnx` (+ `.data`)
#   files to:
#     pillow:<EMBED model dir>/nomic-ai__CodeRankEmbed/   (or wherever
#       EMBED_MODELS points the code-rank-embed ONNX dir on pillow)
#   and recreate the embed-server container. See § Operator next steps
#   at the bottom — NONE of those steps are run by this script.
#
# ── EQUIVALENCE GATE (read before promoting any output) ─────────────
#   A static int8 graph re-exported from source MUST produce vectors
#   numerically equivalent to the shipped dynamic int8 path, or a full
#   go-code reindex is forced. This script's `--no-dynamic-axes` re-fold
#   + the same `quantize_dynamic` recipe the dynamic graph used keeps
#   drift low, but it is NOT proof. Run
#   scripts/equivalence_static_coderank.py AFTER this and confirm
#   cosine ≥ 0.99999 per vector BEFORE rsync'ing anything to pillow.
#
# Usage:
#   ./scripts/export_static_coderank.sh                         # default b32_s512 ONLY
#   BATCH_SIZES="32 4" SEQUENCE_LENGTHS="128 256 512" ./scripts/export_static_coderank.sh
#   OUT=/tmp/coderank-static  MODEL_ID=nomic-ai/CodeRankEmbed \
#     ./scripts/export_static_coderank.sh
set -euo pipefail

# fp32 PyTorch source (MIT). Re-export from source — do NOT reshape the
# shipped dynamic int8 (no Shape/Gather re-fold → partial win + drift
# risk, see plan ADR-001 / Decision 3). nomic_bert is a custom arch →
# `--trust-remote-code` is mandatory.
MODEL_ID="${MODEL_ID:-nomic-ai/CodeRankEmbed}"
OUT="${OUT:-./code-rank-embed-static}"

# Default = ONE bucket: b32_s512. The investigation found 99/135 calls
# are exactly batch=32 × seq=512 (the seq is already pow2-padded+capped
# at 512 by model.rs::round_up_seq_len). Shipping ONLY this bucket first
# is the plan's memory-budget mandate (each static arena ~1-1.5 GiB on
# the shared pillow container; widen only on observed RSS headroom).
BATCH_SIZES="${BATCH_SIZES:-32}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-512}"

VENV="${VENV:-/tmp/venv-embed-export-static-coderank}"

mkdir -p "$OUT"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -q --upgrade pip
# `optimum-cli ... --no-dynamic-axes` pre-folds the runtime shape ops
# into constants. `--task feature-extraction` emits the raw
# last_hidden_state (the dense-embed path mean-pools+normalizes that in
# Rust) — NOT text-classification (that's the reranker). einops is a
# nomic_bert remote-code dependency.
pip install -q "optimum[onnxruntime]>=1.24" "transformers>=4.48" "onnx>=1.17" "onnxruntime>=1.20" "einops>=0.7"

for N in $BATCH_SIZES; do
  for M in $SEQUENCE_LENGTHS; do
    echo
    echo "============================================================"
    echo "Exporting static graph: model=$MODEL_ID batch=$N seq=$M"
    echo "============================================================"

    STAGE="$OUT/_stage_b${N}_s${M}"
    mkdir -p "$STAGE"

    # Phase 1 — fp32 fixed-axes export. `--no-dynamic-axes` plus
    # `--batch_size $N --sequence_length $M` literally bakes `[N, M]`
    # into the graph as constants. Any downstream session.run() with a
    # different shape fails; the Rust loader's exact-match routing (pad
    # UP to the bucket, then slice the batch dim back) is what makes
    # this safe. `--trust-remote-code` loads the nomic_bert custom arch.
    optimum-cli export onnx \
      --model "$MODEL_ID" \
      --task feature-extraction \
      --opset 17 \
      --no-dynamic-axes \
      --batch_size "$N" \
      --sequence_length "$M" \
      --trust-remote-code \
      "$STAGE"

    # Phase 1.5 — FIXED-AXES SELF-CHECK (plan Risk row 3). optimum on a
    # custom arch (`trust_remote_code`) MAY silently emit a still-dynamic
    # graph (dim_param instead of dim_value). Catch it HERE, off-box,
    # before quantizing or shipping — a still-dynamic graph defeats the
    # whole optimization and would silently fall back to arena thrash on
    # pillow. Fails loudly (exit 1) if any input dim is not a literal.
    python3 - "$STAGE/model.onnx" "$N" "$M" <<'PYEOF'
import sys
import onnx

src, want_b, want_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = onnx.load(src, load_external_data=False)
problems = []
seen = []
for inp in m.graph.input:
    dims = inp.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        if d.HasField("dim_value"):
            shape.append(d.dim_value)
        elif d.HasField("dim_param"):
            shape.append(f"<dyn:{d.dim_param}>")
            problems.append(f"  input '{inp.name}' axis is DYNAMIC (dim_param={d.dim_param!r})")
        else:
            shape.append("<unknown>")
            problems.append(f"  input '{inp.name}' has an UNKNOWN dim")
    seen.append(f"  {inp.name}: {shape}")
    # Expect rank-2 [batch, seq] for input_ids / attention_mask.
    if len(shape) >= 2:
        if shape[0] != want_b:
            problems.append(f"  input '{inp.name}' batch dim = {shape[0]}, expected {want_b}")
        if shape[1] != want_s:
            problems.append(f"  input '{inp.name}' seq dim = {shape[1]}, expected {want_s}")

print("[self-check] exported graph inputs:")
print("\n".join(seen))
if problems:
    print("\n[self-check] FAILED — graph is NOT fully fixed-axes:", file=sys.stderr)
    print("\n".join(problems), file=sys.stderr)
    print(
        "\noptimum did not bake literal dims for this custom arch. Fall back\n"
        "to torch.onnx.export with fixed dummy inputs (PyTorch weights are\n"
        "MIT-licensed and available). DO NOT quantize or ship this graph.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"[self-check] OK — inputs are fully fixed at [b={want_b}, s={want_s}]")
PYEOF

    # Phase 2 — int8 dynamic quantization, matching the recipe used to
    # produce the shipped dynamic `model_int8.onnx`
    # (MisterTK/CodeRankEmbed-onnx-int8 quantizes MatMul/Gemm to QInt8).
    # We call `quantize_dynamic` explicitly so we can name the output
    # file AND keep the recipe byte-aligned with the dynamic graph the
    # equivalence gate compares against. `use_external_data_format=True`
    # because nomic_bert > 2 GiB int8 may exceed the protobuf 2 GiB cap
    # and ships its weights as an `.onnx.data` sibling.
    python3 - "$STAGE/model.onnx" "$OUT/model_quantized_static_b${N}_s${M}.onnx" <<'PYEOF'
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
print(f"quantizing {src} -> {dst}")
quantize_dynamic(
    str(src),
    str(dst),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
    use_external_data_format=True,
)
total = dst.stat().st_size
data = dst.with_suffix(dst.suffix + ".data")
if data.exists():
    total += data.stat().st_size
    print(f"  external data: {data.name} ({data.stat().st_size / (1024*1024):.1f} MiB)")
print(f"  graph: {dst.name} ({dst.stat().st_size / (1024*1024):.1f} MiB)")
print(f"  total on-disk: {total / (1024*1024):.1f} MiB")
PYEOF

    # Drop the staging tree once the quantized output is in place.
    rm -rf "$STAGE"
  done
done

echo
echo "============================================================"
echo "DONE — static-shape ONNX files in $OUT"
echo "============================================================"
ls -lah "$OUT"

cat <<'OPERATOR_STEPS'

────────────────────────────────────────────────────────────────────
NEXT STEPS (NOT run by this script):

  1. EQUIVALENCE GATE (mandatory before any deploy). Point the harness
     at the dynamic int8 graph + the new static graph and confirm
     cosine ≥ 0.99999 per vector:
       python3 scripts/equivalence_static_coderank.py \
         --dynamic <dir>/model_int8.onnx \
         --static  ./code-rank-embed-static/model_quantized_static_b32_s512.onnx \
         --tokenizer nomic-ai/CodeRankEmbed
     If it FAILS, a full go-code reindex is forced — report loudly and
     do NOT ship.

  2. rsync the static .onnx (+ .onnx.data sibling if present) to pillow's
     code-rank-embed model dir, then recreate the container:
       rsync -avh ./code-rank-embed-static/model_quantized_static_b32_s512.onnx* \
         pillow:<EMBED code-rank model dir>/
       ssh pillow 'cd <embed-server compose dir> && \
         docker compose up -d --no-deps --force-recreate embed-server'

  3. ENABLE the bucket (Phase 3 — default-off until then). Set on pillow:
       EMBED_STATIC_SHAPES_CODE_RANK_EMBED="b32_s512"
     and recreate. Rollback is instant: unset the env + recreate.

  4. A/B vs the dynamic+env-flip baseline. Promote only on ≥10% p95 win
     AND confirmed memory headroom (plan § Memory budget / canon gate).
────────────────────────────────────────────────────────────────────
OPERATOR_STEPS
