#!/usr/bin/env bash
# Phase H.15 follow-up: ModernBERT-only cells, smaller stress.
# Reuses the existing modernbert-h15-postfix dir if it has files; falls
# back to the new path otherwise.
set -uo pipefail   # no -e so a single timeout doesn't kill the rest

OUT="${1:-docs/benchmarks/2026-05-01-modernbert-h15-postfix}"
mkdir -p "$OUT"

URL="${URL:-http://127.0.0.1:8082/v1/rerank}"
SCENARIOS="${SCENARIOS:-1x10,4x10}"

curl -sf http://127.0.0.1:8082/metrics \
  | grep -E "^embed_rerank_|^embed_token_cache_|^embed_batch_padding_waste_ratio" \
  > "$OUT/prom_snapshot_h15.txt" || true

for size in medium long; do
  for docs in 5 20; do
    tag="gte-modernbert_${size}_docs${docs}"
    echo "→ $tag"
    python3 bench.py \
      --kind rerank \
      --model gte-modernbert \
      --size "$size" \
      --docs-per-req "$docs" \
      --scenarios "$SCENARIOS" \
      --url "$URL" \
      --json \
      > "$OUT/${tag}.json" || echo "  cell timed out — empty file"
  done
done

curl -sf http://127.0.0.1:8082/metrics \
  | grep -E "^embed_rerank_|^embed_token_cache_|^embed_batch_padding_waste_ratio" \
  > "$OUT/prom_snapshot_h15_after.txt" || true

echo "DONE — modernbert cells in $OUT"
ls -la "$OUT"
