#!/usr/bin/env bash
# Phase 2 baseline: gte-multi-rerank vs gte-reranker-modernbert-base
# head-to-head sweep against the live embed-server. Produces JSON
# scenarios per (model, size, docs_per_req, scenarios) — diffable in CI.
#
# Usage: ./scripts/bench_modernbert_vs_mgte.sh [output_dir]
#   default output_dir: docs/benchmarks/2026-05-01-modernbert-baseline/
set -euo pipefail

OUT="${1:-docs/benchmarks/2026-05-01-modernbert-baseline}"
mkdir -p "$OUT"

URL="${URL:-http://127.0.0.1:8082/v1/rerank}"
SCENARIOS="${SCENARIOS:-1x10,4x20}"

# Snapshot Prometheus before the sweep so we can diff sums/counts.
curl -sf http://127.0.0.1:8082/metrics \
  | grep -E "^embed_rerank_|^embed_token_cache_|^embed_batch_padding_waste_ratio" \
  > "$OUT/prom_snapshot_before.txt" || true

for model in gte-multi-rerank gte-modernbert; do
  for size in medium long; do
    for docs in 5 20; do
      tag="${model}_${size}_docs${docs}"
      echo "→ $tag"
      python3 bench.py \
        --kind rerank \
        --model "$model" \
        --size "$size" \
        --docs-per-req "$docs" \
        --scenarios "$SCENARIOS" \
        --url "$URL" \
        --json \
        > "$OUT/${tag}.json"
    done
  done
done

curl -sf http://127.0.0.1:8082/metrics \
  | grep -E "^embed_rerank_|^embed_token_cache_|^embed_batch_padding_waste_ratio" \
  > "$OUT/prom_snapshot_after.txt" || true

echo "DONE — results in $OUT"
ls -la "$OUT"
