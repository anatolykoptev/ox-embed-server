# embed-server — Unified Rust ONNX Embedding + Rerank + SPLADE Sidecar

**Rust / axum** · Docker · Prometheus `/metrics` · branch `main`

Single process serving three model classes concurrently:

| Class    | Default model(s)                                        | Endpoint             | API shape |
|----------|---------------------------------------------------------|----------------------|-----------|
| Dense    | `multilingual-e5-large` (1024d), `jina-code-v2` (768d)  | `POST /v1/embeddings`| OpenAI    |
| Reranker | `gte-multi-rerank`                                      | `POST /v1/rerank`    | Cohere    |
| Sparse   | SPLADE                                                  | `POST /embed_sparse` | TEI       |

## Source layout (`src/`, ~5.6k LOC)

| File                 | LOC  | Role |
|----------------------|-----:|------|
| `main.rs`            |  385 | axum router, startup, graceful SIGTERM |
| `config.rs`          |  694 | env parsing: `EMBED_MODELS`, `RERANKER_MODELS`, batching knobs |
| `types.rs`           |  145 | `AppState`, `ModelEntry`, error types |
| `api.rs`             |  307 | `/v1/embeddings` handler |
| `api_rerank.rs`      |  510 | `/v1/rerank` handler (Cohere-shape) |
| `api_splade.rs`      |  271 | `/embed_sparse` handler (TEI-shape) |
| `model.rs`           |  382 | Dense ONNX session + tokenizer + mean-pool |
| `model_reranker.rs`  |  600 | Cross-encoder ONNX scoring, session pool |
| `model_splade.rs`    |  499 | SPLADE sparse model |
| `batcher.rs`         | 1150 | `DynamicBatcher` + token-budget batcher, carry-over |
| `cache.rs`           |  216 | moka response cache (embeddings only; rerank bypasses) |
| `cache_flow.rs`      |  186 | probe→insert hot path |
| `pool.rs`            |   94 | mean-pool + L2 normalize |
| `metrics.rs`         |  196 | Prometheus exposition helpers |
| `bench.py`           |    — | load harness (run under `rtk` for untruncated output) |

## API

- `POST /v1/embeddings` — OpenAI-compat. Model picked via `model` field
  (fallback: `EMBED_DEFAULT_MODEL`). 503 + `Retry-After: 1` when queue full,
  503 + `Retry-After: 5` during shutdown. Response cache in front.
- `POST /v1/rerank` — Cohere-shape: `{query, documents[], top_n, return_documents}`
  → `{results:[{index, relevance_score}]}`. Documents accepted as plain string
  or `{"text":...}` (untagged serde). **No cache** — `(query, doc)` pairs are
  nearly always unique.
- `POST /embed_sparse` — TEI convention (no `/v1/` prefix by design).
- `GET /health` — plain `ok`.
- `GET /metrics` — Prometheus `embed_*` series. Counters
  (`embed_requests_total`, `embed_rerank_requests_total`,
  `embed_token_cache_total`, etc.), latency histograms
  (`embed_request_duration_seconds`, `embed_inference_duration_seconds`,
  `embed_rerank_request_duration_seconds`,
  `embed_rerank_inference_duration_seconds`,
  `embed_rerank_pool_acquire_duration_seconds`,
  `embed_rerank_tokenizer_duration_seconds`), batch-shape histograms
  (`embed_batch_size`, `embed_rerank_batch_size`,
  `embed_rerank_pairs_per_request`, `embed_batch_tokens`,
  `embed_batch_padding_waste_ratio`,
  `embed_rerank_padding_waste_ratio`), gauges
  (`embed_queue_depth_current`, `embed_rerank_in_flight`,
  `embed_build_info`). All histograms have suffix-matched bucket
  configs in `metrics::init`.

## Environment (live prod values)

| Variable                     | Prod value | Notes |
|------------------------------|------------|------|
| `EMBED_PORT`                 | `8082`     | |
| `EMBED_MODELS`               | `multilingual-e5-large:/models:1024:256:1:false,jina-code-v2:/models-jina:768:512:0:false` | Format: `name:dir:dim:max_len:pad_id:has_tti[:model_file]` |
| `EMBED_DEFAULT_MODEL`        | `multilingual-e5-large` | |
| `EMBED_INTRA_THREADS`        | `4` | ONNX threads per inference |
| `RERANKER_MODELS`            | `gte-multi-rerank:/models-gte-rerank:256:true` | Format: `name:dir:max_len:padded` |
| `RERANKER_INTRA_THREADS`     | `2` | |
| `RERANKER_SESSION_POOL_SIZE` | `2` | |
| `BATCHING_ENABLED`           | `true` | |
| `BATCH_MAX`                  | `32` | Coalesced batch cap |
| `BATCH_MAX_TOKENS`           | `16384` | Token-budget cap (TEI-style, real limiter) |
| `BATCH_WAIT_MS`              | `30` | Coalescing window |
| `MAX_QUEUE_SIZE`             | `256` | Queue cap → 503 |
| `CACHE_MAX_ENTRIES`          | `10000` | moka embedding cache |
| `DRAIN_TIMEOUT_S`            | `10` | SIGTERM drain window |
| `ORT_DYLIB_PATH`             | `/usr/lib/libonnxruntime.so` | required by `ort` with `load-dynamic` |
| `ORT_OPT_LEVEL`              | `3` | |
| `EMBED_VERSION`              | `dev` | stamped into `embed_build_info` |
| `EMBED_ARENA_MAX_MEM_BYTES`  | TBD (default 6442450944 = 6 GiB) | BFCArena hard ceiling; 3→6 GiB bump fixes jina-code-v2 92% error rate (FU-24) |
| `EMBED_ARENA_INITIAL_CHUNK_BYTES` | TBD (default 1048576 = 1 MiB) | First BFCArena allocation block size |
| `EMBED_ARENA_MAX_DEAD_BYTES` | TBD (default 33554432 = 32 MiB) | Dead-bytes threshold for chunk reuse (aggressive vs ORT default 128 MiB) |
| `EMBED_ARENA_EXTEND_STRATEGY` | TBD (default 1 = kSameAsRequested) | 0=kNextPowerOfTwo, 1=kSameAsRequested |

## Deploy

```bash
cd ~/deploy/krolik-server
docker compose build embed-server        # code-only: ~40s (BuildKit cache mounts)
docker compose up -d --no-deps --force-recreate embed-server
```

- `--no-cache` only when `Cargo.toml` / `Cargo.lock` change; regular code hits
  the dummy-main dep layer (`Dockerfile` Layer 1). Cold deps: ~3 min. Warm: ~2s.
- Auto-deploy: push to `main` → dozor webhook (`~/.dozor/deploy-repos.yaml`,
  repo `anatolykoptev/ox-embed-server`).
- **Releases:** release-please on push to `main`. Conventional commits →
  auto-PR + tag. Do not tag manually.

## Gotchas

- `ort` + `load-dynamic` → `libonnxruntime.so` loaded from `ORT_DYLIB_PATH` at
  startup, not linked at build.
- `model_optimized.onnx` (graph-fused via `onnxruntime.transformers.optimizer`)
  is **slower** than `model_quantized.onnx` on ARM Neoverse-N1 for our
  BERT-family models. Do not switch `MODEL_FILE` without benchmarking.
- `ort-sys` downloads the ORT binary on first compile (~30s cold, reused after).
- Batcher has `carry: Option<Item>` — items that would overflow `BATCH_MAX`
  defer to the next batch (not dropped). If log shows `item exceeded
  max_batch after coalesce`, running image predates commit `3598b48` — rebuild.
  Visible as `embed_carry_events_total` in `/metrics`.
- `BATCH_MAX` historical rule was `≤8` on ARM (cache thrash). Prod now runs
  `32` with `BATCH_MAX_TOKENS=16384` as the real cap — re-bench with `bench.py`
  if you change either.

## References

- **Benchmarks:** `docs/benchmarks/` (ARM Neoverse-N1 baselines).
- **Bugs / historical workarounds:** `docs/BUGS.md` (BUG-001 resolved Apr 2026).
- **Roadmap / phase plans:** `docs/ROADMAP.md`, `docs/plans/`.
- **Runbook:** `docs/runbook.md`.
