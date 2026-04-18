# embed-server — Unified Rust ONNX Embedding Sidecar

**Rust** | Docker | OpenAI-compatible `/v1/embeddings` | Prometheus `/metrics`

Single sidecar serving both `multilingual-e5-large` (1024 dim) and
`jina-code-v2` (768 dim). Replaced the Python `embed-jina` Apr 2026.

## Structure

| File | Lines | Role |
|---|---:|---|
| `src/main.rs` | ~130 | Axum server, routes, signal handling, batcher construction |
| `src/api.rs` | ~150 | `/v1/embeddings` handler, request/response wiring |
| `src/types.rs` | ~50 | `AppState`, `ModelEntry`, error/response types |
| `src/model.rs` | ~140 | ONNX session + tokenizer + inference |
| `src/pool.rs` | ~90 | Token pooling (mean-pool + L2 normalize) |
| `src/batcher.rs` | ~240 | `DynamicBatcher` (tokio mpsc + oneshot, wait-window coalesce) |
| `src/metrics.rs` | ~80 | Prometheus exposition helpers |
| `src/config.rs` | ~120 | Env parsing (`EMBED_MODELS`, batching knobs) |
| `bench.py` | — | Load harness (rtk proxy for untruncated output) |

## Models

Two models loaded concurrently in one process. Request selects via
`model` field; `EMBED_DEFAULT_MODEL` (default: first `EMBED_MODELS`
entry) is used when absent.

| Name | Dim | max_len | Notes |
|---|---:|---:|---|
| `multilingual-e5-large` | 1024 | 256 | Default. Multilingual general-purpose. |
| `jina-code-v2` | 768 | 512 | Code / long-context. ~2-3x faster on ARM. |

Perf deltas (ARM Neoverse-N1): 16 short texts single batch — jina
~118 ms/req vs e5 ~357 ms/req. conc=16 single-query — jina ~75 rps
vs e5 ~28 rps.

## API

- `POST /v1/embeddings` — OpenAI-compat. Picks model by `model` field;
  returns 503 + `Retry-After: 1` when queue full, 503 + `Retry-After: 5`
  when shutting down.
- `GET /health` — plain `ok`.
- `GET /metrics` — Prometheus text exposition. Key series:
  `embed_requests_total{model,status}`, `embed_request_duration_seconds`,
  `embed_inference_duration_seconds`, `embed_batch_size`,
  `embed_queue_depth{model}`, `embed_queue_rejected_total{model}`,
  `embed_build_info{version}`.

## Environment

| Variable | Default | Notes |
|---|---|---|
| `EMBED_PORT` | `8082` | |
| `EMBED_MODELS` | required | `name:dir:dim:max_len:pad_id:has_tti[:model_file]` comma-separated. Current: `multilingual-e5-large:/models:1024:256:1:false,jina-code-v2:/models-jina:768:512:0:false` |
| `EMBED_DEFAULT_MODEL` | first model | Name |
| `EMBED_INTRA_THREADS` | `4` | ONNX threads per inference |
| `BATCHING_ENABLED` | `false` | `true` to enable DynamicBatcher |
| `BATCH_MAX` | `32` | Max texts per coalesced ONNX call. On ARM w/o AVX stay at 8 — larger triggers cache thrash |
| `BATCH_WAIT_MS` | `10` | Coalescing window |
| `MAX_QUEUE_SIZE` | `256` | Queue cap → 503 |
| `DRAIN_TIMEOUT_S` | `10` | SIGTERM drain window |
| `ORT_DYLIB_PATH` | `/usr/lib/libonnxruntime.so` | required with `load-dynamic` feature |
| `EMBED_VERSION` | `dev` | Stamped into `embed_build_info` |

## Deploy

```bash
cd ~/deploy/krolik-server
docker compose build embed-server        # code-only change: ~40s (BuildKit cache mounts)
docker compose up -d --no-deps --force-recreate embed-server
```

Use `--no-cache` only when deps (Cargo.toml/Cargo.lock) change — regular
code changes don't need it thanks to the dummy-main dep layer in the
Dockerfile. Code-only rebuild: ~40s. Warm no-change rebuild: ~2s. Cold
rebuild (after Cargo.toml): ~3 min.

## Batcher carry-over (commit `3598b48`, Apr 2026)

`run_worker` holds an internal `carry: Option<Item>`. When a second item
pulled from the channel would push the coalesced batch past `BATCH_MAX`,
it is deferred to the next batch instead of being rejected. Previous
behaviour replied with `item exceeded max_batch after coalesce` and
dropped the item; that error should no longer occur on `master`. If seen
in logs, the running image predates `3598b48` — rebuild and redeploy.
Regression test: `coalesce_overflow_defers_to_next_batch`.

## Releases (release-please)

`.github/workflows/release-please.yml` runs release-please on pushes to
`master`. Config: `release-please-config.json`
(`release-type: rust`, `include-v-in-tag: true`) and
`.release-please-manifest.json` (seed `0.1.0`). Conventional commit
prefixes: `fix:`/`perf:` → patch, `feat:` → minor, `feat!:` or
`BREAKING CHANGE:` → major, others → no bump. The bot opens and
updates a single release PR; merging it creates the `vX.Y.Z` tag,
GitHub release, and version bump in `Cargo.toml` + manifest. First
PR (#1) targets `v0.2.0`. Do not tag manually.

## Gotchas

- `ort` with `load-dynamic` — `libonnxruntime.so` loaded at startup, not linked.
- On ARM Neoverse-N1 without AVX/SVE, `BATCH_MAX > 8` is slower than smaller
  batches due to attention cache thrash. Keep at 8 unless benched otherwise.
- `model_optimized.onnx` (graph-fused via `onnxruntime.transformers.optimizer`)
  is SLOWER than `model_quantized.onnx` on this hardware for our BERT-family
  models. Do not switch MODEL_FILE without benchmarking.
- BUG-001 (ort crate 30 s slowdown for BERT w/ token_type_ids) does NOT fire
  for our jina-code-v2 file — inputs are only `[input_ids, attention_mask]`.
  See `docs/BUGS.md`.
- First-time compile: `ort-sys` downloads the ORT binary at build time; adds
  ~30 s on cold cache. Re-uses after first build.

## Benchmarks (Oracle ARM Neoverse-N1, 4 vCPU, 2 GB RAM)

| Concurrency | e5-large p50 | jina-code-v2 p50 |
|---|---|---|
| 1 | — | 1.7 s |
| 4 | — | 2.4 s (coalesced) |

See `docs/benchmarks/` for baseline + post-migration details.

## History

- Apr 2026: Phase 2 — unified Rust sidecar absorbed jina-code-v2 previously
  served by Python `embed-jina`. BUG-001 workaround proven unnecessary for
  the current model file. Added DynamicBatcher, Prometheus metrics, bounded
  queue + 503 backpressure, graceful SIGTERM. Python `embed-jina` retired.
- Apr 2026: batcher carry-over fix (`3598b48`) — coalesce-overflow items
  deferred instead of dropped. Added release-please workflow (`d217c60`);
  first release PR targets `v0.2.0`.
