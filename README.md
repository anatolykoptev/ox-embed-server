# embed-server

Rust ONNX embedding sidecar. OpenAI-compatible `/v1/embeddings`, Prometheus
`/metrics`, Axum + `ort` + `tokenizers`. Replaces the previous Python
`embed-jina` service.

Port `8082`. Container name `embed-server`. See `CLAUDE.md` for internal
structure and operational notes.

## Models

Two models are loaded simultaneously in one process; pick per request via
the `model` field in the `/v1/embeddings` body.

| Name | Dim | max_len | Notes |
|---|---:|---:|---|
| `multilingual-e5-large` | 1024 | 256 | Default. General-purpose multilingual. |
| `jina-code-v2` | 768 | 512 | Code / long-context. ~2-3x faster on ARM Neoverse-N1. |

Observed deltas (ARM Neoverse-N1, 4 vCPU):

- 16 short texts, single batch: jina ~118 ms/req vs e5 ~357 ms/req.
- Concurrent single-query load @ conc=16: jina ~75 rps vs e5 ~28 rps.

Example:

```bash
curl -s localhost:8082/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"jina-code-v2","input":["fn main() {}"]}'
```

Omitting `model` uses `EMBED_DEFAULT_MODEL` (first entry of `EMBED_MODELS`
by default — currently `multilingual-e5-large`).

## Endpoints

- `POST /v1/embeddings` — OpenAI-compat. 503 + `Retry-After: 1` on full
  queue, 503 + `Retry-After: 5` during shutdown drain.
- `POST /v1/rerank` — Cohere-compatible cross-encoder reranking. See below.
- `GET /health` — `ok`.
- `GET /metrics` — Prometheus text exposition. Series: `embed_requests_total`,
  `embed_request_duration_seconds`, `embed_inference_duration_seconds`,
  `embed_batch_size`, `embed_queue_depth`, `embed_queue_rejected_total`,
  `embed_build_info`.

### `POST /v1/rerank` — cross-encoder reranking

Given a query and a list of documents, returns each document's relevance score. Typical RAG flow: retrieve top-50 via `/v1/embeddings` → rerank via `/v1/rerank` → return top-5 to the LLM.

Cohere-compatible JSON shape.

**Request:**
```json
{
  "model": "bge-reranker-v2-m3",
  "query": "what is a cat",
  "documents": ["a cat is a feline", "pasta is tasty", "cats purr"],
  "top_n": 2
}
```
- `model` — optional. If 1 reranker is configured, it's picked automatically. If 0 or 2+ rerankers, `model` is required.
- `query` — required non-empty string.
- `documents` — required non-empty array.
- `top_n` — optional; if absent, all docs are returned. `top_n=0` yields an empty array; `top_n > len(documents)` is saturated.

**Response:**
```json
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 0, "relevance_score": 5.81},
    {"index": 2, "relevance_score": 3.12}
  ]
}
```
- Results sorted by `relevance_score` DESCENDING.
- `index` is the 0-based position in the original `documents` array.
- Scores are raw logits (any real number) — higher = more relevant. Not a calibrated probability.

**Configure models via env:**
```
RERANKER_MODELS=bge-reranker-v2-m3:/models-reranker:512:true
```
Format: `name:dir:max_len:padded` (comma-separated for multiple).

**Example:**
```bash
curl -s -X POST http://127.0.0.1:8082/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-v2-m3","query":"what is a cat","documents":["a cat is a feline","pasta is tasty"]}'
```

**Notes:**
- The response cache (Phase D) is bypassed for `/v1/rerank` — query/doc pairs are near-unique and caching them would burn RAM for ~0 hit rate.
- Each `(query, doc)` pair is tokenized together by the cross-encoder and scored; batcher coalesces pairs across requests the same way embeddings do.

## Configuration

Environment variables (parsed in `src/config.rs`):

| Var | Default | Notes |
|---|---|---|
| `EMBED_PORT` | `8082` | Listen port. |
| `EMBED_MODELS` | required | `name:dir:dim:max_len:pad_id:has_tti` entries, comma-separated. |
| `EMBED_DEFAULT_MODEL` | first entry | Must match one of `EMBED_MODELS`. |
| `EMBED_INTRA_THREADS` | `4` | ONNX intra-op threads. |
| `BATCHING_ENABLED` | `false` | Set `true`/`1` to enable `DynamicBatcher`. |
| `BATCH_MAX` | `32` | Max coalesced texts per ONNX call. |
| `BATCH_WAIT_MS` | `10` | Coalesce window. |
| `MAX_QUEUE_SIZE` | `256` | Queue cap → 503. |
| `EMBED_MAX_WAITERS` | `8×pool_size` (floor 16) | Worker waiter-queue cap per worker. Set higher (e.g. `64`) to absorb bulk-indexing bursts without touching `EMBED_SESSION_POOL_SIZE`. |
| `DRAIN_TIMEOUT_S` | `10` | SIGTERM drain window. |
| `ORT_DYLIB_PATH` | `/usr/lib/libonnxruntime.so` | Required with `load-dynamic`. |
| `EMBED_VERSION` | `dev` | Stamped into `embed_build_info`. |
| `AUTO_TRUNCATE` | `true` | TEI-compat. Only literal `false` (case-insensitive) disables; `0`/`no`/`off`/empty leave it enabled. |

Current production `EMBED_MODELS`:

```
multilingual-e5-large:/models:1024:256:1:false,jina-code-v2:/models-jina:768:512:0:false
```

## Dynamic batcher

Enabled by `BATCHING_ENABLED=true`. A worker per model coalesces requests
that arrive within `BATCH_WAIT_MS` up to `BATCH_MAX` texts per ONNX call.

Carry-over (as of commit `3598b48`, Apr 2026): if a second item pulled from
the channel would push the already-coalesced batch past `BATCH_MAX`, it is
**deferred** to the next batch via an internal `carry: Option<Item>` in
`run_worker` rather than being rejected. The previous behaviour returned
`item exceeded max_batch after coalesce` and dropped the item. That error
should no longer occur on `master`; if it appears in logs, the running
image was built before `3598b48` — rebuild and redeploy. Regression test:
`coalesce_overflow_defers_to_next_batch` in `src/batcher.rs`.

`BATCH_MAX` stayed at 32; throughput is unchanged by the fix.

Tuning:
- On ARM Neoverse-N1 without AVX/SVE, `BATCH_MAX > 8` can be slower than
  smaller batches due to attention cache thrash. Benchmark before raising.
- `BATCH_WAIT_MS` trades latency for coalescing; 10 ms is a reasonable
  default for mixed workloads.

## Deploy

```bash
docker compose build embed-server
docker compose up -d embed-server
```

Code-only rebuilds reuse the dummy-main dep layer (~40 s). Use `--no-cache`
only when `Cargo.toml`/`Cargo.lock` change (~3 min cold).

## Releases

Release-please workflow (`.github/workflows/release-please.yml`, config in
`release-please-config.json` + `.release-please-manifest.json`, seed
`0.1.0`) opens and maintains a release PR driven by Conventional Commits on
`master`:

- `fix:` / `perf:` → patch bump
- `feat:` → minor bump
- `feat!:` / `BREAKING CHANGE:` in body → major bump
- `docs:` / `chore:` / `ci:` / `refactor:` / `test:` / `build:` → no bump

Cutting a release: merge the open release-please PR. That commit creates
the `vX.Y.Z` tag and GitHub release with the generated changelog, and bumps
the version in `Cargo.toml` and the manifest. No manual tagging.

First release PR (#1) targets `v0.2.0`.
