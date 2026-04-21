# Phase F — `/v1/ner` endpoint design

**Status:** design (approved in brainstorm, not yet planned).
**Supersedes:** ROADMAP.md Phase F entry; previous `scripts/export_gliner.sh` attempt.
**Related:** `docs/plans/2026-04-18-embed-server-phase-2.md`, `docs/ROADMAP.md`.

## 1. Motivation

embed-server ships three endpoints today: `/v1/embeddings`, `/v1/rerank`, `/embed_sparse`. Phase F adds the fourth: **`/v1/ner`** — open-set Named Entity Recognition powered by GLiNER through [`gline-rs 1.0`](https://crates.io/crates/gline-rs) and ONNX Runtime.

### Consumers

| Service | Use case | Text shape |
|---|---|---|
| MemDB (primary) | auto-structure memory entries into `{people, projects, technologies, dates, decisions}` | short, mixed |
| go-search | NER over SERP snippets | short, variable |
| go-nerv | news / narrative paragraphs | medium |
| go-code | commit messages, PR descriptions, docstrings | short, technical |

All four are internal, Rust- or Go-based, live in the same Docker network.

### Why now

The previous export attempt (`scripts/export_gliner.sh`) called `GLiNER.save_pretrained(save_onnx=True)` — a method that only wrote `pytorch_model.bin`, no ONNX file. `deploy/.../models/gliner-small-v2.1/` has been stuck in a half-exported state since 17 Apr. This spec skips local export entirely and pulls a pre-exported ONNX artifact from the `onnx-community` HF org, which is maintained by HF Staff and referenced by `gline-rs` as a tested source.

### Empirical validation (2026-04-21)

Pre-flight experiment in `/tmp/gliner-test` confirmed the stack works:

- `gline-rs 1.0.1` + `ort 2.0.0-rc.9` compiles clean on aarch64-linux.
- `onnx-community/gliner_small-v2.1` INT8 (175 MB) loads in ~1.3 s, infers at 35 ms/text for 2 × 6 labels.
- Entities extracted with 87–95% confidence on persons/companies/locations/dates.

Large model (`gliner-multitask-large-v0.5` INT8, 648 MB) is the production pick — scaled latency estimate ~120 ms/text based on parameter-count ratio; actual to be measured during implementation.

## 2. Goals and non-goals

### In scope (v1)

- `POST /v1/ner` handler with per-request labels (open-set NER).
- Single large ONNX session sized to fit within shared-host RAM budget.
- Label-set-keyed batcher that coalesces requests with identical `labels[]`.
- Pre-exported ONNX deployment flow (no local export step).
- Prometheus metrics aligned with existing `embed_*` naming.
- Warmup at startup to eliminate cold-start penalty on first request.
- Graceful shutdown integrated with existing `drain_batchers` flow.

### Out of scope (deferred)

- GPU inference. Target is CPU-only ARM Neoverse-N1.
- Session pool. Env slot `NER_SESSION_POOL_SIZE` is reserved (default `1`) so a future PR can add pool support without schema break; code path is single-session only.
- Pre-computed label embeddings. Chosen model is UniEncoder Token-mode; this optimization requires a BiEncoder variant (`gliner-x-large`). Not worth the model switch in v1.
- Tokenized-labels LRU cache. Tokenizer is ~1% of request time — rearranging deck chairs while forward pass is the bottleneck. Add only if `embed_ner_tokenize_seconds` histogram flags it.
- Per-label threshold map. Single global `threshold` is enough for v1.
- Fast-path `gliner-small` route alongside the large model. Add if metrics later show that large-model latency is a bottleneck for a specific client.

## 3. Architecture decisions

### D1 — Dispatch: label-bucket DynamicBatcher + single session

**Decision:** one ONNX session (~1.2 GB RSS for large INT8) fronted by a new `NerBatcher` that routes requests into buckets keyed by `hash(sorted(labels))`. Each bucket coalesces items up to `NER_BATCH_MAX` (items) or `NER_BATCH_MAX_TOKENS` (token-budget) before flushing.

**Rationale:**

- **Architectural constraint.** GLiNER concatenates labels with text inside the encoder; two requests with different `labels[]` cannot share a forward pass. Label-bucketing is a necessary condition for any batching, not an optimization.
- **Compute arithmetic.** Transformer attention is `O(N² · hidden_dim)` per forward. On a large model, a coalesced 8-text forward takes ~200 ms (not 8 × 120 ms). This is where throughput comes from.
- **Session pool arithmetic on this host.** Pool of 2 = ~2.4 GB RSS; pool of 4 = ~4.5 GB. Current host already has 9 GB of swap in use and shared with 28 containers. A pool would force cache reclaim on other services. Single session is the right default.
- **CPU contention.** 4 CPUs. A pool of N divides `intra_op_num_threads` N-ways, making each session slower than a single session with full thread count. Pool wins only when N clients run truly in parallel *and* cores exceed sessions. We don't have that.

**Throughput target:** ~40–50 texts/s sustained, single session, batch of 8. Under estimated 4-client steady-state nagatie that is 3–5× headroom.

**Escape hatch:** `NER_SESSION_POOL_SIZE=2` env slot reserved; implementing it is a separate PR (~2 h) using the pattern already in `model_reranker.rs` if metrics ever show saturation.

### D2 — API shape: Cohere-like, consistent with `/v1/rerank`

**Decision:** request/response mirror the `{model, data: [{index, ...}]}` pattern used by `/v1/rerank` and `/embed_sparse`.

#### Request

```json
{
  "model": "gliner-multitask-large-v0.5",
  "texts": ["Anatoly works at Oracle.", "Phase F ships tomorrow."],
  "labels": ["person", "company", "date"],
  "threshold": 0.5
}
```

- `model` — optional when exactly one NER model is configured; required (400 otherwise) when multiple. Same disambiguation contract as `/v1/rerank` and `/embed_sparse`.
- `texts` — non-empty array. Each entry non-empty and non-whitespace-only (400 otherwise).
- `labels` — non-empty array (400 otherwise — open-set NER without labels is undefined).
- `threshold` — optional, float in `[0.0, 1.0]` (400 outside range), default `0.5` (GLiNER's own default).

#### Response 200

```json
{
  "model": "gliner-multitask-large-v0.5",
  "data": [
    {
      "index": 0,
      "entities": [
        {"text": "Anatoly", "label": "person",  "score": 0.93, "start": 0,  "end": 7},
        {"text": "Oracle",  "label": "company", "score": 0.88, "start": 18, "end": 24}
      ]
    },
    {
      "index": 1,
      "entities": [
        {"text": "tomorrow", "label": "date", "score": 0.71, "start": 15, "end": 23}
      ]
    }
  ]
}
```

- `data[].index` preserves original `texts[]` position.
- `data[].entities` sorted by `score` descending (same convention as `/v1/rerank`'s `results[]`).
- `entities[].start`/`end` are byte offsets into the original text (UTF-8 safe by construction — `gline-rs` works in bytes).
- `score` is float in `[0.0, 1.0]`, post-threshold.

#### Error shape

Existing `ErrorResponse` type: `{"error": {"message": "...", "error_type": "..."}}`. No new variant needed. `error_type` values: `invalid_request_error`, `server_error`, `rate_limited`.

### D3 — Label processing: warmup only

**Decision:** at startup, after `NerModel::load()`, run one dummy inference with a representative label-set and a canned sentence to compile the ORT graph and prime arena allocators. Pattern copied from `RerankerModel::warmup()`. No per-request caching.

**Defaults:**

- Warmup labels: `person,organization,location,date` (configurable via `NER_WARMUP_LABELS`, comma-separated).
- Warmup text: hardcoded single sentence inside `model_ner.rs` — short, contains one of each default label type. No env variable for this; it's a one-line change to tune.

**Why not more:**

- **Pre-computed label embeddings:** architecturally unavailable for UniEncoder models. Would require switching to `gliner-x-large` BiEncoder — different model, needs separate quality validation, net win on our workload estimated 5–10% of request latency. Not worth it v1.
- **Tokenized-labels cache:** tokenizer cost is 1–2 ms vs. ~150–200 ms for forward pass. Sub-percent optimization; cargo-cult if added without evidence.

## 4. Components and changes

| Path | Type | LOC (est) | Role |
|---|---|---|---|
| `src/model_ner.rs` | new | ~250 | `NerModel`: `gline-rs` TokenMode wrapper, tokenization, `inference()`, `warmup()` |
| `src/api_ner.rs` | new | ~280 | HTTP handler for `POST /v1/ner`, request/response types, validation, error paths |
| `src/batcher_ner.rs` | new | ~450 | `NerBatcher`: label-bucket DynamicBatcher variant. Derives from existing `DynamicBatcher` where possible |
| `src/config.rs` | modify | +~80 | `NER_MODELS` parser, `NER_*` env knobs, defaults |
| `src/types.rs` | modify | +~40 | `NerEntry { model, batcher }`, `NerModelDef`, extends `AppState` |
| `src/main.rs` | modify | +~40 | load loop (mirror reranker), route wire-up, batcher into `drain_batchers` |
| `src/metrics.rs` | modify | +~30 | `embed_ner_*` counter/histogram/gauge registrations |
| `tests/ner_smoke.rs` | new | ~80 | env-guarded `EMBED_SERVER_URL` integration test, mirror of `tests/rerank_smoke.rs` |
| `Cargo.toml` | modify | +~2 | add `gline-rs = "1"` |
| `Dockerfile` | modify | +~2 | note `/models-ner` mount point |
| `~/deploy/krolik-server/docker-compose.yml` | modify | +~7 | mount `./models/gliner-multitask-large-v0.5` → `/models-ner`, add `NER_*` env vars |
| `~/deploy/krolik-server/models/gliner-multitask-large-v0.5/` | add (external) | — | pre-exported ONNX + tokenizer from `onnx-community` |
| `CLAUDE.md` (embed-server) | modify | +~10 | document new endpoint + env vars in the existing tables |
| `docs/ROADMAP.md` | modify | +~5 | mark Phase F as shipped, add observed metrics post-deploy |
| `scripts/export_gliner.sh` | delete | — | superseded by HF pull; no longer needed |
| **Net new code** | | **~1200 LOC** | incl. tests |

All source files stay within CLAUDE.md's 200-line soft target by organizing along concern boundaries (model / api / batcher / types / config each distinct).

## 5. Data flow

```
Client → POST /v1/ner
   │
   ▼
api_ner::ner_handler
   │  - validate non-empty texts, non-empty labels, threshold ∈ [0,1]
   │  - resolve model name (implicit if 1, explicit required if ≥2)
   │  - check state.shutdown → 503 if cancelled
   │  - for each text: enqueue (labels_hash, text) into NerBatcher, hold oneshot receiver
   │
   ▼
NerBatcher worker (spawned at model load)
   │  loop {
   │    pick a bucket with pending items + elapsed wait_ms OR full budget
   │    drain items up to NER_BATCH_MAX / NER_BATCH_MAX_TOKENS
   │    spawn_blocking:
   │      model.inference(bucket.labels, texts) -> Vec<Vec<Entity>>
   │    send each sub-result back via the oneshot channels
   │  }
   │
   ▼
api_ner::ner_handler
   │  - collect Vec<Entity> per text index
   │  - filter by threshold
   │  - sort entities by score desc
   │  - construct {model, data: [{index, entities}, ...]}
   │
   ▼
JSON response
```

Shutdown path: `main::drain_batchers` receives the `NerBatcher` alongside existing batchers, all drain concurrently within `DRAIN_TIMEOUT_S`.

## 6. Error handling

| Condition | HTTP | `error_type` | Notes |
|---|---|---|---|
| `texts` empty / contains blank | 400 | `invalid_request_error` | |
| `labels` empty / missing | 400 | `invalid_request_error` | |
| `threshold` outside `[0.0, 1.0]` | 400 | `invalid_request_error` | |
| `model` not found | 400 | `invalid_request_error` | |
| `model` missing with ≥2 NER models configured | 400 | `invalid_request_error` | |
| shutdown in progress | 503 | `rate_limited` | `retry-after: 5` |
| queue full (`NER_MAX_QUEUE_SIZE` reached) | 503 | `rate_limited` | `retry-after: 1` |
| ONNX inference failure | 500 | `server_error` | |
| `spawn_blocking` panic / batcher worker died | 500 | `server_error` | tracing::error with stack |

All error responses go through the existing `error_json()` helper in `types.rs`. No new error infrastructure.

## 7. Observability

Metric namespace: `embed_ner_*`, registered in `metrics.rs`.

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `embed_ner_requests_total` | Counter | `model`, `status` | throughput + error rate |
| `embed_ner_duration_seconds` | Histogram | `model` | end-to-end latency |
| `embed_ner_entities_extracted_total` | Counter | `model`, `label` | downstream signal — which labels MemDB/others actually use |
| `embed_ner_bucket_count` | Gauge | `model` | label-set diversity; alert if unexpected fragmentation |
| `embed_ner_queue_depth` | Gauge | `model` | queue pressure, precursor to 503 |
| `embed_ner_batch_items` | Summary | `model` | coalescing effectiveness |
| `embed_ner_batch_tokens` | Summary | `model` | token-budget utilization |
| `embed_ner_carry_events_total` | Counter | `model` | TEI-batcher carry-over signal |
| `embed_ner_tokenize_seconds` | Histogram | `model` | trigger data for deciding if tokenized-labels cache is worth adding |

Structured JSON logs via `tracing` at:

- `info`: model load, warmup complete, shutdown start/drain complete.
- `warn`: queue near-full threshold crossed, batcher Arc still shared on drain.
- `error`: ONNX inference failure, worker panic, tokenizer error.

## 8. Testing strategy

### Unit (in-process, no HTTP server)

- **Request deserialization:** minimal `{texts, labels}` parses; full `{model, texts, labels, threshold}` round-trips; extra fields rejected/ignored per serde policy.
- **Model resolution:** zero / one / ≥2 NER models — assertions on implicit/explicit contract.
- **Threshold edge cases:** `0.0`, `1.0`, `< 0`, `> 1`, default fallback.
- **`labels_hash` determinism:** `["a","b"]` and `["b","a"]` must produce the same bucket key (requires `sort` before `hash`).
- **`NerBatcher`:** bucket creation on first request, coalescing under `NER_BATCH_MAX`, flush by `BATCH_WAIT_MS` timeout, carry-over on item-exceeds-remaining-budget, queue-full 503 semantic.
- **Graceful shutdown:** `NerBatcher::shutdown(timeout)` drains current bucket before exit, does not cut mid-forward.

### Integration smoke

- `tests/ner_smoke.rs`: env-guarded on `EMBED_SERVER_URL`. POST a known sentence with known labels. Assert ≥ one entity extracted at expected `(text, label, score ≥ 0.6)`. Pattern: mirror `tests/rerank_smoke.rs`.

### Manual post-deploy validation

- 100 sequential requests with varied label sets → measure p50/p99 latency, `bucket_count`, `batch_items` distribution.
- Quality spot-check: 10 representative MemDB memories → manually verify entity quality vs expectations.

## 9. Configuration (new env vars)

| Variable | Default | Format / notes |
|---|---|---|
| `NER_MODELS` | (empty) | `name:dir:max_len:mode`, e.g. `gliner-multitask-large-v0.5:/models-ner:384:token`. Empty → `/v1/ner` route not registered. |
| `NER_DEFAULT_MODEL` | (empty) | Optional when exactly one model; required when multiple. |
| `NER_INTRA_THREADS` | `4` | `ort` `intra_op_num_threads` per session. |
| `NER_SESSION_POOL_SIZE` | `1` | Reserved; only `1` supported in v1. Parser accepts any value but errors if > 1 (forward-compat). |
| `NER_BATCH_MAX` | `16` | Soft items-per-forward cap. |
| `NER_BATCH_MAX_TOKENS` | `8192` | Token-budget cap. Real limiter under TEI-style accounting. |
| `NER_BATCH_WAIT_MS` | `20` | Coalescing window. |
| `NER_MAX_QUEUE_SIZE` | `256` | 503 trigger. |
| `NER_WARMUP_LABELS` | `person,organization,location,date` | Comma-separated, used once at startup. |

## 10. Deployment

1. Pull pre-exported model into `~/deploy/krolik-server/models/gliner-multitask-large-v0.5/`:
   ```bash
   # via huggingface-cli (install: pipx install huggingface-hub[cli])
   huggingface-cli download onnx-community/gliner-multitask-large-v0.5 \
     --include "onnx/model_quantized.onnx" "tokenizer.json" "tokenizer_config.json" \
               "config.json" "gliner_config.json" "special_tokens_map.json" "added_tokens.json" \
     --local-dir ~/deploy/krolik-server/models/gliner-multitask-large-v0.5
   ```
   Expected layout: `onnx/model_quantized.onnx` (648 MB) + tokenizer files at the root.
2. Update `~/deploy/krolik-server/docker-compose.yml`: add volume mount + `NER_*` env section.
3. Build + force-recreate:
   ```bash
   cd ~/deploy/krolik-server
   docker compose build --no-cache embed-server
   docker compose up -d --no-deps --force-recreate embed-server
   ```
4. Smoke:
   ```bash
   curl -sS -X POST http://127.0.0.1:8082/v1/ner \
     -H 'Content-Type: application/json' \
     -d '{"texts":["Anatoly works at Oracle from St. Petersburg."], "labels":["person","company","location"]}'
   ```
5. Clean up legacy:
   - Delete `deploy/krolik-server/models/gliner-small-v2.1/pytorch_model.bin` (stale half-export).
   - Remove `scripts/export_gliner.sh` from embed-server repo.
   - Strike Phase F "⚠️ pytorch-only" row in `docs/ROADMAP.md`; add observed post-deploy metrics.

Auto-deploy: push to `main` → dozor webhook → rebuild (standard flow).

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| INT8 quantization degrades NER precision vs FP32 | medium | Smoke test asserts ≥ expected entity with `score ≥ 0.6`. Fallback: switch `NER_MODELS` to point at `model_fp16.onnx` (884 MB). |
| Cold-start delay on first production request after deploy | low (with warmup) | `NerModel::warmup()` at boot, same pattern as reranker. |
| Label-set fragmentation across clients → bucket explosion | medium | Coordinate common taxonomy across MemDB/go-*. Alert on `embed_ner_bucket_count` > 20 (suggests clients send arbitrary labels). |
| `tokenizers` crate version drift between embed-server and `gline-rs` | **medium** | embed-server pulls `tokenizers 0.22.x` via existing deps; `gline-rs 1.0` pins `tokenizers 0.21.x`. Cargo will link both copies by default (major version mismatch = separate crates, code size ~+5 MB). Option 1: accept dual-version cost (simplest). Option 2: downgrade embed-server's transitive `tokenizers` to 0.21 via `[patch.crates-io]` (risky; might break existing api/rerank/splade). Recommendation in implementation plan: Option 1 unless binary-size is a concern. |
| `gline-rs` panic on malformed input | low | `api_ner` handler wraps inference in `spawn_blocking` + join error handling, same pattern as `api_splade`. |
| Inference crashes under concurrent `NerBatcher` drain + new requests | low | Reuse `CancellationToken` gate; no in-flight requests accepted post-cancel (existing pattern). |

## 12. Decision log

- **Model:** `gliner-multitask-large-v0.5` INT8 over `gliner_small-v2.1`. Reason: quality on open-set labels (+10 F1 avg from published evals), disk/RAM budget fits (~1.2 GB RSS in chosen configuration).
- **Dispatch:** single session + label-bucket batcher (D-lean). Session pool rejected for v1 on RAM/CPU arithmetic for this host.
- **API:** Cohere-like `{model, data:[{index, entities}]}`, consistent with `/v1/rerank` and `/embed_sparse`.
- **Label optimization:** warmup only. Pre-compute embeddings architecturally unavailable on UniEncoder; tokenized-labels cache deferred until metrics warrant.
