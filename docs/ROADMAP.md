# embed-server Roadmap

Status of the multi-model Rust inference sidecar on the `krolik` server. Updated as phases ship.

**Live**: `http://embed-server:8082` inside the Docker network, `127.0.0.1:8082` on host. Auto-deployed via dozor webhook on every push to `main`.

**Related doc**: `docs/plans/2026-04-18-embed-server-phase-2.md` — detailed, task-level implementation plan for Phases C–G.

---

## ✅ Shipped

### Phase 0 — Model artifacts on disk

| Model | Path | Size | Status |
|---|---|---|---|
| `multilingual-e5-large` | `models/multilingual-e5-large/` | ~560 MB | ✅ live (default `/v1/embeddings`) |
| `jina-code-v2` | `models/jina-code-v2/` | ~300 MB | ✅ live (`?model=jina-code-v2`) |
| `bge-reranker-v2-m3` | `models/bge-reranker-v2-m3/model_quantized.onnx` | 544 MB | ✅ live (`POST /v1/rerank`, Phase E) |
| `splade-v3-distilbert` | `models/splade-v3-distilbert/model.onnx` | 346 MB | ✅ on disk, **waiting for Phase G endpoint** |
| `gliner-small-v2.1` | `models/gliner-small-v2.1/pytorch_model.bin` | 583 MB | ⚠️ pytorch-only, needs re-export via `torch.onnx.export` |

### Phase A — Batcher warm-up (PR #3 + #5)
- Skip items whose client disconnected before dispatch (saves CPU under load-shedding).
- Explicit `ORT_OPT_LEVEL=3` (max ONNX Runtime graph optimizations).
- `auto_truncate=true` default — silently truncate >max_len inputs at the tokenizer layer. **Side-effect**: fixed a real correctness bug — pre-fix, overlong inputs lost the trailing `[SEP]` token during downstream clipping, producing subtly wrong embeddings.
- +1 metric `embed_batcher_cancelled_items_total`.

### Phase B — Token-budget batcher (PR #6 + #7)
- Pre-tokenize in `api.rs`; `Item` carries `Vec<Vec<u32>>` token ids.
- **Padded-model token accounting** (TEI `core/src/queue.rs` port): `total = max(max_len, entry_max) * (items + entry_items) < BATCH_MAX_TOKENS` (default 16384). Prevents the "1 long item + many shorts = 10× padding waste" antipattern.
- `BATCH_MAX` retained as soft item-count fairness cap.
- Tokenize moved to `spawn_blocking` (perf fix after initial regression).
- +3 metrics: `embed_batch_tokens`, `embed_batch_padding_waste_ratio`, `embed_carry_events_total`.

**Observed metrics (post-deploy)**: padding waste 5-10% — batcher correctly avoids long+short mixing.

### Phase E — `/v1/rerank` endpoint + BGE reranker integration ([PR #12](https://github.com/anatolykoptev/ox-embed-server/pull/12))
- `RerankerModel` struct (parallel to `EmbedModel`), reuses Phase B batcher semantics for `(query, doc)` pair coalescing.
- `RERANKER_MODELS=bge-reranker-v2-m3:/models-reranker:512:true` env parsing.
- Endpoint `POST /v1/rerank` — Cohere-compatible JSON: `{model?, query, documents, top_n?}` → `{model, results: [{index, relevance_score}]}` sorted DESC.
- Response cache (Phase D) is bypassed — query/doc pairs are near-unique; caching would burn RAM for ~0 hit rate.
- Unit + integration tests (`tests/rerank_smoke.rs`, guarded on `EMBED_SERVER_URL`).

**Observed**: Live `/v1/rerank` endpoint serving `bge-reranker-v2-m3`; smoke-verified relevant docs outrank irrelevant (5.77 vs -11.00 spread on "cat" query).

---

## 🔜 Next up

Rough effort = **focused implementation hours** (not calendar time — each phase adds CI wait + dozor build ≈ +5 min overhead).

### Priority 1 — ~5-7 h

#### `B-polish` — close Phase B hygiene
**Effort**: ~2-3 h
**What**:
- Relabel 3 remaining `invalid_request_error` → `server_error` in `api.rs` (server-side errors mis-advertised as client-side)
- Merge tokenize + forward into a single `spawn_blocking` per batch (possibly recovers the −15-30% single-text throughput we saw)
- Small LOW items: empty-item guard in `BatchAccum::fits`; `BATCH_MAX_TOKENS=0` rejection; file-level `#![allow(dead_code)]` → per-item; `retain`/`count` walk fusion; doc comments
- Regression test for spawn_blocking refactor

#### Phase D — Response cache
**Effort**: ~3-4 h
**What**: LRU `HashMap<(String model, [u8; 32] sha256), Vec<f32>>` in a new `src/cache.rs` module. Wire into `api.rs` ahead of the batcher: hit → return instantly, miss → normal path + populate. Env `CACHE_MAX_ENTRIES=10000` (default).
**Payoff**: MemDB re-queries the same memory search strings a lot — cache hit turns a ~200ms forward pass into a ~1ms lookup. Expected hit rate on production ≥15 %.
**Risk**: low — isolated module, pure memoization, deterministic embeddings.

---

### Priority 2 — ~7-10 h

#### MemDB integration of reranker (separate repo, `anatolykoptev/MemDB`)
**Effort**: ~3-4 h
**What**: In `memdb-go`, after retrieving top-50 by embedding, POST to `http://embed-server:8082/v1/rerank`. Flag `MEMDB_RERANKER_ENABLED` for safe rollout.
**Dependency**: Phase E shipped ✅.

#### Phase C — Length bucketing
**Effort**: ~4-6 h
**What**: Replace the single batcher with N sub-batchers keyed by token-length buckets (default `[64, 128, 256, 512]`). Each bucket coalesces only same-ceiling items, killing padding waste for heterogeneous workloads.
**Payoff**: Expected ≥20 % p99 reduction on mixed traffic (BucketServe paper claim ≈ −50 %). Complementary to Phase B — where B says "don't mix bad combinations", C says "never need to mix at all".

---

### Priority 3 — ~9-13 h

#### Phase G — `/v1/sparse` endpoint + SPLADE integration
**Effort**: ~4-6 h
**What**: `SparseModel` → token-weight map. Endpoint returns `{tokens: [[id, weight], ...]}`. Useful for hybrid retrieval.
**Payoff**: In go-code and MemDB, replace or complement the current keyword/BM25 half of hybrid RRF. +5-10 % recall on technical content (exact-term matching).

#### Phase F — `/v1/ner` endpoint + GLiNER integration
**Effort**: ~5-7 h (≈2 h just to re-export ONNX properly)
**What**:
- Export GLiNER to ONNX via `torch.onnx.export` (the `gliner.save_pretrained(save_onnx=True)` path only wrote pytorch weights)
- `TokenClassifierModel` struct
- Endpoint accepts arbitrary labels in the request; returns spans
**Payoff**: Auto-structure MemDB memories into `{people, projects, technologies, dates, decisions}` — memory becomes a graph.

---

## Cumulative sizing

| Scope | Hours | Calendar |
|---|---|---|
| P1 only (polish + cache) | 5-7 h | ~1 day |
| P1 + P2 (+ MemDB wire-up + bucketing) | 12-17 h | ~2 days |
| Everything through P3 | 21-30 h | ~4-5 days |

---

## Decision inputs

**Reasons to prioritize P1 first**:
- Smallest blast radius; closes reviewer findings; might recover the throughput we saw regress on bench.
- Unlocks clean 24 h production metric window to judge Phase B in the real world.

**Reasons to defer P3**:
- SPLADE + GLiNER are nice-to-haves; neither has a blocking downstream consumer yet.
- Better to ship P2 first, gather real usage, then decide if sparse/NER are worth the integration time.

---

## Non-goals (explicit, for this roadmap horizon)

- **GPU inference**. ARM CPU only; no CUDA / no TensorRT.
- **LLM-grade embedders** (e5-mistral-7b, nv-embed-v2). 7 B params don't run on this box.
- **Token-level late-interaction** (ColBERT / ColPali). High quality, but explodes vector storage 10×.
- **Multimodal** (CLIP, SigLIP, image/audio encoders). No retrieval use-case in the stack.
- **Continuous batching / PagedAttention**. vLLM tricks are for autoregressive decoders — not applicable to encoder-only single-forward inference.

---

## How this file is kept current

- Each shipped phase moves its entry from "🔜 Next up" to "✅ Shipped" with PR link and live-metric observations.
- Effort estimates updated post-phase to reflect actual hours, for better future forecasting.
- New follow-ups discovered during execution go to `TaskList` (ephemeral) or to `docs/plans/*` (persistent).

Last refreshed: 2026-04-17 (after Phase E ship).
