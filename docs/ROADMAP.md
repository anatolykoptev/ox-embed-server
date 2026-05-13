# embed-server Roadmap

Status of the multi-model Rust inference sidecar on the `krolik` server. Updated as phases ship.

**Live**: `http://embed-server:8082` inside the Docker network, `127.0.0.1:8082` on host. Auto-deployed via dozor webhook on every push to `main`.

**Related doc**: `docs/plans/2026-04-18-embed-server-phase-2.md` — detailed, task-level implementation plan for Phases C–G.

---

## ✅ Shipped

### Phase 2 multi-process refactor — 2026-05-12 (PR #56 / #57 / #58 + follow-ups #59-#63)

Supervisor + N worker child processes (one per ONNX model). Each worker owns isolated BFCArena — resolves BUG-004 (jina-code-v2 92% error rate from arena fragmentation).

- IPC: postcard tagged-enum frames over UDS (length-prefixed, 64 MiB cap). Cancel-safe per-request connection (PR #62).
- WorkerSupervisor watchdog: auto-restart on exit (clean / SIGABRT 134 / SIGKILL 137 / OOM) with 2s→60s exponential backoff. `embed_worker_restart_total{model}` counter (pre-touched to 0).
- Parallel worker spawn via `tokio::spawn` (PR #61) — startup 3-5× faster (workers warm up in ~3.4 s parallel vs sequential).
- 3 routing paths: `/v1/embeddings` (PR #57), `/v1/rerank` + `/embed_sparse` (PR #58 Wave 2.4b). Legacy in-process path remains for `EMBED_MULTI_PROCESS=0` rollback.
- bincode → postcard (RUSTSEC-2025-0141, upstream unmaintained).
- Dockerfile ships both binaries (`embed-server` + `embed-worker`) via `cargo build --bins`.

Architecture: `docs/architecture/multi-process.md`. Implementation log: `docs/superpowers/plans/2026-05-12-multi-process-refactor.md`.

**Live**: prod compose has `EMBED_MULTI_PROCESS=1` since 2026-05-12. Disable: set to `"0"` + `docker compose up -d --no-deps --force-recreate embed-server` (byte-identical fallback).

**Memory cost**: combined RSS ~5.1 GiB (was ~1.6 GiB monolith). Phase 3.2 (lazy-load / skip in-process when multi_process=1) deferred — requires handler refactor; current overhead acceptable on 24 GiB ARM host.

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

---

## Phase H — Karpathy throughput sprint (2026-05-01 added)

Findings from competitive research (TEI, Infinity, vLLM) + ML expert review during memdb-go LoCoMo eval session. Server is **already best-in-class on ARM ONNX + INT8** per competitive audit; remaining gains come from *application-level* features and *model-level* swaps, NOT batcher microopts.

### H.1 — `RERANKER_BATCH_MAX` env (separate from embed `BATCH_MAX`) ✅ shipped 2026-05-01
- **Why**: shared `batch_max=8` cap blocked rerank coalescing — 5-doc payloads instantly hit cap, killed concurrent throughput
- **Where**: `config.rs` field + `main.rs` reranker batcher init (~25 LOC)
- **Default**: `4 × batch_max` so single env (`BATCH_MAX=8`) preserves embed cache-thrash mitigation while reranker uses 32-64

### H.2 — Tokenize zero-alloc on rerank hot path ✅ shipped 2026-05-01
- **Was**: `Vec<(String, String)>` with `query.to_string() + d.clone()` per doc (2N allocs/call)
- **Now**: `Vec<(&str, &str)>` via `From<(I1, I2)> for EncodeInput` blanket impl
- **Where**: `model_reranker.rs:tokenize_pairs` (~10 LOC)
- **Impact**: ~15% tokenize-phase speedup, zero-risk

### H.3 — `MAX_CONCURRENT_RERANK_REQUESTS` semaphore (TEI pattern) ✅ shipped 2026-05-01
- **Why**: load-shed at HTTP layer with 429 BEFORE tokenizer CPU is spent on requests that will time out upstream
- **Where**: `AppState.rerank_semaphore: Option<Arc<Semaphore>>` + `api_rerank.rs` permit acquire (~40 LOC)
- **Default**: `4` (matches truly-parallel rerank capacity on 4-core ARM)
- **Prior art**: HuggingFace TEI `Infer::try_acquire_permit`

### H.4 — `gte-reranker-modernbert-base` model swap ✅ shipped 2026-05-01
- **Why**: 149M params (3.8× smaller than gte-multi 568M) + 8192 max_seq_len (vs 256, eliminates silent doc truncation) + matches 1.2B Nemotron quality
- **Where**: `RERANKER_MODELS` env now lists both — gte-multi (legacy) + gte-modernbert
- **Files**: `models/gte-reranker-modernbert-base/{config,tokenizer,model_quantized}.{json,onnx}` (151MB INT8 from Alibaba's HF repo)
- **Switch**: set `CROSS_ENCODER_MODEL=gte-modernbert` in memdb-go env

### H.5 — Length-sorted batch packing (Infinity pattern) — TODO
- **Effort**: ~150 LOC + tests
- **What**: After coalesce window closes, drain queue non-blockingly, sort by `max_seq_len`, slice into homogeneous sub-batches before dispatch
- **Where**: `batcher.rs::run_worker` — refactor inner loop
- **Payoff**: 10-25% throughput on **mixed-length** workloads. For our memos (all clustered ~256 tokens) gain near-zero — defer until measurable padding waste appears
- **Prior art**: Infinity `CustomFIFOQueue.pop_optimal_batches`

### H.6 — Fused tokenize+forward в одном `spawn_blocking` — TODO
- **Effort**: ~60 LOC core + 200 LOC test rewrite
- **What**: Move tokenization OUT of `api_rerank.rs` handler, INTO batcher worker so single `spawn_blocking` does both. Eliminates tokio context-switch + L1 cache eviction between tokenize and forward
- **Payoff**: 15-30% single-text latency recovery (matches old ROADMAP B-polish estimate)
- **Risk**: API contract change in `DynamicBatcher::embed_tokens` — needs tests

### H.7 — Tokenizer cache (moka) — TODO
- **Effort**: ~60 LOC
- **What**: `LRU<Sha256<text>, Vec<u32>>` cached input_ids. Memdb-go D7 sub-queries against same docs hit immediately
- **Payoff**: 50ms tokenize → 0ms on cache hit. Estimated 30% rerank latency reduction on repeat-doc loads
- **Where**: extend `cache.rs` (currently embeddings-only) with token cache

### H.8 — Combined `/v1/retrieve_and_rerank` endpoint — TODO
- **Effort**: ~200 LOC
- **What**: Single endpoint accepts query + dense_emb + sparse_emb + candidate_docs → server does RRF fusion + CE rerank. Saves 3 HTTP RTT per memdb-go search
- **Payoff**: 30-100ms × D7 fanout (~3 sub-queries) = 90-300ms p95 reduction per chat turn
- **Risk**: API contract addition; backwards-compat trivial (new endpoint)

### H.9 — Late-interaction (ColBERT-style) endpoint — TODO
- **Effort**: ~400 LOC
- **What**: Score query × docs using stored token-level embeddings + light interaction matrix. NOT full cross-encoder
- **Payoff**: 10× faster than CE for −2pp quality. Drop-in for low-stakes rerank steps (D7 sub-queries)

### H.10 — Adaptive threshold response field — TODO
- **Effort**: ~30 LOC
- **What**: Add `confidence: f32` to rerank response (entropy of top-N scores). Memdb-go can skip docs below threshold without separate quality_floor heuristic
- **Payoff**: cleaner client API, removes duplicated logic

### H.11 — Batch rerank API `/v1/rerank/batch` — TODO
- **Effort**: ~100 LOC
- **What**: Accept array of `{query, documents}` tuples. Server fans out internally, batches efficiently
- **Payoff**: D7 sub-queries в одной HTTP call вместо 3 — saves RTT + lets batcher coalesce across queries

### H.12 — LLM-listwise rerank fallback — TODO
- **Effort**: ~150 LOC
- **What**: When CE quality_floor hit (top-1 < 0.05), call gemini-flash-lite for listwise rerank as fallback
- **Payoff**: Recover degraded queries with smarter model — narrow OOD English where gte struggles
- **Risk**: external LLM dependency, latency variance

### H.13 — gRPC endpoint — TODO (low priority)
- **Effort**: ~200 LOC
- **What**: tonic-based gRPC for `/v1/rerank` + `/v1/embeddings`
- **Payoff**: ~30% wire overhead reduction. Useful only if memdb-go QPS scales 10×

### H.14 — IO binding (ORT pre-allocated tensors) — TODO
- **Effort**: ~80 LOC
- **What**: `SessionInputValue` with reused backing buffers vs per-call malloc
- **Payoff**: 5-15% under sustained load (allocator pressure reduction on 4-core ARM)
- **Prior art**: TEI ORT backend

### Sequencing recommendation

**Sprint 1** (~4 h, immediate user-visible wins): H.7 tokenizer cache + H.10 adaptive threshold + H.11 batch API
**Sprint 2** (~8 h, throughput tier): H.5 length-sort + H.6 fused dispatch + H.14 IO binding
**Sprint 3** (~10 h, architecture): H.8 combined endpoint + H.9 late-interaction + H.12 LLM fallback
**Sprint 4** (≥15 h, hardware tier): H.13 gRPC + Phase F NER + Phase G SPLADE polish

---

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
