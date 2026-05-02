# ModernBERT Reranker Optimization Plan

**Date**: 2026-05-01
**Branch**: `feat/modernbert-optimization`
**Target model**: `Alibaba-NLP/gte-reranker-modernbert-base` (149M params, INT8 ONNX, max_len 256, padded)
**Baseline model**: `Alibaba-NLP/gte-multilingual-reranker-base` (305M, mGTE)
**Hardware**: ARM Neoverse-N1, 4 vCPU, 8 GB container, no AVX/SVE/GPU

## Goal

Make `gte-reranker-modernbert-base` faster than `gte-multi-rerank` in production while preserving rank quality, and produce a defensible benchmark report driven by Prometheus telemetry — not vibes.

## Relationship to Phase H roadmap

This plan continues `docs/ROADMAP.md` Phase H. Items already shipped (carried in this branch's prep commit):

- **H.1** — `RERANKER_BATCH_MAX` env (separate from embed `BATCH_MAX`)
- **H.2** — Zero-alloc tokenize on rerank hot path
- **H.3** — `MAX_CONCURRENT_RERANK_REQUESTS` semaphore (TEI pattern)
- **H.4** — `gte-reranker-modernbert-base` model swap (149M, INT8 ONNX)
- **H.7** — Tokenizer cache (`src/token_cache.rs`)

This plan picks up the deferred items, with one critical insertion: **observability first**. Phase H deferred H.5 / H.6 / H.14 with the pretext "defer until measurable padding waste appears" — but those measurements don't exist (rerank path has zero latency / padding metrics). Phase 1 of this plan creates that visibility, breaking the dependency cycle.

Mapping this plan → Phase H:
- This plan's **Phase 1** (observability) — NEW. Prerequisite for all deferred H items.
- This plan's **3A/3B/3D** (per-model threads, spin-disable, int32 cast) — NEW (not in H).
- This plan's **3C** = **H.5** (length-sort batch packing).
- Future follow-up (out of scope here): H.6 (fused dispatch), H.14 (IO binding) — re-evaluate after Phase 4 numbers.

## Anti-myths (from research agent)

1. **Unpadded sequence packing for ModernBERT requires `flash_attention_2` (CUDA-only).** Not available on ONNX CPU. TEI on CPU also runs padded (Candle). Skip.
2. **ORT transformer optimizer does NOT natively support ModernBERT** (no fusion for alternating local/global attention, RoPE, GeGLU). Falling back to `--model_type bert` produces partial/broken fusions. Skip.
3. **INT4 not supported on Neoverse-N1** for matmul (no dotprod kernel). INT8 is the floor; SDOT/UDOT supported.

## Current code state (from code-audit agent)

Decent:
- Padding done to `max(len_in_batch).min(max_len)`, not to global `max_len` — already efficient.
- `token_type_ids` and `position_ids` not fed to ONNX session — RoPE-friendly.
- Tokenizer agnostic via `tokenizers` crate — handles BPE (ModernBERT) and SentencePiece (mGTE) the same.
- Cross-encoder pairs batched as one `[N, max_seq]` tensor in one `session.run()`.

Missing / wrong:
- Zero per-request metrics for `/v1/rerank` (no counters, no latency histograms).
- `record_inference()` exists but never called for reranker.
- No tokenizer-time, pool-acquire-time, in-flight-count metrics.
- No session input introspection at startup.
- embed-server NOT in `prometheus.yml` — Prometheus is not scraping it at all.
- `bench.py` does not support `/v1/rerank` endpoint.
- CLAUDE.md falsely claims "Duration histograms NOT exposed" (they are registered, just not called for rerank).

## Phases

### Phase 0 — Branch setup + commit prep work [DONE in this commit]

- Branch `feat/modernbert-optimization` from `main`.
- Commit existing H.7 token-cache work as prep commit (~400 LOC, cohesive feature).
- Write this plan as durable artifact.

### Phase 1 — Observability (drives all later decisions)

**1A. Reranker metrics (`src/metrics.rs` + `src/api_rerank.rs` + `src/model_reranker.rs`)**

Add and emit:
- `embed_rerank_request_duration_seconds{model}` — histogram, end-to-end HTTP
- `embed_rerank_inference_duration_seconds{model}` — histogram, ONNX `session.run()` only
- `embed_rerank_tokenizer_duration_seconds{model}` — histogram, `spawn_blocking(tokenize_pairs)`
- `embed_rerank_pool_acquire_duration_seconds{model}` — histogram, mutex wait on session pool
- `embed_rerank_pairs_per_request{model}` — histogram, `documents.len()` distribution
- `embed_rerank_batch_size{model}` — histogram, actual batch passed to `score_pairs`
- `embed_rerank_padding_waste_ratio{model}` — histogram, `1 - real_tokens/(batch*max_seq)`
- `embed_rerank_in_flight{model}` — gauge, currently-processing requests
- `embed_rerank_requests_total{model, status}` — counter

**1B. Session introspection (`src/model_reranker.rs`)**

At `RerankerModel::load`, log `session.inputs()` names and warn if set ≠ `{input_ids, attention_mask}`. One-time startup log.

**1C. Prometheus scrape (`~/deploy/krolik-server/compose/observability.yml` or `prometheus.yml`)**

Add embed-server `:8082/metrics` to scrape jobs. Reload Prometheus.

**1D. bench.py rerank support**

- New `--kind rerank` flag emitting Cohere-shape payload.
- New `--docs-per-req N` for batch-size sweeps.
- New `--size long` fixture (≥400 token docs).
- JSON output mode (`--json`) for CI / diffing.

**1E. Fix stale CLAUDE.md comment** — `Duration histograms NOT exposed` is false. Update.

**Phase 1 verification gate:**
- `cargo build --release` clean.
- Local `cargo test` passes.
- Spin up container, hit `/v1/rerank` 10×, scrape `/metrics`, confirm new series populate with non-zero values for both `gte-multi-rerank` and `gte-modernbert`.
- Prometheus `embed-server` job is `UP`.

### Phase 2 — Baseline measurements

Run `bench.py` on dev/prod against both reranker models:
- Sweep `docs_per_req ∈ {1, 4, 16, 32}` × concurrency `∈ {1, 4, 10}` × `size ∈ {short, medium, long}`.
- Capture: p50/p95/p99 wall latency, throughput, and Prometheus snapshots of new series.
- Write to `docs/benchmarks/2026-05-01-modernbert-baseline.md`.

**Decision gates** (drive Phase 3 priorities):
- If `embed_rerank_pool_acquire_duration_seconds` p95 > 100ms at c=4 → pool size is the bottleneck, raise it before changing threads.
- If `embed_rerank_tokenizer_duration_seconds` > 20% of `request_duration` → tokenizer-bound, ModernBERT arch wins won't matter until tokenize is parallelized.
- If `embed_rerank_padding_waste_ratio` median > 0.4 → length bucketing is high-priority.
- Else → threading config + spin-disable is the lever.

### Phase 3 — Targeted optimizations

Sequenced; each one independently testable.

**3A. Per-model threading config**

Refactor `RERANKER_INTRA_THREADS` / `RERANKER_SESSION_POOL_SIZE` to be **per-model** (extend `RERANKER_MODELS` format or add `RERANKER_THREADS_<model>` overrides). Ship `gte-modernbert` as `intra=4 sessions=1`, keep `gte-multi-rerank` as `intra=2 sessions=2`. Rationale: ModernBERT cross-encoder is dominated by large MatMuls — single-session intra-parallelism beats inter-session contention on 4 cores.

**3B. ORT spin disable**

Set `intra_op_thread_pool_allow_spinning=0` (or `OMP_WAIT_POLICY=PASSIVE` if applicable) on the session — already PASSIVE in env, verify it actually propagates to ORT's intra pool, not just OMP.

**3C. Length bucketing in batcher (`src/batcher.rs`)**

Sort items in a coalesced batch by token length before dispatching. Within the existing token-budget cap (`BATCH_MAX_TOKENS`), grouping similar-length pairs reduces the `max_seq` of each forward pass and cuts padding waste. Conservative implementation: stable sort by length, no batch splitting — only intra-batch reordering. Out-of-order responses already supported via the request-ID/oneshot pattern.

**3D. int32 input_ids**

Cast `input_ids` and `attention_mask` from `i64` → `i32` before `session.run()` if the ONNX graph accepts int32 inputs (verify via Phase 1B introspection log). 1-3% overhead reduction.

**Skipped from priority list (require model artifact regen, not code):**

**3E. Re-export ONNX with optimum-cli + INT8 excluding GeGLU FFN gate layers** — separate workstream, deferred. Current `model_quantized.onnx` (140MB) was already replaced from `.int8.bak` (151MB) on 2026-05-01, so a recipe for this re-export should be documented in `docs/runbook.md` for next iteration.

### Phase 4 — Validation + ship

- Re-run Phase 2 bench harness with all Phase 3 changes applied.
- Diff vs baseline; require regression-free numbers for `gte-multi-rerank` (it's the safe path).
- Require ≥20% p95 improvement OR equivalent latency at higher throughput for `gte-modernbert`.
- If gates pass: `gh pr create`, request review, controller merges.
- If gates fail: revert specific Phase 3 changes, document negative result in benchmark doc.

## Out of scope

- Embedder swap (`multilingual-e5-large` and `jina-code-v2` stay as-is).
- SPLADE changes.
- Replacing `ort` crate or ONNX runtime.
- GPU.
- Re-quantization (deferred to follow-up; documented).

## Risk register

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Per-model threading refactor breaks existing gte-multi-rerank | Medium | Keep current env vars as fallback when per-model override absent. Bench gte-multi-rerank in Phase 2 baseline AND Phase 4 re-bench. |
| Length bucketing reorders responses incorrectly | Medium | Use existing oneshot/request-ID infrastructure (already supports out-of-order). Add unit test that bucketed batch returns correct scores per request. |
| int32 cast breaks ONNX session if graph expects int64 | Low | Phase 1B introspection logs input dtypes. Gate cast on `input.element_type() == TensorElementType::Int32`. |
| Spin-disable hurts dedicated-CPU latency | Low | Make configurable via env (`ORT_ALLOW_SPINNING`, default 0). |
| Prometheus scrape adds load to embed-server | Negligible | 15s scrape interval, `/metrics` is cheap. |

## Commit cadence

One commit per sub-phase (1A, 1B, 1C, 1D, 1E, 3A, 3B, 3C, 3D), conventional commits. Bench results land as `docs/benchmarks/2026-05-01-modernbert-baseline.md` (Phase 2) and `2026-05-01-modernbert-final.md` (Phase 4).
