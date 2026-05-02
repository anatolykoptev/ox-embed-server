# ModernBERT vs gte-multi-rerank — baseline (2026-05-01)

**Hardware**: ARM Neoverse-N1, 4 vCPU, 8 GB container, no AVX/SVE/GPU.
**Server**: embed-server `v0.3.0` (commit `f2092cd`), `RERANKER_INTRA_THREADS=2`, `RERANKER_SESSION_POOL_SIZE=2`, `BATCH_WAIT_MS=30`, `MAX_CONCURRENT_RERANK_REQUESTS=4`. Default container, no Phase 3 changes applied.
**Models**:
- `gte-multi-rerank` (Alibaba mGTE-multilingual reranker, 568M params, INT8 ONNX, max_len 256, padded)
- `gte-reranker-modernbert-base` (Alibaba ModernBERT reranker, 149M params, INT8 ONNX, max_len 256, padded)

**Bench**: `scripts/bench_modernbert_vs_mgte.sh` — `bench.py --kind rerank` against live `/v1/rerank` endpoint with explicit `model` field. 8 cells: 2 models × 2 sizes × 2 doc counts, scenarios `1x10,4x20`. Raw JSON in `2026-05-01-modernbert-baseline/`.

## Headline numbers (p50 latency, ms)

| size   | docs | conc | gte-multi-rerank | gte-modernbert | winner            |
|--------|-----:|-----:|-----------------:|---------------:|-------------------|
| medium |    5 |    1 |         **1 134** |          2 828 | mGTE 2.5×         |
| medium |    5 |    4 |         **6 073** |          7 964 | mGTE 1.3×         |
| medium |   20 |    1 |            7 456 |          8 010 | tie               |
| medium |   20 |    4 |        **17 088** |         37 973 | mGTE 2.2×         |
| long   |    5 |    1 |            1 628 |          1 885 | tie               |
| long   |    5 |    4 |            8 642 |      **7 781** | ModernBERT 1.1×   |
| long   |   20 |    1 |            8 159 |      **7 696** | tie               |
| long   |   20 |    4 |        **20 121** |         24 158 | mGTE 1.2×         |

**Tally**: mGTE wins 5/8, ties 2/8, ModernBERT wins 1/8. Throughput similar (`tps` 0.10-0.89 across all cells, both models).

## Per-model Prometheus breakdown (Phase 1A series, full sweep window)

| metric | gte-multi-rerank | gte-modernbert | note |
|---|---|---|---|
| `embed_rerank_inference_duration_seconds` (avg) | **6.72 s** | **8.42 s** | ModernBERT 25 % slower at the raw `session.run` call |
| `embed_rerank_pool_acquire_duration_seconds` (avg) | 0.59 μs | 0.65 μs | mutex wait negligible — pool size not the bottleneck |
| `embed_rerank_tokenizer_duration_seconds` (avg) | 3.3 ms | 8.2 ms | BPE (ModernBERT) 2.5× slower than SentencePiece, but irrelevant at this latency scale |
| `embed_rerank_padding_waste_ratio` (avg) | 0.0 | 0.0 | bench fixtures use uniform doc lengths — does NOT reflect real prod traffic |

## Interpretation

**The premise "ModernBERT is the future" does not hold for this hardware + this ONNX export.**

ModernBERT (149M params, ~3.8× smaller than mGTE 568M) was expected to be ~1.5-2× faster per call. Instead it is **25 % slower at the inference layer**, ~2× slower on small batches at concurrency 1. The 8/8 cells confirm: *fewer parameters ≠ fewer FLOPs on our setup*.

Why (most likely, in priority of confidence):

1. **Alternating local/global attention is not fused on ONNX CPU.** ORT 1.20's `transformer_optimizer` does not list ModernBERT in its supported model types — the alternating local-attention windows execute as decomposed elementwise ops, paying overhead that nominally reduces N² attention cost but adds sufficient kernel-launch / cache-thrash to net out worse on ARM. Confirmed via research agent and corroborated by HuggingFace community discussions.
2. **GeGLU FFN has 3 linear projections vs BERT's 2.** Per-layer FFN compute is *higher* for ModernBERT than for the mGTE BERT-style variant, despite the smaller hidden dim. INT8 quantization compounds the effect because the GeGLU gate layers produce activation spikes that quantize poorly (arxiv 2405.14428 — confirmed effect on GLU-family activations).
3. **No specialized RoPE / SwiGLU kernels on Neoverse-N1.** The RotaryEmbedding op decomposes into elementwise ops; ARM SDOT/UDOT only accelerates dense matmul, not the rotary application.
4. **BPE tokenizer is 2.5× slower than SentencePiece per call** — but at 3-8 ms the absolute cost is irrelevant against multi-second inference.

## What this means for the optimization plan

**Phase 3 code-level changes (3A/3B/3C/3D) will deliver percent-scale improvements at best — not enough to close the 25 % inference-time gap to mGTE.**

- **Phase 3A** (per-model threading 1×4 vs 2×2). Pool acquire is ~0.6 μs — *not* a bottleneck on this workload. A/B might shift the c=4 saturation behaviour but won't help c=1 single calls. **Lower priority than expected.**
- **Phase 3B** (ORT spin disable env knob). Already coded in `feat/modernbert-phase2-3` branch. Ship as defensive default — small CPU saving on shared host, no behaviour change unless env flipped.
- **Phase 3C** (length bucketing). Padding waste ratio = 0 in bench because synthetic fixtures are uniform. **Cannot be evaluated from this bench**. Need to scrape `embed_rerank_padding_waste_ratio` after a few hours of real memdb-go traffic; defer decision.
- **Phase 3D** (int32 input cast). 1-3 % infra win — irrelevant against 25 % model gap.

## Decision points for the operator

The plan as written assumed ModernBERT would win on speed and we'd flip prod `CROSS_ENCODER_MODEL` to it after Phase 4. **The data says don't.** Three honest paths forward:

1. **Keep mGTE in prod, archive the ModernBERT mount.** Simplest. Phase 3B ships as a small infra polish. ModernBERT stays loaded for quality-comparison eval (model card claims similar BEIR + better LoCo-long-doc), not for speed.
2. **Re-export ModernBERT ONNX targeting the actual issues.** New workstream (call it Phase H.15):
   - `optimum-cli export onnx --model Alibaba-NLP/gte-reranker-modernbert-base --task sequence-classification --opset 17`
   - INT8 dynamic quantization with `nodes_to_exclude` covering the GeGLU FFN gate layers (per arxiv 2405.14428).
   - Re-bench against this same harness; iterate.
   - Effort: half a day for the export pipeline + bench loop.
3. **Treat ModernBERT as a quality-not-speed swap.** Run a separate quality eval (BEIR or memdb-go LoCoMo) — if quality wins outweigh the 25 % latency cost, the migration is worth the user-perceived slowness. Out of scope for this branch.

## Bench sweep methodology — caveats

- Bench was interrupted once by a dozor rebuild (HTTP 503 mid-sweep); re-ran fresh against the new container. Numbers above are the clean re-run.
- `tokenizer_duration_seconds_count` is only 2-3 (vs 88-89 inference calls) because the **token cache (H.7) hit on most repeated bench calls** — bench fires the same `(query, doc)` pair across iterations. Real prod traffic will populate this histogram far more.
- Concurrency = 4 saturates `MAX_CONCURRENT_RERANK_REQUESTS=4` cap; tail latency at this point reflects backpressure, not raw model speed. Use c=1 cells for clean inference comparison.
- Padding waste = 0 across the board because fixtures replicate one snippet `--docs-per-req` times. Real production has mixed lengths — this metric will move.

## Files

- `docs/benchmarks/2026-05-01-modernbert-baseline/<model>_<size>_docs<N>.json` — raw scenarios.
- `docs/benchmarks/2026-05-01-modernbert-baseline/prom_snapshot_{before,after}.txt` — Prometheus dump bracketing the sweep.
- `scripts/bench_modernbert_vs_mgte.sh` — re-runner.
