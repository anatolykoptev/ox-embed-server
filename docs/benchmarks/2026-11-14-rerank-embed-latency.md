# Rerank and embed latency: padding, sub-batching, threads (2026-11-14)

Harness: `examples/bench_infer.rs`, which calls `StandaloneReranker::score` and
`StandaloneEmbedder::infer` exactly as the worker does (tokenization included).
Corpus: deterministic, built from this repo's docs plus a Russian passage, so
token density is that of mixed EN/RU text. Models: `gte-multi-rerank`
(`model_quantized.onnx`, int8) and `multilingual-e5-large` (`model_quantized.onnx`,
int8), ORT 1.24.3, `EMBED_MEMORY_PATTERN_MULTILINGUAL_E5_LARGE=false`.

Host: ARM Neoverse-N1, 4 cores, **shared** with other workloads (1-minute load
2.5-6 during the runs). Absolute seconds are therefore pessimistic and noisy and
are not production latencies; the ratios between interleaved runs are the result.
`min` of N timed iterations is the estimator (contention only ever adds time).

## Where the time goes

The rerank worker already scores all pairs of a request in one batched forward
pass, and the embedder already pads to the batch and sub-batches by length. The
latency is compute: int8 GEMMs over `batch x padded_seq` tokens, so it is linear in
the number of padded tokens. Wins come from fewer padded tokens or more cores,
not from more batching.

## Embedder: pad width

Tensor width was the longest sequence rounded up to the next power of two
(33 tokens ran as 64). 128 short sentences, 2 intra threads, interleaved, 3 rounds
(min seconds):

| width rule | min s (3 interleaved rounds) |
|---|---|
| power of two (before) | 48.2 / 48.8 / 48.1 |
| multiple of 32 | 36.1 (one run) |
| multiple of 16 | 29.4 (one run) |
| multiple of 8 | 26.9 / 26.5 / 26.4 |
| **multiple of 4 (chosen)** | **24.0 / 24.1 / 25.0** |

The 32 and 16 rows come from an earlier single-round sweep on a quiet host
(power of two 48.8, 4: 25.3, 8: 27.9 in that round) and show the same ordering:
the less padding, the faster, so the "power-of-two tiles are cheaper" assumption did
not hold on this CPU.

Embedding quality, measured on the same host (128 sentences, seed 3, old build vs
new build, cosine per vector):

| comparison | min cos | mean cos |
|---|---|---|
| old vs new, embedded as one batch of 128 | 1.0000000 | 1.0000000 |
| old vs new, each text embedded alone (batch of 1) | 0.9922 | 0.9986 |
| old: batch of 128 vs each text alone (existing behaviour) | 0.9891 | 0.9931 |
| new: batch of 128 vs each text alone | 0.9901 | 0.9931 |

The int8 dynamic quantisation takes each activation scale from the whole tensor,
so pad rows move a vector only when the tensor holds little else: in a batch the
real tokens of the other rows dominate the range and the new width changes nothing,
but a lone short text (old: padded to 32 or 64; new: to the next multiple of 4)
shifts by about 0.1% cosine on average, 0.8% at worst. That is smaller than the
batch-composition drift the old code already has (0.993 mean between a text embedded
in a batch and alone). If vectors computed before this change are stored next to
vectors computed after it, expect that order of difference for single-text requests.

## Before / after (2 intra threads, pool 1)

`min` seconds over two interleaved rounds, old = `origin/main`, new = this branch.
Round 2 of `new` ran under heavier host load (load 6), so cells where `new` is
slower than `old` on uniform lengths (where the code path is identical) are noise.

| scenario | old | new | change |
|---|---|---|---|
| rerank 16 docs x 200 ch | 2.02 | 2.03 | same path |
| rerank 16 docs x 1000 ch | 4.73 | 4.75 | same path |
| rerank 32 docs x 200 ch | 4.34 | 4.27 | same path |
| rerank 32 docs x 1000 ch | 9.64 | 10.06 | same path (noise) |
| rerank 32 docs, mixed 80-1400 ch | 9.77 | 8.94 | -8% |
| rerank 32 docs, bimodal 200/1000 ch | 9.87 | 7.00 | -29% |
| embed 32 texts, mixed lengths | 24.4 | 23.6 | -3% |
| embed 32 texts x 1000 ch | 26.0 | 26.9 | same path (noise) |
| embed 128 sentences (120 ch) | 47.8 | 24.5 | **-49%** |
| embed 512 sentences (120 ch) | 193 | 147 | -24% |

Rerank documents that all reach the 256-token cap (the 1000-character rows) are one
band and one pass in both builds: nothing to gain there. The saving is for requests
that mix short and long documents.

## Rerank quality (sub-batching)

Measured on Apple silicon (same ORT build and model files as the N1 runs), not on
the N1 host. int8 dynamic quantisation derives each activation scale from the whole
tensor, so a pair's logit moves a little with its batch-mates and pad width, in the
old code and in the new. Five seeds x 32 docs, one query, `RERANKER_SUBBATCH_RATIO=1.0` (old
behaviour) against `2.0` (new), and each against scoring every pair alone
(no padding, no batch-mates):

| scenario | split vs single pass | single pass vs alone | split vs alone |
|---|---|---|---|
| uniform 1000 ch | identical (Spearman 1.0000) | 0.987 | 0.987 |
| mixed 80-1400 ch | Spearman 0.983 (min 0.955), top-5 overlap 4.6/5, top-1 same 5/5 | 0.980 | 0.987 |
| bimodal 200/1000 ch | Spearman 0.985 (min 0.979), top-5 overlap 4.4/5, top-1 same 3/5 | 0.988 | 0.989 |

The sub-batched scores sit as close to the unbatched reference as the single-pass
scores do (Spearman 0.987 vs 0.980 mixed, 0.989 vs 0.988 bimodal), so the change
adds no deviation beyond what batch composition already causes. Caveat: the corpus is
synthetic and mostly irrelevant to the query, which makes ordering fragile; a
labelled query set would be a stronger check. Acceptance used here: Spearman >= 0.95
and top-5 overlap >= 4/5 against the single pass on every seed.

## Threads (new build, solo requests)

Min seconds, quieter round (load ~2.9):

| setting | 16 rerank docs x 1000 ch | 128 e5 sentences |
|---|---|---|
| 2 intra threads | 4.68 | 24.5 |
| 3 intra threads | 3.47 | - |
| 4 intra threads | 3.34 | 16.2 |
| e5 pool 2, one request split over both sessions (2 threads each; bench `--par 2`, not shipped code) | - | 16.1 |

Raising intra threads helps a solo request by ~30% on a quiet host and nothing on a
busy one (round 1, load 5: 7.5 / 4.5 / 6.4 s for 2 / 3 / 4 threads). Splitting one
request across the two e5 sessions gives the same 16 s as 4 threads on one session,
so there is no case for changing the pool code over raising `EMBED_INTRA_THREADS` /
`RERANKER_INTRA_THREADS`; the cost is oversubscription when several workers are busy
at once, which is the operator's call.

## Not changed

- Precision: both on-disk graphs are already int8 dynamic-quantised
  (`DynamicQuantizeLinear` + `MatMulInteger` nodes; ORT fuses them at load); no
  further quantisation was tried, so there is no precision comparison to report.
- `RERANK_MAX_INPUT_DOCS` is already configurable (default 32).
