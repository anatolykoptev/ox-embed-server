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
| **multiple of 4 (chosen for `memory_pattern=false`)** | **24.0 / 24.1 / 25.0** |
| multiple of 1 (no rounding), Apple silicon only | 5.9 (vs 5.7 for 4, 6.6 for 8, 6.2 for 16, 12.1 for power of two; two rounds) |

The 32 and 16 rows come from an earlier single-round sweep on a quiet host
(power of two 48.8, 4: 25.3, 8: 27.9 in that round) and show the same ordering:
the less padding, the faster, so the "power-of-two tiles are cheaper" assumption did
not hold on this CPU.

### Which rule, when

The pad rule follows the model's `memory_pattern` (`pool::seq_pad_multiple`,
published as `embed_seq_pad_multiple{model}`): multiple of 4 with the pattern off
(production), multiple of 16 with it on (the code default). With the pattern on
ORT keeps one memory plan per distinct shape and the shared arena has a hard cap,
so the shape count is bounded at the source: a 256-token model sees at most 16
distinct widths instead of 64. Measured with the pattern on, 200 single-text
requests of random length on a 4-core Neoverse-N1 host (RSS MiB; the host was
heavily loaded by other tenants, so latency from that session is not reported):

| rule | after load | after 32 / 64 / 96 / 128 / 160 / 192 / 200 requests | errors |
|---|---|---|---|
| power of two (old) | 1384 | 1410 / 1410 / 1410 / 1410 / 1410 / 1410 / 1410 | 0 |
| multiple of 16 (chosen for the pattern on) | 1384 | 1406 / 1411 / 1411 / 1411 / 1410 / 1410 / 1410 | 0 |

RSS is flat once the widths have been seen and the arena counters record no OOM.
Multiples of 8 and 4 were not measured with the pattern on because they are not
used there.

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
That is fine for ranking; a near-duplicate threshold set close to 1.0 should be
checked against it.

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

Measured on the bare bench host, for completeness. **Not a recommendation:** an
earlier trial of higher intra-op thread counts in production showed no gain.

Min seconds, quieter round (load ~2.9):

| setting | 16 rerank docs x 1000 ch | 128 e5 sentences |
|---|---|---|
| 2 intra threads | 4.68 | 24.5 |
| 3 intra threads | 3.47 | - |
| 4 intra threads | 3.34 | 16.2 |
| e5 pool 2, one request split over both sessions (2 threads each; bench `--par 2`, not shipped code) | - | 16.1 |

On a busy host (round 1, load 5) the same rerank rows were 7.5 / 4.5 / 6.4 s for
2 / 3 / 4 threads, i.e. no reliable gain. Splitting one request across the two e5
sessions matched 4 threads on one session, so the pool code was not changed.

## Not changed

- Precision: both on-disk graphs are already int8 dynamic-quantised
  (`DynamicQuantizeLinear` + `MatMulInteger` nodes; ORT fuses them at load); no
  further quantisation was tried, so there is no precision comparison to report.
- `RERANK_MAX_INPUT_DOCS` is already configurable (default 32).
