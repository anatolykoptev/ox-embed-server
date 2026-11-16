# A vector depends on its batch-mates (2026-11-16)
Harness: `examples/embed_batch_drift.rs`, which calls `StandaloneEmbedder::infer`
exactly as the worker does. Model: `multilingual-e5-large` (`model_quantized.onnx`,
int8), ORT 1.24.3, `EMBED_MEMORY_PATTERN_MULTILINGUAL_E5_LARGE=false`, 3 intra-op
threads, ARM Neoverse-N1 host shared with other workloads. Targets: 48 deterministic
texts of 80..1100 characters (about 20..250 tokens), built from this repo's docs.
Reference vector: the target embedded alone (a request of one text). Every figure
is the cosine between that reference and the target's vector inside a bigger
request.

## Mechanism
The graph has 96 `DynamicQuantizeLinear` and 144 `MatMulInteger` nodes: each int8
matmul quantises its input with a scale derived from the min/max of the **whole**
tensor, `[batch, seq, hidden]`. The attention mask and the masked mean pool already
exclude pad rows, so padding is not what moves a vector; another text with larger
activations widens the range and changes the rounding of every text in the tensor.
Evidence from the table: a shorter or identical mate leaves the vector bit-for-bit
unchanged (cosine 1.00000), a long mate moves it.

## Measured (default settings, sub-batching ratio 2.0)
| request around the target | n | mean cos | p05 | min |
|---|---|---|---|---|
| same single-text request twice (determinism) | 12 | 1.00000 | 1.00000 | 1.00000 |
| `[t, t]` | 48 | 1.00000 | 1.00000 | 1.00000 |
| `[t, 17-char text]` | 48 | 1.00000 | 1.00000 | 1.00000 |
| `[t, ~1100-char text]` | 48 | 0.99417 | 0.99180 | 0.99082 |
| 4 texts, random mates 80..1100 chars | 48 | 0.99400 | 0.99218 | 0.99139 |
| 8 texts, random mates, draw A | 48 | 0.99372 | 0.99197 | 0.99152 |
| 8 texts, random mates, draw B | 48 | 0.99351 | 0.99167 | 0.99070 |
| same target, draw A vs draw B | 48 | 0.99327 | 0.99098 | 0.98992 |

The 32-text and 128-text shapes did not finish inside the run budget on the shared
host; the 128-short-sentence shape is in
[2026-11-14-rerank-embed-latency.md](2026-11-14-rerank-embed-latency.md) (mean
0.9931, min 0.9891) and agrees with the rows above: the drift saturates around
0.993 once the batch holds a long text.

Consequence for the response cache: the same text embedded in two different
requests differs by about 0.7% cosine, and the cache serves whichever was computed
first.

## The switch
`EMBED_BATCH_INVARIANT_<MODEL>=1` (or `EMBED_BATCH_INVARIANT=1`) runs every text of
a request in its own forward pass. Same harness, switch on: every row above is
1.00000 (mean, p05 and min), including draw A against draw B. A lone text is
unchanged by the switch, because a lone text already ran alone.

Cost, `bench_infer embed`, 32 texts per request, 3 threads, min of 3 runs on the
shared host (not interleaved, so read it as "no visible penalty", not as a speed-up):

| request | default (s) | invariant (s) |
|---|---|---|
| 32 mixed-length texts | 19.08 | 15.32 |
| 32 texts of 1000 characters | 20.27 | 19.74 |

Consistent with the earlier finding that latency is linear in padded tokens: one
pass per text pads each text to its own width (a multiple of 4) instead of to the
longest text of its group, which offsets the loss of batching. Requests of many very
short texts were not measured; those are the shape where per-pass overhead would
show first.

The switch is off by default because it changes the vectors of every multi-text
request relative to what a running deployment has stored. Turn it on together with a
re-embed of anything that compares vectors across requests.
