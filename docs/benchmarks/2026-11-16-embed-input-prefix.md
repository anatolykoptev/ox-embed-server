# e5 `query: ` / `passage: ` prefix: measured effect (2026-11-16)
`multilingual-e5-large` is trained on prefixed inputs: `query: ` for the search side,
`passage: ` for the indexed side (model card). Until now `input_type` reached the
model as nothing; `EMBED_INPUT_PREFIX_<MODEL>=e5` makes it select the prefix. This
page measures what the prefix is worth on this model and what the wrong prefix costs.

Setup: the production `model_quantized.onnx`, the worker inference path, subsets of
two public BEIR-format sets. Each subset holds the first 60 queries (sorted by id)
that have relevance judgements, every relevant document of those queries, and random
distractors up to 400 documents; documents are `title + text`, truncated by the model
to 256 tokens. Small on purpose (the host is shared): read the intervals, not the
point estimates.

| set, metric (queries / documents) | none / none (today) | query / passage (correct) | passage / passage (switch on, queries untagged) |
|---|---|---|---|
| SciFact (EN) nDCG@10 | 0.8684 | 0.8928 | 0.8595 |
| SciFact Recall@10 | 0.8833 | 0.9333 | 0.8833 |
| RuBQ (RU) nDCG@10 | 0.9641 | 0.9727 | 0.8193 |
| RuBQ Recall@10 | 0.9889 | 0.9889 | 0.8944 |

Paired bootstrap on per-query nDCG@10, difference against none / none (5000
resamples, 95% interval):

| set | variant | difference | interval | queries better / worse / same |
|---|---|---|---|---|
| SciFact | query / passage | +0.0245 | [-0.0267, +0.0819] | 5 / 4 / 51 |
| SciFact | passage / passage | -0.0089 | [-0.0381, +0.0153] | 2 / 2 / 56 |
| RuBQ | query / passage | +0.0086 | [0.0000, +0.0195] | 3 / 0 / 57 |
| RuBQ | passage / passage | -0.1448 | [-0.2096, -0.0824] | 1 / 27 / 32 |

Reading:
- The correct prefix pair helps a little and never hurts here, but on these subsets
  the gain is inside the noise for SciFact and borderline for RuBQ. The model keeps
  most of its quality without prefixes.
- The wrong prefix is the clear risk. With the switch on, a caller that sends queries
  without `input_type: "query"` gets `passage: ` on its queries, which cost 0.145
  nDCG@10 on RuBQ. Turning the switch on is only safe together with callers that tag
  queries.
- A text that already carries `query: ` or `passage: ` is not prefixed again, so
  callers that prefix themselves are unaffected either way.
