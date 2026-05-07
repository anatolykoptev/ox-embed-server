# BFCArena tuning — operator's reference

## Why this doc exists

Between 2026-04-29 and 2026-05-06, the embed-server BFCArena (CPU memory
allocator pool) was the subject of fourteen PRs. Each fixed a real production
incident, but the cumulative churn was a symptom of two architectural gaps:

1. **Magic numbers without observability.** Arena knobs (cap, dead-bytes,
   chunk-size, extend-strategy) shipped with hard-coded defaults but no
   metric to validate the choice.
2. **Memory-pattern × warmup × pool_size interaction was undocumented.**
   ORT's `enable_mem_pattern=1` caches an allocation plan from the FIRST
   inference; if warmup ran at the wrong shape, the first prod request at
   the right shape forced a replan + fallback extend. Combined with a
   shared arena across `pool_size>1`, peak memory doubled.

This document captures the model so the next operator does not re-derive it
from logs.

## Components

```
┌─────────────────────────────────────────────────┐
│ Process                                         │
│  ┌──────────────────────────────────────────┐   │
│  │ Shared CPU BFCArena (env-level)          │   │
│  │   cap = EMBED_ARENA_MAX_MEM_BYTES        │   │
│  │   extend = kSameAsRequested              │   │
│  │   dead-bytes = 64 MiB                    │   │
│  └──────────────────────────────────────────┘   │
│       ↑                ↑                ↑       │
│   ┌───┴───┐        ┌───┴───┐        ┌───┴───┐   │
│   │ e5    │        │ jina  │        │rerank │   │
│   │ pool  │        │ pool  │        │ pool  │   │
│   │ 1..N  │        │ 1..N  │        │ 1..N  │   │
│   │ each  │        │ each  │        │ each  │   │
│   │ owns  │        │ owns  │        │ owns  │   │
│   │ weights│       │ weights│       │ weights│  │
│   │ uses  │        │ uses  │        │ uses  │   │
│   │ shared│        │ shared│        │ shared│   │
│   │ arena │        │ arena │        │ arena │   │
│   │ for   │        │ for   │        │ for   │   │
│   │ scratch│       │ scratch│       │ scratch│  │
│   └───────┘        └───────┘        └───────┘   │
└─────────────────────────────────────────────────┘
```

Per-session resident: model weights only (~400 MiB e5, ~250 MiB jina,
~250 MiB rerank). All scratch (attention QKV, layer outputs, intermediate
matmul scratch) goes through the shared arena.

## Sizing rule

```
container_memory >= weights_total + arena_cap + safety_margin
weights_total    = sum(model_weight × pool_size for each model)
arena_cap        = max_attention_scratch × concurrent_inferences × headroom
max_attention_scratch ≈ B × num_heads × max_seq² × 4 bytes
```

For our prod fleet (e5+jina+rerank, pool_size=2 each):

| Term | Value | Notes |
|------|-------|-------|
| weights_total | ~1.8 GiB | (400+250+250) × 2 |
| max_attention_scratch (jina, B=1, S=512) | ~12 MiB | per inference |
| concurrent_inferences | 6 | 3 models × pool_size=2 |
| arena_cap (with 5× headroom for graph fanout) | ~6 GiB | ✅ matches current default |
| safety_margin | ~2 GiB | tokenizer, axum, OS pages |
| **container limit** | **~10 GiB** | matches compose `memory: 10240M` |

If you raise `pool_size`, `max_seq`, or add a new model, recompute or
expect arena OOM in `/encoder/layer.0/attention/self/Add` (the canonical
forensic location).

## Memory pattern × warmup gotcha

ORT `enable_mem_pattern=1` caches the per-tensor allocation plan from the
first `session.run()`. The plan covers EVERY scratch tensor in the graph,
sized by the first call's input shape.

Subsequent calls at a LARGER shape force a "replan" — ORT walks the graph
again, allocates new scratch from the arena via `BFCArena::Extend`. This
single allocation can be ≥ 1 GiB for a 24-layer BERT encoder at S=512.

The 2026-05-06 incident was exactly this: warmup at S=128, first prod at
S=512, single 1.258 GiB extend overflowed the 4 GiB arena cap.

**Rule:** every model's `EMBED_WARMUP_SEQ_LEN` (or `<MODEL>_WARMUP_SEQ_LEN`
override) MUST equal the model's `max_len`. Pad-once at startup, then the
mem_pattern cache covers every later request.

Per-model override format (added in `feat/per-model-seq-cap-and-warmup`):
```
EMBED_WARMUP_SEQ_LEN_MULTILINGUAL_E5_LARGE=256
EMBED_WARMUP_SEQ_LEN_JINA_CODE_V2=512
```
Convention: uppercase model name, replace `-` with `_`.

## Forensic metrics (added in PR #39)

Every observed knob has a metric. If you tune one, watch the corresponding
counter/histogram before and after.

| Metric | What | Use to check |
|--------|------|--------------|
| `embed_arena_max_mem_bytes` | gauge of cap | env knob took effect |
| `embed_arena_extend_total{model,bin_num}` | counter per BFCArena extend event | which bin extends are common; bin 20 = 1.25 GiB |
| `embed_inference_attention_scratch_bytes` | histogram of `B×heads×S²×4` | which inferences risk arena pressure |
| `embed_inference_peak_bytes` | histogram of RSS delta per inference | catches non-attention scratch surprises |
| `embed_batch_dimensions_total{model,bs_bucket,sl_bucket}` | counter of (B,S) tuples | which shapes prod actually sees |
| `embed_batch_token_budget` | histogram of `B × effective_seq_len` | sizing predictor |
| `embed_inference_failures_total{model,reason,bin_num}` | counter of arena_oom vs other | distinguish OOM from other inference errors |

If `bin_num=20` extends appear in production, immediately:
1. Check `embed_inference_attention_scratch_bytes` p99 — what shape triggered it?
2. Check `embed_arena_max_mem_bytes` vs `weights_total + max_attention × pool_size × headroom` — under-sized?
3. Check warmup logs — was warmup at full `max_len`?

## Knob reference (defaults in `src/arena.rs`)

| Env | Default | Effect |
|-----|---------|--------|
| `EMBED_ARENA_MAX_MEM_BYTES` | 6 GiB | hard ceiling. Above which `BFCArena::Extend` returns `Available memory of X is smaller than requested bytes of Y` |
| `EMBED_ARENA_INITIAL_CHUNK_BYTES` | 1 MiB | first BFCArena block |
| `EMBED_ARENA_MAX_DEAD_BYTES` | 64 MiB | unused-block cap before BFCArena reuses; lower = more reuse, higher = less fragmentation |
| `EMBED_ARENA_EXTEND_STRATEGY` | 1 = kSameAsRequested | 0 = kNextPowerOfTwo (legacy, doubles every extend; we tuned away from it) |
| `EMBED_WARMUP_SEQ_LEN` | 128 | global warmup cap; each `<MODEL>_WARMUP_SEQ_LEN` overrides per-model |
| `BATCH_MAX_SEQ` | 256 | per-batch `max(seq_len)` cap; outliers go into B=1 batches; each `BATCH_MAX_SEQ_<MODEL>` overrides per-model |
| `EMBED_SESSION_POOL_SIZE` | 1 | `N` independent ONNX sessions per model. Each session owns its own weight buffer; all share the arena. With `N>1`, arena pressure scales linearly. |

## Historical PR cascade (do not repeat)

| Date | PR | What | Why repeated |
|------|-----|------|--------------|
| 2026-04-29 | #19 | disable mem_pattern | unbounded growth fix |
| 2026-04-29 | #20 | shared arena kSameAsRequested | undocumented arena interaction |
| 2026-05-01 | #23 | cap 3 GiB | first explicit cap |
| 2026-05-01 | #24 | re-enable mem_pattern (Phase H.17) | reverted #19 — needed for latency |
| 2026-05-02 | #29 | TinyLFU → LRU | unrelated, came in same arena window |
| 2026-05-02 | #31 | revert mem_pattern + histogram buckets | warmup misalignment hit |
| 2026-05-05 | #32 | bump 3→6 GiB | jina ate cap |
| 2026-05-06 | #34 | V2 API + DisableCpuMemArena | upstream ORT bug in V1 CreateArenaCfg |
| 2026-05-06 | #35 | metrics ordering fix | gauges silent before #35 |
| 2026-05-06 | #36 | re-enable mem_pattern + bound warmup + max_seq cap | composite of #24+#31 lessons |
| 2026-05-06 | #37 | malloc_trim 100ms | glibc hoarding pages |
| 2026-05-06 | #38 | session pool + intra=2 | concurrent inference enable |
| 2026-05-06 | #39 | forensic metrics | what this doc references |
| 2026-05-06 | #40 | doc drift fix | CLAUDE.md MAX_DEAD_BYTES |
| 2026-05-06 | #41 | per-model knobs + solo overflow counter | per-model warmup_seq_len + BATCH_MAX_SEQ |

The pattern: each PR fixed a real symptom but lacked a model. After #39 +
this doc, the model is shared. Future arena work must reference both.

## When you must touch arena code

1. Read this doc first.
2. Grep for the knob you want to change in `src/arena.rs` and `src/main.rs`.
3. Before opening a PR, capture pre-change values for every metric in the
   table above.
4. Ship the change.
5. Re-capture metrics after deploy. If a histogram p99 moved by >2x or a
   bin_num counter that was zero now ticks, you broke something.
6. Update this doc with the lesson.
