# Multi-shape Static ONNX Export for ModernBERT Reranker

**Date**: 2026-05-02
**Branch**: `feat/static-multishape`
**Builds on**: PR #27 (commit `c5ac856`, Phase H.20 single-shape static fast-path)
**Target model**: `Alibaba-NLP/gte-reranker-modernbert-base` (ModernBERT, 149M, INT8 ONNX, max_len 256, padded)
**Hardware**: ARM Neoverse-N1, 4 vCPU, 8 GB container

## Why this exists

PR #27 ships a static-shape ONNX session pool for the reranker, fixed to
shape `[1, max_len]`. Operators drop `model_quantized_static.onnx` next
to `model_quantized.onnx` and the loader picks it up — no env config.
The static graph runs ~1.74× faster standalone vs the dynamic graph
because `optimum-cli ... --no-dynamic-axes` pre-folds 701 runtime shape
ops (`Shape`, `Gather`, `Slice`, `Unsqueeze`, `Sqrt`, `Div`) into
constants (graph node count drops 2117 → 1416, -33%).

But the only routing rule is "if `pairs.len() == 1`, use the static
pool". memdb-go — the dominant `/v1/rerank` consumer — fans out D7
sub-queries to **batch=5** calls (`bench.py:198` confirms: "Default 5
mirrors memdb-go's typical D7 sub-query candidate count"). Every prod
call falls through to the dynamic pool today. PR #27's win is real for
quality-eval / single-doc paths only, not for the hot path.

This plan extends the loader to host *multiple* static sessions per
reranker, one per supported batch shape, and routes `score_pairs` to
the matching pool when the runtime batch size matches an exported
shape.

## Anti-myths (preempted)

1. **"Just pad batch=3 up to batch=5 to use the static pool."** No.
   Different optimization, different padding-waste math, different test
   surface. Out of scope. Routing is **exact-match only**; mismatched
   batch sizes fall back to the dynamic pool.
2. **"Use one big static graph with shape `[1..5, max_len]`."** Not
   what `--no-dynamic-axes` gives you. Each batch dim must be a literal
   constant in the ONNX graph for ORT to fold the shape ops; a dynamic
   batch axis re-introduces every `Shape`/`Gather` node we wanted to
   eliminate.
3. **"INT8 quantization on top of static export gives the same 1.74×."**
   Standalone bench yes; production prod-quantized memdb measurement
   was only ~1.04× over dynamic at batch=1 (per task brief). The
   static-graph win is partly dominated by quantize/dequantize node
   cost, which is shape-agnostic. Bench before promising prod uplift.

## Shape policy

Memdb-go realistic shape is `(batch=5, seq=256)`. Recommended baseline
shape set:

| Shape       | Why                                                         | Default? |
|-------------|-------------------------------------------------------------|----------|
| `[1, 256]`  | Existing PR #27 fast-path; quality-eval / D7 single-doc    | yes      |
| `[5, 256]`  | memdb-go D7 hot path                                       | opt-in   |
| `[2, 256]`  | Smaller D7 fanout sub-cases (paired retrieval rerank)      | optional |
| `[10, 256]` | `MAX_CONCURRENT_RERANK_REQUESTS=4` × wider top-K probe     | optional |

`max_len=256` is fixed by the ModernBERT reranker config; we do not
sweep on the seq axis here. Adding `[N, 128]` shapes would be the next
follow-up if Phase 2 padding-waste data justifies it (today the
padding-waste histogram is uninstrumented for the static path).

**Default policy: nothing changes vs PR #27.** The repo ships with
zero `model_quantized_static_b<N>.onnx` files committed (the artifacts
live under `/home/krolik/deploy/krolik-server/models/...`, mounted into
the container, not tracked in git). An operator activates a new shape
by running the export script and dropping the file in place. No code
change, no env flip, no restart with new flags — just a normal
container restart to re-load the model dir. This mirrors PR #27's
"convention over configuration" stance.

## Loader behaviour

### Filename convention

`model_quantized_static_b<N>.onnx`, where `N` is the literal batch
dimension. Examples:
- `model_quantized_static_b1.onnx`
- `model_quantized_static_b5.onnx`

**Backwards compat**: the legacy unsuffixed `model_quantized_static.onnx`
(PR #27) is treated as `b=1`. So existing prod deployments keep
working without operator action.

If both `model_quantized_static.onnx` and `model_quantized_static_b1.onnx`
exist, the suffixed file wins (explicit > implicit) and a `warn` log
flags the duplicate so ops can clean up.

**Justification for filename-over-directory**: directory layouts force
operators to mkdir + mv per shape; flat filenames let a single `cp` or
`scp` deploy a new shape. Filename also carries the shape data
inline — no out-of-band config drift between disk and an env var.

### Discovery

Pure scan over `<dir>/model_quantized_static_b*.onnx` matching
`/^model_quantized_static_b(\d+)\.onnx$/` (plus the legacy
unsuffixed file as `b=1`), returning a `Vec<(usize, PathBuf)>` sorted
by batch size. The scan is a separate function from session
construction so it can be unit-tested against synthetic temp dirs that
contain empty placeholder files.

### Pool layout

```rust
pub(super) static_session_pools: BTreeMap<usize, Vec<Mutex<Session>>>,
pub(super) static_session_cursors: BTreeMap<usize, AtomicUsize>,
```

`BTreeMap` keyed on batch size, value is the per-shape session pool.
Pool size hard-coded to 2 per shape (mirrors PR #27 and the dynamic
pool default; matches the typical 4-core / 4-inflight config). Each
shape gets its own round-robin cursor so a busy `b=5` pool can't stall
the `b=1` fast path.

If discovery returns empty, the field is an empty `BTreeMap` — same
runtime semantics as `Option::None` in PR #27. Routing collapses to
the dynamic-only path, no new branching needed.

## Routing

In `score_pairs`:

```rust
if let Some(pool) = self.static_session_pools.get(&token_ids.len()) {
    return self.score_pairs_static(token_ids, &pool);
}
// fall through to dynamic
```

**Exact match only.** A batch=3 call with static pools `{1, 5}`
available falls through to the dynamic pool. We do **not** pad-up to
batch=5 — that's a different optimization (changes padding-waste
calculus, introduces a new branch in score reading, hurts the
static-graph latency advantage by inflating compute).

`score_pairs_static` is generalised so it works at any batch size, not
hard-coded to 1: it builds an `[N, max_len]` tensor exactly the same
way the dynamic path does, but always pads to `self.max_len` (because
that's literally the only seq the static graph accepts). The output
shape assertion becomes `[N, 1]` instead of `[1, 1]`.

## Memory budget

Per #27 commit msg, one ModernBERT static session = ~255 MiB; pool
size 2 = ~510 MiB extra per shape.

| Configuration | Dynamic pool | Static `b=1` | Static `b=5` | Total ModernBERT |
|---------------|-------------:|-------------:|-------------:|-----------------:|
| Pre-#27 (dynamic only) | 510 MiB | — | — | ~510 MiB |
| Post-#27 ({b=1}) — current prod | 510 MiB | 510 MiB | — | ~1.0 GiB |
| Post-#27 + multi {b=1, b=5} | 510 MiB | 510 MiB | 510 MiB | ~1.5 GiB |
| Hypothetical {b=1, b=2, b=5, b=10} | 510 MiB | 510 MiB | 510 MiB × 3 | ~2.5 GiB |

Container shared CPU arena cap is 3 GiB (Phase H.16). The {b=1, b=5}
configuration fits with ~1.5 GiB of headroom for embed-server's other
models (multilingual-e5-large, jina-code-v2). The four-shape
configuration leaves only ~500 MiB headroom — risky on a box that also
runs SPLADE, OTel exporters, and the dynamic batcher arenas. Hold the
four-shape variant until Phase 2 data justifies it.

## Disk budget

Each `model_quantized_static_b<N>.onnx` is ~250 MiB (per PR #27 stage:
the existing `model_quantized_static.onnx` was 245 MiB). Adding b=5
adds ~250 MiB to the model dir. Two-shape policy {b=1, b=5}: ~500 MiB
of static graphs alongside the ~140 MiB dynamic file. The model dir is
in `/home/krolik/deploy/krolik-server/models/...` — disk is cheap,
but operators tracking models in git-LFS would feel the size. Keep the
files out of git (no `.onnx` should land on this branch).

## Padding waste analysis

PR #27's static path always pads to `max_len=256` (no per-call
truncation). Multi-shape inherits this — the static graph can't accept
a smaller seq.

Realistic shape distribution from memdb-go (per Phase 2 baseline plans
+ hot-path inspection):

| Inbound batch size | Frequency | Path with {b=1, b=5} static pool |
|--------------------|-----------|----------------------------------|
| 1                  | ~30%      | static b=1 ✓ (1.74× standalone, ~1.04× prod) |
| 2-4                | ~10%      | dynamic (no pad-up)              |
| 5                  | ~50%      | static b=5 ✓ (target uplift)     |
| 6-10               | ~10%      | dynamic                          |

So 80% of calls land on a static pool with the {1, 5} policy. The
"missed" 20% (batches 2/3/4 and 6-10) remain on the dynamic pool —
which is by design, see Routing. Dynamic-pool latency for those bins
is unaffected (mutex contention is the only shared resource and the
static pool has its own).

## Risks

1. **Quantized int8 ARM Neoverse-N1: real prod win may be small.**
   PR #27's standalone bench showed 1.74× but the prod measurement on
   quantized int8 was only ~1.04× — most of the static-graph win is
   dominated by quantize/dequantize nodes that don't get folded.
   Recommendation: bench {b=5} export against the production dynamic
   pool **before** committing to deploying the file. If the prod
   speedup is <10%, do not promise the optimization to memdb. Ship
   the loader plumbing (zero-risk, gated by file existence) but hold
   the export deploy.

2. **Disk: each static `.onnx` is ~250 MiB**, +~1 GiB across
   {b=1, b=2, b=5, b=10}. Operator burden: backup discipline,
   `du`-watch on the models volume. Mitigation: strict shape policy
   defaults to `{b=1}` only; new shapes need explicit operator deploy.

3. **Padding waste** for `[N, 256]` static shapes when real seqs are
   shorter — the static graph compute is pinned at `max_len=256`. For
   memdb-go this rarely bites because the production `MaxCharsPerDoc=0`
   config means real docs frequently saturate the 256-token cap. But
   for callers passing short queries (e.g. quality eval), the static
   path may be slower than the dynamic path's tighter `max_seq` bound.
   Same trade-off PR #27 documented.

4. **Export time and disk**: `optimum-cli ... --no-dynamic-axes
   --batch_size 5 --sequence_length 256` for ModernBERT takes
   significant CPU + tens of GB of intermediate artifacts during
   conversion. Run the export off-prod (dev box, not krolik) and rsync
   the resulting file. The export script documents this; do **not**
   run it from this branch's CI.

5. **ORT graph rejection**: a static graph compiled at
   `[5, 256]` literally rejects any other input shape. Routing must
   never path a `len() != 5` call through the b=5 pool, otherwise
   `session.run()` returns an opaque ORT error. Covered by exact-match
   routing + unit test.

6. **Concurrent batch=3 flood**: imagine memdb-go suddenly switches
   sub-query fanout to 3. With `{b=1, b=5}` static, every batch=3 call
   falls through to dynamic. The dynamic pool has only 2 sessions —
   under c=4 concurrency we'd see mutex contention spike on dynamic
   while the static pools sit idle. Mitigation: monitor
   `embed_rerank_pool_acquire_duration_seconds{model="gte-modernbert"}`
   p95 + the upcoming `static_pools` gauge; if a new dominant batch
   size emerges, export a matching shape.

## Open questions

1. **gte-multi-rerank** has no static export today (`static_sessions:0`).
   Loader plumbing supports it for free (the scan returns empty, the
   `BTreeMap` stays empty). If memdb cuts over from ModernBERT to
   mGTE, we'd want the same `{b=1, b=5}` pipeline. Not in this
   branch's scope; tracked here as a future workstream.

2. **Per-model pool size**. PR #27 hard-codes 2 sessions per static
   pool. With 4 shapes × 2 sessions × 510 MiB we'd be at the arena
   cap. A future env knob `RERANK_STATIC_POOL_SIZE_PER_SHAPE` could
   tune this, but it's out of scope here.

3. **`optimum-cli` batched static export verification**: the `--batch_size`
   flag plus `--no-dynamic-axes` *should* fix the batch axis, per
   `optimum-cli` docs (verified via context7). But we have not yet
   produced a batched static export end-to-end on krolik — the export
   script is shipped as documentation of the recipe, not as a verified
   build. First operator to run it should record the int8 quantization
   recipe used (the PR #27 file used a manual dynamic-quant pass — we
   document the same recipe).

## Why this is convention-only, not env-driven

The task brief floats `RERANK_STATIC_SHAPES=<model>:1,5;...` as one
option. We rejected it for the same reasons PR #27 rejected
`RERANK_ENABLE_STATIC=1`:

- File presence on disk **is** the config — no parsing, no validation,
  no drift between the env var and what's actually deployable.
- One operator action ("rsync the .onnx into the model dir + restart")
  to enable a shape. With env config, the operator has to remember to
  also flip the env var.
- No env-format migration burden if we add new shape axes (e.g. seq
  length) later.

## Implementation checklist

- [ ] Pure discovery function `discover_static_shape_files(dir) -> BTreeMap<usize, PathBuf>` with unit tests over synthetic temp dirs.
- [ ] Loader builds `BTreeMap<usize, Vec<Mutex<Session>>>` from the discovered map.
- [ ] `score_pairs` exact-match routes to per-shape pool; falls through to dynamic on miss.
- [ ] `score_pairs_static` generalised to any `[N, max_len]` shape (was hard-coded `[1, max_len]`).
- [ ] Per-shape `tracing::info!` at load: `model=X batch=N count=2`.
- [ ] Backwards compat: `model_quantized_static.onnx` still treated as `b=1`.
- [ ] Tests: discovery, routing, fallback, backwards compat.
- [ ] `scripts/export_static_modernbert.sh` documenting the export recipe (do **not** run on this branch).
- [ ] `docs/runbook.md` entry: how operators deploy a new shape.

## Out of scope

- Embedder static export (e.g. `multilingual-e5-large`). Different
  callers, different shape distribution; would need its own design.
- Pad-up routing (batch=3 → static b=5). Different cost calculus.
- Shape-axis sweeps on `seq_len` (only batch axis here).
- Production `CROSS_ENCODER_MODEL` flip (still gated on Phase 4 of the
  2026-05-01 plan).
- Auto-export at startup. Operators run the export off-box.

## Ship vs hold (honest take)

**Ship:** the loader plumbing — it's strictly additive, off by default
(activated by file presence), and zero-risk on the existing
single-shape deployment. Backwards-compat unit tests guarantee PR #27's
prod path keeps working byte-for-byte.

**Hold:** the actual `b=5` deployment until a focused bench shows
≥10% prod p95 reduction over the dynamic pool at the realistic
`(batch=5, seq=256)` shape. PR #27's prod measurement (1.04× at
batch=1, vs the standalone 1.74×) is the data point that should make
us cautious. Batch amortisation already extracts most of the
parallelism that static folding would extract — it's plausible the
incremental win at batch=5 is in the noise. A negative result is fine;
revert is `rm` on the static file.
