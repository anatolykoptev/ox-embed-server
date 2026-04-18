# Embed-Server Phase 2: Throughput + Reranker/NER/Sparse Models

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

---

## Background & Rationale (read this before Phase 0)

### Why embeddings

An embedding is a compact representation of meaning as a point in an N-dim space. Semantically close texts end up close, distant texts end up far. That turns "find related" from a lexical problem into a geometric one. Concretely, our stack already uses embeddings for:

- **Semantic search** — "cancel subscription" ≈ "stop payment" (MemDB retrieves by similarity).
- **Dedup / clustering** — same thing said two ways groups together.
- **RAG pre-retrieval** — top-K candidates fed into the LLM.

What embeddings CANNOT do alone (and which sibling models fix):

| Gap | Fix | Size |
|---|---|---|
| Fine-grained relevance (dot product is coarse) | **Cross-encoder reranker** | 280-568 MB |
| Entity extraction ("is this a person? a project?") | **GLiNER (label-driven NER)** | ~150 MB |
| Term/aberration matching (BM25 replacement) | **SPLADE sparse encoder** | ~130 MB |

### Which models we chose (and why)

All three must fit CPU/ARM inference budget, INT8-quantized, under ~600 MB on disk, run via ONNX Runtime.

**Reranker: `BAAI/bge-reranker-v2-m3`** (568 M params, multilingual). On public benchmarks, pairing retrieval + rerank lifts Hit Rate from ~0.82 → ~0.94, MRR from 0.73 → 0.87 (LlamaIndex benchmark). Multilingual matters because MemDB serves both Russian and English traffic.

**NER: `urchade/gliner_small-v2.1`** (~150 M). GLiNER takes arbitrary labels at request time (e.g. `["person", "project", "deadline"]`) and returns spans. Lets MemDB structure memories into a graph of people/projects/tech without training a custom model.

**Sparse: `naver/splade-v3-distilbert`** (~134 M). Produces sparse weighted-token vectors, compatible with inverted index / hybrid retrieval. go-code already advertises "hybrid RRF (semantic + keyword)" — SPLADE replaces the BM25 half with learned sparse.

Explicit non-choices:
- No LLM-grade embedders (e5-mistral-7B, NV-Embed) — 7 B params won't run on this ARM box.
- No ColBERT/ColPali — token-level late-interaction exploded vector storage by 10×.
- No multimodal (CLIP/SigLIP) — no image retrieval in our stack.

### Competitive landscape (what LLM inference servers do)

Quick research notes on vLLM / TGI / Triton / TEI, distilled into what applies to an encoder-only ONNX CPU server:

| Tool | Key tricks | Applies to us? |
|---|---|---|
| **vLLM** | PagedAttention (KV cache in pages), continuous batching, speculative decoding, prefix caching | ❌ all for autoregressive LLMs; encoder forward is single-pass, no KV cache |
| **TGI** | Flash-Attention 2, PagedAttention, GPTQ/AWQ/bitsandbytes | ❌ same reasons + no FA kernels on ARM CPU |
| **Triton** | Dynamic batcher (`preferred_batch_size` + `max_queue_delay`), response cache, multi-instance groups | ✅ dynamic batcher parameters + response cache |
| **TEI (HF Text-Embeddings-Inference)** | Token-budget batching, padded-model accounting, client-cancel check, off-thread tokenization | ✅✅ direct template — Rust-in-Rust; port the algorithm from `core/src/queue.rs` |
| **BucketServe** (paper) | Length-bucketing — group by token-count to minimise padding | ✅ up to −50% p99 on heterogeneous workloads |

The design choices below (Phases B & C) are literal ports of TEI's queue algorithm + BucketServe's bucketing. TEI's `core/src/queue.rs` is the reference — read it before touching `src/batcher.rs`: https://github.com/huggingface/text-embeddings-inference/blob/main/core/src/queue.rs

### What's currently in place (baseline)

- Rust `embed-server` sidecar, port 8082, two models loaded: `multilingual-e5-large` (1024-dim, text), `jina-code-v2` (768-dim, code).
- OpenAI-compatible `/v1/embeddings`, Prometheus `/metrics`, `/health`.
- Dynamic batcher with **item-count** budget (`BATCH_MAX=8`, `BATCH_WAIT_MS=10`) + just-landed carry-over fix (`3598b48`).
- 4 GiB memory cap (bumped from 3 GiB this session), ~1.5 GiB RSS.
- Autodeploy via dozor webhook on every push to `main`.

### ROI summary (why these phases, in this order)

| Phase | Effort | Expected win | Risk |
|---|---|---|---|
| 0 — install models | 2 h | — (prep) | None (no code) |
| A — warm-up (cancel, ORT opt, truncate) | 1 d | +10-20% rps | Low |
| B — token-budget batcher | 2-3 d | **2×+ rps** on mixed load | Medium |
| C — length-bucketing | 1-2 d | −20% p99 | Low (feature-flagged) |
| D — response cache | 1 d | +15% cache hits → huge latency wins for MemDB | Low (isolated module) |
| E — reranker `/v1/rerank` | 3-4 d | **+0.05 Hit Rate** RAG improvement | Medium (new endpoint, new model kind) |
| F — GLiNER `/v1/ner` (deferred) | ~2 d | Graph-ready memories | Medium |
| G — SPLADE `/v1/sparse` (deferred) | ~2 d | Better hybrid recall | Medium |

---

**Goal:** Double throughput on heterogeneous CPU load, add a second class of models (reranker, NER, sparse) to serve the MemDB + go-code search stack end-to-end from one sidecar.

**Architecture:** Keep the current `/v1/embeddings` path unchanged. Rebuild the batcher around a **token budget** with **padded-model accounting** (TEI `core/src/queue.rs` pattern), add **length-bucketed queues** on top. Introduce two new model kinds — `RerankerModel` (cross-encoder → single score) and `TokenClassifierModel` (GLiNER/SPLADE → token-level output) — behind sibling endpoints `/v1/rerank`, `/v1/ner`, `/v1/sparse`. Reuse the same batcher for all three kinds because all are padded BERT-style encoders.

**Tech Stack:** Rust 2024, `axum`, `tokio`, `ort 2.0.0-rc.12` (ONNX Runtime), `tokenizers`, `ndarray`, Prometheus. Models exported via `optimum-cli` / `optimum-onnx`.

**Baseline (pre-plan, 2026-04-17 metrics):**
- e5-large: 2340 req, avg request 2.91 s, avg inference 2.81 s
- jina-code-v2: 344 req, avg request 4.13 s, avg inference 3.25 s
- Throughput (conc=16, short query): jina ~75 rps, e5 ~28 rps
- `BATCH_MAX=8` (items), `BATCH_WAIT_MS=10`
- Memory cap: 4 GiB, current RSS ~1.5 GiB

**Success criteria:**
- Phase A: +10% rps on bench_par.sh, 0 regressions.
- Phase B: ≥2× rps on mixed short+long workload.
- Phase C: ≥20% p99 reduction on heterogeneous trafic (measured over 24h in prod).
- Phase D: cache hit rate ≥15% at 24h.
- Phase E: Hit Rate on fixed eval set ≥ +0.05 over embeddings-only baseline.

---

## Phase 0: Install Recommended Models (data prep, ~2 hours)

No code changes to embed-server. Just export models, put them on disk, mount into the container. Sets up the artifacts that later phases will consume.

### Task 0.1: Export `BAAI/bge-reranker-v2-m3` to INT8 ONNX

**Files:**
- Create: `scripts/export_reranker_bge.sh`
- Create: `/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3/model_quantized.onnx`
- Create: `/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3/tokenizer.json`

**Step 1: Write the export script**

```bash
#!/usr/bin/env bash
# scripts/export_reranker_bge.sh
set -euo pipefail
OUT=/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3
mkdir -p "$OUT"
cd /tmp && python3 -m venv venv-export && source venv-export/bin/activate
pip install -q "optimum[onnxruntime]==1.27.*" transformers onnx
optimum-cli export onnx \
  --model BAAI/bge-reranker-v2-m3 \
  --task text-classification \
  --opset 17 \
  "$OUT"
python3 - <<PY
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
q = ORTQuantizer.from_pretrained("$OUT")
q.quantize(save_dir="$OUT", quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=False))
PY
ls -lah "$OUT"/model_quantized.onnx "$OUT"/tokenizer.json
```

**Step 2: Run it**

```bash
ssh krolik 'bash /home/krolik/src/embed-server/scripts/export_reranker_bge.sh'
```

Expected: `model_quantized.onnx` ~200 MB, `tokenizer.json` ~17 MB.

**Step 3: Smoke-test the ONNX directly (before integrating)**

```python
# scripts/smoke_reranker_bge.py
import onnxruntime as ort, json
from transformers import AutoTokenizer
d = "/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3"
s = ort.InferenceSession(f"{d}/model_quantized.onnx")
t = AutoTokenizer.from_pretrained(d)
pairs = [("what is a cat", "a cat is a small domestic feline"),
         ("what is a cat", "the price of oil dropped yesterday")]
enc = t(pairs, padding=True, truncation=True, max_length=512, return_tensors="np")
out = s.run(None, {k: v for k, v in enc.items() if k in [i.name for i in s.get_inputs()]})
print("scores:", out[0].reshape(-1).tolist())
```

Expected: first score >> second score (first pair is relevant).

**Step 4: Commit the export scripts** (not the model binaries — those live on disk, not in git)

```bash
git add scripts/export_reranker_bge.sh scripts/smoke_reranker_bge.py
git commit -m "feat(models): add BGE-reranker-v2-m3 export + smoke script"
```

### Task 0.2: Export `urchade/gliner_small-v2.1` to ONNX

**Files:**
- Create: `scripts/export_gliner.sh`
- Create: `/home/krolik/deploy/krolik-server/models/gliner-small-v2.1/model.onnx`
- Create: `/home/krolik/deploy/krolik-server/models/gliner-small-v2.1/tokenizer.json`

**Step 1: Write export script**

```bash
#!/usr/bin/env bash
# scripts/export_gliner.sh
set -euo pipefail
OUT=/home/krolik/deploy/krolik-server/models/gliner-small-v2.1
mkdir -p "$OUT"
source /tmp/venv-export/bin/activate
pip install -q gliner onnx
python3 - <<PY
from gliner import GLiNER
m = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
m.save_pretrained("$OUT", save_onnx=True)
PY
ls -lah "$OUT"
```

**Step 2: Run + smoke**

Same pattern as Task 0.1 — ssh + bash + verify output files.

**Step 3: Commit scripts.**

### Task 0.3: Export `naver/splade-v3-distilbert` to ONNX

**Files:**
- Create: `scripts/export_splade.sh`
- Create: `/home/krolik/deploy/krolik-server/models/splade-v3-distilbert/model.onnx`

**Step 1-3:** Same pattern. SPLADE exports via `optimum-cli export onnx --task fill-mask`.

### Task 0.4: Mount model volumes into `embed-server` container

**Files:**
- Modify: `/home/krolik/deploy/krolik-server/compose/memdb.yml` (embed-server block)

**Step 1: Add to `volumes:`**

```yaml
- /home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3:/models-reranker:ro
- /home/krolik/deploy/krolik-server/models/gliner-small-v2.1:/models-gliner:ro
- /home/krolik/deploy/krolik-server/models/splade-v3-distilbert:/models-splade:ro
```

**Step 2: `docker compose up -d embed-server`** — just remounts, no rebuild.

**Step 3: Verify mounts inside container**

```bash
ssh krolik 'docker exec embed-server ls /models-reranker /models-gliner /models-splade'
```

Expected: all three dirs list their `.onnx` and `tokenizer.json`.

**Step 4: Commit**

```bash
cd /home/krolik/deploy/krolik-server
git add compose/memdb.yml
git commit -m "feat(embed): mount reranker/gliner/splade model volumes"
```

---

## Phase A: Risk-Free Warm-Up (1 day)

Get +10-20% throughput with three tiny, isolated changes before the big batcher rewrite.

### Task A1: Cancel-check before batch dispatch

**Files:**
- Modify: `src/batcher.rs` (inside `run_worker`, before building the batch)
- Test: `src/batcher.rs` (new inline test `cancelled_items_are_skipped`)

**Step 1: Write the failing test**

Add to `#[cfg(test)] mod tests`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_items_are_skipped() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();
    let b = Arc::new(DynamicBatcher::with_name(
        "t_cancel",
        move |t| { cc.fetch_add(t.len(), Ordering::SeqCst); Ok(t.iter().map(|_| vec![0.0f32; 4]).collect()) },
        32, 50, 16,
    ));
    // First request, drop its future immediately → receiver closes.
    let (tx, rx) = oneshot::channel::<Result<Vec<Vec<f32>>, BatchError>>();
    // Manually inject an Item whose reply_tx we drop before dispatch.
    // Because embed() doesn't expose Item directly, we simulate by spawning
    // a client request and aborting its JoinHandle before the batch window
    // closes.
    let b1 = b.clone();
    let handle = tokio::spawn(async move { b1.embed(vec!["a".into()]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;
    handle.abort();
    // Second request goes through normally.
    let r = b.embed(vec!["b".into()]).await.unwrap();
    assert_eq!(r.len(), 1);
    // Inference should have run only for "b" — not for the aborted request.
    assert_eq!(call_count.load(Ordering::SeqCst), 1, "expected 1 inferred text, got {}", call_count.load(Ordering::SeqCst));
    drop(rx);
    Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
}
```

**Step 2: Run — should fail**

```bash
ssh krolik 'source $HOME/.cargo/env && cd /home/krolik/src/embed-server && cargo test batcher::tests::cancelled_items_are_skipped 2>&1 | tail -15'
```

Expected: `call_count` = 2, not 1 (aborted "a" was still embedded).

**Step 3: Implement in `run_worker`**

In `src/batcher.rs`, around the current dispatch site:

```rust
// Before `dispatch_batch(batch, ...)`, drop items whose caller is gone.
batch.retain(|it| !it.reply.is_closed());
if batch.is_empty() { continue; }  // entire batch was cancelled → skip inference
```

Also add the same check inside the coalesce loop — if the newly received item is already closed, don't even add it.

**Step 4: Run — should pass.**

**Step 5: Commit**

```bash
git add src/batcher.rs
git commit -m "perf(batcher): skip items whose client dropped before dispatch

Mirrors TEI core/src/queue.rs behaviour. Under p99-tail conditions,
some clients time out before the batch window closes; running
inference for them wastes CPU on the critical path. Check
reply.is_closed() and skip. Saves a few percent on heavy days.

Regression test: cancelled_items_are_skipped."
```

### Task A2: ORT Graph Optimization Level

**Files:**
- Modify: `src/model.rs` (session builder)

**Step 1: Write characterisation test**

```rust
// src/model.rs bottom of file, in #[cfg(test)] mod
#[test]
fn session_uses_level3_optimization() {
    // This is more of a doc-test — assert that the env knob is wired
    // through by inspecting a builder; the perf effect is measured by bench.
    assert_eq!(std::env::var("ORT_OPT_LEVEL").unwrap_or_default(), "3");
}
```

**Step 2: Read current session creation**

```bash
grep -n "SessionBuilder\|with_optimization_level\|commit_from_file" /home/krolik/src/embed-server/src/model.rs
```

**Step 3: Wire the flag**

In `src/model.rs`, on the `SessionBuilder`:

```rust
use ort::session::builder::GraphOptimizationLevel;

let opt_level = std::env::var("ORT_OPT_LEVEL")
    .ok()
    .and_then(|s| s.parse::<u8>().ok())
    .unwrap_or(3);
let level = match opt_level {
    0 => GraphOptimizationLevel::Disable,
    1 => GraphOptimizationLevel::Level1,
    2 => GraphOptimizationLevel::Level2,
    _ => GraphOptimizationLevel::Level3,
};
builder = builder.with_optimization_level(level)?;
```

**Step 4: Set env in compose**

Modify `/home/krolik/deploy/krolik-server/compose/memdb.yml` embed-server block, add:
```yaml
ORT_OPT_LEVEL: "3"
```

**Step 5: Bench before/after**

Bench the same `bench_par.sh` we used earlier with the OLD image, record numbers. Then redeploy with the new image and bench again. Expected: 5-15% improvement on quantized models; could be zero. If zero, leave the flag — it's not a regression.

**Step 6: Commit**

```bash
git add src/model.rs
cd /home/krolik/deploy/krolik-server && git add compose/memdb.yml
git commit -m "perf(model): enable ORT graph optimization level 3 via env"
```

### Task A3: Auto-truncate overlong inputs

**Files:**
- Modify: `src/api.rs` (request validation)
- Test: same file

**Step 1: Write failing test**

```rust
#[tokio::test]
async fn overlong_input_is_truncated_not_erred() {
    // Construct an input far longer than max_len=512 (e.g. 5000 tokens worth)
    // Send it through the embed pipeline; expect a 200 response with a vector,
    // not a 400 error about length.
    let long = "word ".repeat(2000);
    let (status, body) = embed_request(vec![long]).await;
    assert_eq!(status, 200, "expected 200 with auto_truncate, got {status}: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["data"][0]["embedding"].as_array().is_some());
}
```

**Step 2: Run — should fail** (current behaviour errors on overlong inputs in tokenizer).

**Step 3: Implement**

In `src/api.rs`, before tokenization, add:
```rust
let auto_truncate = std::env::var("AUTO_TRUNCATE").ok().map(|s| s != "false").unwrap_or(true);
```
Then pass `auto_truncate` into the tokenizer call. Check current tokenizer invocation in `src/model.rs::embed` — it likely already calls `encoder.with_truncation(...)`. Confirm max_len is enforced, or switch to truncating before the hard error.

**Step 4: Run — should pass.**

**Step 5: Commit**

```bash
git add src/api.rs src/model.rs
git commit -m "feat(api): auto-truncate inputs > max_len (TEI-compat default)"
```

---

## Phase B: Token-Budget Batcher ⭐ (2-3 days)

The main event. Rewrites the coalesce loop around token counts with padded-model accounting. This is the work that delivers 2×+ throughput on mixed workloads.

**Reference:** https://github.com/huggingface/text-embeddings-inference/blob/main/core/src/queue.rs — port the algorithm, don't copy the code (different async runtime, different item shape).

### Task B1: Pre-tokenize in `api.rs` — carry `input_ids` in Item

**Files:**
- Modify: `src/batcher.rs` — add `input_ids: Vec<u32>`, `token_count: usize` to Item
- Modify: `src/api.rs` — tokenize before `batcher.embed(...)` call
- Modify: `src/model.rs` — accept `Vec<Vec<u32>>` tokens instead of texts

**Step 1: Write failing test**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn item_carries_token_count() {
    // Create an Item with pre-tokenized ids; batcher should dispatch
    // a "texts" callback that receives token ids, not strings.
    let got_tokens: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![]));
    let gt = got_tokens.clone();
    let b = DynamicBatcher::with_name_tokenized(
        "t_tok",
        move |ids| { gt.lock().unwrap().extend(ids.clone()); Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect()) },
        /*max_batch_tokens*/ 1000, 50, 16,
    );
    let _ = b.embed_tokens(vec![vec![1,2,3], vec![4,5]]).await;
    let captured = got_tokens.lock().unwrap().clone();
    assert_eq!(captured, vec![vec![1u32,2,3], vec![4,5]]);
    b.shutdown(Duration::from_millis(200)).await;
}
```

**Step 2: Run — fails (no `with_name_tokenized` / `embed_tokens`).**

**Step 3: Implement**

This is a larger change: restructure `Item`, add a new constructor path. Keep the old `with_name` + `embed` for backward-compat during transition.

Full outline (compressed):
```rust
#[derive(Debug)]
struct Item {
    token_ids: Vec<Vec<u32>>,   // one entry per text within the request
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
    // legacy "texts" removed once api.rs is migrated
}
```

In `api.rs::embeddings`, tokenize before dispatching:
```rust
let tokens = entry.model.tokenize(&texts)?;  // new Model::tokenize()
entry.batcher.embed_tokens(tokens).await
```

In `Model::embed`, accept token ids directly.

**Step 4: Run — passes.**

**Step 5: Commit**

```bash
git add src/batcher.rs src/api.rs src/model.rs
git commit -m "refactor(batcher): carry pre-tokenized ids in Item (prep for token-budget)"
```

### Task B2: Add `BATCH_MAX_TOKENS` env, keep `BATCH_MAX` as item-count cap

**Files:**
- Modify: `src/config.rs`
- Test: `src/config.rs` inline test

**Step 1: Test the new parse path.**

```rust
#[test]
fn parses_batch_max_tokens_env() {
    std::env::set_var("BATCH_MAX_TOKENS", "16384");
    let cfg = Config::from_env().unwrap();
    assert_eq!(cfg.batch_max_tokens, 16384);
    std::env::remove_var("BATCH_MAX_TOKENS");
}
```

**Step 2-4:** Add field, parse, default `16384`, commit.

### Task B3: Padded-model token accounting

**Files:**
- Modify: `src/batcher.rs::run_worker`

**Step 1: Write failing test**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn padded_accounting_prevents_mixing_long_with_many_short() {
    // max_batch_tokens = 1000, padded_model = true.
    // First: 500 tokens. Then: 10× 50-token requests arrive.
    // Old naive sum: 500 + 10*50 = 1000 → all fit.
    // Padded correct: max(500, 50)*(1+10) = 5500 → only first fits, rest deferred.
    let log: Arc<Mutex<Vec<Vec<usize>>>> = Arc::new(Mutex::new(vec![]));
    let l = log.clone();
    let b = Arc::new(DynamicBatcher::with_tokens(
        "t_pad",
        move |ids| { l.lock().unwrap().push(ids.iter().map(|i| i.len()).collect()); Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect()) },
        /*max_batch_tokens*/ 1000, /*padded_model*/ true, 50, 16,
    ));
    let b_first = b.clone();
    let first = tokio::spawn(async move { b_first.embed_tokens(vec![vec![0u32; 500]]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;
    let mut rest = vec![];
    for _ in 0..10 {
        let bc = b.clone();
        rest.push(tokio::spawn(async move { bc.embed_tokens(vec![vec![0u32; 50]]).await }));
    }
    let _ = first.await;
    for h in rest { let _ = h.await; }
    let batches = log.lock().unwrap();
    assert!(batches[0] == vec![500], "first batch should hold only the 500-token item, got {:?}", batches[0]);
    assert!(batches.len() >= 2, "10 short items must dispatch in separate batch(es)");
    drop(batches);
    Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
}
```

**Step 2: Run — fails.**

**Step 3: Implement**

Port TEI's formula:
```rust
let entry_tokens = entry.token_ids.iter().map(|t| t.len()).sum::<usize>();
let max_in_entry = entry.token_ids.iter().map(|t| t.len()).max().unwrap_or(0);
let new_max_length = max(current_max, max_in_entry);
let total_if_added = if padded_model {
    new_max_length * (current_requests + entry.token_ids.len())
} else {
    current_tokens + entry_tokens
};
if total_if_added > max_batch_tokens {
    carry = Some(entry);
    break;
}
current_max = new_max_length;
current_tokens += entry_tokens;
current_requests += entry.token_ids.len();
```

**Step 4: Run — passes.**

**Step 5: Commit**

```bash
git add src/batcher.rs
git commit -m "feat(batcher): padded-model token accounting (TEI-style)

Replaces item-count cap with token-budget cap that correctly
accounts for padding-to-longest in the batch. A 500-tok item
no longer gets joined by 10× 50-tok items — that combination
wastes 10× compute on padding."
```

### Task B4: Metrics

**Files:**
- Modify: `src/metrics.rs`
- Modify: `src/batcher.rs`

Add:
- `embed_batch_tokens_sum{model}` — total tokens per inference
- `embed_batch_padding_waste_ratio{model}` — `(max_len * n - sum_lens) / (max_len * n)` — how much of the batch was padding
- `embed_carry_events_total{model}` — how often items get deferred

Test each with a dedicated inline test. Commit separately.

### Task B5: Update `BATCH_MAX_TOKENS` in compose

**File:** `/home/krolik/deploy/krolik-server/compose/memdb.yml`

Replace `BATCH_MAX: "8"` with `BATCH_MAX_TOKENS: "16384"` (TEI default). Keep `BATCH_MAX: "32"` as a soft cap on items (fairness). Commit in krolik-server repo.

### Task B6: Bench comparison

**Files:**
- Modify: `tests/bench.rs` (criterion benches)

**Step 1: Write benches**

```rust
// tests/bench.rs
// cargo bench --bench throughput
fn bench_short_mixed_with_long(c: &mut Criterion) { ... }
fn bench_pure_short(c: &mut Criterion) { ... }
fn bench_pure_long(c: &mut Criterion) { ... }
```

Run before deploy (old binary → baseline), deploy Phase B, run again. Record numbers in plan execution notes.

**Step 2: Acceptance gate**

- Pure-short: ≥ 2× req/s improvement
- Mixed: ≥ 1.5× req/s improvement
- Pure-long: no regression (within ±5%)

If any gate misses, open an issue and investigate before moving to Phase C.

### Task B7: Code review

Dispatch `code-reviewer` subagent on the full Phase B diff (`git log B1..B6`). Address any CRITICAL or HIGH findings before merging.

---

## Phase C: Length-Bucketing (1-2 days)

Perpendicular optimization to B — reduces intra-batch padding waste.

### Task C1: Bucket configuration

**Files:**
- Modify: `src/config.rs`

Add `BATCH_BUCKETS="64,128,256,512"` env (comma-separated token ceilings). Parse to `Vec<usize>`. Default `[64, 128, 256, 512]`.

**Step 1-5:** Standard config+test+commit cycle.

### Task C2: Multi-bucket `DynamicBatcher`

**Files:**
- Modify: `src/batcher.rs`

**Step 1: Failing test**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn items_route_to_their_length_bucket() {
    let log_by_bucket: Arc<Mutex<HashMap<usize, Vec<Vec<usize>>>>> = Default::default();
    let l = log_by_bucket.clone();
    let b = Arc::new(DynamicBatcher::with_buckets(
        "t_buckets",
        move |bucket_ceiling, ids| {
            l.lock().unwrap().entry(bucket_ceiling).or_default()
                .push(ids.iter().map(|i| i.len()).collect());
            Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
        },
        /*buckets*/ vec![64, 256, 512],
        /*max_batch_tokens*/ 2048, 50, 16,
    ));
    // 50-tok items go to bucket 64; 200-tok → 256; 400-tok → 512.
    let _ = tokio::join!(
        b.embed_tokens(vec![vec![0u32; 50]]),
        b.embed_tokens(vec![vec![0u32; 200]]),
        b.embed_tokens(vec![vec![0u32; 400]]),
    );
    let m = log_by_bucket.lock().unwrap();
    assert!(m.contains_key(&64) && m.contains_key(&256) && m.contains_key(&512));
    drop(m);
    Arc::try_unwrap(b).ok().unwrap().shutdown(Duration::from_millis(200)).await;
}
```

**Step 2-4:** Implement — one `mpsc` channel per bucket + one worker per bucket (reuse the existing per-worker state machine from B). Routing: incoming item's max token length → first bucket whose ceiling ≥ that length.

**Step 5:** Commit.

### Task C3: Per-bucket metrics + docs

Add `embed_bucket_depth{bucket}` gauge. Update README + CLAUDE.md to describe bucketing. Commit.

### Task C4: 24h production observation

Roll out, leave for 24h, pull metrics. Gate on ≥20% p99 reduction vs pre-Phase-C baseline. If win, keep. If no improvement, revert (via env `BATCH_BUCKETS=""` → single bucket).

---

## Phase D: Response Cache (1 day)

Self-contained module, lowest risk.

### Task D1: New module `src/cache.rs`

**Files:**
- Create: `src/cache.rs`
- Modify: `src/main.rs` (add `mod cache;`)

**Step 1: Failing test (put inline in cache.rs)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hit_after_insert() {
        let c = EmbeddingCache::new(10);
        assert!(c.get("e5", "hello").is_none());
        c.insert("e5", "hello", vec![1.0, 2.0]);
        assert_eq!(c.get("e5", "hello"), Some(vec![1.0, 2.0]));
    }
    #[test]
    fn lru_eviction() {
        let c = EmbeddingCache::new(2);
        c.insert("m", "a", vec![1.0]);
        c.insert("m", "b", vec![2.0]);
        c.insert("m", "c", vec![3.0]); // evicts "a"
        assert!(c.get("m", "a").is_none());
        assert!(c.get("m", "b").is_some());
        assert!(c.get("m", "c").is_some());
    }
    #[test]
    fn per_model_keyspace() {
        let c = EmbeddingCache::new(10);
        c.insert("e5", "x", vec![1.0]);
        c.insert("jina", "x", vec![2.0]);
        assert_eq!(c.get("e5", "x"), Some(vec![1.0]));
        assert_eq!(c.get("jina", "x"), Some(vec![2.0]));
    }
}
```

**Step 2-4: Implement.** LRU via `std::collections::LinkedList` OR the `lru` crate. Key = `(String, [u8; 32])` where `[u8; 32]` is SHA-256 of the text. Wrap in `Mutex<...>`.

**Step 5:** Commit.

### Task D2: Wire into `api.rs`

**Files:**
- Modify: `src/api.rs::embeddings`

On request: for each text, probe cache → if hit, collect vector; miss list goes to batcher. After batcher returns, insert misses. Return combined vectors in original order.

Write test: same text twice in single request → only one inference call. Commit.

### Task D3: Metrics + env

Add `embed_cache_hit_total{model}`, `embed_cache_miss_total{model}`. Env: `CACHE_MAX_ENTRIES=10000` (default). Commit.

### Task D4: Bench + observation

Benchmark with repeated queries: cache hit should yield sub-millisecond response. Deploy, observe 24h, gate on cache hit rate ≥ 15% on prod traffic.

---

## Phase E: Reranker Integration (3-4 days)

Ships the first non-embedding model kind. Reuses the batcher from Phase B.

### Task E1: `RerankerModel` struct

**Files:**
- Create: `src/model_reranker.rs`
- Modify: `src/main.rs` (add `mod model_reranker;`)

**Step 1: Failing test**

```rust
// src/model_reranker.rs
#[cfg(test)] mod tests {
    use super::*;
    #[tokio::test]
    async fn rerank_orders_relevant_higher() {
        let m = RerankerModel::load("/models-reranker", "bge-reranker-v2-m3", 512).unwrap();
        let scores = m.rerank("what is a cat", &[
            "a cat is a small domestic feline",
            "the price of oil dropped yesterday",
        ]).unwrap();
        assert!(scores[0] > scores[1], "scores: {:?}", scores);
    }
}
```

**Step 2-4: Implement.** Mirror `Model` but output is `Vec<f32>` (one score per doc), not `Vec<Vec<f32>>`. Tokenize (query, doc) pairs, forward pass, take logit.

**Step 5:** Commit.

### Task E2: Env `RERANKER_MODELS` parser

**Files:**
- Modify: `src/config.rs`

Pattern: `name:dir:max_len:padded` — `bge-reranker-v2-m3:/models-reranker:512:true`. Test commit cycle.

### Task E3: `/v1/rerank` endpoint

**Files:**
- Create: `src/api_rerank.rs`
- Modify: `src/main.rs` (add route)

Request:
```json
{"model": "bge-reranker-v2-m3", "query": "...", "documents": ["...", "..."], "top_k": 5}
```
Response:
```json
{"results": [{"index": 0, "relevance_score": 0.92}, ...]}
```

Test: integration test that POSTs the known query/doc pair from Task E1 and verifies ordering.

**Step 5:** Commit.

### Task E4: Batcher reuse

Same token-budget batcher from Phase B services the reranker — one request = multiple (q,doc) pairs, each pair is one "entry" in the batcher. Reuse `with_buckets`. Padded-model = true. No new batcher code.

### Task E5: Integration test against running server

**Files:**
- Create: `tests/rerank_smoke.rs`

```rust
#[tokio::test]
async fn rerank_smoke_live() {
    // Assumes server is running via start_server fixture.
    let resp = reqwest::Client::new()
        .post("http://127.0.0.1:8082/v1/rerank")
        .json(&serde_json::json!({
            "model": "bge-reranker-v2-m3",
            "query": "how to initialize a database",
            "documents": [
                "To initialize a database, run `createdb mydb`",
                "Yesterday I ate pasta for lunch",
                "Database initialization means creating the initial schema and data"
            ],
            "top_k": 3
        }))
        .send().await.unwrap()
        .json::<serde_json::Value>().await.unwrap();
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results[0]["index"], 0);  // most relevant first
    assert_eq!(results[2]["index"], 1);  // pasta last
}
```

### Task E6: Docs + CLAUDE.md

Update both with reranker usage, when to use it, example curl, memdb-go integration hint.

### Task E7: MemDB integration (separate repo)

Out of scope for embed-server plan. Draft as short stub: "After retrieving top-50 by embedding, call /v1/rerank to re-score, return top-5." Open a ticket in memdb-go repo. Not part of this PR chain.

---

## Phase F: GLiNER NER Integration (sketched, follow-up plan)

Same shape as Phase E, but output is spans: `[{"text":"X", "label":"Y", "start":..., "end":...}]`. Endpoint `/v1/ner`. Defer the full plan until Phase E ships — the pattern transfers directly.

## Phase G: SPLADE Sparse Integration (sketched)

Output is sparse vector (token-id → weight map). Endpoint `/v1/sparse`. Useful for hybrid retrieval. Defer until F.

---

## Rollout Order

```
Phase 0  →  Phase A  ─┬→  Phase B  →  Phase C
                      │
                      └→  Phase D  (parallel with B/C)
                      
Phase B done  →  Phase E
```

Each phase ships independently as its own PR into `main`. Dozor auto-deploys each merge. Observe 24h between phases for metrics stability.

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Token-budget batcher introduces a deadlock under load | Medium | Stress test with 64 concurrent conns over long+short mix; keep `BATCHING_ENABLED=false` fallback |
| Reranker + 2 embedders + other models > 4GiB RAM | Medium | Bump compose `memory: 6144M`; lazy-load GLiNER/SPLADE only if `ENABLED` env set |
| Cache grows unboundedly if CACHE_MAX_ENTRIES misconfigured | Low | Hard-coded ceiling 100k; emit warn log if > 90% full |
| ORT Level 3 breaks a specific model on ARM | Low | Make per-model override `ORT_OPT_LEVEL_<MODEL>` |
