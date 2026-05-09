# Known Issues & Bugs

## BUG-001: ort crate 1000x slowdown on ARM for BERT models with token_type_ids

**Status:** RESOLVED (2026-04, jina-code-v2 served natively by Rust embed-server)
**Severity:** Critical
**Date:** 2026-03-06 (opened) / 2026-04 (closed)
**Component:** pykeio/ort v2.0.0-rc.12, ONNX Runtime 1.24.x, ARM Neoverse-N1

### Resolution

The current `jina-code-v2` ONNX file ships with only `[input_ids, attention_mask]`
inputs (no `token_type_ids`), which avoids the 3-named-input code path in `ort`
that triggered the 1000x slowdown. With that file, `embed-server` runs jina
natively in Rust — the Python `embed-jina` sidecar was retired in April 2026
and archived at `github.com:anatolykoptev/ox-embed-jina` (tag `retired-2026-04-17`).

Verified in prod: jina-code-v2 p50 ~1.7 s single-query, ~2.4 s at conc=4
(Oracle ARM Neoverse-N1, 4 vCPU). No 30 s outliers observed.

If a future ONNX export re-adds `token_type_ids`, the slowdown may return —
keep the 2-input file until `ort` ships a verified fix upstream. Monitoring
pykeio/ort releases remains prudent.

### Problem

The Rust `ort` crate produces 1000x slower inference for certain BERT-family ONNX models on ARM (Neoverse-N1 / Oracle Cloud A1). Specifically:

| Model | ort (Rust) | Python onnxruntime | Go onnxruntime_go |
|-------|-----------|-------------------|-------------------|
| multilingual-e5-large (XLM-RoBERTa) | 100ms | ~80ms | ~200ms |
| jina-code-v2 (BERT with token_type_ids) | **30,000ms** | **60ms** | ~150ms |

Same ONNX files, same ORT version (1.24.3), same machine. The slowdown is specific to:
- ARM architecture (not reproduced on x86)
- Models requiring `token_type_ids` input (BERT-family)
- `ort` crate specifically (Python and Go bindings work fine)

### What was tested

1. **ORT versions**: 1.24.1 (even slower: 43s jina, 2.2s e5), 1.24.3 (30s jina)
2. **token_type_ids=false**: Model crashes ("Missing Input: token_type_ids")
3. **Internalized token_type_ids in ONNX graph**: Still slow (10-19s)
4. **fp32 model (no quantization)**: Progressively slower (25s->153s, memory thrashing)
5. **jina-only (no e5 loaded)**: Still 24-33s with only 604MiB memory
6. **Thread configs** (auto, 4 intra + 1 inter): No improvement
7. **ONNX graph optimization** (Gelu/LayerNorm/SkipLayerNorm fusions): No improvement for ort

### Workaround

Deployed a Python sidecar (`embed-jina`) for jina-code-v2 using native `onnxruntime` package. The Rust `embed-server` handles only e5-large (which works fine in ort).

### Root cause hypothesis

Likely a bug in ort crate's tensor memory layout or session execution path for models with 3 named inputs on ARM. The Python and Go ORT bindings use different FFI approaches that don't trigger this issue.

### To monitor

- pykeio/ort releases: check if fixed in future versions
- Test with ort v3.x when available
- Consider filing upstream issue at https://github.com/pykeio/ort/issues

---

## BUG-002: ONNX models require graph optimization for ARM

**Status:** RESOLVED
**Severity:** High
**Date:** 2026-03-06

### Problem

Unoptimized ONNX models run ~50x slower on ARM without AVX instructions. The ONNX Runtime graph optimizer fuses operations (Gelu, LayerNormalization, SkipLayerNormalization) that are critical for ARM NEON performance.

### Solution

Pre-optimize models using `onnxruntime.transformers.optimizer`:

```python
from onnxruntime.transformers.optimizer import optimize_model

# jina-code-v2
m = optimize_model('model_quantized.onnx', model_type='bert',
                   num_heads=12, hidden_size=768, opt_level=0)
m.save_model_to_file('model_optimized.onnx')

# multilingual-e5-large
m = optimize_model('model_quantized.onnx', model_type='bert',
                   num_heads=16, hidden_size=1024, opt_level=0)
m.save_model_to_file('model_optimized.onnx')
```

Note: `opt_level=0` skips torch-dependent ORT optimization. The graph fusions alone provide the speedup. The `onnx` pip package is required (not just `onnxruntime`).

### Fusions applied (jina-code-v2)

- Gelu: 12
- LayerNormalization: 24
- SkipLayerNormalization: 37

---

## BUG-003: Python wsgiref.simple_server doesn't flush responses with ThreadingMixIn

**Status:** RESOLVED
**Severity:** High
**Date:** 2026-03-06

### Problem

`wsgiref.simple_server.WSGIServer` combined with `socketserver.ThreadingMixIn` accepts HTTP requests and processes them, but never sends the response back to the client. The client times out despite the server completing the work.

This happens because:
1. `WSGIServer` defaults to HTTP/1.0
2. `ThreadingMixIn` spawns threads that don't properly close connections
3. Without explicit `Content-Length` headers, HTTP/1.0 relies on connection close to signal end of response
4. The threading layer interferes with connection lifecycle

### Solution

Replace `wsgiref` WSGI app with `http.server.BaseHTTPRequestHandler`:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # Important: use HTTP/1.1

    def do_POST(self):
        # Always set Content-Length explicitly
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
```

Key requirements:
- Use `HTTP/1.1` protocol version
- Always send explicit `Content-Length` header
- Use `BaseHTTPRequestHandler` (not WSGI)

---

## BUG-004: jina-code-v2 BFCArena OOM under unbounded `input` arrays + pool=2 concurrency

**Status:** RESOLVED (2026-05-09, 3-layer fix shipped same day)
**Severity:** Critical (~1 failure/min in prod for ~30 min)
**Date:** 2026-05-09 (incident discovered, escalated, fully fixed)
**Component:** ox-embed-server batcher + ONNX Runtime BFCArena + downstream HTTP clients (memdb-go, go-code via go-kit/embed.Client)

### TL;DR

`POST /v1/embeddings` accepted unbounded `input: []` arrays. Memdb-go shipped requests with up to **100 docs/call**. Each became one batcher Item with `n_texts=100` — admitted unconditionally by the batcher's first-item starvation guard. The invariant lives in `BatchAccum::seq_capped_by`'s doc comment (`src/batcher.rs`): *"the empty-batch path admits the first item unconditionally so a single long-doc request never starves"*. The bypass also covers `BatchAccum::fits` because the worker only calls `fits` after the first item is already admitted. Result: `BATCH_MAX=8` is silently exceeded by any single Item with `n_texts > 8`. For jina-code-v2 (12 heads, max_len=512), B=100 allocates:

```
B × H × S² × 4 bytes  =  100 × 12 × 512² × 4  =  1,258,291,200 bytes  (1.258 GB)
```

…of attention scratch per inference. With `EMBED_SESSION_POOL_SIZE=2` (PR #99 latency win), two parallel slots required ~2.5 GB scratch — exceeding the BFCArena cap (3 GiB per PR #98) → `BFCArena::AllocateRawInternal` failure → HTTP 500 + `embed_inference_failures_total{model="jina-code-v2",reason="arena_oom"}`.

### Trigger conditions

| Layer | Pre-incident value | Effect on bug |
|---|---|---|
| memdb-go `texts []string` | unbounded | sends 100-doc arrays |
| ox-embed-server batcher | first-item-admit unconditional | bypasses BATCH_MAX=8 |
| jina max_len | **512** (PR #80, May 7) | quadratic scratch growth — was 256 before |
| `EMBED_ARENA_MAX_MEM_BYTES` | 3 GiB (PR #98) | leaves only ~1.16 GiB free for scratch |
| `EMBED_SESSION_POOL_SIZE` | **2** (PR #99) | 2 concurrent jina batches → 2.5 GiB scratch |

Each PR was safe in isolation. The combination + the latent unbounded-array bug in clients produced emergent failure. Failures appeared after PR #48 (EvictablePool) deploy + PR #99 (pool=2) recreate cleared the BFCArena and ramp-up exposed the worst case.

### Discovery

`mcp__go-code__debug_investigate` on the embed-server service window after PR #48 deploy showed:

- `embed_request_duration_seconds_p99` spike **×93.77**
- `embed_inference_failures_total` rate **+Inf** (counter appeared from zero)
- `EmbedBatchWaitHigh` alert: `p95 batch wait > 500ms for 5m (model=jina-code-v2)`
- 76× ERROR logs in 13 sec on `encoder/layer.0/attention/self/Add` node

Three passes of `debug_investigate` traced the root cause from latency symptom → metrics → code:

1. **Pass 1**: caught the failure spike + log pattern. Initially suspected EvictablePool eviction.
2. **Pass 2 (after PR #102 EvictablePool rollback)**: failures **continued** — proof EvictablePool was a victim, not the cause. Pointed at pool=2 + arena cap interaction.
3. **Pass 3 (after PR #103 arena 3→4 GiB raise)**: failures STILL continued at lower rate. `embed_inference_attention_scratch_bytes` histogram showed 35/53 jina inferences in the 1-4 GB bucket. Reverse-math: `1.258 GB / (12 × 4) = B × S²` → with S=512, `B=100`. With BATCH_MAX=8, this is impossible from coalesced batches — must be **single Item with n_texts=100**.

`mcp__go-code__understand` on `EmbedModel.embed_tokens` + `BatchAccum.fits` confirmed the batcher first-item-admit gate.

### Resolution — 3-layer fix shipped 2026-05-09

**Layer 1 — Server-side cap (`ox-embed-server` PR #49, commit `4b5b72e`):**

- New env `EMBED_MAX_INPUT_ARRAY` (default 32, matches arena headroom: `32 × 12 × 512² × 4 = 402 MiB ≪ 4 GiB - weights`).
- `POST /v1/embeddings` rejects `input.len() > cap` with **HTTP 400** + JSON body `{error:{type,code,message,cap,received}}`. Permanent client misuse, no retry. Note: `/embed_sparse` (SPLADE) carries the `embed_max_input_array` value on `AppState` but does NOT enforce the cap in its handler — pre-existing gap, see Known Limitations item 5.
- Three new metrics:
  - `embed_input_array_size{model}` histogram — natural distribution
  - `embed_input_array_rejected_total{model,reason}` counter — rejects observed
  - `embed_batcher_first_item_oversize_total{model,reason}` — fires when first-item-admit accepts an Item that would have failed normal `fits()`

**Layer 2 — Client-side chunking in `go-kit/embed.Client` (go-kit PR #48 → tagged v0.49.0; v0.50.0 includes it):**

- Transparent client-side chunking: `Client.Embed()` and `Client.EmbedWithResult()` auto-split `len(texts) > chunkSize` into sequential sub-batches (default 32, env `GOKIT_EMBED_CHUNK_SIZE`).
- Chunking gate placed **above** fallback routing (BLOCKER fix from review): fallback-wired clients also chunk. `dispatchChunk()` helper preserves fallback semantics per chunk.
- Cache placement: **above** chunking — per-text cache keys retain granularity (chunk where every text is cached returns 0 backend calls).
- Sequential dispatch (not parallel) — server-side batcher already coalesces concurrent calls; parallel client chunks would just cause batcher contention.
- `ErrDimMismatch.Index` plumbed through: `validateDim` returns per-vector index within chunk; `embedChunked` maps to absolute caller-facing position via `i + de.Index`.
- New metrics: `embed_chunks_per_call{model}` (1× per call, all paths), `embed_chunk_size{model}` (per dispatched sub-batch).

**Layer 3 — Downstream consumers:**

- `memdb-go` PR #311 (commit `63ad3876`): bumped go-kit to v0.50.0, removed wrapper-level chunking duplicate from `internal/embedder/http.go` (PR #310 was a temporary defense-in-depth pre-go-kit-lift). `MEMDB_EMBED_CHUNK_SIZE` aliased to `WithChunkSize` opt at construction so existing operator config keeps working.
- `go-code` PR #91 (commit `dc9d25a`): already on go-kit v0.50.0 (transparent protection — no code change at call sites).

**krolik-server compose updates (PR sequence):**

- PR #98 (pre-incident): `EMBED_ARENA_MAX_MEM_BYTES` 6→3 GiB
- PR #99 (pre-incident): `EMBED_SESSION_POOL_SIZE` 1→2 (latency win)
- PR #102 (incident response v5): `EMBED_IDLE_EVICT_SECS` 0→600 — **ROLLBACK** (caused 38 cold-start re-init OOM cascade)
- PR #103 (incident response v6): arena 3→4 GiB, mem 6→7 GiB — partial relief but failures continued (root cause not yet identified)
- PR #104 (incident response v6 emergency): `EMBED_SESSION_POOL_SIZE` 2→1 — **band-aid** stopped failures by serialising scratch
- PR #107 (incident close v7): pool 1→2 restored after layered fix shipped — verified 0 failures + 0 server rejects in 30+ min before restore

### Verification commands

After deploy, verify the 3-layer fix is wired:

```bash
# Server-side cap working:
curl -s -X POST http://127.0.0.1:8082/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"jina-code-v2\",\"input\":[$(seq -s, -f '"x"' 1 33)]}"
# Expects: HTTP 400 + {error:{code:"input_array_too_large",cap:32,received:33}}

# Server cap counter:
curl -s http://127.0.0.1:8082/metrics | grep embed_input_array_rejected_total

# Client chunking metric (memdb-go side, after first chunked call):
curl -s http://127.0.0.1:8080/metrics | grep embed_chunks_per_call
# Expects: histogram with sample count > 0; sum equals total chunk count
```

### Lessons

1. **Multiple safe PRs can produce emergent failure.** Each of #80 / #98 / #99 was reviewed individually and looked safe. The combination + the latent unbounded-array bug (existed for ~year+ without surfacing because old config absorbed it) gave the OOM.
2. **`debug_investigate` cross-references metrics + traces + symbols across boundaries.** Single-source observation (logs only, metrics only) would not have pinpointed the batcher first-item-admit path. Three iterative passes narrowed from latency spike → arena cap interaction → unbounded-array root cause.
3. **Defense in depth.** Server-side cap protects from any future client. Client-side chunking in shared lib protects all current consumers transparently. Either layer alone would have stopped this incident.
4. **Emergency band-aids must be marked TEMPORARY.** PR #104 (pool=1) stopped the bleeding but degraded p95 latency. The fix isn't complete until the band-aid is removed (PR #107).
5. **Reviewer is gold for concurrency code.** The go-kit PR #48 reviewer caught a BLOCKER (WithFallback bypassing chunking gate) that would have caused silent failures for any caller using fallback chains. Tests passed before the fix — review caught what tests didn't.

### Cross-references

- ox-embed-server: PRs #46 (ALiBi precompute), #47 (per-model ONNX cache + arena docs), #48 (EvictablePool port), #49 (server cap), #50 (this BUG-004 doc)
- go-kit: PR #48 → tag v0.49.0/v0.50.0 (lift chunking to shared Client)
- memdb-go: PR #310 (wrapper chunking — temporary), #311 (cleanup + bump v0.50.0)
- go-code: PR #91 (bump go-kit v0.50.0)
- go-search: PR #19 (bump go-kit v0.37.1 → v0.50.0), #20 (explicit `WithChunkSize(32)` opt + dead-code cleanup)
- krolik-server: PRs #98, #99, #102 (rollback), #103, #104 (band-aid), #107 (restore)

### Service version map (post-incident)

All consumers of `go-kit/embed.Client` should be on **v0.50.0** to get transparent chunking:

| Service | go-kit version | Notes |
|---|---|---|
| `ox-embed-server` | n/a (Rust) | Server-side `EMBED_MAX_INPUT_ARRAY=32` cap |
| `memdb-go` | v0.50.0 | PR #311 — wrapper chunking removed, delegates to go-kit |
| `go-code` | v0.50.0 | PR #91 — transparent protection |
| `go-search` | v0.50.0 | PR #19 + #20 — explicit `WithChunkSize(32)` |

If a future consumer is added, bump go-kit ≥ v0.49.0 (chunking) or v0.50.0 (chunking + tracing/httpmw) at construction.

### Known limitations (followup-tracked)

These are gaps deliberately left open at the time of BUG-004 closure. Each has a specific followup PR scope:

1. **`/v1/rerank` documents-array cap** — ✅ **CLOSED by PR #52** (commit on main). Added `RERANK_MAX_INPUT_DOCS` env (default 32), HTTP 400 on overflow, new metrics `embed_rerank_input_docs_size{model="unknown"}` + `embed_rerank_input_docs_rejected_total{model="unknown",reason}` (the `model="unknown"` label is a placeholder for schema symmetry with embed cap counters — the cap fires before model resolution from request body, so the actual model is not yet known at recording time).

2. **`embed_chunks_per_call` histogram low informativeness for go-search** — go-search's pipeline never exceeds 32 texts per call (`MAX_FETCH_URLS+1=9` max, `embedding_answer` caps at 24). The histogram will always show `chunks_per_call=1` for go-search labels. Operators watching `/metrics` see new series with no signal. Not harmful — Noted only.

3. **Verification command portability** — the `seq -s, -f '"x"' 1 33` pattern in the verification block above is bash/Linux only (different `seq` flag semantics on macOS). Operators on macOS workstations should generate the payload differently. Noted; not blocking.

4. **`go-kit/embed` env parsing inconsistency** — go-kit PR #48 introduced env reading inline in `client_v2.go` (warn-on-failure pattern) rather than going through a shared factory.go-style helper. Not a bug — just style drift inside the lib. Noted in PR #48 review NIT.

5. **`/embed_sparse` (SPLADE) input-array cap not enforced** — `src/api_splade.rs` carries `embed_max_input_array` field on `AppState` (test constructor sets it), but the live handler never checks `req.input.len() > cap`. PR #49 originally claimed both `/v1/embeddings` and `/embed_sparse` were capped — fact-check during PR #52 review showed only `/v1/embeddings` enforces. SPLADE uses a different pool (smaller models, lower scratch) so the immediate OOM risk is lower than embed/rerank, but the same class-of-bug recurrence-risk pattern applies for any future model swap. **Followup PR**: mirror PR #49's pattern in `api_splade.rs` (reuse same `EMBED_MAX_INPUT_ARRAY` env, same `embed_input_array_rejected_total{model,reason}` counter labels — distinguishes via `model` label which already varies per endpoint).
