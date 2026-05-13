# Multi-Process Architecture

Since 2026-05-12 (Phase 2 refactor, PR #57 + #58 + follow-ups) embed-server runs as **supervisor + N child workers** instead of a single monolithic process. This document is the canonical architecture reference.

## Why multi-process

Pre-refactor: 4 ONNX models (e5-large, jina-code-v2, gte-multi-rerank, splade-v3-distilbert) loaded in the same process. Sessions shared one ORT BFCArena (CPU allocator). `jina-code-v2` (variable-seq, ALiBi attention) caused arena fragmentation cycles — under concurrent load with the other models present, `jina-code-v2` hit `arena_oom` errors on **92 % of requests** (BUG-004, incident 2026-05-12).

Per-process arena = per-model. Worker for `jina-code-v2` has its own arena, not shared with `e5-large` / `gte-multi-rerank` / `splade`. Fragmentation in one worker does not poison the others.

## Topology

```
  HTTP clients (memdb-go, go-code, go-nerv, etc.)
                          │
                          ▼
                ┌─────────────────────┐
                │  embed-server PID 1 │  (axum HTTP, in-process models still loaded for legacy fallback)
                │  :8082              │
                └─────────┬───────────┘
                          │
                  ┌───────┴───────┐
                  │  WorkerPool   │  HashMap<model_name, Arc<WorkerSupervisor>>
                  └───────┬───────┘
                          │ per-request UDS connect (cancel-safe)
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ embed-worker  │ │ embed-worker  │ │ embed-worker  │ │ embed-worker  │
│ KIND=embed    │ │ KIND=embed    │ │ KIND=rerank   │ │ KIND=splade   │
│ e5-large      │ │ jina-code-v2  │ │ gte-multi-rer.│ │ splade-v3-d.  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘

UDS sockets: /tmp/embed-workers/<model>.sock
```

## Process boundaries

### Supervisor (`embed-server` binary)
- Owns axum HTTP server (`:8082`), routing, metrics, response cache.
- Loads in-process model sessions (legacy path — used when `EMBED_MULTI_PROCESS=0`, fallback when worker_pool dispatch returns "no worker for model X").
- Spawns + monitors workers (`tokio::process::Command` with `kill_on_drop(true)`).
- Routes `/v1/embeddings`, `/v1/rerank`, `/embed_sparse` to workers when `worker_pool.is_some()`.

### Worker (`embed-worker` binary)
- One process per model. `EMBED_WORKER_KIND` env (`embed` / `rerank` / `splade`) selects which `StandaloneXxx` wrapper to load.
- Owns: one OrtEnv, one BFCArena, N ONNX Sessions (from `EMBED_WORKER_POOL_SIZE`), tokenizer.
- Listens on UDS socket, accepts one connection per request, processes via `spawn_blocking` (ONNX is sync CPU-bound), returns response, closes connection.
- On panic or OOM-kill: exits. Supervisor watchdog respawns with backoff 2s→60s.

## IPC protocol — `WorkerRequest` / `WorkerResponse`

Wire format: **postcard** (varint-encoded, serde-compat). bincode 2.x was replaced after RUSTSEC-2025-0141 (upstream unmaintained).

Frame layout: `u32 LE length` + `payload bytes` (max 64 MiB per frame, enforced both sides).

Tagged enums (defined in `src/ipc/protocol.rs`):

```rust
pub enum WorkerRequest {
    Embed(EmbedRequest),     // { request_id, model, texts, max_seq_len }
    Rerank(RerankRequest),   // { request_id, model, query, documents, max_seq_len }
    Splade(SpladeRequest),   // { request_id, model, texts, max_seq_len, top_k, min_weight }
}

pub enum WorkerResponse {
    Embed(EmbedResponseOk),  // { request_id, vectors: Vec<Vec<f32>>, dims }
    Rerank(RerankResponseOk),// { request_id, scores: Vec<f32> }
    Splade(SpladeResponseOk),// { request_id, sparse: Vec<Vec<(u32, f32)>> }
    Err { request_id, message },
}
```

Invariants:
- `WorkerClient.send_request` verifies `response.request_id() == request.request_id()`. Mismatch → `io::Error::InvalidData` (cancel-safety guard).
- Each `dispatch_*` method on `WorkerClient` verifies cardinality (e.g. `vectors.len() == texts.len()` for embed; `scores.len() == documents.len()` for rerank).
- Per-request UDS connection (PR #62). Persistent pool was cancel-unsafe — `Mutex<UnixStream>` across `write_frame + read_frame` left a buffered response when the caller's future was cancelled, poisoning the next caller. UDS local-domain connect ~10–100 µs vs ONNX inference 5–50 ms — negligible overhead.

## Supervisor lifecycle

`WorkerSupervisor` (in `src/supervisor/handle.rs`) is an actor that owns the worker `Child` for its entire lifetime.

```
launch() ──► spawn_one() ──► return Arc<Self>
                │
                └── tokio::spawn(watchdog_loop)
                        │
                        ▼
                   ┌─ loop:
                   │   await child.wait()         // worker exited
                   │   log exit code + signal
                   │   clear client_slot          // dispatchers see "unavailable"
                   │   sleep(backoff)             // 2s → 4s → 8s … 60s cap
                   │   spawn_one() retry
                   │   on success: restore client_slot, increment restart_count, reset backoff
                   │   on failure: keep backoff growing
                   └─ (forever — actor is process-lifetime)
```

- `Arc<WorkerSupervisor>` cycle: watchdog task holds `Arc<Self>` strong ref. Supervisor never drops while task runs. Intentional (process-lifetime).
- Exit codes 134 (SIGABRT from `panic=abort` profile) and 137 (SIGKILL / OOM) handled identically — any exit triggers respawn.
- `client_slot: RwLock<Option<Arc<WorkerClient>>>` — `None` during respawn. `WorkerPool::dispatch_*` polls 200 ms for client up to `dispatch_timeout` (default 30 s), then returns `"worker for model X unavailable after Ns"`.

## Memory cost

ort 2.0-rc.12 does NOT mmap weights from disk — each ONNX session allocates its own private anonymous memory for weights. **Workers do NOT share weight buffers.** This was assumed-positive in the original plan, verified-negative empirically (Phase 3.1 — `pmap` showed `shared: 0` per worker).

Combined RSS post-deploy:
- 4 worker processes: ~2.7 GiB (e5: ~1.4 GiB, jina: ~0.6 GiB, rerank: ~0.4 GiB, splade: ~0.2 GiB).
- Supervisor with in-process duplicate: ~2.4 GiB.
- **Total: ~5.1 GiB** (was ~1.6 GiB pre-refactor; cost is ~3.5 GiB, of which ~2.4 GiB is the in-process duplicate and ~1.1 GiB is real worker overhead).

Phase 3.2 (deferred) would skip in-process loading when `EMBED_MULTI_PROCESS=1`, saving ~2.4 GiB. Requires handler refactor — out of scope for the current ship.

Host-level: swap usage dropped from 15 GiB pegged (pre-refactor jina arena thrashing) to ~7 GiB after multi-process deploy.

## Observability

| Metric | Type | Notes |
|--------|------|-------|
| `embed_worker_restart_total{model}` | counter | Pre-touched to 0 at supervisor launch. Increments on watchdog respawn. |
| `embed_requests_total{model,status}` | counter | Shared with legacy path. |
| `embed_inference_duration_seconds{model}` | histogram | Pre-existing. Includes IPC round-trip when routed via worker. |
| `embed_build_info{version}` | gauge | Set to `EMBED_VERSION` env, default `phase-2-multi-process`. |
| `embed_arena_*` | gauges/counters | Per-supervisor (in-process arena), not per-worker. |

Worker-side metrics are NOT exposed (workers don't run their own Prometheus exporter — would require multi-process aggregation). Operationally tracked via supervisor logs + per-process RSS via `docker top` + `/proc/<pid>/status`.

## Deploy + rollback

Enable: `EMBED_MULTI_PROCESS=1` in `~/deploy/krolik-server/compose/memdb.yml` (live default since 2026-05-12).

Disable: set to `"0"` and `docker compose up -d --no-deps --force-recreate embed-server`. Behaviour reverts byte-identical to pre-2026-05-12 monolith — in-process `EmbedModel`/`RerankerModel`/`SpladeModel` handle inference, workers don't spawn.

Image: dozor rebuilds on every push to `main` (~3 min). Smoke timeout was tuned to 120 s (was 30 s, caused false-rollback before parallel worker spawn landed in PR #61).

## Related files

| File | Role |
|------|------|
| `src/bin/worker.rs` | Worker binary entrypoint |
| `src/supervisor/handle.rs` | `WorkerSupervisor` actor + watchdog |
| `src/supervisor/pool.rs` | `WorkerPool` routing |
| `src/ipc/protocol.rs` | Wire types |
| `src/ipc/client.rs` | Supervisor-side `WorkerClient` |
| `src/ipc/frame.rs` | postcard frame codec |
| `src/main.rs` | Parallel worker spawn in `main()` |
| `src/api*.rs` | Handler dispatch through `worker_pool` |
| `docs/superpowers/plans/2026-05-12-multi-process-refactor.md` | Original implementation plan + execution log |
| `docs/runbook.md` | Operational symptoms + responses |
| `docs/BUGS.md` | BUG-004 (motivating incident) |

## History

- 2026-05-12 (PR #56) — Phase 1: IPC scaffold, worker binary skeleton, `StandaloneEmbedder`. No behaviour change.
- 2026-05-12 (PR #57) — Phase 2: supervisor + worker_pool + watchdog + embed routing + Dockerfile ships both binaries.
- 2026-05-12 (PR #58) — Wave 2.4b: IPC tagged enums (rerank, splade variants), generic `LoadedModel` worker dispatch, `api_rerank.rs` + `api_splade.rs` route through workers.
- 2026-05-12 (PR #59, #60) — Dockerfile hotfixes: stub all 3 cargo targets + touch `src/lib.rs` to invalidate empty stub artifact in Layer 2.
- 2026-05-12 (PR #61) — Parallel worker spawn via `tokio::spawn` (startup 3-5× faster, was sequential await).
- 2026-05-12 (PR #62) — **Critical**: per-request UDS conn replaces persistent pool. Fixes rerank cancel-safety regression (100% failure rate when downstream timeouts cancelled futures mid-write).
- 2026-05-12 (PR #63) — Pre-touch `embed_worker_restart_total` counter to 0 at launch. Observability gap.
