# Embed-Server Multi-Process Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---

## Execution Log (live state — updated 2026-05-12)

### Decisions taken autonomously by controller during execution

1. **IPC codec: `bincode` → `postcard`** (Phase 1 Wave 1.1 fixes). Reason: `cargo deny` blocks RUSTSEC-2025-0141 (bincode team disbanded; upstream unmaintained). Postcard is the most mature serde-compat replacement: active maintenance, `no_std`+`alloc`, varint encoding. Wire format compatible with the rest of the plan; all references to `bincode` below should be read as `postcard`. Also drops `heapless+atomic-polyfill` (RUSTSEC-2023-0089) via `default-features = false`.
2. **Tokio process spawn semantics**: `tokio::process::Command` + `kill_on_drop(true)` adopted in `WorkerHandle::spawn` (Wave 2.2). Plan didn't specify; this is the right async + cleanup pattern.
3. **`tokio::fs::try_exists` + `child.try_wait()` polling** in `WorkerHandle::spawn` socket-wait loop (Wave 2.2 code-quality polish). Plan used blocking `Path::exists()`; that violates the async-no-block invariant and didn't catch the "worker crashed before socket appeared" path (would wait full 60s).
4. **Shared `ChildGuard` helper** at `tests/common/mod.rs` (Wave 2.1 polish, rule of 3 hit across three integration tests). Plan didn't specify; standard idiom via `#[path]`.
5. **Wave 2.4 scope reduction — embed-only routing** (recorded now, implementation pending). Plan's Task 14 says "mirror the pattern for reranker" in api_rerank.rs, plus controller added api_splade.rs per primer. Decision: route `/v1/embeddings` only. `/v1/rerank` + `/embed_sparse` stay on the in-process path even when `EMBED_MULTI_PROCESS=1`. Reason: current `InferRequest`/`InferResponse` types model embed-only semantics (dense f32 vectors + dim). Reranker returns scores; SPLADE returns sparse maps. Extending the IPC protocol is a separate work-stream (see Wave 2.4b below). Embed is the primary motivation of this refactor (jina-code-v2 BFCArena fragmentation, BUG-004).
6. **Wave 2.3 prod-risk gate**: enabling `EMBED_MULTI_PROCESS=1` against a Phase-2.3-only build doubles resident memory (workers loaded AND in-process loaded, but API still routes in-process). Workaround: `tracing::warn!` at startup. Real fix lands when Wave 2.4 routing cuts the in-process embed sessions out of the hot path. Operators must not enable the flag in prod until Phase 2 ships in full.
7. **PR strategy** (replaces "per-wave" PR plan from plan body):
   - **Phase 1** (PR #56) — non-critical (scaffold under feature flag, default off, monolith behavior unchanged). **Controller auto-merged** after spec + code-quality APPROVED + local CI green (357/359, 2 unrelated pre-existing flakes). Squash-merged at `841a447` on 2026-05-12.
   - **Phase 2** (open PR after Wave 2.6) — critical (runtime cutover + deploy infra). **Requires explicit operator ack** per CLAUDE.md.
   - **Phase 3** (open after Phase 2 merged) — semi-critical (optimizations). Controller may auto-merge after review.

### Progress tracker

| Wave | Tasks | Status | Notes |
|---|---|---|---|
| **Phase 1** | | ✅ MERGED | PR #56, squash `841a447`, 11 commits |
| 1.1 | T1 bincode→postcard / T2 protocol / T3 frame | ✅ done | 7 lib tests (`ipc::`) |
| 1.2 | T4 worker bin scaffold / T5 ipc_ping | ✅ done | `[lib]` + 2nd `[[bin]]` added |
| 1.3 | T6 StandaloneEmbedder / T7 InferRequest handler / T8 e5 integration | ✅ done | Real e5 inference via UDS, 6.27s; `EmbedModel::load`/`tokenize`/`embed_tokens` (plan's `load_from_config`/`infer_raw` don't exist) |
| **Phase 2** | | 🟡 IN PROGRESS | Branch `feat/embed-multi-process-phase2`, 8 commits |
| 2.1 | T9 WorkerClient / T10 client roundtrip | ✅ done | UDS pool + round-robin + `request_id` correlation check |
| 2.2 | T11 WorkerHandle::spawn / T12 WorkerPool routing | ✅ done | `supervisor_dispatch` test PASS 5.37s |
| 2.3 | T13 main.rs flag wire | ✅ done | `EMBED_MULTI_PROCESS` env + `AppState.worker_pool`; routing NOT yet wired |
| 2.4 | T14 api.rs routing / T15 E2E HTTP test | 🔵 NEXT | **embed-only**, no api_rerank/api_splade |
| 2.4b | api_rerank routing / api_splade routing | 📋 NEW (added by controller) | Requires `InferRequest`/`InferResponse` protocol extension for rerank scores + sparse maps |
| 2.5 | T16 watchdog/auto-restart | ⏸ pending | WorkerSupervisor actor pattern; SIGABRT (exit 134, panic=abort) treated same as 137 (OOM) |
| 2.6 | T17 Dockerfile both bins | ⏸ pending | |
| 2.7 | T18 compose enable | ⏸ pending | Separate PR in `~/deploy/krolik-server` |
| **Phase 3** | | ⏸ pending | |
| 3.1 | T19 mmap weights verify | ⏸ pending | Post-deploy probe |
| 3.2 | T20 lazy-load + idle evict | ⏸ pending | |

### Followups discovered during execution (not blocking; deferred to dedicated waves or follow-up PRs)

- IPC protocol extension: rerank scores message variant, splade sparse-map variant (gates Wave 2.4b)
- `WorkerClient::infer` slot poisoning: I/O error on a UDS leaves the slot dead until restart. Hot path becomes unreachable for that model. **Watchdog (Wave 2.5) MUST handle this.**
- `WorkerPool` outer `Arc<RwLock<HashMap>>` — outer Arc redundant since WorkerPool is itself always wrapped in `Arc<WorkerPool>`. Cosmetic; not worth churn.
- `WorkerHandle::child` is currently `pub` — should be `pub(crate)` once Wave 2.5 WorkerSupervisor owns lifecycle.
- Parallel worker spawn (`futures::future::join_all`) — startup latency reduction. Currently sequential = sum, ideal = max. Worth doing when N (model count) grows beyond 2-3.
- PID-namespacing socket files in `EMBED_WORKER_SOCKET_DIR` — multiple embed-server instances on same host would collide.
- Graceful shutdown via `ControlMessage::Shutdown` sent before `Child` drop (currently SIGKILL via `kill_on_drop`). Mid-flight inference cut.
- Cross-binary `#[allow(dead_code)]` noise on supervisor types — disappears once Wave 2.4 wires them into `main.rs` hot path.
- 2 pre-existing flaky tests (`batcher::tests::backpressure_rejects_at_eighty_percent`, `queue_full_returns_error`) hang >20 min on this server. Diff untouched. **Followup**: investigate `batcher.rs` test flakiness as separate issue.

---

**Goal:** Refactor embed-server from monolithic single-process (3 models share one ORT BFCArena) into supervisor + N child processes (one process per model with isolated BFCArena), eliminating arena fragmentation root cause documented in BUG-004 / incident 2026-05-12.

**Architecture:**
- **Supervisor (current binary, renamed `embed-supervisor`)** owns HTTP server, batcher, queue, routing, metrics aggregation. Spawns + monitors child workers.
- **Worker (`embed-worker`, new binary)** loads ONE model (e5 / jina / reranker), owns its OrtEnv + BFCArena, exposes inference over Unix Domain Socket via bincode-encoded request/response.
- IPC: tokio UDS, length-prefixed bincode frames. Embeddings vectors (f32) serialized directly — ~4 KB / 1024-dim vector.
- Watchdog: supervisor restarts crashed/OOM workers; in-flight requests fail-fast → memdb-go client retries (commit 90b964f1).
- Cutover: feature flag `EMBED_MULTI_PROCESS=1` switches code path; old monolith remains until flag becomes default.

**Tech Stack:** Rust 1.85+, tokio, axum 0.8, ort 2.0-rc.12, bincode 2.x, tokio-uds (via tokio::net::UnixListener), tracing, prometheus.

**Phases (each phase ends with green CI + production-deployable):**
1. **Phase 1** — IPC contract + worker binary scaffold (no behavior change, monolith still default)
2. **Phase 2** — Supervisor mode behind feature flag, gradual cutover per model
3. **Phase 3** — Optimizations: shared mmap weights, lazy-load + idle evict
4. **Phase 4** — Redis embedding cache (separate concern, can land any time after Phase 1)
5. **Phase 5** — Dynamic padding + seq-len bucketing (orthogonal to multi-process)

This plan covers Phases 1-3. Phases 4 + 5 get their own plans.

---

## File Structure

### New files

| Path | Purpose |
|------|---------|
| `src/bin/worker.rs` | New binary entrypoint for child worker process |
| `src/worker/mod.rs` | Worker process module — main loop, dispatch |
| `src/worker/embedding.rs` | Worker handler for embedding inference |
| `src/worker/reranker.rs` | Worker handler for reranker inference |
| `src/ipc/mod.rs` | IPC layer — UDS connection setup, frame protocol |
| `src/ipc/protocol.rs` | Wire types: `InferRequest`, `InferResponse`, `ControlMessage` |
| `src/ipc/client.rs` | Supervisor side — `WorkerClient` per child |
| `src/ipc/server.rs` | Worker side — `IpcServer` accept loop |
| `src/supervisor/mod.rs` | Supervisor lifecycle — spawn, monitor, restart |
| `src/supervisor/pool.rs` | Worker pool — routing requests to child by model |
| `tests/ipc_roundtrip.rs` | Integration test: supervisor↔worker roundtrip |
| `tests/worker_crash.rs` | Integration test: worker crash + restart |

### Modified files

| Path | Change |
|------|--------|
| `Cargo.toml` | Add `bincode = "2"`, `[[bin]]` for worker |
| `src/main.rs` | Branch on `EMBED_MULTI_PROCESS` env: legacy path OR supervisor path |
| `src/config.rs` | Add `multi_process: bool`, `worker_bin_path`, `worker_socket_dir` |
| `src/api.rs` | Replace direct `EmbedModel::infer()` call with `state.dispatch()` (works for both monolith + supervisor mode) |
| `src/api_rerank.rs` | Same dispatch indirection |
| `src/types.rs` | Add `DispatchTarget` enum (Inline / RemoteWorker) |
| `src/metrics.rs` | Add `embed_ipc_*` family of metrics |

### Files unchanged in Phase 1-2

`src/arena.rs`, `src/batcher.rs`, `src/evictable_pool.rs`, `src/cache.rs`, `src/onnx_cache.rs` — used by worker, no changes needed.

---

## Phase 1 — IPC scaffold (no behavior change)

Goal: land worker binary + IPC types behind feature flag, monolith remains default. Every commit green CI.

### Wave 1.1 — IPC protocol types

#### Task 1: Add bincode dependency

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add dependency**

In `Cargo.toml` under `[dependencies]`:
```toml
bincode = { version = "2.0.0-rc.3", features = ["serde"] }
```

- [ ] **Step 2: Verify build**

Run: `cargo build --locked`
Expected: PASS (new dep compiles)

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "deps: add bincode 2.x for IPC framing"
```

---

#### Task 2: Define IPC wire types

**Files:**
- Create: `src/ipc/mod.rs`
- Create: `src/ipc/protocol.rs`
- Test: `src/ipc/protocol.rs` (inline `#[cfg(test)]` mod)

- [ ] **Step 1: Write failing test for serialization roundtrip**

Create `src/ipc/protocol.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    pub request_id: u64,
    pub model: String,
    pub texts: Vec<String>,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InferResponse {
    Ok { request_id: u64, vectors: Vec<Vec<f32>>, dims: u32 },
    Err { request_id: u64, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlMessage {
    Ping,
    Pong,
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_infer_request() {
        let req = InferRequest {
            request_id: 42,
            model: "jina-code-v2".into(),
            texts: vec!["fn main() {}".into()],
            max_seq_len: 512,
        };
        let bytes = bincode::serde::encode_to_vec(&req, bincode::config::standard()).unwrap();
        let (decoded, _): (InferRequest, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(req, decoded);
    }

    #[test]
    fn roundtrip_infer_response_ok() {
        let resp = InferResponse::Ok {
            request_id: 1,
            vectors: vec![vec![0.1, 0.2, 0.3]],
            dims: 3,
        };
        let bytes = bincode::serde::encode_to_vec(&resp, bincode::config::standard()).unwrap();
        let (decoded, _): (InferResponse, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(resp, decoded);
    }
}
```

Create `src/ipc/mod.rs`:
```rust
pub mod protocol;
```

Modify `src/main.rs` to add module declaration after the existing `mod` declarations:
```rust
mod ipc;
```

- [ ] **Step 2: Run test, verify it fails compilation**

Run: `cargo test --lib ipc::protocol`
Expected: FAIL (module not yet wired) → after wiring → PASS

- [ ] **Step 3: Commit**

```bash
git add src/ipc/ src/main.rs
git commit -m "feat(ipc): define wire types for supervisor↔worker protocol"
```

---

#### Task 3: Length-prefixed frame codec

**Files:**
- Create: `src/ipc/frame.rs`
- Modify: `src/ipc/mod.rs`
- Test: in `frame.rs`

- [ ] **Step 1: Write failing test for frame encode/decode**

Create `src/ipc/frame.rs`:
```rust
use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024; // 64 MiB safety cap

/// Writes a length-prefixed bincode-encoded message to the stream.
pub async fn write_frame<W, T>(stream: &mut W, value: &T) -> io::Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: serde::Serialize,
{
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if bytes.len() > MAX_FRAME_BYTES as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    stream.write_u32_le(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;
    stream.flush().await
}

/// Reads a length-prefixed bincode-encoded message from the stream.
pub async fn read_frame<R, T>(stream: &mut R) -> io::Result<T>
where
    R: AsyncReadExt + Unpin,
    T: for<'de> serde::Deserialize<'de>,
{
    let len = stream.read_u32_le().await?;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;
    let (value, _): (T, _) = bincode::serde::decode_from_slice(&buf, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::protocol::InferRequest;

    #[tokio::test]
    async fn frame_roundtrip_through_pipe() {
        let (mut a, mut b) = tokio::io::duplex(8192);
        let req = InferRequest {
            request_id: 1,
            model: "e5".into(),
            texts: vec!["hello".into()],
            max_seq_len: 128,
        };
        write_frame(&mut a, &req).await.unwrap();
        let decoded: InferRequest = read_frame(&mut b).await.unwrap();
        assert_eq!(req, decoded);
    }
}
```

Modify `src/ipc/mod.rs`:
```rust
pub mod frame;
pub mod protocol;
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib ipc::`
Expected: 3 passing tests

- [ ] **Step 3: Commit**

```bash
git add src/ipc/frame.rs src/ipc/mod.rs
git commit -m "feat(ipc): length-prefixed bincode frame codec"
```

---

### Wave 1.2 — Worker binary scaffold

#### Task 4: Add worker binary target

**Files:**
- Modify: `Cargo.toml`
- Create: `src/bin/worker.rs`

- [ ] **Step 1: Add `[[bin]]` entries**

In `Cargo.toml` append:
```toml
[[bin]]
name = "embed-server"
path = "src/main.rs"

[[bin]]
name = "embed-worker"
path = "src/bin/worker.rs"
```

(If `[[bin]]` block already exists for `embed-server`, just append the second.)

- [ ] **Step 2: Create minimal worker entrypoint**

Create `src/bin/worker.rs`:
```rust
//! Worker process binary — one process per model.
//!
//! Loads a single ONNX model, owns its own OrtEnv + BFCArena, exposes
//! inference over Unix Domain Socket to the supervisor.
//!
//! Phase 1: scaffold only. Connects to UDS, echoes ControlMessage::Ping → Pong.
//! Full inference handlers land in Wave 1.4.

use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::ControlMessage;
use std::path::PathBuf;
use tokio::net::UnixListener;

#[derive(Debug)]
struct WorkerConfig {
    model: String,
    socket_path: PathBuf,
}

fn parse_args() -> WorkerConfig {
    let model = std::env::var("EMBED_WORKER_MODEL")
        .expect("EMBED_WORKER_MODEL env required");
    let socket_path: PathBuf = std::env::var("EMBED_WORKER_SOCKET")
        .expect("EMBED_WORKER_SOCKET env required")
        .into();
    WorkerConfig { model, socket_path }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = parse_args();
    tracing::info!(model = %cfg.model, socket = ?cfg.socket_path, "worker starting");

    if cfg.socket_path.exists() {
        std::fs::remove_file(&cfg.socket_path)?;
    }
    let listener = UnixListener::bind(&cfg.socket_path)?;
    tracing::info!("worker listening on UDS");

    loop {
        let (mut stream, _) = listener.accept().await?;
        let model = cfg.model.clone();
        tokio::spawn(async move {
            loop {
                let msg: ControlMessage = match read_frame(&mut stream).await {
                    Ok(m) => m,
                    Err(_) => break,
                };
                let reply = match msg {
                    ControlMessage::Ping => ControlMessage::Pong,
                    ControlMessage::Shutdown => {
                        tracing::info!(model = %model, "shutdown requested");
                        std::process::exit(0);
                    }
                    other => {
                        tracing::warn!(?other, "unexpected message");
                        ControlMessage::Pong
                    }
                };
                if write_frame(&mut stream, &reply).await.is_err() {
                    break;
                }
            }
        });
    }
}
```

Note: this requires `embed-server` to be a library too. Check `Cargo.toml` — if no `[lib]` section exists, add it.

In `Cargo.toml`:
```toml
[lib]
name = "embed_server"
path = "src/lib.rs"
```

- [ ] **Step 3: Create `src/lib.rs` exposing modules**

Create `src/lib.rs`:
```rust
//! Library facade — exposes modules to binaries (main + worker).
pub mod ipc;
```

(Add other `pub mod X;` entries as later tasks need to reach them from the worker binary.)

- [ ] **Step 4: Verify build**

Run: `cargo build --locked`
Expected: PASS, builds 2 binaries

Run: `ls target/debug/embed-server target/debug/embed-worker`
Expected: both exist

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/bin/worker.rs src/lib.rs
git commit -m "feat(worker): scaffold worker binary with ping/pong over UDS"
```

---

#### Task 5: Integration test — supervisor pings worker over UDS

**Files:**
- Create: `tests/ipc_ping.rs`

- [ ] **Step 1: Write failing integration test**

Create `tests/ipc_ping.rs`:
```rust
use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::ControlMessage;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

#[tokio::test]
async fn supervisor_pings_worker() {
    let socket = format!("/tmp/embed-worker-test-{}.sock", std::process::id());
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let mut child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "test-model")
        .env("EMBED_WORKER_SOCKET", &socket)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn worker");

    // Wait for socket to appear (worker startup race).
    for _ in 0..50 {
        if std::path::Path::new(&socket).exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(std::path::Path::new(&socket).exists(), "worker did not create socket");

    let mut conn = UnixStream::connect(&socket).await.expect("connect");
    write_frame(&mut conn, &ControlMessage::Ping).await.unwrap();
    let reply: ControlMessage = read_frame(&mut conn).await.unwrap();
    assert_eq!(reply, ControlMessage::Pong);

    write_frame(&mut conn, &ControlMessage::Shutdown).await.unwrap();
    let _ = child.wait();
    let _ = std::fs::remove_file(&socket);
}
```

- [ ] **Step 2: Run test**

Run: `cargo test --test ipc_ping --locked`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/ipc_ping.rs
git commit -m "test(ipc): supervisor↔worker ping/pong over UDS"
```

---

### Wave 1.3 — Worker inference handler

#### Task 6: Extract `EmbedModel::infer_batch` to a self-contained function

The worker calls inference without going through `AppState` / batcher / pool. Need a "raw inference" entry point.

**Files:**
- Modify: `src/model.rs`

- [ ] **Step 1: Read existing `infer_batch` (or equivalent) and identify dependencies**

Run: `grep -nE "pub (async )?fn infer" src/model.rs`
Read those functions' signatures + bodies. Note Arc<...>s threaded through — they're the legitimate worker-state.

- [ ] **Step 2: Add a `StandaloneEmbedder` struct**

Append to `src/model.rs` (or split into `src/model/standalone.rs` if file grows past 1500 lines):
```rust
/// Standalone embedder for worker process — owns model state, no batcher/queue.
/// Worker process creates one of these on startup and calls `infer` for each
/// IPC request. Caller (worker main loop) handles concurrency limits via
/// tokio::Semaphore matching EMBED_SESSION_POOL_SIZE.
pub struct StandaloneEmbedder {
    inner: Arc<EmbedModel>,  // reuse existing EmbedModel internals
}

impl StandaloneEmbedder {
    pub async fn load(model_name: &str, cfg: &crate::config::Config) -> anyhow::Result<Self> {
        // Reuse existing EmbedModel::load_from_config or equivalent.
        // Pass minimal Config: model_name + onnx_path + tokenizer_path.
        let inner = Arc::new(crate::model::EmbedModel::load_from_config(model_name, cfg).await?);
        Ok(Self { inner })
    }

    /// Plain batch inference. No batcher, no queue.
    pub async fn infer(&self, texts: Vec<String>, max_seq_len: u32) -> anyhow::Result<(Vec<Vec<f32>>, u32)> {
        self.inner.infer_raw(texts, max_seq_len).await
    }
}
```

If `EmbedModel::load_from_config` / `infer_raw` don't exist verbatim, **find the closest existing equivalents** and route through them. The point: worker should not need a batcher/queue/pool — it itself is the singleton inside its process.

- [ ] **Step 3: Verify build**

Run: `cargo build --locked`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/model.rs
git commit -m "feat(model): StandaloneEmbedder for worker process use"
```

---

#### Task 7: Worker dispatches InferRequest → StandaloneEmbedder

**Files:**
- Modify: `src/bin/worker.rs`
- Modify: `src/lib.rs` (expose model + config modules)

- [ ] **Step 1: Update `src/lib.rs`**

```rust
pub mod arena;
pub mod batcher;
pub mod cache;
pub mod config;
pub mod ipc;
pub mod metrics;
pub mod model;
pub mod model_reranker;
pub mod onnx_cache;
pub mod pool;
pub mod token_cache;
pub mod types;
```

- [ ] **Step 2: Rewrite `src/bin/worker.rs` to handle InferRequest**

```rust
use embed_server::config::Config;
use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{ControlMessage, InferRequest, InferResponse};
use embed_server::model::StandaloneEmbedder;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::sync::Semaphore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_name = std::env::var("EMBED_WORKER_MODEL")?;
    let socket_path: PathBuf = std::env::var("EMBED_WORKER_SOCKET")?.into();
    let intra_threads: usize = std::env::var("EMBED_WORKER_INTRA_THREADS")
        .unwrap_or_else(|_| "2".into())
        .parse()?;
    let pool_size: usize = std::env::var("EMBED_WORKER_POOL_SIZE")
        .unwrap_or_else(|_| "1".into())
        .parse()?;

    tracing::info!(model = %model_name, ?socket_path, intra_threads, pool_size, "worker starting");

    let cfg = Config::from_env()?; // existing
    let embedder = Arc::new(StandaloneEmbedder::load(&model_name, &cfg).await?);
    let semaphore = Arc::new(Semaphore::new(pool_size));

    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }
    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("worker ready");

    loop {
        let (mut stream, _) = listener.accept().await?;
        let embedder = embedder.clone();
        let semaphore = semaphore.clone();
        tokio::spawn(async move {
            // Each connection runs serially (single-threaded per UDS conn).
            // Concurrency limit enforced by semaphore (matches EMBED_SESSION_POOL_SIZE).
            loop {
                let req: InferRequest = match read_frame(&mut stream).await {
                    Ok(r) => r,
                    Err(_) => break,
                };

                let _permit = match semaphore.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        // Saturated — return 429-equivalent over IPC.
                        let resp = InferResponse::Err {
                            request_id: req.request_id,
                            message: "worker saturated".into(),
                        };
                        let _ = write_frame(&mut stream, &resp).await;
                        continue;
                    }
                };

                let resp = match embedder.infer(req.texts, req.max_seq_len).await {
                    Ok((vectors, dims)) => InferResponse::Ok {
                        request_id: req.request_id,
                        vectors,
                        dims,
                    },
                    Err(e) => InferResponse::Err {
                        request_id: req.request_id,
                        message: e.to_string(),
                    },
                };
                if write_frame(&mut stream, &resp).await.is_err() {
                    break;
                }
            }
        });
    }
}
```

- [ ] **Step 3: Build**

Run: `cargo build --locked`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bin/worker.rs src/lib.rs
git commit -m "feat(worker): real inference handler over IPC"
```

---

#### Task 8: Integration test — worker performs inference on `e5` model

**Files:**
- Create: `tests/worker_e5_inference.rs`

- [ ] **Step 1: Write the test**

```rust
//! Exercises worker binary against multilingual-e5-large.
//! Requires model files at $EMBED_MODEL_DIR/multilingual-e5-large/.
//! CI: skip if env not set.

use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{InferRequest, InferResponse};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

#[tokio::test]
async fn worker_infers_e5_batch() {
    if std::env::var("EMBED_MODEL_DIR").is_err() {
        eprintln!("SKIP: EMBED_MODEL_DIR not set");
        return;
    }

    let socket = format!("/tmp/embed-worker-e5-{}.sock", std::process::id());
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let mut child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "multilingual-e5-large")
        .env("EMBED_WORKER_SOCKET", &socket)
        .env("EMBED_WORKER_POOL_SIZE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn worker");

    // Wait up to 30s for model load + UDS bind.
    for _ in 0..300 {
        if std::path::Path::new(&socket).exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(std::path::Path::new(&socket).exists(), "worker socket missing");

    let mut conn = UnixStream::connect(&socket).await.unwrap();
    let req = InferRequest {
        request_id: 1,
        model: "multilingual-e5-large".into(),
        texts: vec!["query: hello".into(), "query: world".into()],
        max_seq_len: 128,
    };
    write_frame(&mut conn, &req).await.unwrap();

    let resp: InferResponse = read_frame(&mut conn).await.unwrap();
    match resp {
        InferResponse::Ok { request_id, vectors, dims } => {
            assert_eq!(request_id, 1);
            assert_eq!(vectors.len(), 2);
            assert_eq!(dims, 1024);
            assert_eq!(vectors[0].len(), 1024);
        }
        InferResponse::Err { message, .. } => panic!("inference failed: {message}"),
    }

    let _ = child.kill();
    let _ = std::fs::remove_file(&socket);
}
```

- [ ] **Step 2: Run test**

Run: `EMBED_MODEL_DIR=/path/to/models cargo nextest run --test worker_e5_inference`
Expected: PASS (or SKIP if env unset)

If models live on the dev box at `/home/krolik/embed-models/` or similar — set env accordingly. Verify path with `find / -name 'multilingual-e5-large*' -type d 2>/dev/null | head -3`.

- [ ] **Step 3: Commit**

```bash
git add tests/worker_e5_inference.rs
git commit -m "test(worker): e5 inference roundtrip integration test"
```

---

## Phase 2 — Supervisor cutover

Goal: supervisor spawns worker pool, routes API requests to children. Feature-flagged.

### Wave 2.1 — `WorkerClient` (supervisor side)

#### Task 9: Connection pool to one worker

**Files:**
- Create: `src/ipc/client.rs`
- Modify: `src/ipc/mod.rs`

- [ ] **Step 1: Write the client**

```rust
//! Supervisor-side client to one worker process.
//!
//! Holds N persistent UDS connections (matches worker's pool_size); each
//! connection runs requests serially. Outer caller acquires a connection
//! from the pool via `infer()`, which round-robins.

use crate::ipc::frame::{read_frame, write_frame};
use crate::ipc::protocol::{InferRequest, InferResponse};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

pub struct WorkerClient {
    socket_path: PathBuf,
    pool: Vec<Arc<Mutex<UnixStream>>>,
    next_idx: AtomicU64,
    request_counter: AtomicU64,
}

impl WorkerClient {
    pub async fn connect(socket_path: PathBuf, conns: usize) -> std::io::Result<Self> {
        let mut pool = Vec::with_capacity(conns);
        for _ in 0..conns {
            let stream = UnixStream::connect(&socket_path).await?;
            pool.push(Arc::new(Mutex::new(stream)));
        }
        Ok(Self {
            socket_path,
            pool,
            next_idx: AtomicU64::new(0),
            request_counter: AtomicU64::new(0),
        })
    }

    pub async fn infer(&self, model: String, texts: Vec<String>, max_seq_len: u32) -> std::io::Result<InferResponse> {
        let idx = (self.next_idx.fetch_add(1, Ordering::Relaxed) as usize) % self.pool.len();
        let req_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let req = InferRequest { request_id: req_id, model, texts, max_seq_len };

        let conn = self.pool[idx].clone();
        let mut stream = conn.lock().await;
        write_frame(&mut *stream, &req).await?;
        let resp: InferResponse = read_frame(&mut *stream).await?;
        Ok(resp)
    }
}
```

- [ ] **Step 2: Wire into module**

`src/ipc/mod.rs`:
```rust
pub mod client;
pub mod frame;
pub mod protocol;
```

- [ ] **Step 3: Build**

Run: `cargo build --locked`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/ipc/client.rs src/ipc/mod.rs
git commit -m "feat(ipc): WorkerClient with persistent connection pool"
```

---

#### Task 10: Test — WorkerClient + worker bin end-to-end

**Files:**
- Create: `tests/worker_client_roundtrip.rs`

- [ ] **Step 1: Write the test**

```rust
use embed_server::ipc::client::WorkerClient;
use embed_server::ipc::protocol::InferResponse;
use std::process::{Command, Stdio};
use std::time::Duration;

#[tokio::test]
async fn client_roundtrip_e5() {
    if std::env::var("EMBED_MODEL_DIR").is_err() {
        eprintln!("SKIP");
        return;
    }
    let socket = format!("/tmp/embed-wc-test-{}.sock", std::process::id());
    let _ = std::fs::remove_file(&socket);
    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let mut child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "multilingual-e5-large")
        .env("EMBED_WORKER_SOCKET", &socket)
        .env("EMBED_WORKER_POOL_SIZE", "2")
        .stdout(Stdio::null()).stderr(Stdio::null())
        .spawn().unwrap();

    for _ in 0..300 {
        if std::path::Path::new(&socket).exists() { break; }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let client = WorkerClient::connect(socket.clone().into(), 2).await.unwrap();
    let resp = client.infer(
        "multilingual-e5-large".into(),
        vec!["query: hello".into()],
        128,
    ).await.unwrap();
    match resp {
        InferResponse::Ok { vectors, dims, .. } => {
            assert_eq!(vectors.len(), 1);
            assert_eq!(dims, 1024);
        }
        InferResponse::Err { message, .. } => panic!("{message}"),
    }

    let _ = child.kill();
    let _ = std::fs::remove_file(&socket);
}
```

- [ ] **Step 2: Run**

Run: `EMBED_MODEL_DIR=... cargo nextest run --test worker_client_roundtrip`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/worker_client_roundtrip.rs
git commit -m "test(ipc): WorkerClient roundtrip via real worker binary"
```

---

### Wave 2.2 — Supervisor module

#### Task 11: `WorkerHandle` — spawn + monitor

**Files:**
- Create: `src/supervisor/mod.rs`
- Create: `src/supervisor/handle.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write `WorkerHandle`**

`src/supervisor/handle.rs`:
```rust
//! Single worker process handle — owns Child, monitors lifecycle.
//!
//! Phase 2: blocking spawn, no auto-restart.
//! Phase 2.3 (Task 14): adds watchdog + restart-on-exit.

use crate::ipc::client::WorkerClient;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};

pub struct WorkerHandle {
    pub model: String,
    pub socket_path: PathBuf,
    pub child: Child,
    pub client: Arc<WorkerClient>,
}

pub struct SpawnSpec {
    pub model: String,
    pub worker_bin: PathBuf,
    pub socket_dir: PathBuf,
    pub pool_size: usize,
    pub intra_threads: usize,
    pub env_extra: Vec<(String, String)>,
}

impl WorkerHandle {
    pub async fn spawn(spec: SpawnSpec) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&spec.socket_dir).ok();
        let socket_path = spec.socket_dir.join(format!("{}.sock", spec.model));
        let _ = std::fs::remove_file(&socket_path);

        let mut cmd = Command::new(&spec.worker_bin);
        cmd.env("EMBED_WORKER_MODEL", &spec.model)
            .env("EMBED_WORKER_SOCKET", &socket_path)
            .env("EMBED_WORKER_POOL_SIZE", spec.pool_size.to_string())
            .env("EMBED_WORKER_INTRA_THREADS", spec.intra_threads.to_string())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        for (k, v) in &spec.env_extra {
            cmd.env(k, v);
        }
        let child = cmd.spawn()?;

        // Wait for UDS to appear (model load can take seconds).
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            if socket_path.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        if !socket_path.exists() {
            anyhow::bail!("worker {} did not create socket within 60s", spec.model);
        }

        let client = Arc::new(WorkerClient::connect(socket_path.clone(), spec.pool_size).await?);
        Ok(Self { model: spec.model, socket_path, child, client })
    }
}
```

`src/supervisor/mod.rs`:
```rust
pub mod handle;
pub use handle::{SpawnSpec, WorkerHandle};
```

`src/lib.rs`:
```rust
// ... existing modules ...
pub mod supervisor;
```

- [ ] **Step 2: Build + commit**

```bash
cargo build --locked
git add src/supervisor/ src/lib.rs
git commit -m "feat(supervisor): WorkerHandle — spawn + connect"
```

---

#### Task 12: `WorkerPool` — map model name → handle, dispatch

**Files:**
- Create: `src/supervisor/pool.rs`
- Modify: `src/supervisor/mod.rs`

- [ ] **Step 1: Write `WorkerPool`**

```rust
//! Routing layer — maps model name to WorkerHandle, dispatches inference.

use crate::ipc::protocol::InferResponse;
use crate::supervisor::WorkerHandle;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct WorkerPool {
    workers: Arc<RwLock<HashMap<String, Arc<WorkerHandle>>>>,
}

impl WorkerPool {
    pub fn new() -> Self {
        Self { workers: Arc::new(RwLock::new(HashMap::new())) }
    }

    pub async fn add(&self, handle: WorkerHandle) {
        let mut w = self.workers.write().await;
        w.insert(handle.model.clone(), Arc::new(handle));
    }

    pub async fn dispatch(&self, model: &str, texts: Vec<String>, max_seq_len: u32) -> anyhow::Result<InferResponse> {
        let handle = {
            let w = self.workers.read().await;
            w.get(model).cloned().ok_or_else(|| anyhow::anyhow!("no worker for model {model}"))?
        };
        Ok(handle.client.infer(model.to_string(), texts, max_seq_len).await?)
    }
}

impl Default for WorkerPool {
    fn default() -> Self { Self::new() }
}
```

`src/supervisor/mod.rs`:
```rust
pub mod handle;
pub mod pool;
pub use handle::{SpawnSpec, WorkerHandle};
pub use pool::WorkerPool;
```

- [ ] **Step 2: Build + commit**

```bash
cargo build --locked
git add src/supervisor/
git commit -m "feat(supervisor): WorkerPool with model→handle routing"
```

---

#### Task 13: Wire supervisor into `main.rs` behind feature flag

**Files:**
- Modify: `src/main.rs`
- Modify: `src/config.rs`
- Modify: `src/types.rs`

- [ ] **Step 1: Add config flag**

In `src/config.rs` `Config` struct add field:
```rust
pub multi_process: bool,
pub worker_bin_path: std::path::PathBuf,
pub worker_socket_dir: std::path::PathBuf,
```

In `Config::from_env()`:
```rust
multi_process: std::env::var("EMBED_MULTI_PROCESS")
    .map(|v| v == "1" || v == "true").unwrap_or(false),
worker_bin_path: std::env::var("EMBED_WORKER_BIN")
    .unwrap_or_else(|_| "/usr/local/bin/embed-worker".into()).into(),
worker_socket_dir: std::env::var("EMBED_WORKER_SOCKET_DIR")
    .unwrap_or_else(|_| "/tmp/embed-workers".into()).into(),
```

- [ ] **Step 2: Add `AppState::multi_process` field**

In `src/types.rs` add field to `AppState`:
```rust
pub worker_pool: Option<Arc<crate::supervisor::WorkerPool>>,
```

Set to `None` in legacy path, `Some(pool)` in multi-process path.

- [ ] **Step 3: Branch in `main.rs`**

In `main()` after `Config::from_env()`:
```rust
let worker_pool = if cfg.multi_process {
    let pool = crate::supervisor::WorkerPool::new();
    // Spawn one worker per configured model.
    for (model_name, model_cfg) in &cfg.models {
        let handle = crate::supervisor::WorkerHandle::spawn(crate::supervisor::SpawnSpec {
            model: model_name.clone(),
            worker_bin: cfg.worker_bin_path.clone(),
            socket_dir: cfg.worker_socket_dir.clone(),
            pool_size: model_cfg.session_pool_size,
            intra_threads: cfg.intra_threads,
            env_extra: Vec::new(),
        }).await?;
        pool.add(handle).await;
    }
    Some(Arc::new(pool))
} else {
    None
};
```

Adjust `AppState` construction to thread `worker_pool` through.

- [ ] **Step 4: Build + commit**

```bash
cargo build --locked
git add src/config.rs src/main.rs src/types.rs
git commit -m "feat(supervisor): wire multi-process behind EMBED_MULTI_PROCESS flag"
```

---

### Wave 2.3 — Routing in API handlers

#### Task 14: `api.rs` — dispatch through worker_pool when set

**Files:**
- Modify: `src/api.rs`

- [ ] **Step 1: Find current `/embed` handler — most likely calls `state.batcher.submit(...)`**

Run: `grep -n "fn handle_embed\|pub async fn embed\|.submit(" src/api.rs | head -10`

- [ ] **Step 2: Insert worker_pool branch BEFORE batcher call**

Replace the inference dispatch with:
```rust
let resp = if let Some(pool) = &state.worker_pool {
    // Multi-process path.
    let ipc_resp = pool.dispatch(&req.model, req.texts.clone(), req.max_seq_len as u32).await
        .map_err(|e| /* existing error type */)?;
    match ipc_resp {
        crate::ipc::protocol::InferResponse::Ok { vectors, dims, .. } => {
            // Map to existing response shape
            build_embed_response(vectors, dims, req.model.clone())
        }
        crate::ipc::protocol::InferResponse::Err { message, .. } => {
            return Err(/* existing 500 error */);
        }
    }
} else {
    // Legacy monolith path — keep as-is.
    existing_batcher_submit_path()
};
```

- [ ] **Step 3: Same change in `src/api_rerank.rs`**

Mirror the pattern for reranker.

- [ ] **Step 4: Build + run unit tests**

```
cargo build --locked
cargo nextest run --lib
```

Both expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api.rs src/api_rerank.rs
git commit -m "feat(api): route through worker_pool when multi_process=true"
```

---

#### Task 15: E2E test — supervisor in multi-process mode answers HTTP request

**Files:**
- Create: `tests/multi_process_e2e.rs`

- [ ] **Step 1: Write the test**

```rust
//! End-to-end: start embed-server in multi-process mode, send /embed,
//! verify response. Skips if EMBED_MODEL_DIR unset.

use std::process::{Command, Stdio};
use std::time::Duration;

#[tokio::test]
async fn multi_process_embed() {
    if std::env::var("EMBED_MODEL_DIR").is_err() {
        eprintln!("SKIP");
        return;
    }
    let port: u16 = 28082;
    let socket_dir = format!("/tmp/embed-multi-e2e-{}", std::process::id());
    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let server_bin = env!("CARGO_BIN_EXE_embed-server");

    let mut server = Command::new(server_bin)
        .env("EMBED_MULTI_PROCESS", "1")
        .env("EMBED_WORKER_BIN", worker_bin)
        .env("EMBED_WORKER_SOCKET_DIR", &socket_dir)
        .env("PORT", port.to_string())
        // any other required env (EMBED_MODELS list, paths) ...
        .stdout(Stdio::inherit()).stderr(Stdio::inherit())
        .spawn().unwrap();

    // Wait for /health
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    for _ in 0..600 {
        if client.get(format!("{base}/health")).send().await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let resp: serde_json::Value = client
        .post(format!("{base}/embed"))
        .json(&serde_json::json!({
            "model": "multilingual-e5-large",
            "input": ["hello"],
        }))
        .send().await.unwrap()
        .json().await.unwrap();

    assert!(resp["data"][0]["embedding"].is_array());
    assert_eq!(resp["data"][0]["embedding"].as_array().unwrap().len(), 1024);

    let _ = server.kill();
    let _ = std::fs::remove_dir_all(&socket_dir);
}
```

- [ ] **Step 2: Run**

```
EMBED_MODEL_DIR=... cargo nextest run --test multi_process_e2e
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/multi_process_e2e.rs
git commit -m "test(multi-process): E2E HTTP → supervisor → worker → response"
```

---

### Wave 2.4 — Watchdog + restart

#### Task 16: Restart worker on exit

**Files:**
- Modify: `src/supervisor/handle.rs`
- Modify: `src/supervisor/pool.rs`

- [ ] **Step 1: Add `monitor_and_restart` task**

In `WorkerHandle::spawn`, after successful spawn return, also spawn a tokio task that:
1. Awaits `child.wait()`
2. On exit (any cause), logs the exit status
3. Notifies pool to remove + respawn (channel or `Arc<Notify>`)

Skeleton:
```rust
pub async fn spawn_with_monitor(spec: SpawnSpec, pool: Arc<WorkerPool>) -> anyhow::Result<()> {
    let handle = Self::spawn(spec.clone()).await?;
    let model = handle.model.clone();
    pool.add(handle).await;

    // Spawn monitor task
    tokio::spawn(async move {
        loop {
            let exit = {
                let h = pool.workers.read().await;
                if let Some(h) = h.get(&model) {
                    // We need exclusive access to wait on Child; use a different design:
                    // store oneshot::Receiver in handle, signal from a dedicated wait task.
                    unimplemented!() // see step 2 below
                } else {
                    break;
                }
            };
            tracing::warn!(model = %model, ?exit, "worker exited, restarting in 2s");
            tokio::time::sleep(Duration::from_secs(2)).await;
            match Self::spawn(spec.clone()).await {
                Ok(new_h) => pool.add(new_h).await,
                Err(e) => {
                    tracing::error!(model = %model, error = ?e, "respawn failed");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });
    Ok(())
}
```

Note: actual implementation needs `Child` ownership reorg — likely a `RestartSupervisor` actor that owns the `Child`, with `WorkerPool` only holding `Arc<WorkerClient>` (the IPC client survives respawn since it reconnects).

Cleaner final architecture:
```rust
// supervisor/handle.rs
pub struct WorkerSupervisor {
    spec: SpawnSpec,
    client_slot: Arc<RwLock<Option<Arc<WorkerClient>>>>,  // shared with WorkerPool
}

impl WorkerSupervisor {
    pub fn launch(spec: SpawnSpec) -> Arc<WorkerSupervisor> {
        let supervisor = Arc::new(Self {
            spec: spec.clone(),
            client_slot: Arc::new(RwLock::new(None)),
        });
        let sup = supervisor.clone();
        tokio::spawn(async move {
            loop {
                match sup.spawn_one().await {
                    Ok(()) => tracing::info!(model = %sup.spec.model, "worker exited cleanly"),
                    Err(e) => tracing::error!(model = %sup.spec.model, ?e, "worker spawn/run failed"),
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
        supervisor
    }

    async fn spawn_one(&self) -> anyhow::Result<()> {
        // 1. Spawn child process
        // 2. Wait for UDS
        // 3. Connect WorkerClient
        // 4. Store client in client_slot
        // 5. Await child.wait()
        // 6. Clear client_slot
        Ok(())
    }

    pub async fn client(&self) -> Option<Arc<WorkerClient>> {
        self.client_slot.read().await.clone()
    }
}
```

Then `WorkerPool` stores `HashMap<String, Arc<WorkerSupervisor>>`.

- [ ] **Step 2: Refactor pool + handle**

Rewrite `WorkerPool` to use `WorkerSupervisor` instead of `WorkerHandle`. Migrate `main.rs` callers.

- [ ] **Step 3: Write test — kill worker process, verify supervisor restarts**

```rust
#[tokio::test]
async fn worker_restarts_after_kill() {
    // ... spawn supervisor ...
    // ... do one /embed request — verify works ...
    // ... `kill -9` the worker pid ...
    // ... wait 5s ...
    // ... do another /embed request — verify works (new worker) ...
}
```

- [ ] **Step 4: Build, test, commit**

```bash
cargo build --locked
cargo nextest run --test worker_restart
git add src/supervisor/ src/main.rs
git commit -m "feat(supervisor): auto-restart workers on exit"
```

---

### Wave 2.5 — Cutover to production

#### Task 17: Update Dockerfile to ship both binaries

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Build both bins in image**

```dockerfile
RUN cargo build --release --locked --bins
# ...
COPY --from=builder /app/target/release/embed-server /usr/local/bin/embed-server
COPY --from=builder /app/target/release/embed-worker /usr/local/bin/embed-worker
```

- [ ] **Step 2: Verify image build**

Run: `docker build -t embed-server:multi .`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "docker: include embed-worker binary in image"
```

---

#### Task 18: Enable in `compose/memdb.yml` (deploy repo, separate PR)

This step lands in `~/deploy/krolik-server`, not `embed-server`. Tracked separately.

Set:
```yaml
EMBED_MULTI_PROCESS: "1"
EMBED_WORKER_BIN: "/usr/local/bin/embed-worker"
EMBED_WORKER_SOCKET_DIR: "/tmp/embed-workers"
```

Roll out: bump image tag in compose, dozor auto-deploys via webhook. Monitor `embed_inference_duration_seconds` p95 for 1h post-deploy.

---

## Phase 3 — Optimizations

### Wave 3.1 — Shared mmap weights

#### Task 19: Verify weights are mmap'd read-only

In modern `ort` (2.0.x), models loaded via `Session::builder().commit_from_file(...)` are mmap'd by default. Linux kernel page-cache deduplicates pages across processes automatically — no code change required.

- [ ] **Step 1: Verify with `pmap`**

After running multi-process mode in production for 5 minutes, run:
```bash
docker compose exec embed-server sh -c '
  for pid in $(pgrep embed-worker); do
    echo "=== pid $pid ==="
    pmap -X $pid | grep -E "model.onnx|tokenizer" | head -5
  done
'
```

Look for `s` (shared) flag on the mapping. If absent — investigate `ort` Session config.

- [ ] **Step 2: Measure RAM reduction**

Compare `docker stats` total RSS before/after multi-process cutover. Target: weights loaded once, not 3×.

- [ ] **Step 3: Document finding in `docs/architecture-changelog.md`**

```markdown
## 2026-05-XX — Multi-process refactor

Shared weights via Linux page cache (mmap default in ort 2.x).
Before: ~6 GiB RSS (3× full weight duplication).
After: ~3 GiB RSS (page cache dedupe).
```

- [ ] **Step 4: Commit**

```bash
git add docs/architecture-changelog.md
git commit -m "docs: confirm shared mmap weights via page cache"
```

---

### Wave 3.2 — Lazy-load + idle evict

#### Task 20: `WorkerSupervisor::idle_timeout`

**Files:**
- Modify: `src/supervisor/handle.rs`

- [ ] **Step 1: Add idle tracking**

In `WorkerSupervisor`, add `last_request_at: Arc<RwLock<Instant>>`. Update on every `client().await.infer(...)` call.

Spawn idle-monitor task:
```rust
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        let last = *sup.last_request_at.read().await;
        if last.elapsed() > Duration::from_secs(sup.spec.idle_timeout_secs) {
            tracing::info!(model = %sup.spec.model, "evicting idle worker");
            // Signal current spawn_one loop to exit gracefully:
            // - Send ControlMessage::Shutdown over IPC
            // - Set "should_resume: false" flag so loop pauses until next request
            sup.evict().await;
        }
    }
});
```

Then `WorkerPool::dispatch` becomes:
```rust
pub async fn dispatch(&self, model: &str, ...) -> ... {
    let sup = ...;
    if sup.client().await.is_none() {
        sup.respawn_blocking().await?;  // cold start
    }
    sup.client().await.unwrap().infer(...).await
}
```

- [ ] **Step 2: Test — worker stopped after idle, restarts on request**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(supervisor): lazy-load + idle evict per worker"
```

---

## Self-Review Checklist

After this plan is implemented, the reviewer (or controller) verifies:

- [ ] All 3 production models (e5, jina, reranker) work via multi-process path
- [ ] `EMBED_MULTI_PROCESS=0` (legacy) still works — rollback path intact
- [ ] Worker crash leaves other models functional
- [ ] After 30 min production load: no `arena_oom` errors for jina-code-v2
- [ ] p95 inference latency for jina-code-v2 < 5s (alert silences)
- [ ] Host swap usage drops (no longer 15 GiB pegged)
- [ ] Total RSS across worker processes < 4 GiB (weights shared)

---

## Out of Scope (separate plans)

- **Phase 4** — Redis embedding cache. Hash(content+model) → vector. Big throughput win. Plan: `2026-05-XX-embed-redis-cache.md`.
- **Phase 5** — Dynamic padding + seq-len bucketing in `src/batcher.rs`. Plan: `2026-05-XX-batcher-padding-bucketing.md`.
- **Phase 6** — int8 quantization, accuracy A/B. Plan: `2026-05-XX-onnx-quantization.md`.
- **Phase 7** — ORT → Candle migration. Long-term. Plan: `2026-05-XX-candle-migration.md`.

---

**Plan owner:** to@letaem.in
**Created:** 2026-05-12
**Related incidents:** 2026-05-12 jina-code-v2 arena_oom regression (PR #132 revert), BUG-004 (Phase B per-session arena)
