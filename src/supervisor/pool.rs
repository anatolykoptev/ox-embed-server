//! Routing layer — model name → WorkerSupervisor. Dispatches inference.
//!
//! Wave 2.5: pool now holds `Arc<WorkerSupervisor>` (not `Arc<WorkerHandle>`).
//! Dispatch methods poll for a live client during respawn, returning an error
//! after `dispatch_timeout` (default `DISPATCH_TIMEOUT_SECS`, overridable via
//! `EMBED_DISPATCH_TIMEOUT_SECS`).

use crate::ipc::protocol::WorkerResponse;
use crate::supervisor::WorkerSupervisor;
use crate::supervisor::util::resolve_duration_secs_env;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Typed dispatch error — lets the API layer distinguish a transient
/// "worker respawning" timeout (→ 503 + Retry-After, clients retry) from
/// a hard inference failure (→ 500). Without this, `get_client` timeout
/// was mapped to a generic `anyhow::Error` and surfaced as 500, breaking
/// client retry logic (issue #97).
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// Worker for `model` not registered in the pool. Programmer error —
    /// config/build issue, not transient.
    #[error("no worker for model {model}")]
    NoWorker { model: String },
    /// Worker is respawning and did not become available within
    /// `dispatch_timeout`. Transient — clients should retry after a
    /// short backoff. Maps to HTTP 503 + Retry-After.
    #[error("worker for model {model} unavailable after {timeout_secs}s (respawn in progress)")]
    Timeout { model: String, timeout_secs: u64 },
    /// Underlying IPC error (UDS read/write failure, frame decode error).
    /// May be transient (worker crashed mid-dispatch) or permanent.
    #[error(transparent)]
    Ipc(#[from] std::io::Error),
}

trait IpcErrorExt {
    /// `InvalidData`/`InvalidInput` come from a live worker answering with the
    /// wrong response id or cardinality (`ipc::client`) or an undecodable
    /// frame: deterministic, so a retry only repeats it. Connect, EOF and
    /// broken-pipe errors mean the worker is gone.
    fn is_protocol_violation(&self) -> bool;
}

impl IpcErrorExt for std::io::Error {
    fn is_protocol_violation(&self) -> bool {
        matches!(
            self.kind(),
            std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput
        )
    }
}

impl DispatchError {
    /// Bounded `reason` label for `embed_inference_failures_total`.
    pub fn failure_reason(&self) -> &'static str {
        match self {
            DispatchError::NoWorker { .. } => "no_worker",
            DispatchError::Timeout { .. } => "worker_unavailable",
            DispatchError::Ipc(e) if e.is_protocol_violation() => "ipc_protocol",
            DispatchError::Ipc(_) => "worker_died",
        }
    }

    /// True when a retry can succeed once the worker has respawned: the
    /// handlers answer 503 + `Retry-After` instead of a permanent-looking 500.
    /// `NoWorker` is a config error and stays 500.
    pub fn is_transient(&self) -> bool {
        match self {
            DispatchError::Timeout { .. } => true,
            DispatchError::Ipc(e) => !e.is_protocol_violation(),
            DispatchError::NoWorker { .. } => false,
        }
    }
}

// ── pool tuning defaults ───────────────────────────────────────────────────────

/// Maximum time to wait for a respawning worker to become available.
///
/// 30 s is chosen as the worst-case bound for a model reload:
///   - ONNX graph load: ~5–15 s on ARM for the largest models (jina-code-v2).
///   - Socket wait poll: up to SOCKET_WAIT_POLL_INTERVAL × 300 iterations.
///
/// Callers (the downstream consumer) retry on error, so a bounded 30 s timeout surfaces a
/// clear error rather than queuing silently forever.
/// Overridable via `EMBED_DISPATCH_TIMEOUT_SECS`. Captured at startup;
/// restart the container to change.
const DISPATCH_TIMEOUT_SECS: u64 = 30;

/// How often to poll for a live client while a worker is respawning.
///
/// 200 ms is fine-grained enough to detect recovery within one round-trip
/// (model reload ≈ 5–15 s) without burning CPU. Not operator-tunable:
/// changing the poll granularity doesn't meaningfully affect P99 latency.
const DISPATCH_POLL_INTERVAL: Duration = Duration::from_millis(200);

pub struct WorkerPool {
    workers: Arc<RwLock<HashMap<String, Arc<WorkerSupervisor>>>>,
    /// Maximum time to wait for a respawning worker to become available.
    dispatch_timeout: Duration,
}

impl WorkerPool {
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            dispatch_timeout: resolve_duration_secs_env(
                "EMBED_DISPATCH_TIMEOUT_SECS",
                Duration::from_secs(DISPATCH_TIMEOUT_SECS),
                &format!("DISPATCH_TIMEOUT_SECS ({DISPATCH_TIMEOUT_SECS})"),
            ),
        }
    }

    /// Test-only: build a pool with an explicit `dispatch_timeout`,
    /// bypassing the env-var resolution in [`new`]. Used by the #150
    /// tests to force `DispatchError::Timeout` quickly (e.g. 1 ms)
    /// against a `for_test_no_client` supervisor without waiting 30 s.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn with_dispatch_timeout(dispatch_timeout: Duration) -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            dispatch_timeout,
        }
    }

    pub async fn add(&self, supervisor: Arc<WorkerSupervisor>) {
        let mut w = self.workers.write().await;
        w.insert(supervisor.model().to_string(), supervisor);
    }

    /// Shared helper: look up the supervisor for `model` and poll until a
    /// live client is available (or `dispatch_timeout` expires).
    ///
    /// Returns both the supervisor (so the caller can stamp
    /// [`WorkerSupervisor::record_success`] on a successful dispatch) and the
    /// client.
    async fn get_client(
        &self,
        model: &str,
    ) -> Result<(Arc<WorkerSupervisor>, Arc<crate::ipc::client::WorkerClient>), DispatchError> {
        let supervisor = {
            let w = self.workers.read().await;
            w.get(model)
                .cloned()
                .ok_or_else(|| DispatchError::NoWorker {
                    model: model.to_string(),
                })?
        };

        let deadline = std::time::Instant::now() + self.dispatch_timeout;
        loop {
            if let Some(client) = supervisor.client().await {
                return Ok((supervisor, client));
            }
            if std::time::Instant::now() >= deadline {
                return Err(DispatchError::Timeout {
                    model: model.to_string(),
                    timeout_secs: self.dispatch_timeout.as_secs(),
                });
            }
            tokio::time::sleep(DISPATCH_POLL_INTERVAL).await;
        }
    }

    /// Dispatch an embed request to the worker registered for `model`.
    ///
    /// On a **successful** inference (non-`Err` [`WorkerResponse`]), stamps
    /// the supervisor's last-success timestamp so the heartbeat freshness
    /// check can suppress false-positive probe failures under load (#149).
    pub async fn dispatch_embed(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> Result<WorkerResponse, DispatchError> {
        let (supervisor, client) = self.get_client(model).await?;
        let result = client
            .dispatch_embed(model.to_string(), texts, max_seq_len)
            .await
            .map_err(DispatchError::Ipc)?;
        if !matches!(result, crate::ipc::protocol::WorkerResponse::Err { .. }) {
            supervisor.record_success();
        }
        Ok(result)
    }

    /// Dispatch a rerank request to the worker registered for `model`.
    ///
    /// On a **successful** inference (non-`Err` [`WorkerResponse`]), stamps
    /// the supervisor's last-success timestamp — see [`dispatch_embed`].
    pub async fn dispatch_rerank(
        &self,
        model: &str,
        query: String,
        documents: Vec<String>,
        max_seq_len: u32,
    ) -> Result<WorkerResponse, DispatchError> {
        let (supervisor, client) = self.get_client(model).await?;
        let result = client
            .dispatch_rerank(model.to_string(), query, documents, max_seq_len)
            .await
            .map_err(DispatchError::Ipc)?;
        if !matches!(result, crate::ipc::protocol::WorkerResponse::Err { .. }) {
            supervisor.record_success();
        }
        Ok(result)
    }

    /// Dispatch a splade request to the worker registered for `model`.
    ///
    /// On a **successful** inference (non-`Err` [`WorkerResponse`]), stamps
    /// the supervisor's last-success timestamp — see [`dispatch_embed`].
    pub async fn dispatch_splade(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
        top_k: u32,
        min_weight: f32,
    ) -> Result<WorkerResponse, DispatchError> {
        let (supervisor, client) = self.get_client(model).await?;
        let result = client
            .dispatch_splade(model.to_string(), texts, max_seq_len, top_k, min_weight)
            .await
            .map_err(DispatchError::Ipc)?;
        if !matches!(result, crate::ipc::protocol::WorkerResponse::Err { .. }) {
            supervisor.record_success();
        }
        Ok(result)
    }

    /// Currently registered model names. Useful for /health, /metrics.
    #[allow(dead_code)] // used by health/metrics endpoints
    pub async fn models(&self) -> Vec<String> {
        self.workers.read().await.keys().cloned().collect()
    }

    /// Models whose worker has no live connection right now (not yet loaded,
    /// or between respawns), sorted. Empty means every worker is up.
    pub async fn down_models(&self) -> Vec<String> {
        let sups: Vec<Arc<WorkerSupervisor>> =
            self.workers.read().await.values().cloned().collect();
        let mut down = Vec::new();
        for sup in sups {
            if sup.client().await.is_none() {
                down.push(sup.model().to_string());
            }
        }
        down.sort();
        down
    }

    /// Snapshot of (model_name, pid) pairs for currently live workers.
    ///
    /// Workers between respawns have `pid == 0` and are excluded from the
    /// returned list — the RSS-poll loop has nothing to read for them.
    pub async fn worker_pids(&self) -> Vec<(String, u32)> {
        self.workers
            .read()
            .await
            .values()
            .filter_map(|sup| {
                let pid = sup.current_pid();
                if pid != 0 {
                    Some((sup.model().to_string(), pid))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{DISPATCH_POLL_INTERVAL, DISPATCH_TIMEOUT_SECS};
    use crate::supervisor::util::resolve_duration_secs_env;
    use serial_test::serial;
    use std::time::Duration;

    fn spec(model: &str) -> crate::supervisor::handle::SpawnSpec {
        crate::supervisor::handle::SpawnSpec {
            model: model.to_string(),
            kind: crate::supervisor::handle::WorkerKind::Embed,
            worker_bin: std::path::PathBuf::from("/bin/true"),
            socket_dir: std::path::PathBuf::from("/tmp"),
            pool_size: 1,
            intra_threads: 1,
            env_extra: vec![],
        }
    }

    /// `/ready` trusts this list: it must name exactly the workers without a
    /// live client, sorted, and nothing else.
    #[tokio::test]
    async fn down_models_lists_only_workers_without_a_client() {
        use crate::supervisor::handle::WorkerSupervisor;
        let pool = super::WorkerPool::new();
        assert!(pool.down_models().await.is_empty());

        let sock = std::env::temp_dir().join(format!("pool-down-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let _listener = tokio::net::UnixListener::bind(&sock).expect("bind");
        let client = std::sync::Arc::new(
            crate::ipc::client::WorkerClient::connect(sock.clone(), 1)
                .await
                .expect("connect"),
        );
        pool.add(WorkerSupervisor::for_test(spec("up"), client))
            .await;
        pool.add(WorkerSupervisor::for_test_no_client(spec("zeta")))
            .await;
        pool.add(WorkerSupervisor::for_test_no_client(spec("alpha")))
            .await;

        assert_eq!(pool.down_models().await, vec!["alpha", "zeta"]);
        let _ = std::fs::remove_file(&sock);
    }

    #[test]
    fn dispatch_error_reason_and_retryability() {
        use super::DispatchError;
        let ipc = DispatchError::Ipc(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
        let timeout = DispatchError::Timeout {
            model: "m".into(),
            timeout_secs: 1,
        };
        let none = DispatchError::NoWorker { model: "m".into() };
        assert_eq!(ipc.failure_reason(), "worker_died");
        let proto = DispatchError::Ipc(std::io::Error::from(std::io::ErrorKind::InvalidData));
        assert_eq!(proto.failure_reason(), "ipc_protocol");
        assert!(
            !proto.is_transient(),
            "a protocol violation is not retryable"
        );
        for kind in [
            std::io::ErrorKind::NotFound,
            std::io::ErrorKind::ConnectionRefused,
            std::io::ErrorKind::UnexpectedEof,
            std::io::ErrorKind::BrokenPipe,
        ] {
            let e = DispatchError::Ipc(std::io::Error::from(kind));
            assert_eq!(e.failure_reason(), "worker_died", "{kind:?}");
            assert!(e.is_transient(), "{kind:?}");
        }
        assert_eq!(timeout.failure_reason(), "worker_unavailable");
        assert_eq!(none.failure_reason(), "no_worker");
        assert!(ipc.is_transient() && timeout.is_transient());
        assert!(!none.is_transient(), "a missing worker is a config error");
        for e in [&ipc, &proto, &timeout, &none] {
            assert!(
                crate::metrics::INFERENCE_FAILURE_REASONS.contains(&e.failure_reason()),
                "reason must be pre-touched: {}",
                e.failure_reason()
            );
        }
    }

    fn resolve_dispatch_timeout() -> Duration {
        resolve_duration_secs_env(
            "EMBED_DISPATCH_TIMEOUT_SECS",
            Duration::from_secs(DISPATCH_TIMEOUT_SECS),
            &format!("DISPATCH_TIMEOUT_SECS ({DISPATCH_TIMEOUT_SECS})"),
        )
    }

    #[test]
    fn poll_interval_is_subzero() {
        // Sanity: poll interval < 1s so respawn detection latency is bounded.
        assert!(DISPATCH_POLL_INTERVAL.as_millis() < 1000);
    }

    #[test]
    #[serial]
    fn dispatch_timeout_default() {
        let prev = std::env::var("EMBED_DISPATCH_TIMEOUT_SECS").ok();
        unsafe { std::env::remove_var("EMBED_DISPATCH_TIMEOUT_SECS") };
        let d = resolve_dispatch_timeout();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_DISPATCH_TIMEOUT_SECS") },
        }
        assert_eq!(d.as_secs(), DISPATCH_TIMEOUT_SECS);
    }

    #[test]
    #[serial]
    fn dispatch_timeout_env_override() {
        let prev = std::env::var("EMBED_DISPATCH_TIMEOUT_SECS").ok();
        unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", "120") };
        let d = resolve_dispatch_timeout();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_DISPATCH_TIMEOUT_SECS") },
        }
        assert_eq!(d.as_secs(), 120);
    }

    #[test]
    #[serial]
    fn dispatch_timeout_zero_falls_back() {
        let prev = std::env::var("EMBED_DISPATCH_TIMEOUT_SECS").ok();
        unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", "0") };
        let d = resolve_dispatch_timeout();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_DISPATCH_TIMEOUT_SECS") },
        }
        assert_eq!(
            d.as_secs(),
            DISPATCH_TIMEOUT_SECS,
            "zero falls back to default"
        );
    }

    #[test]
    #[serial]
    fn dispatch_timeout_invalid_falls_back() {
        let prev = std::env::var("EMBED_DISPATCH_TIMEOUT_SECS").ok();
        unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", "not-a-number") };
        let d = resolve_dispatch_timeout();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_DISPATCH_TIMEOUT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_DISPATCH_TIMEOUT_SECS") },
        }
        assert_eq!(
            d.as_secs(),
            DISPATCH_TIMEOUT_SECS,
            "invalid falls back to default"
        );
    }
}
