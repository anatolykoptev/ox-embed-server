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

    pub async fn add(&self, supervisor: Arc<WorkerSupervisor>) {
        let mut w = self.workers.write().await;
        w.insert(supervisor.model().to_string(), supervisor);
    }

    /// Shared helper: look up the supervisor for `model` and poll until a
    /// live client is available (or `dispatch_timeout` expires).
    async fn get_client(
        &self,
        model: &str,
    ) -> Result<Arc<crate::ipc::client::WorkerClient>, DispatchError> {
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
                return Ok(client);
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
    pub async fn dispatch_embed(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> Result<WorkerResponse, DispatchError> {
        let client = self.get_client(model).await?;
        client
            .dispatch_embed(model.to_string(), texts, max_seq_len)
            .await
            .map_err(DispatchError::Ipc)
    }

    /// Dispatch a rerank request to the worker registered for `model`.
    pub async fn dispatch_rerank(
        &self,
        model: &str,
        query: String,
        documents: Vec<String>,
        max_seq_len: u32,
    ) -> Result<WorkerResponse, DispatchError> {
        let client = self.get_client(model).await?;
        client
            .dispatch_rerank(model.to_string(), query, documents, max_seq_len)
            .await
            .map_err(DispatchError::Ipc)
    }

    /// Dispatch a splade request to the worker registered for `model`.
    pub async fn dispatch_splade(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
        top_k: u32,
        min_weight: f32,
    ) -> Result<WorkerResponse, DispatchError> {
        let client = self.get_client(model).await?;
        client
            .dispatch_splade(model.to_string(), texts, max_seq_len, top_k, min_weight)
            .await
            .map_err(DispatchError::Ipc)
    }

    /// Currently registered model names. Useful for /health, /metrics.
    #[allow(dead_code)] // used by health/metrics endpoints
    pub async fn models(&self) -> Vec<String> {
        self.workers.read().await.keys().cloned().collect()
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
