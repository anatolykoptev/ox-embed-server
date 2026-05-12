//! Routing layer — model name → WorkerSupervisor. Dispatches inference.
//!
//! Wave 2.5: pool now holds `Arc<WorkerSupervisor>` (not `Arc<WorkerHandle>`).
//! `dispatch` polls for a live client during respawn, returning an error after
//! `dispatch_timeout` (default 30s).

use crate::ipc::protocol::InferResponse;
use crate::supervisor::WorkerSupervisor;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub struct WorkerPool {
    workers: Arc<RwLock<HashMap<String, Arc<WorkerSupervisor>>>>,
    /// Maximum time to wait for a respawning worker to become available.
    dispatch_timeout: Duration,
}

impl WorkerPool {
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            dispatch_timeout: Duration::from_secs(30),
        }
    }

    pub async fn add(&self, supervisor: Arc<WorkerSupervisor>) {
        let mut w = self.workers.write().await;
        w.insert(supervisor.model().to_string(), supervisor);
    }

    pub async fn dispatch(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> anyhow::Result<InferResponse> {
        let supervisor = {
            let w = self.workers.read().await;
            w.get(model)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no worker for model {model}"))?
        };

        // Poll until a live client is available or the timeout expires.
        // The supervisor clears client_slot to None while respawn is in
        // progress; callers simply wait here rather than getting an instant
        // error, which prevents thundering-herd reconnect attempts.
        let deadline = std::time::Instant::now() + self.dispatch_timeout;
        loop {
            if let Some(client) = supervisor.client().await {
                return Ok(client.infer(model.to_string(), texts, max_seq_len).await?);
            }
            if std::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "worker for model {model} unavailable after {}s (respawn in progress)",
                    self.dispatch_timeout.as_secs()
                );
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Currently registered model names. Useful for /health, /metrics.
    #[allow(dead_code)] // used by health/metrics endpoints
    pub async fn models(&self) -> Vec<String> {
        self.workers.read().await.keys().cloned().collect()
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}
