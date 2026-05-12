//! Routing layer — model name → WorkerSupervisor. Dispatches inference.
//!
//! Wave 2.5: pool now holds `Arc<WorkerSupervisor>` (not `Arc<WorkerHandle>`).
//! Dispatch methods poll for a live client during respawn, returning an error
//! after `dispatch_timeout` (default 30s).

use crate::ipc::protocol::WorkerResponse;
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

    /// Shared helper: look up the supervisor for `model` and poll until a
    /// live client is available (or `dispatch_timeout` expires).
    async fn get_client(
        &self,
        model: &str,
    ) -> anyhow::Result<Arc<crate::ipc::client::WorkerClient>> {
        let supervisor = {
            let w = self.workers.read().await;
            w.get(model)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no worker for model {model}"))?
        };

        let deadline = std::time::Instant::now() + self.dispatch_timeout;
        loop {
            if let Some(client) = supervisor.client().await {
                return Ok(client);
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

    /// Dispatch an embed request to the worker registered for `model`.
    pub async fn dispatch_embed(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> anyhow::Result<WorkerResponse> {
        let client = self.get_client(model).await?;
        Ok(client
            .dispatch_embed(model.to_string(), texts, max_seq_len)
            .await?)
    }

    /// Dispatch a rerank request to the worker registered for `model`.
    pub async fn dispatch_rerank(
        &self,
        model: &str,
        query: String,
        documents: Vec<String>,
        max_seq_len: u32,
    ) -> anyhow::Result<WorkerResponse> {
        let client = self.get_client(model).await?;
        Ok(client
            .dispatch_rerank(model.to_string(), query, documents, max_seq_len)
            .await?)
    }

    /// Dispatch a splade request to the worker registered for `model`.
    pub async fn dispatch_splade(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> anyhow::Result<WorkerResponse> {
        let client = self.get_client(model).await?;
        Ok(client
            .dispatch_splade(model.to_string(), texts, max_seq_len)
            .await?)
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
