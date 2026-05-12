//! Routing layer — model name → WorkerHandle. Dispatches inference.

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
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add(&self, handle: WorkerHandle) {
        let mut w = self.workers.write().await;
        w.insert(handle.model.clone(), Arc::new(handle));
    }

    pub async fn dispatch(
        &self,
        model: &str,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> anyhow::Result<InferResponse> {
        let handle = {
            let w = self.workers.read().await;
            w.get(model)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no worker for model {model}"))?
        };
        Ok(handle
            .client
            .infer(model.to_string(), texts, max_seq_len)
            .await?)
    }

    /// Currently registered model names. Useful for /health, /metrics.
    pub async fn models(&self) -> Vec<String> {
        self.workers.read().await.keys().cloned().collect()
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}
