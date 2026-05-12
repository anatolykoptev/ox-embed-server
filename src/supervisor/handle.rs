//! Single worker process handle — owns Child, connects WorkerClient.
//!
//! Phase 2 scaffold: blocking spawn, no auto-restart.
//! Wave 2.5 (Task 16) will switch to actor-pattern WorkerSupervisor with watchdog.

use crate::ipc::client::WorkerClient;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};

#[derive(Debug, Clone)]
pub struct SpawnSpec {
    pub model: String,
    pub worker_bin: PathBuf,
    pub socket_dir: PathBuf,
    pub pool_size: usize,
    pub intra_threads: usize,
    /// Extra env vars to pass to the worker (e.g. EMBED_MODELS, ORT_DYLIB_PATH).
    pub env_extra: Vec<(String, String)>,
}

pub struct WorkerHandle {
    pub model: String,
    pub socket_path: PathBuf,
    pub child: Child,
    pub client: Arc<WorkerClient>,
}

impl WorkerHandle {
    pub async fn spawn(spec: SpawnSpec) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&spec.socket_dir).ok();
        let socket_path = spec.socket_dir.join(format!("{}.sock", spec.model));
        let _ = std::fs::remove_file(&socket_path);

        tracing::info!(model = %spec.model, ?socket_path, pool_size = spec.pool_size, "spawning worker");

        let mut cmd = Command::new(&spec.worker_bin);
        cmd.env("EMBED_WORKER_MODEL", &spec.model)
            .env("EMBED_WORKER_SOCKET", &socket_path)
            .env("EMBED_WORKER_POOL_SIZE", spec.pool_size.to_string())
            .env("EMBED_WORKER_INTRA_THREADS", spec.intra_threads.to_string())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for (k, v) in &spec.env_extra {
            cmd.env(k, v);
        }
        let child = cmd.spawn().map_err(|e| {
            tracing::error!(model = %spec.model, error = %e, "worker spawn failed");
            e
        })?;

        // Wait up to 60s for the worker to create its UDS (model cold load can take seconds).
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

        let client = Arc::new(
            WorkerClient::connect(socket_path.clone(), spec.pool_size)
                .await
                .map_err(|e| anyhow::anyhow!("worker client connect failed: {e}"))?,
        );

        tracing::info!(model = %spec.model, "worker handle ready");
        Ok(Self {
            model: spec.model,
            socket_path,
            child,
            client,
        })
    }
}
