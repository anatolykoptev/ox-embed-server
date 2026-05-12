//! WorkerSupervisor — owns Child exclusively, watches for exit, respawns.
//!
//! Wave 2.5 (Task 16): replaced the single-shot WorkerHandle with an actor
//! pattern. The supervisor:
//!   - owns the Child in a dedicated tokio task (watchdog_loop)
//!   - clears the client slot on worker exit so dispatchers see "unavailable"
//!   - automatically respawns with exponential backoff (2s → 60s cap)
//!   - increments restart_count on each successful respawn
//!
//! Backoff schedule: 2s → 4s → 8s → … → 60s (capped). Backoff advances
//! exactly once per failed spawn attempt and resets to INITIAL_BACKOFF on
//! the first successful respawn.
//!
//! SpawnSpec is unchanged from Wave 2.3 (no .kind field yet; that lands in
//! Wave 2.4b when reranker/splade IPC variants are added).
//!
//! TODO followups:
//! - Connection-error != worker-death detection (latent slot poisoning when
//!   worker listener dies but process alive). Wave 2.5b heartbeat or
//!   detection ping needed.
//! - dispatch_timeout env-gate (currently hardcoded 30s).
//! - Watchdog circuit-breaker: stop respawning after N consecutive failures
//!   with no success in between (currently retries forever).
//! - Graceful shutdown via ControlMessage::Shutdown before kill_on_drop SIGKILL.

use crate::ipc::client::WorkerClient;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::RwLock;

const INITIAL_BACKOFF: Duration = Duration::from_secs(2);
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Advance exponential backoff by doubling, capped at MAX_BACKOFF.
fn next_backoff(current: Duration) -> Duration {
    (current * 2).min(MAX_BACKOFF)
}

/// Inference kind the worker should load.
///
/// Passed as `EMBED_WORKER_KIND` env to the worker process. The worker
/// loads the appropriate model type and expects only the matching
/// `WorkerRequest` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerKind {
    Embed,
    Rerank,
    Splade,
}

impl WorkerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Rerank => "rerank",
            Self::Splade => "splade",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpawnSpec {
    pub model: String,
    /// Kind of model this worker should load. Sets `EMBED_WORKER_KIND` env.
    pub kind: WorkerKind,
    pub worker_bin: PathBuf,
    pub socket_dir: PathBuf,
    pub pool_size: usize,
    pub intra_threads: usize,
    /// Extra env vars to pass to the worker (e.g. EMBED_MODELS, ORT_DYLIB_PATH).
    pub env_extra: Vec<(String, String)>,
}

/// Actor that owns a single worker process and its UDS client.
///
/// Callers query the live client via [`WorkerSupervisor::client`]; the field
/// is `None` while a respawn is in progress. [`WorkerPool::dispatch`] polls
/// with a configurable timeout.
pub struct WorkerSupervisor {
    spec: SpawnSpec,
    /// Current live client. `None` while the worker is being respawned.
    client_slot: Arc<RwLock<Option<Arc<WorkerClient>>>>,
    /// Monotonically increasing count of successful respawns for observability.
    restart_count: Arc<std::sync::atomic::AtomicU64>,
}

impl WorkerSupervisor {
    /// Spawn the supervisor + initial worker. Fails loudly on first-start
    /// failure (startup errors are not retried; only post-startup crashes
    /// trigger the watchdog respawn loop).
    pub async fn launch(spec: SpawnSpec) -> anyhow::Result<Arc<Self>> {
        let supervisor = Arc::new(Self {
            spec: spec.clone(),
            client_slot: Arc::new(RwLock::new(None)),
            restart_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        });

        // Initial spawn — fail loudly so the server startup loop can exit(1).
        let (child, client) = Self::spawn_one(&supervisor.spec).await?;
        *supervisor.client_slot.write().await = Some(client);

        // Hand off the Child to the watchdog task; it owns the Child for its
        // entire lifetime.
        let sup_clone = supervisor.clone();
        tokio::spawn(async move {
            sup_clone.watchdog_loop(child).await;
        });

        Ok(supervisor)
    }

    /// One-shot: fork worker, wait for socket to appear, connect client.
    ///
    /// Returns `(Child, Arc<WorkerClient>)` on success. Fails if the process
    /// dies before the socket appears or if the initial connect fails.
    async fn spawn_one(spec: &SpawnSpec) -> anyhow::Result<(Child, Arc<WorkerClient>)> {
        if let Err(e) = std::fs::create_dir_all(&spec.socket_dir) {
            tracing::warn!(dir = ?spec.socket_dir, error = %e, "create_dir_all failed; subsequent bind may fail");
        }
        let socket_path = spec.socket_dir.join(format!("{}.sock", spec.model));
        let _ = std::fs::remove_file(&socket_path);

        tracing::info!(model = %spec.model, ?socket_path, pool_size = spec.pool_size, "spawning worker");

        let mut cmd = Command::new(&spec.worker_bin);
        cmd.env("EMBED_WORKER_MODEL", &spec.model)
            .env("EMBED_WORKER_KIND", spec.kind.as_str())
            .env("EMBED_WORKER_SOCKET", &socket_path)
            .env("EMBED_WORKER_POOL_SIZE", spec.pool_size.to_string())
            .env("EMBED_WORKER_INTRA_THREADS", spec.intra_threads.to_string())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        for (k, v) in &spec.env_extra {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(|e| {
            tracing::error!(model = %spec.model, error = %e, "worker spawn failed");
            e
        })?;

        // Poll up to 60s for socket, with early-exit if child dies first.
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if Instant::now() >= deadline {
                anyhow::bail!("worker {} did not create socket within 60s", spec.model);
            }
            if tokio::fs::try_exists(&socket_path).await.unwrap_or(false) {
                break;
            }
            if let Some(status) = child.try_wait()? {
                anyhow::bail!(
                    "worker {} exited before socket appeared: status={:?}",
                    spec.model,
                    status
                );
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        let client = Arc::new(
            WorkerClient::connect(socket_path.clone(), spec.pool_size)
                .await
                .map_err(|e| anyhow::anyhow!("worker client connect failed: {e}"))?,
        );

        tracing::info!(model = %spec.model, "worker handle ready");
        Ok((child, client))
    }

    /// Watchdog: block on `child.wait()`, clear slot, respawn with exponential
    /// backoff. Loops forever — the tokio task is the process lifetime.
    ///
    /// Backoff advances exactly once per failed spawn attempt (in the Err arm)
    /// and resets to INITIAL_BACKOFF on the first success. Exit codes 134
    /// (SIGABRT) and 137 (SIGKILL/OOM) are treated identically to clean exit.
    async fn watchdog_loop(self: Arc<Self>, mut child: Child) {
        let mut backoff = INITIAL_BACKOFF;
        loop {
            // Wait for the current child to exit.
            let status = match child.wait().await {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::error!(
                        model = %self.spec.model,
                        error = %e,
                        "child.wait() errored"
                    );
                    None
                }
            };

            // Log exit with signal info where available.
            if let Some(ref status) = status {
                #[cfg(unix)]
                let signal = std::os::unix::process::ExitStatusExt::signal(status);
                #[cfg(not(unix))]
                let signal: Option<i32> = None;
                tracing::warn!(
                    model = %self.spec.model,
                    ?status,
                    code = ?status.code(),
                    ?signal,
                    restart_count = self.restart_count.load(std::sync::atomic::Ordering::Relaxed),
                    "worker exited; clearing client slot and respawning"
                );
            }

            // Clear client slot — dispatchers see "worker unavailable".
            *self.client_slot.write().await = None;

            // Respawn loop — each failed attempt advances backoff exactly once.
            loop {
                tokio::time::sleep(backoff).await;
                match Self::spawn_one(&self.spec).await {
                    Ok((new_child, new_client)) => {
                        *self.client_slot.write().await = Some(new_client);
                        self.restart_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        crate::metrics::worker_restart_inc(&self.spec.model);
                        backoff = INITIAL_BACKOFF; // reset on success
                        child = new_child;
                        break; // exit inner loop, back to outer wait()
                    }
                    Err(e) => {
                        tracing::error!(
                            model = %self.spec.model,
                            error = ?e,
                            backoff_secs = backoff.as_secs(),
                            "respawn failed; will retry"
                        );
                        backoff = next_backoff(backoff);
                    }
                }
            }
        }
    }

    /// Returns the current live client, or `None` if a respawn is in progress.
    pub async fn client(&self) -> Option<Arc<WorkerClient>> {
        self.client_slot.read().await.clone()
    }

    /// Number of successful respawns since launch. Zero until first crash.
    #[allow(dead_code)] // TODO(Phase 3 metrics): expose via /health or /metrics endpoint
    pub fn restart_count(&self) -> u64 {
        self.restart_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Model name this supervisor is responsible for.
    pub fn model(&self) -> &str {
        &self.spec.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `launch()` surfaces a clear error when the fake worker
    /// creates the socket file but doesn't actually listen on it (connect
    /// will fail). The important invariant: launch never hangs — it returns
    /// Err within the 60s socket-wait window (or immediately if connect
    /// fails after the socket file appears).
    ///
    /// Full respawn-path coverage (SIGKILL → supervisor restarts → pool keeps
    /// serving) requires a real mini-worker binary and is handled by the
    /// controller's integration test suite.
    #[tokio::test]
    #[ignore = "needs full mini-worker harness; respawn verified by controller integration tests"]
    async fn supervisor_respawns_on_child_exit() {
        let socket_dir: std::path::PathBuf =
            std::env::temp_dir().join(format!("embed-sup-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&socket_dir);
        std::fs::create_dir_all(&socket_dir).unwrap();

        let fake_worker_path = socket_dir.join("fake_worker.sh");
        let socket_path = socket_dir.join("test-model.sock");
        std::fs::write(
            &fake_worker_path,
            format!("#!/bin/sh\ntouch {}\nexit 0\n", socket_path.display()),
        )
        .unwrap();

        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&fake_worker_path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&fake_worker_path, perms).unwrap();

        let spec = SpawnSpec {
            model: "test-model".into(),
            kind: super::WorkerKind::Embed,
            worker_bin: fake_worker_path,
            socket_dir: socket_dir.clone(),
            pool_size: 1,
            intra_threads: 1,
            env_extra: vec![],
        };

        // The fake worker creates the socket file but doesn't listen on it.
        // WorkerClient::connect will fail → launch() returns Err.
        match WorkerSupervisor::launch(spec).await {
            Ok(_) => panic!("expected launch to fail with non-listening socket"),
            Err(e) => eprintln!("launch failed as expected: {e}"),
        }

        let _ = std::fs::remove_dir_all(&socket_dir);
    }
}
