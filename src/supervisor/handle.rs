//! WorkerSupervisor — owns Child exclusively, watches for exit, respawns.
//!
//! Wave 2.5 (Task 16): replaced the single-shot WorkerHandle with an actor
//! pattern. The supervisor:
//!   - owns the Child in a dedicated tokio task (watchdog_loop)
//!   - clears the client slot on worker exit so dispatchers see "unavailable"
//!   - automatically respawns with exponential backoff (INITIAL_BACKOFF → MAX_BACKOFF)
//!   - increments restart_count on each successful respawn
//!
//! Backoff schedule: INITIAL_BACKOFF → 2× → … → MAX_BACKOFF (capped). Backoff
//! advances exactly once per failed spawn attempt and resets to INITIAL_BACKOFF
//! on the first successful respawn.
//!
//! SpawnSpec is unchanged from Wave 2.3 (no .kind field yet; that lands in
//! Wave 2.4b when reranker/splade IPC variants are added).
//!
//! TODO followups:
//! - Connection-error != worker-death detection (latent slot poisoning when
//!   worker listener dies but process alive). Wave 2.5b heartbeat or
//!   detection ping needed.
//! - Watchdog circuit-breaker: stop respawning after N consecutive failures
//!   with no success in between (currently retries forever).
//! - Graceful shutdown via ControlMessage::Shutdown before kill_on_drop SIGKILL.

use crate::ipc::client::WorkerClient;
use crate::supervisor::util::resolve_duration_secs_env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt as _, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::RwLock;

// ── supervisor timing constants ────────────────────────────────────────────────

/// Initial respawn backoff delay after a worker crash.
///
/// 2 s gives the OS time to release the socket file and release process
/// resources before we attempt to fork again. Not operator-tunable: the
/// watchdog handles crashes asynchronously and callers already see
/// "unavailable" via the client_slot; adjusting this doesn't affect
/// caller-visible latency.
const INITIAL_BACKOFF: Duration = Duration::from_secs(2);

/// Maximum respawn backoff (exponential doubling caps here).
///
/// 60 s prevents permanent crash-loops from consuming CPU without bound.
/// Operators can observe restart_count via /metrics to detect flapping
/// workers.
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Default timeout for waiting for a freshly-spawned worker's socket to appear.
///
/// Model load times:
///   - e5-large INT8: ~3–5 s on ARM
///   - jina-code-v2: ~8–12 s (large BERT variant)
///   - SPLADE: ~2–4 s
///
/// 60 s provides 4–5× headroom on the slowest model. Overridable via
/// `EMBED_WORKER_SOCKET_WAIT_SECS`. Captured at startup; restart the
/// container to change.
const SOCKET_WAIT_SECS: u64 = 60;

/// Poll granularity for the socket-file existence check during worker startup.
///
/// 200 ms is fine-grained enough that the per-model startup delay is within
/// one poll interval of the actual socket appearance time. Not operator-tunable.
const SOCKET_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(200);

// ── heartbeat defaults ────────────────────────────────────────────────────────

/// Default heartbeat probe interval. The supervisor sends a 1-word inference
/// probe to the worker every this often. 0 disables heartbeat.
///
/// 30s balances detection latency against probe overhead: one 1-word
/// inference every 30s is negligible.
///
/// Worst-case wedge detection, with `HEARTBEAT_MAX_FAILS=3` and a 15s probe
/// timeout: each cycle is `sleep(30s)` plus a probe bounded at 15s = 45s, three
/// cycles to spend the fail budget = 135s, plus up to one further interval if
/// the wedge starts just after a good beat — so **135-165s**. (It was ~96-126s
/// under the old 2s probe timeout; the comment previously said "~90s", which
/// was wrong even then.) Over that window a wedged worker parks four to five
/// rounds of client requests, each bounded by `DISPATCH_TIMEOUT_SECS` (30s) in
/// `supervisor/pool.rs`, so callers see dispatch timeouts throughout — they are
/// not silently hanging.
///
/// That cost is accepted deliberately: a FALSE kill triggers a 5-15s cold model
/// reload, which lengthens the queue, which makes the next probe likelier to
/// time out — a positive feedback loop. 40 extra seconds on a genuinely wedged
/// worker that is already erroring is strictly the better failure.
///
/// Overridable via `EMBED_WORKER_HEARTBEAT_INTERVAL_SECS`. Captured at
/// startup; restart the container to change.
const HEARTBEAT_INTERVAL_SECS: u64 = 30;

/// Default consecutive heartbeat failures before killing the worker.
///
/// 3 consecutive fails (each with `HEARTBEAT_PROBE_TIMEOUT_MS` timeout)
/// filters transient slowness — a single slow batch under contention
/// should not trigger a kill. Only a sustained wedge (all 3 probes fail)
/// forces a restart.
///
/// Overridable via `EMBED_WORKER_HEARTBEAT_MAX_FAILS`. Must be > 0.
const HEARTBEAT_MAX_FAILS: u32 = 3;

/// Default timeout for each heartbeat inference probe.
///
/// 15s. The previous value was 2s, justified by "a healthy worker completes a
/// 1-word inference in 5-50ms" — which is true only of an IDLE worker. The
/// probe is dispatched through the same UDS and queues behind the worker's
/// session pool, so under real traffic it measures QUEUE DEPTH, not liveness.
/// `docs/BUGS.md` (BUG-001) records p50 ~1.7s single-query and ~2.4s at conc=4
/// for jina-code-v2 on this exact hardware — i.e. the old default sat BELOW the
/// measured median under load.
///
/// Live consequence on pillow between 2026-08-01 and 2026-08-04: 156 probe
/// timeouts and 32 SIGKILLs, all of healthy workers (`code-rank-embed` 23,
/// with `gte-multi-rerank` repeatedly logging "heartbeat recovered after
/// failures" — a genuinely wedged worker never recovers). Each kill costs a
/// 5-15s cold model reload, which lengthens the queue, which makes the next
/// probe more likely to time out. A wedge detector that fires hardest at peak
/// load is a load amplifier.
///
/// 15s is ~6x the documented p50-at-conc-4 and still an order of magnitude
/// below the time a genuinely wedged worker stays stuck (forever). The
/// remaining false-positive window — sustained saturation deep enough to delay
/// a probe past 15s — is closed properly by suppressing a failure when the
/// worker has completed a real inference within the interval; that is tracked
/// as a follow-up rather than bundled here.
///
/// Overridable via `EMBED_WORKER_HEARTBEAT_PROBE_TIMEOUT_MS`. Must be > 0.
/// NOTE: pillow's compose.yml sets this explicitly, so raising the default
/// alone does not change production — the deployed value must be raised too.
const HEARTBEAT_PROBE_TIMEOUT_MS: u64 = 15_000;

/// jina-code-v2 p50 at concurrency 4 on the deployment hardware, from
/// `docs/BUGS.md` BUG-001. Not a tuning knob — a recorded measurement.
const MEASURED_JINA_P50_CONC4_MS: u64 = 2_400;

/// The probe timeout may be retuned, but never back under the latency it is
/// probing. A compile-time assertion rather than a test on purpose: the failure
/// this guards is a one-character edit to a constant, and a binary whose wedge
/// detector will kill healthy workers should not build at all — waiting for a
/// test run gives it a window to be shipped in.
const _: () = assert!(
    HEARTBEAT_PROBE_TIMEOUT_MS >= 4 * MEASURED_JINA_P50_CONC4_MS,
    "HEARTBEAT_PROBE_TIMEOUT_MS must leave at least 4x headroom over the measured \
     jina-code-v2 p50-at-conc-4 (docs/BUGS.md BUG-001), or the wedge detector kills \
     healthy workers under load — as it did 32 times on pillow in three days"
);

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

/// What one [`WorkerSupervisor::heartbeat_tick`] cycle did.
///
/// Exists so a test can assert on the decision rather than on a side effect.
/// `heartbeat_loop` ignores it — the value is the observable seam that makes
/// the loop's logic testable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeartbeatOutcome {
    /// No probe was sent: the worker is between respawns (`client_slot` is
    /// `None`), or the fail budget was spent but the worker had already exited.
    /// Never counts as a failure.
    Skipped,
    /// Probe answered successfully. Fail counter reset.
    Ok,
    /// Probe timed out, failed to dispatch, or the worker answered with an
    /// error — but the fail budget is not yet spent.
    Failed,
    /// Fail budget spent; the worker was SIGKILLed for the watchdog to respawn.
    Killed,
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
    /// How long to wait for a freshly-spawned worker's socket to appear.
    ///
    /// Resolved once from `EMBED_WORKER_SOCKET_WAIT_SECS` at [`launch`] time;
    /// restart the container to pick up a changed value.
    socket_wait: Duration,
    /// PID of the currently live worker child process.
    ///
    /// Updated atomically after every successful [`spawn_one`] and cleared to 0
    /// when the child exits (before respawn). The RSS-poll loop reads this to
    /// call `/proc/<pid>/status`; 0 means worker is between respawns.
    current_pid: Arc<std::sync::atomic::AtomicU32>,
    /// Heartbeat probe interval. 0 disables heartbeat (for tests / opt-out).
    heartbeat_interval: Duration,
    /// Number of consecutive heartbeat failures before killing the worker.
    heartbeat_max_fails: u32,
    /// Timeout for each heartbeat inference probe.
    heartbeat_probe_timeout: Duration,
}

impl WorkerSupervisor {
    /// Spawn the supervisor + initial worker. Fails loudly on first-start
    /// failure (startup errors are not retried; only post-startup crashes
    /// trigger the watchdog respawn loop).
    pub async fn launch(spec: SpawnSpec) -> anyhow::Result<Arc<Self>> {
        // Resolve env-tunable values once here so every respawn uses the same
        // values captured at startup. Restart the container to change them.
        let socket_wait = resolve_duration_secs_env(
            "EMBED_WORKER_SOCKET_WAIT_SECS",
            Duration::from_secs(SOCKET_WAIT_SECS),
            &format!("SOCKET_WAIT_SECS ({SOCKET_WAIT_SECS})"),
        );
        let heartbeat_interval = resolve_duration_secs_env(
            "EMBED_WORKER_HEARTBEAT_INTERVAL_SECS",
            Duration::from_secs(HEARTBEAT_INTERVAL_SECS),
            &format!("HEARTBEAT_INTERVAL_SECS ({HEARTBEAT_INTERVAL_SECS})"),
        );
        let heartbeat_max_fails = match std::env::var("EMBED_WORKER_HEARTBEAT_MAX_FAILS") {
            Ok(s) => match s.trim().parse::<u32>() {
                Ok(0) => {
                    tracing::warn!(
                        "EMBED_WORKER_HEARTBEAT_MAX_FAILS=0 is invalid; defaulting to {HEARTBEAT_MAX_FAILS}"
                    );
                    HEARTBEAT_MAX_FAILS
                }
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!(
                        "EMBED_WORKER_HEARTBEAT_MAX_FAILS={s:?} is not a valid u32; defaulting to {HEARTBEAT_MAX_FAILS}"
                    );
                    HEARTBEAT_MAX_FAILS
                }
            },
            Err(_) => HEARTBEAT_MAX_FAILS,
        };
        let heartbeat_probe_timeout = Duration::from_millis(
            match std::env::var("EMBED_WORKER_HEARTBEAT_PROBE_TIMEOUT_MS") {
                Ok(s) => match s.trim().parse::<u64>() {
                    Ok(0) => {
                        tracing::warn!(
                            "EMBED_WORKER_HEARTBEAT_PROBE_TIMEOUT_MS=0 is invalid; defaulting to {HEARTBEAT_PROBE_TIMEOUT_MS}"
                        );
                        HEARTBEAT_PROBE_TIMEOUT_MS
                    }
                    Ok(v) => v,
                    Err(_) => {
                        tracing::warn!(
                            "EMBED_WORKER_HEARTBEAT_PROBE_TIMEOUT_MS={s:?} is not a valid u64; defaulting to {HEARTBEAT_PROBE_TIMEOUT_MS}"
                        );
                        HEARTBEAT_PROBE_TIMEOUT_MS
                    }
                },
                Err(_) => HEARTBEAT_PROBE_TIMEOUT_MS,
            },
        );

        let supervisor = Arc::new(Self {
            spec: spec.clone(),
            client_slot: Arc::new(RwLock::new(None)),
            restart_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            socket_wait,
            current_pid: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            heartbeat_interval,
            heartbeat_max_fails,
            heartbeat_probe_timeout,
        });

        // Initial spawn — fail loudly so the server startup loop can exit(1).
        let (child, client) = Self::spawn_one(&supervisor.spec, supervisor.socket_wait).await?;
        *supervisor.client_slot.write().await = Some(client);
        // Store child PID so the RSS-poll loop can read /proc/<pid>/status.
        // child.id() returns None only if the process has already been awaited,
        // which cannot happen here (we just spawned it).
        if let Some(pid) = child.id() {
            supervisor
                .current_pid
                .store(pid, std::sync::atomic::Ordering::Relaxed);
        }

        // Pre-touch the restart counter to 0 — makes "healthy / no restarts"
        // observable as a present-but-zero series, not absent (which
        // Prometheus operators read as "metric not wired").
        crate::metrics::worker_restart_touch(&supervisor.spec.model);
        // Pre-touch RSS gauge to 0 — same rationale as restart counter:
        // makes "healthy worker, RSS not yet sampled" visible in Prometheus
        // as 0 rather than absent.
        crate::metrics::worker_rss_touch(&supervisor.spec.model);
        // Pre-touch heartbeat counter so "no heartbeats yet" is visible as 0.
        crate::metrics::worker_heartbeat_touch(&supervisor.spec.model);

        // Hand off the Child to the watchdog task; it owns the Child for its
        // entire lifetime.
        let sup_clone = supervisor.clone();
        tokio::spawn(async move {
            sup_clone.watchdog_loop(child).await;
        });

        // Spawn the heartbeat liveness probe. This is the wedge detector for
        // issue #90: a worker can spin at 220% CPU with zero throughput without
        // exiting, so watchdog_loop (which blocks on child.wait()) never wakes.
        // The heartbeat sends periodic inference probes; after N consecutive
        // timeouts it kills the child via SIGKILL, which wakes watchdog_loop →
        // respawn. Disabled when heartbeat_interval is 0 (tests / opt-out).
        if !supervisor.heartbeat_interval.is_zero() {
            let sup_clone = supervisor.clone();
            tokio::spawn(async move {
                sup_clone.heartbeat_loop().await;
            });
        }

        Ok(supervisor)
    }

    /// One-shot: fork worker, wait for socket to appear, connect client.
    ///
    /// Returns `(Child, Arc<WorkerClient>)` on success. Fails if the process
    /// dies before the socket appears or if the initial connect fails.
    async fn spawn_one(
        spec: &SpawnSpec,
        socket_wait: Duration,
    ) -> anyhow::Result<(Child, Arc<WorkerClient>)> {
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
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        for (k, v) in &spec.env_extra {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(|e| {
            tracing::error!(model = %spec.model, error = %e, "worker spawn failed");
            e
        })?;

        // Spawn log-forwarding tasks: pipe worker stdout/stderr to supervisor
        // stdout/stderr with a `[<model>]` prefix so `docker logs` shows
        // per-model lines.
        //
        // Back-pressure note: worker uses a synchronous tracing writer. If the
        // OS pipe buffer fills (64 KiB), the worker's sync write blocks, which
        // can stall the tokio runtime. This is acceptable: the pipe empties
        // as fast as the supervisor's stdout can drain (docker logs; no
        // intermediate buffer). Under a log flood the worker rate-limits
        // itself — a useful natural back-pressure rather than a hazard.
        // The tasks are fire-and-forget (no join handle): they exit naturally
        // when the child closes its end of the pipe on exit.
        if let Some(child_stdout) = child.stdout.take() {
            let model_tag = format!("[{}] ", spec.model);
            tokio::spawn(async move {
                let reader = BufReader::new(child_stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // println! is fine here: each forwarded line is terminated by
                    // the newline from the worker's tracing output. The supervisor
                    // has a single stdout writer; no double-buffering concern.
                    println!("{}{}", model_tag, line);
                }
            });
        }
        if let Some(child_stderr) = child.stderr.take() {
            let model_tag = format!("[{}] ", spec.model);
            tokio::spawn(async move {
                let reader = BufReader::new(child_stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    eprintln!("{}{}", model_tag, line);
                }
            });
        }

        // Poll up to socket_wait for socket, with early-exit if child dies first.
        let deadline = Instant::now() + socket_wait;
        loop {
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "worker {} did not create socket within {}s",
                    spec.model,
                    socket_wait.as_secs()
                );
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
            tokio::time::sleep(SOCKET_WAIT_POLL_INTERVAL).await;
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

            // Clear worker state on exit — pid FIRST, then client slot.
            // Ordering matters: clearing pid before client_slot ensures the
            // RSS-poll loop sees pid=0 (skip) before dispatchers see
            // "unavailable", avoiding the epoch-exit gap where a dead PID
            // is still readable while the slot is already None.
            self.clear_on_exit().await;

            // Respawn loop — each failed attempt advances backoff exactly once.
            loop {
                tokio::time::sleep(backoff).await;
                match Self::spawn_one(&self.spec, self.socket_wait).await {
                    Ok((new_child, new_client)) => {
                        // Store new PID before handing child back to wait().
                        if let Some(pid) = new_child.id() {
                            self.current_pid
                                .store(pid, std::sync::atomic::Ordering::Relaxed);
                        }
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

    /// Clear worker state on exit — `current_pid` first, then `client_slot`.
    ///
    /// Ordering invariant: `current_pid` is set to 0 BEFORE `client_slot`
    /// becomes None. This ensures:
    ///   - The RSS-poll loop sees pid=0 (skip) before dispatchers see
    ///     "unavailable" — no window where it reads `/proc/<dead_pid>/status`.
    ///
    /// Scope of the benefit, stated honestly: this narrows a sub-window, it
    /// does not close a race. The dominant window is process-death ->
    /// `child.wait()` returns -> this runs, which is milliseconds and entirely
    /// unaffected. What the ordering removes is the interval between the atomic
    /// store and the `RwLock` write acquisition — sub-microsecond and
    /// uncontended. Concretely it saves at most one `/proc/<dead_pid>/status`
    /// read returning ENOENT per worker exit, in `supervisor/pool.rs`'s
    /// `worker_pids`. Worth keeping; not worth believing more of.
    ///
    /// It is NOT an invariant that observers never see `pid != 0 &&
    /// client_slot == None` — the respawn path in `watchdog_loop` deliberately
    /// uses the opposite order (PID set first, then `client_slot`) and so
    /// produces exactly that state on every restart. Reading a
    /// live-but-not-ready PID is harmless (memory stats only), and dispatchers
    /// correctly see "unavailable" until `client_slot` is set.
    async fn clear_on_exit(&self) {
        // Clear PID FIRST — RSS-poll loop sees pid=0 (skip) before
        // dispatchers see "unavailable" via client_slot=None.
        self.current_pid
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // Clear client slot — dispatchers see "worker unavailable".
        *self.client_slot.write().await = None;
    }

    /// Returns the current live client, or `None` if a respawn is in progress.
    pub async fn client(&self) -> Option<Arc<WorkerClient>> {
        self.client_slot.read().await.clone()
    }

    /// Heartbeat liveness loop — wedge detector for issue #90.
    ///
    /// `watchdog_loop` blocks on `child.wait()` and only wakes when the worker
    /// **exits**. A wedged worker (CPU spin, zero throughput) does not exit,
    /// so the watchdog sleeps forever and `embed_worker_restart_total` stays 0.
    ///
    /// This loop sends a 1-word inference probe to the worker every
    /// `heartbeat_interval`. If `heartbeat_max_fails` consecutive probes fail
    /// (timeout or error), it kills the child via SIGKILL. `watchdog_loop`
    /// then wakes from `child.wait()`, clears the client slot, and respawns.
    ///
    /// Why inference probe, not a lightweight ping: the worker runs ONNX in
    /// `spawn_blocking`, so its async runtime stays responsive while ONNX
    /// spins. A ping would succeed against a wedged worker. Only a real
    /// inference that queues behind the stuck `spawn_blocking` call can
    /// detect the wedge.
    ///
    /// The loop tolerates transient `client_slot = None` (respawn in
    /// progress) by skipping that probe without counting it as a failure.
    async fn heartbeat_loop(self: Arc<Self>) {
        let mut consecutive_fails: u32 = 0;
        loop {
            tokio::time::sleep(self.heartbeat_interval).await;
            self.heartbeat_tick(&mut consecutive_fails).await;
        }
    }

    /// One heartbeat cycle: probe, classify, and kill if the fail budget is
    /// spent. Returns what happened so a test can assert on it.
    ///
    /// Split out of [`heartbeat_loop`] because the loop is untestable by
    /// construction — it sleeps first and never returns, so no test can call
    /// it. The test that was supposed to prove the #90 fix worked
    /// (`heartbeat_kills_wedged_worker`) therefore asserted against a
    /// hand-inlined copy of this body instead: deleting the whole
    /// `tokio::spawn(heartbeat_loop)` call site left it green, and the
    /// per-kind dispatch below (#135) had no test coverage at all. Everything
    /// except the sleep lives here so the tests drive the real code.
    async fn heartbeat_tick(&self, consecutive_fails: &mut u32) -> HeartbeatOutcome {
        // Skip if worker is between respawns (client_slot = None).
        // This is not a heartbeat failure — the watchdog is already
        // handling a restart. Reset the fail counter so the freshly
        // respawned worker starts with a clean slate.
        let client = match self.client().await {
            Some(c) => c,
            None => {
                *consecutive_fails = 0;
                return HeartbeatOutcome::Skipped;
            }
        };

        // Dispatch the right probe kind for this worker — embed probe
        // to embed workers, rerank probe to rerank workers, splade probe
        // to splade workers. A kind mismatch (e.g. embed probe to a
        // rerank worker) would be a permanent error → false kill.
        let probe = tokio::time::timeout(
            self.heartbeat_probe_timeout,
            match self.spec.kind {
                WorkerKind::Embed => Box::pin(client.dispatch_embed(
                    self.spec.model.clone(),
                    vec!["test".to_string()],
                    8,
                ))
                    as std::pin::Pin<
                        Box<
                            dyn std::future::Future<
                                    Output = std::io::Result<crate::ipc::protocol::WorkerResponse>,
                                > + Send,
                        >,
                    >,
                WorkerKind::Rerank => Box::pin(client.dispatch_rerank(
                    self.spec.model.clone(),
                    "test".to_string(),
                    vec!["doc".to_string()],
                    8,
                ))
                    as std::pin::Pin<
                        Box<
                            dyn std::future::Future<
                                    Output = std::io::Result<crate::ipc::protocol::WorkerResponse>,
                                > + Send,
                        >,
                    >,
                WorkerKind::Splade => Box::pin(client.dispatch_splade(
                    self.spec.model.clone(),
                    vec!["test".to_string()],
                    8,
                    64,
                    0.01,
                ))
                    as std::pin::Pin<
                        Box<
                            dyn std::future::Future<
                                    Output = std::io::Result<crate::ipc::protocol::WorkerResponse>,
                                > + Send,
                        >,
                    >,
            },
        )
        .await;

        let failed = match probe {
            // A worker that ANSWERS with an error is not healthy. The old
            // catch-all `Ok(Ok(_))` matched `WorkerResponse::Err` too, so a
            // worker whose every inference failed (arena OOM, unloaded model,
            // corrupted session) replied promptly with an error, was counted
            // as a good beat, and was never killed — while
            // `embed_worker_heartbeat_total{result="error"}` sat at exactly 0
            // (verified on pillow: 0 across 51,447 beats). The wedge detector
            // was blind to the failure mode it exists to catch. `/ready`
            // already branched on this variant correctly; the heartbeat did
            // not. This arm must stay ABOVE the success arm.
            Ok(Ok(crate::ipc::protocol::WorkerResponse::Err { message, .. })) => {
                *consecutive_fails += 1;
                tracing::warn!(
                    model = %self.spec.model,
                    consecutive_fails = *consecutive_fails,
                    max_fails = self.heartbeat_max_fails,
                    error = %message,
                    "heartbeat probe: worker returned an error"
                );
                self.record_heartbeat("error");
                true
            }
            Ok(Ok(_)) => {
                if *consecutive_fails > 0 {
                    tracing::info!(
                        model = %self.spec.model,
                        consecutive_fails = *consecutive_fails,
                        "heartbeat recovered after failures"
                    );
                }
                *consecutive_fails = 0;
                self.record_heartbeat("ok");
                false
            }
            Ok(Err(e)) => {
                *consecutive_fails += 1;
                tracing::warn!(
                    model = %self.spec.model,
                    consecutive_fails = *consecutive_fails,
                    max_fails = self.heartbeat_max_fails,
                    error = ?e,
                    "heartbeat probe: dispatch failed"
                );
                self.record_heartbeat("error");
                true
            }
            Err(_) => {
                *consecutive_fails += 1;
                tracing::warn!(
                    model = %self.spec.model,
                    consecutive_fails = *consecutive_fails,
                    max_fails = self.heartbeat_max_fails,
                    timeout_ms = self.heartbeat_probe_timeout.as_millis(),
                    "heartbeat probe timed out (worker may be wedged)"
                );
                self.record_heartbeat("timeout");
                true
            }
        };

        if !failed {
            return HeartbeatOutcome::Ok;
        }
        if *consecutive_fails < self.heartbeat_max_fails {
            return HeartbeatOutcome::Failed;
        }

        let pid = self.current_pid();
        if pid == 0 {
            // Worker already exited / between respawns — watchdog is
            // handling it. Reset and wait for the next cycle.
            *consecutive_fails = 0;
            return HeartbeatOutcome::Skipped;
        }
        tracing::error!(
            model = %self.spec.model,
            pid,
            consecutive_fails = *consecutive_fails,
            max_fails = self.heartbeat_max_fails,
            "heartbeat: killing wedged worker (SIGKILL) — watchdog will respawn"
        );
        self.record_heartbeat("kill");
        // SIGKILL the wedged worker. watchdog_loop's child.wait()
        // will return with the killed status → clear client slot →
        // respawn. We use libc::kill (not child.kill()) because the
        // Child is owned by watchdog_loop, not by us.
        #[cfg(unix)]
        unsafe {
            let _ = libc::kill(pid as i32, libc::SIGKILL);
        }
        // Reset the counter — the next cycle will see client_slot =
        // None (watchdog clearing it) and skip, then the respawned
        // worker gets a fresh start.
        *consecutive_fails = 0;
        HeartbeatOutcome::Killed
    }

    /// Record a heartbeat result labelled with THIS worker's model.
    ///
    /// `embed_worker_restart_total` has carried a `{model}` label since it was
    /// introduced; `embed_worker_heartbeat_total` did not, so a rising
    /// `result="kill"` told the operator that some worker was being killed but
    /// not which one — on a 4-worker deployment that is the first question
    /// asked, and answering it meant reading container logs.
    fn record_heartbeat(&self, result: &str) {
        crate::metrics::record_worker_heartbeat(&self.spec.model, result);
    }

    /// Number of successful respawns since launch. Zero until first crash.
    #[allow(dead_code)] // TODO(Phase 3 metrics): expose via /health or /metrics endpoint
    pub fn restart_count(&self) -> u64 {
        self.restart_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// PID of the currently live worker process, or 0 if between respawns.
    ///
    /// Reads the atomic with Relaxed ordering — the RSS poll loop only uses
    /// this for a best-effort `/proc/<pid>/status` read; a briefly stale PID
    /// at most causes one skipped or misattributed sample, not correctness loss.
    pub fn current_pid(&self) -> u32 {
        self.current_pid.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Model name this supervisor is responsible for.
    pub fn model(&self) -> &str {
        &self.spec.model
    }

    /// Test-only: create a supervisor with a pre-connected client and no
    /// child process. The watchdog loop is not spawned — the supervisor
    /// is a static fixture for testing dispatch paths (e.g. /ready probe
    /// timeout against a hung worker socket).
    #[cfg(test)]
    #[allow(dead_code)] // used by api_health tests in the binary crate
    pub(crate) fn for_test(spec: SpawnSpec, client: Arc<WorkerClient>) -> Arc<Self> {
        Arc::new(Self {
            spec,
            client_slot: Arc::new(RwLock::new(Some(client))),
            restart_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            socket_wait: Duration::from_secs(60),
            current_pid: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            heartbeat_interval: Duration::ZERO,
            heartbeat_max_fails: 3,
            heartbeat_probe_timeout: Duration::from_millis(100),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::util::resolve_duration_secs_env;
    use serial_test::serial;
    use std::time::Duration;

    fn resolve_socket_wait() -> Duration {
        resolve_duration_secs_env(
            "EMBED_WORKER_SOCKET_WAIT_SECS",
            Duration::from_secs(SOCKET_WAIT_SECS),
            &format!("SOCKET_WAIT_SECS ({SOCKET_WAIT_SECS})"),
        )
    }

    #[test]
    #[serial]
    fn socket_wait_default() {
        let prev = std::env::var("EMBED_WORKER_SOCKET_WAIT_SECS").ok();
        unsafe { std::env::remove_var("EMBED_WORKER_SOCKET_WAIT_SECS") };
        let d = resolve_socket_wait();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_WORKER_SOCKET_WAIT_SECS") },
        }
        assert_eq!(d.as_secs(), SOCKET_WAIT_SECS);
    }

    #[test]
    #[serial]
    fn socket_wait_env_override() {
        let prev = std::env::var("EMBED_WORKER_SOCKET_WAIT_SECS").ok();
        unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", "120") };
        let d = resolve_socket_wait();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_WORKER_SOCKET_WAIT_SECS") },
        }
        assert_eq!(d.as_secs(), 120);
    }

    #[test]
    #[serial]
    fn socket_wait_zero_falls_back() {
        let prev = std::env::var("EMBED_WORKER_SOCKET_WAIT_SECS").ok();
        unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", "0") };
        let d = resolve_socket_wait();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_WORKER_SOCKET_WAIT_SECS") },
        }
        assert_eq!(d.as_secs(), SOCKET_WAIT_SECS, "zero falls back to default");
    }

    #[test]
    #[serial]
    fn socket_wait_invalid_falls_back() {
        let prev = std::env::var("EMBED_WORKER_SOCKET_WAIT_SECS").ok();
        unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", "bad") };
        let d = resolve_socket_wait();
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_WORKER_SOCKET_WAIT_SECS", p) },
            None => unsafe { std::env::remove_var("EMBED_WORKER_SOCKET_WAIT_SECS") },
        }
        assert_eq!(
            d.as_secs(),
            SOCKET_WAIT_SECS,
            "invalid falls back to default"
        );
    }

    /// Verifies the epoch-exit ordering invariant: on worker exit,
    /// `current_pid` is cleared to 0 BEFORE `client_slot` becomes None.
    ///
    /// This prevents the RSS-poll loop from reading `/proc/<dead_pid>/status`
    /// (exit gap) and ensures a consistent state for concurrent observers.
    ///
    /// The test is deterministic: it holds a read lock on `client_slot` so
    /// the write in `clear_on_exit` blocks, then inspects `current_pid`.
    /// With the correct ordering (pid-first), pid is already 0 when the
    /// write blocks. With the buggy ordering (client-first), pid is still
    /// the old value because the write blocked before the pid store ran.
    #[tokio::test]
    async fn clear_on_exit_clears_pid_before_client_slot() {
        let supervisor = Arc::new(WorkerSupervisor {
            spec: SpawnSpec {
                model: "test-model".into(),
                kind: WorkerKind::Embed,
                worker_bin: PathBuf::from("/bin/true"),
                socket_dir: PathBuf::from("/tmp"),
                pool_size: 1,
                intra_threads: 1,
                env_extra: vec![],
            },
            client_slot: Arc::new(RwLock::new(None)),
            restart_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            socket_wait: Duration::from_secs(60),
            current_pid: Arc::new(std::sync::atomic::AtomicU32::new(42)),
            heartbeat_interval: Duration::ZERO,
            heartbeat_max_fails: 3,
            heartbeat_probe_timeout: Duration::from_millis(150),
        });

        // Hold a read lock on client_slot so the write in clear_on_exit
        // blocks at the RwLock acquisition point.
        let _read_guard = supervisor.client_slot.read().await;

        // Spawn clear_on_exit. With correct ordering (pid-first), it stores
        // pid=0 then blocks on the write lock. With buggy ordering
        // (client-first), it blocks on the write lock immediately, leaving
        // pid=42.
        let sup_clone = supervisor.clone();
        let clear_handle = tokio::spawn(async move {
            sup_clone.clear_on_exit().await;
        });

        // Let the clear task run up to the blocking write().await.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // With the fix: pid is already 0 (stored before the write attempt).
        // With the bug: pid is still 42 (write blocked before pid store).
        let pid = supervisor.current_pid();
        assert_eq!(
            pid, 0,
            "current_pid must be cleared to 0 before client_slot becomes None \
             (epoch-exit ordering gap)"
        );

        // Release the read lock so clear_on_exit can complete.
        drop(_read_guard);
        clear_handle.await.unwrap();

        // Final state: both cleared.
        assert_eq!(supervisor.current_pid(), 0);
        assert!(supervisor.client().await.is_none());
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let b1 = next_backoff(INITIAL_BACKOFF);
        assert_eq!(b1, Duration::from_secs(4));
        let b2 = next_backoff(b1);
        assert_eq!(b2, Duration::from_secs(8));

        // Boundary: 32 s × 2 = 64 s → clamped to MAX_BACKOFF (60 s).
        assert_eq!(
            next_backoff(Duration::from_secs(32)),
            Duration::from_secs(60),
            "32s doubles to 64s which must clamp to MAX_BACKOFF=60s"
        );

        // Ensure cap is respected for values already beyond MAX_BACKOFF.
        let huge = Duration::from_secs(1000);
        assert_eq!(next_backoff(huge), MAX_BACKOFF);
    }

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

    // ── heartbeat wedge detector (#90, #134, #135) ──────────────────────────
    //
    // These drive `heartbeat_tick` — the real decision code — rather than a
    // hand-inlined copy of it. The previous test in this slot replicated the
    // loop body inline and asserted that `tokio::time::timeout` fires against
    // a silent socket and that `libc::kill` kills `sleep 999`. Neither is a
    // property of this crate: deleting the whole `tokio::spawn(heartbeat_loop)`
    // call site left it green, and the per-kind probe dispatch (#135) had no
    // coverage at all because the test hard-coded `dispatch_embed`.

    /// A mock worker that speaks the real IPC framing, records the kind of
    /// request it received, and replies with whatever the caller chose.
    ///
    /// This is what lets a test assert on the probe DISPATCH (#135) instead of
    /// only on its timing.
    struct MockWorker {
        socket_path: std::path::PathBuf,
        received: Arc<std::sync::Mutex<Vec<String>>>,
        task: tokio::task::JoinHandle<()>,
    }

    /// How the mock replies to a probe.
    #[derive(Clone, Copy, PartialEq)]
    enum MockReply {
        /// Reply with the matching success variant.
        Ok,
        /// Reply with `WorkerResponse::Err` — a worker that ANSWERS but is
        /// broken.
        WorkerError,
        /// Accept the connection and never reply — a wedged worker.
        Hang,
    }

    impl MockWorker {
        async fn start(tag: &str, reply: MockReply) -> Self {
            use crate::ipc::frame::{read_frame, write_frame};
            use crate::ipc::protocol::{
                EmbedResponseOk, RerankResponseOk, SpladeResponseOk, WorkerRequest, WorkerResponse,
            };

            let socket_path = std::env::temp_dir().join(format!(
                "embed-hb-{}-{}-{:?}.sock",
                tag,
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_file(&socket_path);
            let listener = tokio::net::UnixListener::bind(&socket_path).expect("bind mock worker");

            let received = Arc::new(std::sync::Mutex::new(Vec::new()));
            let recv = received.clone();
            let task = tokio::spawn(async move {
                while let Ok((mut stream, _)) = listener.accept().await {
                    let recv = recv.clone();
                    tokio::spawn(async move {
                        if reply == MockReply::Hang {
                            // Hold the connection open forever, never reply.
                            std::future::pending::<()>().await;
                            return;
                        }
                        let req: WorkerRequest = match read_frame(&mut stream).await {
                            Ok(r) => r,
                            Err(_) => return,
                        };
                        recv.lock().unwrap().push(req.kind().to_string());
                        let id = req.request_id();
                        let resp = match (reply, &req) {
                            (MockReply::WorkerError, _) => WorkerResponse::Err {
                                request_id: id,
                                message: "arena allocation failed".to_string(),
                            },
                            (_, WorkerRequest::Embed(_)) => {
                                WorkerResponse::Embed(EmbedResponseOk {
                                    request_id: id,
                                    vectors: vec![vec![0.0; 4]],
                                    dims: 4,
                                })
                            }
                            (_, WorkerRequest::Rerank(_)) => {
                                WorkerResponse::Rerank(RerankResponseOk {
                                    request_id: id,
                                    scores: vec![0.5],
                                })
                            }
                            (_, WorkerRequest::Splade(_)) => {
                                WorkerResponse::Splade(SpladeResponseOk {
                                    request_id: id,
                                    sparse: vec![vec![(1u32, 0.5f32)]],
                                })
                            }
                        };
                        let _ = write_frame(&mut stream, &resp).await;
                    });
                }
            });

            Self {
                socket_path,
                received,
                task,
            }
        }

        fn kinds_received(&self) -> Vec<String> {
            self.received.lock().unwrap().clone()
        }
    }

    impl Drop for MockWorker {
        fn drop(&mut self) {
            self.task.abort();
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }

    async fn supervisor_for(
        mock: &MockWorker,
        kind: WorkerKind,
        pid: u32,
    ) -> Arc<WorkerSupervisor> {
        use crate::ipc::client::WorkerClient;
        let client = Arc::new(
            WorkerClient::connect(mock.socket_path.clone(), 1)
                .await
                .expect("connect to mock worker"),
        );
        Arc::new(WorkerSupervisor {
            spec: SpawnSpec {
                model: "test-heartbeat".to_string(),
                kind,
                worker_bin: std::path::PathBuf::from("/bin/true"),
                socket_dir: mock.socket_path.parent().unwrap().to_path_buf(),
                pool_size: 1,
                intra_threads: 1,
                env_extra: vec![],
            },
            client_slot: Arc::new(RwLock::new(Some(client))),
            restart_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            socket_wait: Duration::from_secs(60),
            current_pid: Arc::new(std::sync::atomic::AtomicU32::new(pid)),
            heartbeat_interval: Duration::from_millis(50),
            heartbeat_max_fails: 1,
            heartbeat_probe_timeout: Duration::from_millis(150),
        })
    }

    /// The #90 claim, driven through the real code path: a worker that accepts
    /// the connection but never answers must be SIGKILLed so the watchdog
    /// respawns it.
    ///
    /// RED-proven: deleting the `libc::kill` call, or the
    /// `*consecutive_fails >= self.heartbeat_max_fails` branch, fails this.
    #[tokio::test]
    async fn heartbeat_tick_kills_wedged_worker() {
        let mock = MockWorker::start("wedged", MockReply::Hang).await;
        let mut child = std::process::Command::new("sleep")
            .arg("999")
            .spawn()
            .expect("spawn sleep");
        let sup = supervisor_for(&mock, WorkerKind::Embed, child.id()).await;

        let mut fails = 0;
        let outcome = sup.heartbeat_tick(&mut fails).await;

        assert_eq!(
            outcome,
            HeartbeatOutcome::Killed,
            "a worker that never answers must be killed once the fail budget is spent"
        );
        let status = child.wait().expect("wait child");
        assert!(
            !status.success(),
            "the real child process must have been SIGKILLed, not left running"
        );
        assert_eq!(fails, 0, "the fail counter resets after a kill");
    }

    /// A worker that ANSWERS with an error is not healthy.
    ///
    /// This is the regression guard for the bug where the catch-all
    /// `Ok(Ok(_))` arm swallowed `WorkerResponse::Err`: a worker whose every
    /// inference failed replied promptly, was counted as a good beat, and was
    /// never killed. On pillow this showed as
    /// `embed_worker_heartbeat_total{result="error"}` sitting at exactly 0
    /// across 51,447 beats.
    ///
    /// RED-proven: removing the `WorkerResponse::Err` arm makes this return
    /// `Ok` and the assertion fails.
    #[tokio::test]
    async fn heartbeat_tick_counts_worker_error_as_failure() {
        let mock = MockWorker::start("errreply", MockReply::WorkerError).await;
        let mut child = std::process::Command::new("sleep")
            .arg("999")
            .spawn()
            .expect("spawn sleep");
        let sup = supervisor_for(&mock, WorkerKind::Embed, child.id()).await;

        let mut fails = 0;
        let outcome = sup.heartbeat_tick(&mut fails).await;

        assert_eq!(
            outcome,
            HeartbeatOutcome::Killed,
            "an error reply must count as a failed beat (max_fails=1 -> kill), not a healthy one"
        );
        let status = child.wait().expect("wait child");
        assert!(!status.success(), "the broken worker must be killed");
    }

    /// A healthy worker answers, and the tick reports Ok without killing
    /// anything — the counterweight to the two tests above, so "kill on
    /// everything" cannot pass them all.
    #[tokio::test]
    async fn heartbeat_tick_ok_on_healthy_worker() {
        let mock = MockWorker::start("healthy", MockReply::Ok).await;
        let mut child = std::process::Command::new("sleep")
            .arg("999")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();
        let sup = supervisor_for(&mock, WorkerKind::Embed, pid).await;

        let mut fails = 3;
        let outcome = sup.heartbeat_tick(&mut fails).await;

        assert_eq!(outcome, HeartbeatOutcome::Ok);
        assert_eq!(
            fails, 0,
            "a good beat must reset the consecutive-fail count"
        );

        // The child must STILL be running — nothing was killed.
        //
        // `try_wait`, not `libc::kill(pid, 0)`: a SIGKILLed-but-unreaped child
        // is a zombie, and `kill(pid, 0)` returns 0 for zombies, so that check
        // would pass even if the kill HAD fired. `try_wait` returning `None`
        // is the only thing that actually distinguishes "running" from
        // "killed a moment ago".
        assert!(
            child.try_wait().expect("try_wait").is_none(),
            "a healthy worker must not be signalled — the child has exited"
        );
        let _ = child.kill();
        let _ = child.wait();
        let _ = pid;
    }

    /// The heartbeat LOOP must actually probe — repeatedly.
    ///
    /// `heartbeat_tick` is well covered by the tests above, but a tick nobody
    /// calls is worth nothing: before this, `heartbeat_loop` had exactly one
    /// caller (production) and zero test callers, which is the same shape as
    /// the defect this PR fixes, one level up. This drives the real loop
    /// against a mock worker and asserts the counter moves.
    ///
    /// RED-proven: replacing the `self.heartbeat_tick(..).await` call in
    /// `heartbeat_loop` with a no-op leaves the counter at 0 and fails here.
    #[tokio::test]
    async fn heartbeat_loop_probes_repeatedly() {
        let mock = MockWorker::start("loop", MockReply::Ok).await;
        let sup = supervisor_for(&mock, WorkerKind::Embed, 0).await;

        let task = tokio::spawn(sup.clone().heartbeat_loop());
        // The supervisor is built with a 50ms interval; give it room for
        // several cycles without making the test slow.
        tokio::time::sleep(Duration::from_millis(400)).await;
        task.abort();

        let probes = mock.kinds_received().len();
        assert!(
            probes >= 2,
            "heartbeat_loop must keep probing on its interval; saw {probes} probe(s) in 400ms \
             at a 50ms interval"
        );
    }

    /// #135: each worker kind must receive ITS OWN probe variant. An embed
    /// probe sent to a rerank worker is a permanent error, so a mismatch would
    /// make the detector kill healthy workers forever.
    ///
    /// The previous test hard-coded `WorkerKind::Embed` and called
    /// `dispatch_embed` by hand, so the Rerank and Splade arms — the entire
    /// content of #135 — were never executed by any test.
    ///
    /// RED-proven: pointing any arm of the `match self.spec.kind` at the wrong
    /// dispatcher fails the corresponding case here.
    #[tokio::test]
    async fn heartbeat_tick_dispatches_probe_matching_worker_kind() {
        for (kind, expected) in [
            (WorkerKind::Embed, "embed"),
            (WorkerKind::Rerank, "rerank"),
            (WorkerKind::Splade, "splade"),
        ] {
            let mock = MockWorker::start(expected, MockReply::Ok).await;
            let sup = supervisor_for(&mock, kind, 0).await;

            let mut fails = 0;
            let outcome = sup.heartbeat_tick(&mut fails).await;

            assert_eq!(
                outcome,
                HeartbeatOutcome::Ok,
                "{expected} worker answered its own probe kind, so the beat is healthy"
            );
            assert_eq!(
                mock.kinds_received(),
                vec![expected.to_string()],
                "a {kind:?} worker must be probed with a {expected} request, not another kind"
            );
        }
    }

    /// A worker between respawns (`client_slot == None`) is the watchdog's
    /// business, not the heartbeat's: no probe, no failure, and the counter
    /// resets so the fresh worker starts clean.
    #[tokio::test]
    async fn heartbeat_tick_skips_while_client_slot_empty() {
        let sup = Arc::new(WorkerSupervisor {
            spec: SpawnSpec {
                model: "test-heartbeat".to_string(),
                kind: WorkerKind::Embed,
                worker_bin: std::path::PathBuf::from("/bin/true"),
                socket_dir: std::env::temp_dir(),
                pool_size: 1,
                intra_threads: 1,
                env_extra: vec![],
            },
            client_slot: Arc::new(RwLock::new(None)),
            restart_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            socket_wait: Duration::from_secs(60),
            current_pid: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            heartbeat_interval: Duration::from_millis(50),
            heartbeat_max_fails: 1,
            heartbeat_probe_timeout: Duration::from_millis(150),
        });

        let mut fails = 2;
        let outcome = sup.heartbeat_tick(&mut fails).await;

        assert_eq!(outcome, HeartbeatOutcome::Skipped);
        assert_eq!(
            fails, 0,
            "a respawn window must not count against the worker"
        );
    }
}
