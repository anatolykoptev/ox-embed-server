//! Worker process — one process per model. Loads one ONNX model, exposes
//! inference over UDS to the supervisor.
//!
//! Supported model kinds (set via `EMBED_WORKER_KIND` env, default "embed"):
//!   - "embed"  — dense embedding via StandaloneEmbedder
//!   - "rerank" — cross-encoder reranker via StandaloneReranker
//!   - "splade" — SPLADE sparse encoder via StandaloneSplade
//!
//! The worker loads the model once at startup, then loops accepting UDS
//! connections and serving WorkerRequest frames. Each request must match the
//! worker's loaded kind; a kind mismatch returns WorkerResponse::Err.

use axum::Router;
use embed_server::config::Config;
use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{
    EmbedResponseOk, RerankResponseOk, SpladeResponseOk, WorkerRequest, WorkerResponse,
};
use embed_server::model::{StandaloneEmbedder, StandaloneReranker, StandaloneSplade};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::UnixListener;
use tokio::sync::Semaphore;

// ── per-worker defaults ────────────────────────────────────────────────────────

/// Default number of ONNX intra-op threads per session.
///
/// 2 is the empirically-tuned value for ARM cores on the krolik host
/// (24 vCPU / 3 models × pool_size sessions). Increasing beyond 2 stalls
/// other sessions under concurrent inference. Overridable via
/// `EMBED_WORKER_INTRA_THREADS`.
const INTRA_THREADS: usize = 2;

/// Default ONNX session pool size per worker (i.e. max concurrent inferences).
///
/// 1 session is the safe baseline — each session holds ~650 MiB of model
/// weights in the shared BFCArena. Increase only when a model fits within
/// the arena budget with headroom. Overridable via `EMBED_WORKER_POOL_SIZE`.
const POOL_SIZE: usize = 1;

// Waiter-queue constants (WAITERS_POOL_MULTIPLIER, WAITERS_FLOOR) and the
// per-model resolver live in `embed_server::worker_waiters` so integration
// tests under `tests/` can reach them without spawning a real worker.

// Counter for tasks waiting to acquire the per-worker semaphore.
// When this exceeds max_waiters, new requests get an immediate
// "worker queue overflow" error instead of joining an unbounded
// wait list. The limit is configurable via EMBED_MAX_WAITERS env;
// the default formula is WAITERS_POOL_MULTIPLIER × pool_size
// (minimum WAITERS_FLOOR), which gives breathing room for short-burst
// convoys without runaway memory. tokio::sync::Semaphore has no built-in
// max-waiter cap, so acquire_owned().await would otherwise queue requests
// without bound under pathological bursts.
static WAITERS: AtomicUsize = AtomicUsize::new(0);

// resolve_max_waiters_for_model is re-exported from embed_server::worker_waiters.
// The worker binary calls it at startup (line ~245) passing pool_size + model_name.

/// Install a Prometheus metrics recorder in the worker process and, if
/// `EMBED_WORKER_METRICS_PORT` is set, spawn a lightweight HTTP server
/// that exposes `/metrics` for Prometheus scraping.
///
/// Port assignment: the supervisor sets `EMBED_WORKER_METRICS_PORT` to
/// `EMBED_WORKER_METRICS_PORT_BASE + <worker-index>` (default base 8200).
/// Workers are indexed in spawn order across embed + rerank + splade pools.
/// Example with two embed models: jina-code-v2 → 8200, e5-large → 8201.
///
/// If the env var is unset, the recorder is still installed (so arena /
/// batcher counters accumulate) but no HTTP port is opened — back-compat
/// for existing deploys.
fn install_worker_metrics(model_name: &str) -> PrometheusHandle {
    // Use the SHARED bucket config (single authority in `metrics.rs`) so
    // worker-side histograms — `embed_worker_inference_duration_seconds`,
    // `embed_worker_queue_wait_duration_seconds` — land in the same buckets as the
    // supervisor's. A bare `PrometheusBuilder::new()` here would silently fall
    // back to library-default buckets, making cross-process comparison wrong.
    let handle = embed_server::metrics::apply_histogram_buckets(PrometheusBuilder::new())
        .install_recorder()
        .expect("install Prometheus recorder in worker");

    let port_str = std::env::var("EMBED_WORKER_METRICS_PORT").ok();
    if let Some(raw) = port_str {
        match raw.trim().parse::<u16>() {
            Ok(port) => {
                let handle_clone = handle.clone();
                // Bind address — env-overridable, defaults to all interfaces
                // (0.0.0.0) so Prometheus from sibling Docker container can
                // scrape. Operator may set EMBED_WORKER_METRICS_BIND to
                // "127.0.0.1" for host-only / single-process dev where
                // exposure is undesired.
                let bind = std::env::var("EMBED_WORKER_METRICS_BIND")
                    .unwrap_or_else(|_| "0.0.0.0".to_string());
                let addr: std::net::SocketAddr = match format!("{}:{}", bind, port).parse() {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::error!(
                            model = %model_name, port, bind = %bind, error = %e,
                            "EMBED_WORKER_METRICS_BIND invalid; falling back to 0.0.0.0"
                        );
                        ([0u8, 0, 0, 0], port).into()
                    }
                };
                let model = model_name.to_string();
                tokio::spawn(async move {
                    let app = Router::new().route(
                        "/metrics",
                        axum::routing::get(move || {
                            let h = handle_clone.clone();
                            async move {
                                let body = h.render();
                                (
                                    [(
                                        axum::http::header::CONTENT_TYPE,
                                        "text/plain; version=0.0.4",
                                    )],
                                    body,
                                )
                            }
                        }),
                    );
                    let listener = match tokio::net::TcpListener::bind(addr).await {
                        Ok(l) => {
                            tracing::info!(
                                model = %model,
                                port,
                                "worker metrics HTTP server started"
                            );
                            l
                        }
                        Err(e) => {
                            tracing::warn!(
                                model = %model,
                                port,
                                error = %e,
                                "worker metrics HTTP server bind failed; metrics will not be scraped"
                            );
                            return;
                        }
                    };
                    if let Err(e) = axum::serve(listener, app).await {
                        tracing::warn!(
                            model = %model,
                            error = %e,
                            "worker metrics HTTP server exited"
                        );
                    }
                });
            }
            Err(_) => {
                tracing::warn!(
                    EMBED_WORKER_METRICS_PORT = %raw.trim(),
                    "EMBED_WORKER_METRICS_PORT is not a valid port; metrics HTTP server disabled"
                );
            }
        }
    }

    handle
}

/// Handle the result of shared CPU arena registration.
///
/// Returns `Err` on failure (aborting worker startup), matching the
/// reranker/splade pattern which panics via
/// `assert_arena_registered_before_session`. Previously the embed path only
/// warned and continued, silently falling back to per-session BFCArena with
/// unbounded memory growth.
///
/// Incrementing `embed_arena_registration_failed_total` before returning the
/// error makes the failure observable in `/metrics` even though the worker
/// exits immediately afterwards (the supervisor's `/metrics` endpoint
/// scrapes the worker's last render).
fn handle_arena_registration(result: Result<(), String>) -> anyhow::Result<()> {
    match result {
        Ok(()) => {
            tracing::info!("shared CPU arena registered (worker)");
            Ok(())
        }
        Err(e) => {
            embed_server::metrics::record_arena_registration_failed();
            Err(anyhow::anyhow!("shared arena registration failed: {e}"))
        }
    }
}

/// Register the shared CPU arena for this worker's model.
///
/// Pre-touches the `embed_arena_registration_failed_total` counter to 0 so
/// it is visible in `/metrics` from startup even when no failure occurs,
/// then delegates to [`handle_arena_registration`] for the fail/continue
/// decision.
fn register_arena_for_worker(model_name: &str) -> anyhow::Result<()> {
    embed_server::metrics::arena_registration_failed_touch();
    handle_arena_registration(embed_server::arena::register_shared_cpu_arena_for_model(
        model_name,
    ))
}

fn require_env(var: &str) -> anyhow::Result<String> {
    std::env::var(var).map_err(|_e| {
        tracing::error!(var, "required environment variable missing");
        anyhow::anyhow!("required env var missing: {var}")
    })
}

/// Enum wrapping the three model kinds the worker can load.
enum LoadedModel {
    Embed(StandaloneEmbedder),
    Rerank(StandaloneReranker),
    Splade(StandaloneSplade),
}

impl LoadedModel {
    fn kind(&self) -> &'static str {
        match self {
            Self::Embed(_) => "embed",
            Self::Rerank(_) => "rerank",
            Self::Splade(_) => "splade",
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let kind = std::env::var("EMBED_WORKER_KIND").unwrap_or_else(|_| "embed".into());
    let model_name = require_env("EMBED_WORKER_MODEL")?;
    let socket_path: PathBuf = require_env("EMBED_WORKER_SOCKET")?.into();
    let intra_threads: usize = std::env::var("EMBED_WORKER_INTRA_THREADS")
        .unwrap_or_else(|_| INTRA_THREADS.to_string())
        .parse()?;
    let pool_size: usize = std::env::var("EMBED_WORKER_POOL_SIZE")
        .unwrap_or_else(|_| POOL_SIZE.to_string())
        .parse()?;

    // Resolve max_waiters once at startup -- cheaper than re-reading env
    // on every request, and captured by copy into the spawned async tasks.
    let max_waiters =
        embed_server::worker_waiters::resolve_max_waiters_for_model(pool_size, &model_name);

    tracing::info!(
        kind = %kind,
        model = %model_name,
        ?socket_path,
        intra_threads,
        pool_size,
        max_waiters,
        "worker starting"
    );

    let cfg = Config::from_env().map_err(|e| {
        tracing::error!(error = %e, "config load failed");
        anyhow::anyhow!("config: {e}")
    })?;

    // Install Prometheus recorder. If EMBED_WORKER_METRICS_PORT is set, also
    // spawns a lightweight HTTP /metrics server for per-worker scraping.
    // Must happen before arena registration (which emits gauges).
    // Held for process lifetime: dropping this handle shuts down the metrics
    // HTTP server, making /metrics unreachable for the rest of the worker life.
    let _metrics_handle = install_worker_metrics(&model_name);

    // Register the shared CPU arena BEFORE any Session::builder() call.
    // Each worker is a fresh process — the parent supervisor's registration
    // doesn't carry over. RerankerModel::load and SpladeModel::load both
    // assert this via `assert_arena_registered_before_session` (which panics).
    // The embed path (StandaloneEmbedder::load) does NOT assert — without a
    // shared arena it silently falls back to per-session BFCArena, causing
    // unbounded memory growth. So registration failure MUST abort startup,
    // not warn-and-continue.
    // Uses per-model EMBED_ARENA_MAX_MEM_BYTES_<KEY> override if set.
    register_arena_for_worker(&model_name)?;

    let loaded = match kind.as_str() {
        "embed" => LoadedModel::Embed(
            StandaloneEmbedder::load(&model_name, &cfg, intra_threads, pool_size).map_err(|e| {
                tracing::error!(error = %e, model = %model_name, "embed model load failed");
                anyhow::anyhow!("load failed: {e}")
            })?,
        ),
        "rerank" => LoadedModel::Rerank(
            StandaloneReranker::load(&model_name, &cfg, intra_threads, pool_size).map_err(|e| {
                tracing::error!(error = %e, model = %model_name, "reranker load failed");
                anyhow::anyhow!("load failed: {e}")
            })?,
        ),
        "splade" => LoadedModel::Splade(
            StandaloneSplade::load(&model_name, &cfg, intra_threads, pool_size).map_err(|e| {
                tracing::error!(error = %e, model = %model_name, "splade load failed");
                anyhow::anyhow!("load failed: {e}")
            })?,
        ),
        other => {
            tracing::error!(kind = %other, "unknown EMBED_WORKER_KIND");
            anyhow::bail!("unknown EMBED_WORKER_KIND: {other}");
        }
    };

    let loaded = Arc::new(loaded);
    let semaphore = Arc::new(Semaphore::new(pool_size));

    if socket_path.exists()
        && let Err(e) = std::fs::remove_file(&socket_path)
    {
        tracing::error!(path = ?socket_path, error = %e, "stale socket cleanup failed");
        return Err(e.into());
    }
    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!(kind = %loaded.kind(), "worker ready");

    loop {
        let (mut stream, _) = listener.accept().await.map_err(|e| {
            tracing::error!(error = %e, "accept failed");
            e
        })?;
        let loaded = loaded.clone();
        let semaphore = semaphore.clone();
        let model_name = model_name.clone();
        tokio::spawn(async move {
            loop {
                let req: WorkerRequest = match read_frame(&mut stream).await {
                    Ok(r) => r,
                    Err(_) => break,
                };

                // Wall-clock start of the queue-wait window: from a fully-read
                // request frame until the inference permit is acquired. Recorded
                // as `embed_worker_queue_wait_duration_seconds` so operators can separate
                // "queue is deep" (rising wait) from "model is slow" (rising
                // `embed_worker_inference_duration_seconds`). See the jina-code-v2
                // backpressure incident dossier.
                let queue_wait_start = std::time::Instant::now();

                // Bounded-queue admission: await permit instead of instant
                // reject. With per-request UDS conn (PR #62), burst load
                // would mass-trigger try_acquire failures + 500-spam to
                // memdb-go, which retries → amplification cascade. acquire()
                // queues requests at the worker; supervisor-side
                // WorkerPool::dispatch_timeout (EMBED_DISPATCH_TIMEOUT_SECS)
                // caps the upper bound, so a stuck worker still surfaces as
                // a real error after the configured timeout.
                //
                // Waiter cap: tokio::Semaphore has no built-in max-waiter
                // limit. Under pathological bursts, unbounded acquire_owned
                // futures queue memory without bound. WAITERS tracks in-flight
                // waiters; once > max_waiters (resolved at startup from
                // EMBED_MAX_WAITERS env, default WAITERS_POOL_MULTIPLIER×pool_size
                // floor WAITERS_FLOOR), new requests fail fast with "worker
                // queue overflow" so callers get an immediate error instead
                // of accumulating silently.
                let waiters_now = WAITERS.fetch_add(1, Ordering::Relaxed);
                embed_server::metrics::set_worker_queue_depth(&model_name, waiters_now + 1);
                if waiters_now >= max_waiters {
                    WAITERS.fetch_sub(1, Ordering::Relaxed);
                    embed_server::metrics::set_worker_queue_depth(&model_name, waiters_now);
                    tracing::warn!(
                        waiters = waiters_now,
                        max_waiters,
                        "worker queue overflow — rejecting request"
                    );
                    let resp = WorkerResponse::Err {
                        request_id: req.request_id(),
                        message: "worker queue overflow".into(),
                    };
                    let _ = write_frame(&mut stream, &resp).await;
                    continue;
                }

                // Acquired permit drops at end of iteration — automatic release.
                let _permit = match semaphore.clone().acquire_owned().await {
                    Ok(p) => {
                        let depth = WAITERS.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                        embed_server::metrics::set_worker_queue_depth(&model_name, depth);
                        // Permit acquired — close the queue-wait window. Under
                        // pool_size=1 this equals the time the previous
                        // inference held the single slot (head-of-line wait).
                        embed_server::metrics::record_worker_queue_wait(
                            &model_name,
                            queue_wait_start.elapsed(),
                        );
                        p
                    }
                    Err(_) => {
                        // Only fires if the semaphore was explicitly closed —
                        // we never call close(), so this is unreachable in
                        // current code. Future-proof: surface as a clear error.
                        let depth = WAITERS.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                        embed_server::metrics::set_worker_queue_depth(&model_name, depth);
                        let resp = WorkerResponse::Err {
                            request_id: req.request_id(),
                            message: "worker semaphore closed".into(),
                        };
                        let _ = write_frame(&mut stream, &resp).await;
                        continue;
                    }
                };

                // All model inference is sync/CPU-bound (ONNX). Run on the
                // blocking pool so the async runtime thread is not stalled.
                let req_id = req.request_id();
                let loaded_ref = loaded.clone();
                // Embed-path batch size for `embed_worker_batch_size` — captured
                // before `req` is moved into the closure. Only the Embed arm
                // records worker-side inference timing (see below), so only its
                // batch size is needed.
                let embed_batch_size = match &req {
                    WorkerRequest::Embed(r) => Some(r.texts.len()),
                    WorkerRequest::Rerank(_) | WorkerRequest::Splade(_) => None,
                };
                // Pure-inference window: just the spawn_blocking ONNX forward
                // pass, EXCLUDING the queue wait recorded above. This is the
                // worker-side `embed_worker_inference_duration_seconds` that the
                // supervisor's conflated round-trip metric could not isolate.
                let infer_start = std::time::Instant::now();
                let resp = tokio::task::spawn_blocking(move || match (&*loaded_ref, req) {
                    (LoadedModel::Embed(m), WorkerRequest::Embed(r)) => {
                        match m.infer(r.texts, r.max_seq_len) {
                            Ok((vectors, dims)) => WorkerResponse::Embed(EmbedResponseOk {
                                request_id: r.request_id,
                                vectors,
                                dims,
                            }),
                            Err(e) => WorkerResponse::Err {
                                request_id: r.request_id,
                                message: e,
                            },
                        }
                    }
                    (LoadedModel::Rerank(m), WorkerRequest::Rerank(r)) => {
                        match m.score(r.query, r.documents, r.max_seq_len) {
                            Ok(scores) => WorkerResponse::Rerank(RerankResponseOk {
                                request_id: r.request_id,
                                scores,
                            }),
                            Err(e) => WorkerResponse::Err {
                                request_id: r.request_id,
                                message: e,
                            },
                        }
                    }
                    (LoadedModel::Splade(m), WorkerRequest::Splade(r)) => {
                        match m.encode(r.texts, r.max_seq_len, r.top_k, r.min_weight) {
                            Ok(sparse) => WorkerResponse::Splade(SpladeResponseOk {
                                request_id: r.request_id,
                                sparse,
                            }),
                            Err(e) => WorkerResponse::Err {
                                request_id: r.request_id,
                                message: e,
                            },
                        }
                    }
                    (loaded_model, req) => WorkerResponse::Err {
                        request_id: req.request_id(),
                        message: format!(
                            "kind mismatch: model is {}, request is {}",
                            loaded_model.kind(),
                            req.kind()
                        ),
                    },
                })
                .await
                .unwrap_or_else(|e| WorkerResponse::Err {
                    request_id: req_id,
                    message: format!("spawn_blocking join error: {e}"),
                });

                // Record the pure ONNX forward-pass time for completed EMBED
                // inferences only. Scoped to the Embed arm on purpose: the
                // supervisor only emits `embed_inference_duration_seconds`
                // (the round-trip to subtract against) on the embed path, so a
                // rerank/splade worker series would have no counterpart and
                // would also collide with the existing `embed_rerank_*`
                // namespace. An `Err` variant (worker queue overflow / model
                // error / join failure) is tracked by the supervisor's
                // `embed_inference_failure` counter and would pollute the
                // latency histogram with near-zero / partial durations, so it
                // is excluded here.
                if let (Some(batch_size), false) =
                    (embed_batch_size, matches!(resp, WorkerResponse::Err { .. }))
                {
                    embed_server::metrics::record_worker_inference(
                        &model_name,
                        infer_start.elapsed(),
                        batch_size,
                    );
                }

                if write_frame(&mut stream, &resp).await.is_err() {
                    break;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{INTRA_THREADS, POOL_SIZE, handle_arena_registration};
    use embed_server::worker_waiters::{
        WAITERS_FLOOR, WAITERS_POOL_MULTIPLIER, resolve_max_waiters_for_model,
    };
    use serial_test::serial;

    // All tests in this module mutate EMBED_MAX_WAITERS via
    // std::env::set_var / remove_var, which is process-global state.
    // std::env::set_var is non-reentrant (glibc setenv) and officially
    // unsafe since Rust 1.82. The #[serial] attribute (serial_test crate)
    // ensures these tests never run concurrently, making the mutations safe.

    // ── arena registration error propagation ──────────────────────────────────

    #[test]
    fn arena_registration_failure_propagates_as_error() {
        // When register_shared_cpu_arena_for_model returns Err, the worker
        // MUST propagate it as an error (aborting startup) — NOT warn and
        // continue. Warn-and-continue caused the embed path to silently fall
        // back to per-session BFCArena with unbounded memory growth (issue #92).
        let result = handle_arena_registration(Err("simulated ORT failure".into()));
        assert!(
            result.is_err(),
            "arena registration failure must abort startup, not warn-and-continue"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("shared arena registration failed"),
            "error message must mention arena registration failure, got: {msg}"
        );
    }

    #[test]
    fn arena_registration_success_returns_ok() {
        // Happy path: registration succeeds → Ok(()), worker continues.
        let result = handle_arena_registration(Ok(()));
        assert!(
            result.is_ok(),
            "successful arena registration must not abort startup"
        );
    }

    // ── existing tests ─────────────────────────────────────────────────────────

    #[test]
    // These are compile-time `const` values, so clippy's
    // `assertions_on_constants` lint fires; the assertions are intentional —
    // they guard against a future edit setting a constant to a nonsense value
    // and are documented as such. Keep them as runtime asserts (the message
    // text is the point) rather than rewriting as `const { }` blocks.
    #[allow(clippy::assertions_on_constants)]
    fn constants_have_sane_values() {
        // Guard against accidentally editing constants to nonsense values.
        assert!(INTRA_THREADS >= 1, "intra_threads must be ≥1");
        assert!(POOL_SIZE >= 1, "pool_size must be ≥1");
        assert!(
            WAITERS_POOL_MULTIPLIER >= 1,
            "waiters multiplier must be ≥1"
        );
        assert!(
            WAITERS_FLOOR >= WAITERS_POOL_MULTIPLIER,
            "floor should be at least as large as one pool's worth of waiters"
        );
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_default_formula() {
        // Without EMBED_MAX_WAITERS or per-model env set, formula applies.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        unsafe {
            std::env::remove_var("EMBED_MAX_WAITERS");
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        }
        let val = resolve_max_waiters_for_model(4, "test-model");
        unsafe {
            match prev {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
            match prev_per {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
        }
        assert_eq!(val, 4 * WAITERS_POOL_MULTIPLIER, "4 × multiplier");
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_floor() {
        // pool_size=1 → 1×8=8 < 16 → floor kicks in. Per-model unset.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        unsafe {
            std::env::remove_var("EMBED_MAX_WAITERS");
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        }
        let val = resolve_max_waiters_for_model(1, "test-model");
        unsafe {
            match prev {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
            match prev_per {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
        }
        assert_eq!(val, WAITERS_FLOOR, "floor = WAITERS_FLOOR");
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_env_override() {
        // EMBED_MAX_WAITERS=64 overrides formula when per-model unset.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        unsafe {
            std::env::set_var("EMBED_MAX_WAITERS", "64");
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        }
        let val = resolve_max_waiters_for_model(2, "test-model");
        unsafe {
            match prev {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
            match prev_per {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
        }
        assert_eq!(val, 64);
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_env_invalid_falls_back() {
        // Non-numeric EMBED_MAX_WAITERS warns and falls back to formula (per-model unset).
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        unsafe {
            std::env::set_var("EMBED_MAX_WAITERS", "not-a-number");
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        }
        let val = resolve_max_waiters_for_model(3, "test-model");
        unsafe {
            match prev {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
            match prev_per {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
        }
        assert_eq!(val, 3 * WAITERS_POOL_MULTIPLIER, "3 × multiplier");
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_zero_falls_back() {
        // EMBED_MAX_WAITERS=0 → fall back to formula (per-model also unset).
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        unsafe {
            std::env::set_var("EMBED_MAX_WAITERS", "0");
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        }
        let val = resolve_max_waiters_for_model(2, "test-model");
        unsafe {
            match prev {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
            match prev_per {
                Some(p) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", p),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
        }
        // 2 × 8 = 16, which also hits the floor(16), so either way WAITERS_FLOOR.
        assert_eq!(
            val, WAITERS_FLOOR,
            "zero falls back to formula (2×mult=16, floor=16)"
        );
    }
}
