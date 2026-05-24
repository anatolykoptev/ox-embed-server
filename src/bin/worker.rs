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
    let handle = PrometheusBuilder::new()
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
    let max_waiters = embed_server::worker_waiters::resolve_max_waiters_for_model(pool_size, &model_name);

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
    // assert this; EmbedModel::load silently falls back to per-session
    // BFCArena without it (less efficient).
    // Uses per-model EMBED_ARENA_MAX_MEM_BYTES_<KEY> override if set.
    if let Err(e) = embed_server::arena::register_shared_cpu_arena_for_model(&model_name) {
        tracing::warn!(error = %e, "shared arena registration failed; sessions will use per-session BFCArena");
    } else {
        tracing::info!("shared CPU arena registered (worker)");
    }

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

                if write_frame(&mut stream, &resp).await.is_err() {
                    break;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{INTRA_THREADS, POOL_SIZE};
    use embed_server::worker_waiters::{
        resolve_max_waiters_for_model, WAITERS_FLOOR, WAITERS_POOL_MULTIPLIER,
    };
    use serial_test::serial;

    // All tests in this module mutate EMBED_MAX_WAITERS via
    // std::env::set_var / remove_var, which is process-global state.
    // std::env::set_var is non-reentrant (glibc setenv) and officially
    // unsafe since Rust 1.82. The #[serial] attribute (serial_test crate)
    // ensures these tests never run concurrently, making the mutations safe.

    #[test]
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
