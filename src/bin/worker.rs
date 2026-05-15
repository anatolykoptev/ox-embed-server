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

use embed_server::config::Config;
use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{
    EmbedResponseOk, RerankResponseOk, SpladeResponseOk, WorkerRequest, WorkerResponse,
};
use embed_server::model::{StandaloneEmbedder, StandaloneReranker, StandaloneSplade};
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
const DEFAULT_INTRA_THREADS: usize = 2;

/// Default ONNX session pool size per worker (i.e. max concurrent inferences).
///
/// 1 session is the safe baseline — each session holds ~650 MiB of model
/// weights in the shared BFCArena. Increase only when a model fits within
/// the arena budget with headroom. Overridable via `EMBED_WORKER_POOL_SIZE`.
const DEFAULT_POOL_SIZE: usize = 1;

/// Multiplier applied to pool_size when computing the default max-waiters cap.
///
/// 8× gives ample burst headroom while keeping the waiter queue bounded.
/// The formula is `pool_size × WAITERS_POOL_MULTIPLIER`, floored at
/// `WAITERS_FLOOR`.
const WAITERS_POOL_MULTIPLIER: usize = 8;

/// Minimum max-waiters cap regardless of pool_size.
///
/// Prevents `pool_size=1` (or very small) from producing a cap so low (8)
/// that brief single-connection bursts trigger "worker queue overflow".
/// 16 ensures at least one full round of requests per ONNX-inference batch
/// can queue without rejection.
const WAITERS_FLOOR: usize = 16;

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

/// Resolve the max-waiters limit once at startup.
///
/// Reads `EMBED_MAX_WAITERS` env:
/// - If unset: falls back to `pool_size × WAITERS_POOL_MULTIPLIER` (floor WAITERS_FLOOR).
/// - If set but unparseable or zero: warns and falls back to the formula.
///   A zero value would cause 100% rejection of all incoming requests, so it
///   is treated as a misconfiguration rather than an intentional setting.
fn resolve_max_waiters(pool_size: usize) -> usize {
    let default = || {
        pool_size
            .saturating_mul(WAITERS_POOL_MULTIPLIER)
            .max(WAITERS_FLOOR)
    };
    match std::env::var("EMBED_MAX_WAITERS") {
        Err(_) => default(),
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) if n > 0 => n,
            Ok(_) => {
                // Parsed as zero — would silently reject every request.
                tracing::warn!(
                    EMBED_MAX_WAITERS = %raw,
                    fallback = default(),
                    "EMBED_MAX_WAITERS=0 would reject all requests; using formula fallback"
                );
                default()
            }
            Err(_) => {
                tracing::warn!(
                    EMBED_MAX_WAITERS = %raw,
                    fallback = default(),
                    "EMBED_MAX_WAITERS is not a valid usize; using formula fallback"
                );
                default()
            }
        },
    }
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
        .unwrap_or_else(|_| DEFAULT_INTRA_THREADS.to_string())
        .parse()?;
    let pool_size: usize = std::env::var("EMBED_WORKER_POOL_SIZE")
        .unwrap_or_else(|_| DEFAULT_POOL_SIZE.to_string())
        .parse()?;

    // Resolve max_waiters once at startup -- cheaper than re-reading env
    // on every request, and captured by copy into the spawned async tasks.
    let max_waiters = resolve_max_waiters(pool_size);

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

    // Register the shared CPU arena BEFORE any Session::builder() call.
    // Each worker is a fresh process — the parent supervisor's registration
    // doesn't carry over. RerankerModel::load and SpladeModel::load both
    // assert this; EmbedModel::load silently falls back to per-session
    // BFCArena without it (less efficient).
    if let Err(e) = embed_server::arena::register_shared_cpu_arena() {
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
                if waiters_now >= max_waiters {
                    WAITERS.fetch_sub(1, Ordering::Relaxed);
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
                        WAITERS.fetch_sub(1, Ordering::Relaxed);
                        p
                    }
                    Err(_) => {
                        // Only fires if the semaphore was explicitly closed —
                        // we never call close(), so this is unreachable in
                        // current code. Future-proof: surface as a clear error.
                        WAITERS.fetch_sub(1, Ordering::Relaxed);
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
    use super::{
        resolve_max_waiters, DEFAULT_INTRA_THREADS, DEFAULT_POOL_SIZE, WAITERS_FLOOR,
        WAITERS_POOL_MULTIPLIER,
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
        assert!(DEFAULT_INTRA_THREADS >= 1, "intra_threads must be ≥1");
        assert!(DEFAULT_POOL_SIZE >= 1, "pool_size must be ≥1");
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
        // Without EMBED_MAX_WAITERS set, formula = pool_size × WAITERS_POOL_MULTIPLIER, floor WAITERS_FLOOR.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        unsafe { std::env::remove_var("EMBED_MAX_WAITERS") };
        let val = resolve_max_waiters(4);
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_MAX_WAITERS", p) },
            None => unsafe { std::env::remove_var("EMBED_MAX_WAITERS") },
        }
        assert_eq!(val, 4 * WAITERS_POOL_MULTIPLIER, "4 × multiplier");
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_floor() {
        // pool_size=1 -> 1×8=8 < 16 -> floor kicks in.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        unsafe { std::env::remove_var("EMBED_MAX_WAITERS") };
        let val = resolve_max_waiters(1);
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_MAX_WAITERS", p) },
            None => unsafe { std::env::remove_var("EMBED_MAX_WAITERS") },
        }
        assert_eq!(val, WAITERS_FLOOR, "floor = WAITERS_FLOOR");
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_env_override() {
        // EMBED_MAX_WAITERS=64 overrides formula regardless of pool_size.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        unsafe { std::env::set_var("EMBED_MAX_WAITERS", "64") };
        let val = resolve_max_waiters(2);
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_MAX_WAITERS", p) },
            None => unsafe { std::env::remove_var("EMBED_MAX_WAITERS") },
        }
        assert_eq!(val, 64);
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_env_invalid_falls_back() {
        // Non-numeric EMBED_MAX_WAITERS warns and falls back to formula.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        unsafe { std::env::set_var("EMBED_MAX_WAITERS", "not-a-number") };
        let val = resolve_max_waiters(3);
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_MAX_WAITERS", p) },
            None => unsafe { std::env::remove_var("EMBED_MAX_WAITERS") },
        }
        assert_eq!(val, 3 * WAITERS_POOL_MULTIPLIER, "3 × multiplier");
    }

    #[test]
    #[serial]
    fn resolve_max_waiters_zero_falls_back() {
        // EMBED_MAX_WAITERS=0 would reject every request; must fall back to formula.
        let prev = std::env::var("EMBED_MAX_WAITERS").ok();
        unsafe { std::env::set_var("EMBED_MAX_WAITERS", "0") };
        let val = resolve_max_waiters(2);
        match prev {
            Some(p) => unsafe { std::env::set_var("EMBED_MAX_WAITERS", p) },
            None => unsafe { std::env::remove_var("EMBED_MAX_WAITERS") },
        }
        // 2 × 8 = 16, which also hits the floor(16), so either way WAITERS_FLOOR.
        assert_eq!(
            val, WAITERS_FLOOR,
            "zero falls back to formula (2×mult=16, floor=16)"
        );
    }
}
