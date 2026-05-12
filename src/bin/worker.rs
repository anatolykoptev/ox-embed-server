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
use tokio::net::UnixListener;
use tokio::sync::Semaphore;

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
        .unwrap_or_else(|_| "2".into())
        .parse()?;
    let pool_size: usize = std::env::var("EMBED_WORKER_POOL_SIZE")
        .unwrap_or_else(|_| "1".into())
        .parse()?;

    tracing::info!(
        kind = %kind,
        model = %model_name,
        ?socket_path,
        intra_threads,
        pool_size,
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

                let _permit = match semaphore.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        let resp = WorkerResponse::Err {
                            request_id: req.request_id(),
                            message: "worker saturated".into(),
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
