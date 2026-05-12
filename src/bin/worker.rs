//! Worker process — one process per model. Loads one ONNX model, exposes
//! inference over UDS to the supervisor. Phase 1 of multi-process refactor.
//!
//! Control messages (Ping/Pong/Shutdown) are handled in Wave 2 via a separate
//! channel. This binary handles InferRequest only.

use embed_server::config::Config;
use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{InferRequest, InferResponse};
use embed_server::model::StandaloneEmbedder;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::sync::Semaphore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_name = std::env::var("EMBED_WORKER_MODEL")?;
    let socket_path: PathBuf = std::env::var("EMBED_WORKER_SOCKET")?.into();
    let intra_threads: usize = std::env::var("EMBED_WORKER_INTRA_THREADS")
        .unwrap_or_else(|_| "2".into())
        .parse()?;
    let pool_size: usize = std::env::var("EMBED_WORKER_POOL_SIZE")
        .unwrap_or_else(|_| "1".into())
        .parse()?;

    tracing::info!(model = %model_name, ?socket_path, intra_threads, pool_size, "worker starting");

    let cfg = Config::from_env().map_err(|e| anyhow::anyhow!("config error: {e}"))?;
    let embedder = Arc::new(
        StandaloneEmbedder::load(&model_name, &cfg, intra_threads, pool_size)
            .map_err(|e| anyhow::anyhow!("model load failed: {e}"))?,
    );
    let semaphore = Arc::new(Semaphore::new(pool_size));

    if socket_path.exists() {
        if let Err(e) = std::fs::remove_file(&socket_path) {
            tracing::error!(path = ?socket_path, error = %e, "stale socket cleanup failed");
            return Err(e.into());
        }
    }
    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("worker ready");

    loop {
        let (mut stream, _) = listener.accept().await.map_err(|e| {
            tracing::error!(error = %e, "accept failed");
            e
        })?;
        let embedder = embedder.clone();
        let semaphore = semaphore.clone();
        tokio::spawn(async move {
            loop {
                let req: InferRequest = match read_frame(&mut stream).await {
                    Ok(r) => r,
                    Err(_) => break,
                };

                let _permit = match semaphore.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        let resp = InferResponse::Err {
                            request_id: req.request_id,
                            message: "worker saturated".into(),
                        };
                        let _ = write_frame(&mut stream, &resp).await;
                        continue;
                    }
                };

                // tokenize + embed_tokens are sync/CPU-bound (ONNX inference).
                // Run on the blocking pool so the async runtime thread is not stalled.
                let req_id = req.request_id;
                let texts = req.texts;
                let max_seq_len = req.max_seq_len;
                let emb = embedder.clone();
                let resp = match tokio::task::spawn_blocking(move || {
                    emb.infer(texts, max_seq_len)
                })
                .await
                {
                    Ok(Ok((vectors, dims))) => InferResponse::Ok {
                        request_id: req_id,
                        vectors,
                        dims,
                    },
                    Ok(Err(e)) => InferResponse::Err {
                        request_id: req_id,
                        message: e,
                    },
                    Err(e) => InferResponse::Err {
                        request_id: req_id,
                        message: format!("spawn_blocking join error: {e}"),
                    },
                };
                if write_frame(&mut stream, &resp).await.is_err() {
                    break;
                }
            }
        });
    }
}
