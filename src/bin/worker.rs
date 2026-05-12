//! Worker process binary — one process per model.
//!
//! Loads a single ONNX model, owns its own OrtEnv + BFCArena, exposes
//! inference over Unix Domain Socket to the supervisor.
//!
//! Phase 1: scaffold only. Connects to UDS, echoes ControlMessage::Ping → Pong.
//! Full inference handlers land in Wave 1.3.

use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::ControlMessage;
use std::path::PathBuf;
use tokio::net::UnixListener;

#[derive(Debug)]
struct WorkerConfig {
    model: String,
    socket_path: PathBuf,
}

fn parse_args() -> WorkerConfig {
    let model = std::env::var("EMBED_WORKER_MODEL").expect("EMBED_WORKER_MODEL env required");
    let socket_path: PathBuf = std::env::var("EMBED_WORKER_SOCKET")
        .expect("EMBED_WORKER_SOCKET env required")
        .into();
    WorkerConfig { model, socket_path }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = parse_args();
    tracing::info!(model = %cfg.model, socket = ?cfg.socket_path, "worker starting");

    if cfg.socket_path.exists() {
        std::fs::remove_file(&cfg.socket_path).map_err(|e| {
            tracing::error!(path = ?cfg.socket_path, error = %e, "failed to remove stale socket");
            e
        })?;
    }
    let listener = UnixListener::bind(&cfg.socket_path)?;
    tracing::info!("worker listening on UDS");

    loop {
        let (mut stream, _) = listener.accept().await.map_err(|e| {
            tracing::error!(error = %e, "accept failed");
            e
        })?;
        let model = cfg.model.clone();
        tokio::spawn(async move {
            loop {
                let msg: ControlMessage = match read_frame(&mut stream).await {
                    Ok(m) => m,
                    Err(_) => break,
                };
                let reply = match msg {
                    ControlMessage::Ping => ControlMessage::Pong,
                    ControlMessage::Shutdown => {
                        tracing::info!(model = %model, "shutdown requested");
                        std::process::exit(0);
                    }
                    other => {
                        tracing::warn!(?other, "unexpected control message, closing connection");
                        break;
                    }
                };
                if write_frame(&mut stream, &reply).await.is_err() {
                    break;
                }
            }
        });
    }
}
