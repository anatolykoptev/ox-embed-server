//! Exercises worker binary against multilingual-e5-large.
//!
//! Requires model files reachable via Config::from_env() — set EMBED_MODELS,
//! EMBED_DEFAULT_MODEL, ORT_DYLIB_PATH before running.
//!
//! Skip gracefully if env not set (CI-safe).

#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{InferRequest, InferResponse};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

#[tokio::test]
async fn worker_infers_e5_batch() {
    let Some(models_env) = std::env::var("EMBED_MODELS").ok() else {
        eprintln!("SKIP: EMBED_MODELS not set");
        return;
    };
    let ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if ort_path.is_empty() || !std::path::Path::new(&ort_path).exists() {
        eprintln!("SKIP: ORT_DYLIB_PATH not set or file missing (got: {ort_path:?})");
        return;
    }

    let socket: PathBuf =
        std::env::temp_dir().join(format!("embed-worker-e5-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "multilingual-e5-large")
        .env("EMBED_WORKER_SOCKET", &socket)
        .env("EMBED_WORKER_POOL_SIZE", "1")
        .env("EMBED_MODELS", &models_env)
        .env("ORT_DYLIB_PATH", &ort_path)
        .env("EMBED_DEFAULT_MODEL", "multilingual-e5-large")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn worker binary");

    let _guard = ChildGuard {
        child: Some(child),
        socket: socket.clone(),
    };

    // Wait up to 30 s for the worker to create the socket (model load ~5-10 s).
    for _ in 0..300 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(socket.exists(), "worker socket missing after 30 s");

    let mut conn = UnixStream::connect(&socket)
        .await
        .expect("connect to worker UDS");
    // NOTE: max_seq_len is not yet honoured by the worker (Phase 5 wires it).
    // Send 128 to match plan, but model uses its configured max_len internally.
    let req = InferRequest {
        request_id: 1,
        model: "multilingual-e5-large".into(),
        texts: vec!["query: hello".into(), "query: world".into()],
        max_seq_len: 128,
    };
    write_frame(&mut conn, &req)
        .await
        .expect("send InferRequest");

    let resp: InferResponse = read_frame(&mut conn).await.expect("recv InferResponse");
    match resp {
        InferResponse::Ok {
            request_id,
            vectors,
            dims,
        } => {
            assert_eq!(request_id, 1, "request_id round-trip");
            assert_eq!(vectors.len(), 2, "two texts → two vectors");
            assert_eq!(dims, 1024, "multilingual-e5-large is 1024-d");
            assert_eq!(vectors[0].len(), 1024, "vector length matches dims");
            // Sanity: L2-normalised vectors should have norm ≈ 1.0.
            let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "vector should be L2-normalised, got norm={norm}"
            );
        }
        InferResponse::Err { message, .. } => panic!("inference failed: {message}"),
    }
    // `_guard` drops here — ChildGuard::drop calls kill() + wait() + socket cleanup.
}
