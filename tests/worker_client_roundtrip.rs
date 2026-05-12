//! Drives the WorkerClient against a spawned embed-worker binary
//! loading multilingual-e5-large. Skips if env not set.
//!
//! Wave 2.4b: migrated from client.infer() + InferResponse to
//! dispatch_embed() + WorkerResponse.

#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use embed_server::ipc::client::WorkerClient;
use embed_server::ipc::protocol::WorkerResponse;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

#[tokio::test]
async fn client_roundtrip_e5() {
    let Some(models_env) = std::env::var("EMBED_MODELS").ok() else {
        eprintln!("SKIP: EMBED_MODELS not set");
        return;
    };
    let ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if ort_path.is_empty() || !std::path::Path::new(&ort_path).exists() {
        eprintln!("SKIP: ORT_DYLIB_PATH not set or missing");
        return;
    }

    let socket: PathBuf =
        std::env::temp_dir().join(format!("embed-wc-test-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "multilingual-e5-large")
        .env("EMBED_WORKER_KIND", "embed")
        .env("EMBED_WORKER_SOCKET", &socket)
        .env("EMBED_WORKER_POOL_SIZE", "2")
        .env("EMBED_MODELS", &models_env)
        .env("ORT_DYLIB_PATH", &ort_path)
        .env("EMBED_DEFAULT_MODEL", "multilingual-e5-large")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn worker");

    let _guard = ChildGuard {
        child: Some(child),
        socket: socket.clone(),
    };

    // Wait up to 30s for the worker to create the socket (cold model load ~5-10s).
    for _ in 0..300 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(socket.exists(), "worker socket missing");

    let client = WorkerClient::connect(socket.clone(), 2)
        .await
        .expect("connect to worker");

    // First request
    let resp = client
        .dispatch_embed(
            "multilingual-e5-large".into(),
            vec!["query: hello".into()],
            128,
        )
        .await
        .expect("io");
    match resp {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.vectors.len(), 1);
            assert_eq!(ok.dims, 1024);
            assert_eq!(ok.vectors[0].len(), 1024);
        }
        WorkerResponse::Err { message, .. } => panic!("infer failed: {message}"),
        other => panic!("unexpected response variant: {other:?}"),
    }

    // Second request — round-robins to second pool conn
    let resp2 = client
        .dispatch_embed(
            "multilingual-e5-large".into(),
            vec!["query: world".into(), "query: again".into()],
            128,
        )
        .await
        .expect("io");
    match resp2 {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.vectors.len(), 2);
            assert_eq!(ok.dims, 1024);
        }
        WorkerResponse::Err { message, .. } => panic!("infer 2 failed: {message}"),
        other => panic!("unexpected response variant: {other:?}"),
    }
}
