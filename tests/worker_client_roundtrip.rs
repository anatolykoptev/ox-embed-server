//! Drives `WorkerClient` (the supervisor-side IPC client) against a spawned
//! `embed-worker` loading the tiny embed fixture, with a 2-connection pool.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).

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
    let Some(env) = common::setup() else {
        return;
    };

    let socket: PathBuf =
        std::env::temp_dir().join(format!("embed-wc-test-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);

    let child = Command::new(env!("CARGO_BIN_EXE_embed-worker"))
        .env("EMBED_WORKER_MODEL", common::EMBED_NAME)
        .env("EMBED_WORKER_KIND", "embed")
        .env("EMBED_WORKER_SOCKET", &socket)
        .env("EMBED_WORKER_POOL_SIZE", "2")
        .env("EMBED_MODELS", env.embed_spec())
        .env("ORT_DYLIB_PATH", &env.ort)
        .env("EMBED_DEFAULT_MODEL", common::EMBED_NAME)
        .env(common::MEMORY_PATTERN_ENV.0, common::MEMORY_PATTERN_ENV.1)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn worker");
    let _guard = ChildGuard {
        child: Some(child),
        socket: socket.clone(),
    };

    for _ in 0..600 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(socket.exists(), "worker socket missing after 60 s");

    let client = WorkerClient::connect(socket.clone(), 2, Duration::from_secs(60))
        .await
        .expect("connect to worker");

    let first = match client
        .dispatch_embed(
            common::EMBED_NAME.into(),
            vec!["hello".into()],
            common::MAX_LEN as u32,
        )
        .await
        .expect("io")
    {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.vectors.len(), 1);
            assert_eq!(ok.dims as usize, common::EMBED_DIM);
            assert_eq!(ok.vectors[0].len(), common::EMBED_DIM);
            ok.vectors[0].clone()
        }
        WorkerResponse::Err { message, .. } => panic!("infer failed: {message}"),
        other => panic!("unexpected response variant: {other:?}"),
    };

    // Second request round-robins to the second pool connection; it carries the
    // same text first, so the two connections must agree on its vector.
    match client
        .dispatch_embed(
            common::EMBED_NAME.into(),
            vec!["hello".into(), "world again".into()],
            common::MAX_LEN as u32,
        )
        .await
        .expect("io")
    {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.vectors.len(), 2);
            assert_eq!(ok.dims as usize, common::EMBED_DIM);
            let d = common::max_abs_diff(&ok.vectors[0], &first);
            assert!(d < 1e-4, "same text, different pool connection: diff {d}");
        }
        WorkerResponse::Err { message, .. } => panic!("infer 2 failed: {message}"),
        other => panic!("unexpected response variant: {other:?}"),
    }
}
