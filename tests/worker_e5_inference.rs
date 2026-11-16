//! Drives the real `embed-worker` binary over its Unix socket against the tiny
//! embed fixture (`tests/fixtures/tiny/embed`).
//!
//! Beyond shape, this pins the properties the worker's batching path must keep:
//! a text's vector must not depend on its batch-mates (padding is masked out of
//! the mean pool) and must come back at the caller's position even when the
//! planner splits the request into length-homogeneous sub-batches.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).
#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;
use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::{EmbedRequest, WorkerRequest, WorkerResponse};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

async fn embed(conn: &mut UnixStream, request_id: u64, texts: Vec<String>) -> Vec<Vec<f32>> {
    let n = texts.len();
    let req = WorkerRequest::Embed(EmbedRequest {
        request_id,
        model: common::EMBED_NAME.into(),
        texts,
        max_seq_len: common::MAX_LEN as u32,
    });
    write_frame(conn, &req).await.expect("send WorkerRequest");
    let resp: WorkerResponse = read_frame(conn).await.expect("recv WorkerResponse");
    match resp {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.request_id, request_id, "request_id round-trip");
            assert_eq!(ok.vectors.len(), n, "one vector per text");
            assert_eq!(ok.dims as usize, common::EMBED_DIM, "dims from the fixture");
            ok.vectors
        }
        WorkerResponse::Err { message, .. } => panic!("inference failed: {message}"),
        other => panic!("unexpected response variant: {other:?}"),
    }
}

#[tokio::test]
async fn worker_infers_e5_batch() {
    let Some(env) = common::setup() else {
        return;
    };
    let socket: PathBuf =
        std::env::temp_dir().join(format!("embed-worker-e5-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);
    let child = Command::new(env!("CARGO_BIN_EXE_embed-worker"))
        .env("EMBED_WORKER_MODEL", common::EMBED_NAME)
        .env("EMBED_WORKER_KIND", "embed")
        .env("EMBED_WORKER_SOCKET", &socket)
        .env("EMBED_WORKER_POOL_SIZE", "1")
        .env("EMBED_MODELS", env.embed_spec())
        .env("ORT_DYLIB_PATH", &env.ort)
        .env("EMBED_DEFAULT_MODEL", common::EMBED_NAME)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn worker binary");
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
    let mut conn = UnixStream::connect(&socket)
        .await
        .expect("connect to worker UDS");

    let texts = common::probe_texts();

    // Reference: each text alone.
    let mut solo: Vec<Vec<f32>> = Vec::new();
    for (i, t) in texts.iter().enumerate() {
        solo.push(
            embed(&mut conn, 100 + i as u64, vec![t.clone()])
                .await
                .remove(0),
        );
    }
    assert_ne!(
        common::max_abs_diff(&solo[0], &solo[4]),
        0.0,
        "fixture must give distinct texts distinct vectors"
    );
    for v in &solo {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "vector should be L2-normalised, got norm={norm}"
        );
    }

    // Every text, in every probe batch, must equal its solo vector and sit at
    // the caller's position.
    for (n, batch) in common::PROBE_BATCHES.iter().enumerate() {
        let req: Vec<String> = batch.iter().map(|&i| texts[i].clone()).collect();
        let got = embed(&mut conn, n as u64, req).await;
        for (pos, &i) in batch.iter().enumerate() {
            let d = common::max_abs_diff(&got[pos], &solo[i]);
            assert!(
                d < 1e-4,
                "batch {batch:?}, position {pos} (text {i}): vector depends on its \
                 batch-mates or is out of order (max abs diff {d})"
            );
        }
    }
}
