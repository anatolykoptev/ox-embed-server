//! End-to-end: WorkerSupervisor launches a real worker on the tiny embed
//! fixture, dispatches inference via the pool, verifies the response.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).
#[path = "common/mod.rs"]
mod common;
use embed_server::ipc::protocol::WorkerResponse;
use embed_server::supervisor::{SpawnSpec, WorkerKind, WorkerPool, WorkerSupervisor};
use std::path::PathBuf;

#[tokio::test]
async fn pool_dispatches_to_spawned_worker() {
    let Some(env) = common::setup() else {
        return;
    };
    let socket_dir: PathBuf =
        std::env::temp_dir().join(format!("embed-pool-test-{}", std::process::id()));
    let worker_bin: PathBuf = env!("CARGO_BIN_EXE_embed-worker").into();
    let spec = SpawnSpec {
        model: common::EMBED_NAME.into(),
        kind: WorkerKind::Embed,
        worker_bin,
        socket_dir: socket_dir.clone(),
        pool_size: 1,
        intra_threads: 2,
        env_extra: vec![
            ("EMBED_MODELS".into(), env.embed_spec()),
            ("ORT_DYLIB_PATH".into(), env.ort.display().to_string()),
            ("EMBED_DEFAULT_MODEL".into(), common::EMBED_NAME.into()),
        ],
    };
    let supervisor = WorkerSupervisor::launch(spec)
        .await
        .expect("launch supervisor");
    let pool = WorkerPool::new();
    pool.add(supervisor).await;
    let resp = pool
        .dispatch_embed(
            common::EMBED_NAME,
            vec!["hello supervisor".into()],
            common::MAX_LEN as u32,
        )
        .await
        .expect("dispatch");
    match resp {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.vectors.len(), 1);
            assert_eq!(ok.dims as usize, common::EMBED_DIM);
        }
        WorkerResponse::Err { message, .. } => panic!("infer failed: {message}"),
        other => panic!("unexpected response variant: {other:?}"),
    }
    // Dispatch to unknown model errors out.
    let err = pool
        .dispatch_embed("nonexistent-model", vec!["x".into()], 64)
        .await
        .expect_err("should error on unknown model");
    let msg = format!("{err}");
    assert!(msg.contains("no worker for model"), "got: {msg}");
    // Verify models() lists the registered model.
    let models = pool.models().await;
    assert!(
        models.contains(&common::EMBED_NAME.to_string()),
        "models() should include the registered model, got: {models:?}"
    );
    assert_eq!(
        models.len(),
        1,
        "only one model registered, got: {models:?}"
    );
    let _ = std::fs::remove_dir_all(&socket_dir);
}
