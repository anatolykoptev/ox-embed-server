//! End-to-end: WorkerSupervisor launches a real worker, dispatches inference via
//! the pool, verifies response. Skips if model env not set.
//!
//! Wave 2.4b: migrated from dispatch() + InferResponse to dispatch_embed() + WorkerResponse.

use embed_server::ipc::protocol::WorkerResponse;
use embed_server::supervisor::{SpawnSpec, WorkerKind, WorkerPool, WorkerSupervisor};
use std::path::PathBuf;

#[tokio::test]
async fn pool_dispatches_to_spawned_worker() {
    let Some(models_env) = std::env::var("EMBED_MODELS").ok() else {
        eprintln!("SKIP: EMBED_MODELS not set");
        return;
    };
    let ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if ort_path.is_empty() || !std::path::Path::new(&ort_path).exists() {
        eprintln!("SKIP: ORT_DYLIB_PATH not set");
        return;
    }

    let socket_dir: PathBuf =
        std::env::temp_dir().join(format!("embed-pool-test-{}", std::process::id()));

    let worker_bin: PathBuf = env!("CARGO_BIN_EXE_embed-worker").into();

    let spec = SpawnSpec {
        model: "multilingual-e5-large".into(),
        kind: WorkerKind::Embed,
        worker_bin,
        socket_dir: socket_dir.clone(),
        pool_size: 1,
        intra_threads: 2,
        env_extra: vec![
            ("EMBED_MODELS".into(), models_env),
            ("ORT_DYLIB_PATH".into(), ort_path),
            ("EMBED_DEFAULT_MODEL".into(), "multilingual-e5-large".into()),
        ],
    };

    let supervisor = WorkerSupervisor::launch(spec)
        .await
        .expect("launch supervisor");

    let pool = WorkerPool::new();
    pool.add(supervisor).await;

    let resp = pool
        .dispatch_embed(
            "multilingual-e5-large",
            vec!["query: hello supervisor".into()],
            128,
        )
        .await
        .expect("dispatch");
    match resp {
        WorkerResponse::Embed(ok) => {
            assert_eq!(ok.vectors.len(), 1);
            assert_eq!(ok.dims, 1024);
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
        models.contains(&"multilingual-e5-large".to_string()),
        "models() should include the registered model, got: {models:?}"
    );
    assert_eq!(
        models.len(),
        1,
        "only one model registered, got: {models:?}"
    );

    let _ = std::fs::remove_dir_all(&socket_dir);
}
