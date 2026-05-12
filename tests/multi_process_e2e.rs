//! End-to-end test: spin up embed-server in multi-process mode on a random
//! port, send /v1/embeddings, verify the response contains a 1024-dim vector.
//!
//! Skips gracefully when EMBED_MODELS or ORT_DYLIB_PATH are unset / the model
//! files are missing, so the standard `cargo nextest run --lib` suite stays
//! green in CI.
//!
//! Run with real model:
//!   EMBED_MODELS="multilingual-e5-large:/home/krolik/deploy/krolik-server/models/multilingual-e5-large:1024:256:1:false" \
//!   ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//!   cargo nextest run --locked --test multi_process_e2e

#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn multi_process_embed_e2e() {
    let Some(models_env) = std::env::var("EMBED_MODELS").ok() else {
        eprintln!("SKIP: EMBED_MODELS not set");
        return;
    };
    let ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if ort_path.is_empty() || !std::path::Path::new(&ort_path).exists() {
        eprintln!("SKIP: ORT_DYLIB_PATH not set or missing (got: {ort_path:?})");
        return;
    }

    // Random-ish high port to avoid conflict with running prod embed-server (:8082).
    let port: u16 = 28082 + (std::process::id() % 10) as u16;
    let socket_dir: PathBuf =
        std::env::temp_dir().join(format!("embed-multi-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&socket_dir);
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let server_bin = env!("CARGO_BIN_EXE_embed-server");

    let server = Command::new(server_bin)
        .env("EMBED_MULTI_PROCESS", "1")
        .env("EMBED_WORKER_BIN", worker_bin)
        .env("EMBED_WORKER_SOCKET_DIR", &socket_dir)
        .env("EMBED_PORT", port.to_string())
        .env("EMBED_MODELS", &models_env)
        .env("ORT_DYLIB_PATH", &ort_path)
        .env("EMBED_DEFAULT_MODEL", "multilingual-e5-large")
        // One worker per model is enough; suppress batcher to keep resident
        // memory lower during tests.
        .env("BATCHING_ENABLED", "false")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn embed-server");

    // Pass a placeholder socket path that does not exist — ChildGuard will
    // attempt remove_file on it at drop, which is a no-op on ENOENT.
    // We clean up socket_dir manually at the end of the test.
    let placeholder_socket = socket_dir.join("__guard_noop__");
    let _guard = ChildGuard {
        child: Some(server),
        socket: placeholder_socket,
    };

    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("build reqwest client");

    // Wait up to 60 s for /health — model cold load can be slow.
    let mut ready = false;
    for _ in 0..600 {
        if client
            .get(format!("{base}/health"))
            .send()
            .is_ok_and(|r| r.status().is_success())
        {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(ready, "embed-server did not become ready on port {port}");

    let resp: serde_json::Value = client
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "model": "multilingual-e5-large",
            "input": ["hello multi-process world"],
        }))
        .send()
        .expect("POST /v1/embeddings")
        .json()
        .expect("parse JSON response");

    let embedding = &resp["data"][0]["embedding"];
    assert!(
        embedding.is_array(),
        "embedding field should be array, got: {resp}"
    );
    let dim = embedding.as_array().unwrap().len();
    assert_eq!(
        dim, 1024,
        "e5-large should produce 1024-dim vector, got {dim}"
    );

    let total_tokens = resp["usage"]["total_tokens"].as_u64().unwrap_or(0);
    assert!(
        total_tokens > 0,
        "total_tokens should be non-zero for non-empty input, got: {total_tokens}"
    );

    // Explicit cleanup of socket dir (_guard uses remove_file on placeholder).
    let _ = std::fs::remove_dir_all(&socket_dir);
}
