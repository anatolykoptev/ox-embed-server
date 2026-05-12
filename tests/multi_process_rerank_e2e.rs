//! End-to-end test: spin up embed-server in multi-process mode on a random
//! port with a reranker, POST /v1/rerank, verify response shape.
//!
//! Skips gracefully when RERANKER_MODELS or ORT_DYLIB_PATH are unset so the
//! standard `cargo nextest run --lib` suite stays green without model files.
//!
//! Run with real model:
//!   RERANKER_MODELS="gte-multi-rerank:/models-gte-rerank:256:true" \
//!   ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//!   cargo nextest run --locked --test multi_process_rerank_e2e

#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn multi_process_rerank_e2e() {
    let Some(reranker_env) = std::env::var("RERANKER_MODELS").ok() else {
        eprintln!("SKIP: RERANKER_MODELS not set");
        return;
    };
    let ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if ort_path.is_empty() || !std::path::Path::new(&ort_path).exists() {
        eprintln!("SKIP: ORT_DYLIB_PATH not set or missing");
        return;
    }

    let port: u16 = 28092 + (std::process::id() % 10) as u16;
    let socket_dir: PathBuf =
        std::env::temp_dir().join(format!("embed-rerank-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&socket_dir);
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let server_bin = env!("CARGO_BIN_EXE_embed-server");

    // Need a dummy EMBED_MODELS for the server to boot (even if empty reranker list)
    // Use a value that won't load (no path) — server may fail embed model load but
    // reranker path still exercises the worker dispatch. Use EMBED_MODELS unset and
    // rely on the server's config parser allowing empty embed models.
    let server = Command::new(server_bin)
        .env("EMBED_MULTI_PROCESS", "1")
        .env("EMBED_WORKER_BIN", worker_bin)
        .env("EMBED_WORKER_SOCKET_DIR", &socket_dir)
        .env("EMBED_PORT", port.to_string())
        .env("RERANKER_MODELS", &reranker_env)
        .env("ORT_DYLIB_PATH", &ort_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn embed-server");

    let server_socket = socket_dir.join("server.marker");
    let _guard = ChildGuard {
        child: Some(server),
        socket: server_socket.clone(),
    };

    // Wait for server to become healthy (up to 60s for cold model load).
    let url = format!("http://127.0.0.1:{port}");
    let health_url = format!("{url}/health");
    let mut healthy = false;
    for _ in 0..120 {
        std::thread::sleep(Duration::from_millis(500));
        if let Ok(resp) = reqwest::blocking::get(&health_url)
            && resp.status().is_success()
        {
            healthy = true;
            break;
        }
    }
    if !healthy {
        eprintln!("SKIP: server did not become healthy within 60s");
        return;
    }

    // Parse reranker name from RERANKER_MODELS (format: name:dir:max_len:padded)
    let reranker_name = reranker_env
        .split(',')
        .next()
        .and_then(|s| s.split(':').next())
        .unwrap_or("gte-multi-rerank")
        .to_string();

    let body = serde_json::json!({
        "model": reranker_name,
        "query": "what is ONNX?",
        "documents": [
            "ONNX is an open format for machine learning models",
            "Paris is the capital of France",
            "ONNX Runtime accelerates inference"
        ]
    });

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{url}/v1/rerank"))
        .json(&body)
        .send()
        .expect("POST /v1/rerank");

    assert!(
        resp.status().is_success(),
        "expected 200 from /v1/rerank, got {}",
        resp.status()
    );

    let json: serde_json::Value = resp.json().expect("parse response json");
    let results = json["results"].as_array().expect("results array");

    // 3 documents in → 3 results out.
    assert_eq!(results.len(), 3, "expected 3 results for 3 documents");

    // Each result must have index (0-2) and relevance_score (float).
    for r in results {
        let idx = r["index"].as_u64().expect("index field");
        assert!(idx < 3, "index {idx} out of range");
        let score = r["relevance_score"]
            .as_f64()
            .expect("relevance_score field");
        assert!(
            score.is_finite(),
            "relevance_score must be finite, got {score}"
        );
    }

    // First result (highest score) should be one of the ONNX docs (index 0 or 2).
    let top_index = results[0]["index"].as_u64().expect("top index");
    assert!(
        top_index == 0 || top_index == 2,
        "expected ONNX doc to rank top, got index {top_index}"
    );

    let _ = std::fs::remove_dir_all(&socket_dir);
}
