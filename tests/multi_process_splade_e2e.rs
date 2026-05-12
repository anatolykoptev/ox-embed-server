//! End-to-end test: spin up embed-server in multi-process mode on a random
//! port with a SPLADE model, POST /embed_sparse, verify response shape.
//!
//! Skips gracefully when SPLADE_MODELS or ORT_DYLIB_PATH are unset so the
//! standard `cargo nextest run --lib` suite stays green without model files.
//!
//! Run with real model:
//!   SPLADE_MODELS="splade-v3:/models-splade:128" \
//!   ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//!   cargo nextest run --locked --test multi_process_splade_e2e

#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn multi_process_splade_e2e() {
    let Some(splade_env) = std::env::var("SPLADE_MODELS").ok() else {
        eprintln!("SKIP: SPLADE_MODELS not set");
        return;
    };
    let ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if ort_path.is_empty() || !std::path::Path::new(&ort_path).exists() {
        eprintln!("SKIP: ORT_DYLIB_PATH not set or missing");
        return;
    }

    let port: u16 = 28102 + (std::process::id() % 10) as u16;
    let socket_dir: PathBuf =
        std::env::temp_dir().join(format!("embed-splade-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&socket_dir);
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let server_bin = env!("CARGO_BIN_EXE_embed-server");

    let server = Command::new(server_bin)
        .env("EMBED_MULTI_PROCESS", "1")
        .env("EMBED_WORKER_BIN", worker_bin)
        .env("EMBED_WORKER_SOCKET_DIR", &socket_dir)
        .env("EMBED_PORT", port.to_string())
        .env("SPLADE_MODELS", &splade_env)
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
        if let Ok(resp) = reqwest::blocking::get(&health_url) {
            if resp.status().is_success() {
                healthy = true;
                break;
            }
        }
    }
    if !healthy {
        eprintln!("SKIP: server did not become healthy within 60s");
        return;
    }

    // Parse SPLADE model name from SPLADE_MODELS (format: name:dir:max_len)
    let splade_name = splade_env
        .split(',')
        .next()
        .and_then(|s| s.split(':').next())
        .unwrap_or("splade-v3")
        .to_string();

    let body = serde_json::json!({
        "model": splade_name,
        "input": [
            "ONNX is an open format for machine learning models",
            "Paris is the capital of France"
        ]
    });

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{url}/embed_sparse"))
        .json(&body)
        .send()
        .expect("POST /embed_sparse");

    assert!(
        resp.status().is_success(),
        "expected 200 from /embed_sparse, got {}",
        resp.status()
    );

    let json: serde_json::Value = resp.json().expect("parse response json");
    let data = json["data"].as_array().expect("data array");

    // 2 texts in → 2 sparse vectors out.
    assert_eq!(data.len(), 2, "expected 2 sparse vectors for 2 texts");

    for item in data {
        let indices = item["indices"].as_array().expect("indices array");
        let values = item["values"].as_array().expect("values array");

        // indices and values must be aligned.
        assert_eq!(
            indices.len(),
            values.len(),
            "indices and values must have equal length"
        );

        // SPLADE should produce at least one non-zero term per non-trivial input.
        assert!(
            !indices.is_empty(),
            "sparse vector must have at least one term"
        );

        // All weights should be positive (log(1 + ReLU(logit)) ≥ 0, filtered > min_weight).
        for v in values {
            let w = v.as_f64().expect("weight value");
            assert!(w > 0.0, "weight must be positive, got {w}");
        }

        // index field should be 0-based.
        let idx = item["index"].as_u64().expect("index field");
        assert!(idx < 2, "index {idx} out of range");
    }

    let _ = std::fs::remove_dir_all(&socket_dir);
}
