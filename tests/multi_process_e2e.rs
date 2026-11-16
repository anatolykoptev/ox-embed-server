//! End-to-end: `embed-server` in multi-process mode on a free port, workers
//! spawned from the real `embed-worker` binary, `/v1/embeddings` answered from
//! the tiny embed fixture through the full supervisor -> UDS -> worker path.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).
#[path = "common/mod.rs"]
mod common;
use std::time::Duration;

fn embed(client: &reqwest::blocking::Client, base: &str, inputs: &[String]) -> serde_json::Value {
    client
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({ "model": common::EMBED_NAME, "input": inputs }))
        .send()
        .expect("POST /v1/embeddings")
        .error_for_status()
        .expect("embeddings status")
        .json()
        .expect("parse JSON response")
}

fn vec_at(resp: &serde_json::Value, i: usize) -> Vec<f32> {
    resp["data"][i]["embedding"]
        .as_array()
        .unwrap_or_else(|| panic!("embedding {i} should be an array, got: {resp}"))
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

#[test]
fn multi_process_embed_e2e() {
    let Some(env) = common::setup() else {
        return;
    };
    let server = common::spawn_server(
        &env,
        "multi-e2e",
        env!("CARGO_BIN_EXE_embed-server"),
        env!("CARGO_BIN_EXE_embed-worker"),
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build reqwest client");

    let texts = common::probe_texts();
    let solo: Vec<Vec<f32>> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let resp = embed(&client, &server.base, std::slice::from_ref(t));
            if i == 0 {
                let total_tokens = resp["usage"]["total_tokens"].as_u64().unwrap_or(0);
                assert!(
                    total_tokens > 0,
                    "total_tokens should be non-zero for non-empty input, got: {total_tokens}"
                );
            }
            let v = vec_at(&resp, 0);
            assert_eq!(
                v.len(),
                common::EMBED_DIM,
                "fixture embedder is {}-dim",
                common::EMBED_DIM
            );
            v
        })
        .collect();

    // Every text, in every probe batch, must equal its solo vector and sit at
    // the caller's position.
    for batch in common::PROBE_BATCHES {
        let req: Vec<String> = batch.iter().map(|&i| texts[i].clone()).collect();
        let resp = embed(&client, &server.base, &req);
        for (pos, &i) in batch.iter().enumerate() {
            let d = common::max_abs_diff(&vec_at(&resp, pos), &solo[i]);
            assert!(
                d < 1e-4,
                "batch {batch:?}, position {pos} (text {i}): vector depends on its \
                 batch-mates or is out of order (max abs diff {d})"
            );
        }
    }
}
