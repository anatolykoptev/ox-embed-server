//! Manual smoke test for POST /v1/rerank against a RUNNING embed-server that
//! has a real, trained reranker (semantic ordering needs real weights).
//!
//! Ignored by default so it shows up as "ignored" rather than a vacuous pass.
//! Run:
//!   EMBED_SERVER_URL=http://127.0.0.1:8082 cargo test --test rerank_smoke -- --ignored --nocapture
//!
//! When run explicitly, an unset URL or an unreachable server FAILS the test.
//! The `/v1/rerank` request/response contract, `top_n` and batch independence
//! are covered on CI by `multi_process_rerank_e2e` against the tiny fixture.

use serde_json::json;

fn base_url() -> String {
    std::env::var("EMBED_SERVER_URL")
        .expect("EMBED_SERVER_URL must name the running embed-server for this smoke test")
}

fn server_ready(base: &str) -> bool {
    match reqwest::blocking::Client::new()
        .get(format!("{base}/health"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
    {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

#[test]
#[ignore = "manual smoke: needs EMBED_SERVER_URL and a trained reranker"]
fn rerank_orders_relevant_above_unrelated() {
    let base = base_url();
    assert!(server_ready(&base), "server at {base} not reachable");

    let body = json!({
        "model": "bge-reranker-v2-m3",
        "query": "what is a cat",
        "documents": [
            "a cat is a small domestic feline mammal",
            "the price of oil dropped yesterday",
            "cats purr when content"
        ]
    });

    let resp = reqwest::blocking::Client::new()
        .post(format!("{base}/v1/rerank"))
        .json(&body)
        .send()
        .expect("POST failed");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "status: {}",
        resp.status()
    );

    let payload: serde_json::Value = resp.json().expect("parse response");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 3);

    // First result must be index 0 (feline doc). Second either index 2 (cats purr)
    // or equal to first — they're both cat-related.
    let first_idx = results[0]["index"].as_i64().unwrap();
    assert_eq!(first_idx, 0, "feline doc should rank first, got: {payload}");

    // Oil (index 1) must NOT be the top result.
    let last_idx = results[2]["index"].as_i64().unwrap();
    assert_eq!(last_idx, 1, "oil doc should rank last, got: {payload}");

    // Scores must be monotonically decreasing.
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["relevance_score"].as_f64().unwrap())
        .collect();
    for w in scores.windows(2) {
        assert!(w[0] >= w[1], "scores not desc: {scores:?}");
    }
}

#[test]
#[ignore = "manual smoke: needs EMBED_SERVER_URL and a trained reranker"]
fn rerank_respects_top_n() {
    let base = base_url();
    assert!(server_ready(&base), "server at {base} not reachable");

    let body = json!({
        "model": "bge-reranker-v2-m3",
        "query": "animals",
        "documents": ["dog", "cat", "car", "bicycle"],
        "top_n": 2
    });

    let resp = reqwest::blocking::Client::new()
        .post(format!("{base}/v1/rerank"))
        .json(&body)
        .send()
        .expect("POST failed");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let payload: serde_json::Value = resp.json().unwrap();
    let results = payload["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        2,
        "top_n=2 but got {} results",
        results.len()
    );
}
