//! Integration test for POST /v1/rerank. Requires a running embed-server.
//!
//! Run:
//!   EMBED_SERVER_URL=http://127.0.0.1:8082 cargo test --test rerank_smoke -- --nocapture
//!
//! If EMBED_SERVER_URL is unset or the server is unreachable, tests print a
//! visible skip message and return Ok — so the main test suite stays green.

use serde_json::json;

fn base_url() -> Option<String> {
    std::env::var("EMBED_SERVER_URL").ok()
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
fn rerank_orders_relevant_above_unrelated() {
    let Some(base) = base_url() else {
        eprintln!("SKIP rerank_orders_relevant_above_unrelated: EMBED_SERVER_URL unset");
        return;
    };
    if !server_ready(&base) {
        eprintln!("SKIP rerank_orders_relevant_above_unrelated: server at {base} not reachable");
        return;
    }

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
fn rerank_respects_top_n() {
    let Some(base) = base_url() else {
        eprintln!("SKIP rerank_respects_top_n: EMBED_SERVER_URL unset");
        return;
    };
    if !server_ready(&base) {
        eprintln!("SKIP rerank_respects_top_n: server at {base} not reachable");
        return;
    }

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
