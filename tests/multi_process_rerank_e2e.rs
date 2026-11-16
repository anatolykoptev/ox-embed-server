//! End-to-end: `embed-server` in multi-process mode, `POST /v1/rerank` served by
//! a real reranker worker loading the tiny cross-encoder fixture.
//!
//! The fixture's weights are random, so the test checks the response contract
//! and that a document's score does not depend on its batch-mates, not which
//! document ranks first.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).
#[path = "common/mod.rs"]
mod common;
use std::time::Duration;

fn rerank(
    client: &reqwest::blocking::Client,
    base: &str,
    query: &str,
    docs: &[String],
    top_n: Option<usize>,
) -> Vec<(usize, f64)> {
    let mut body = serde_json::json!({
        "model": common::RERANK_NAME,
        "query": query,
        "documents": docs,
    });
    if let Some(n) = top_n {
        body["top_n"] = n.into();
    }
    let resp = client
        .post(format!("{base}/v1/rerank"))
        .json(&body)
        .send()
        .expect("POST /v1/rerank");
    assert!(
        resp.status().is_success(),
        "expected 200 from /v1/rerank, got {}",
        resp.status()
    );
    let json: serde_json::Value = resp.json().expect("parse response json");
    json["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|r| {
            (
                r["index"].as_u64().expect("index field") as usize,
                r["relevance_score"].as_f64().expect("relevance_score"),
            )
        })
        .collect()
}

#[test]
fn multi_process_rerank_e2e() {
    let Some(env) = common::setup() else {
        return;
    };
    let server = common::spawn_server(
        &env,
        "rerank-e2e",
        env!("CARGO_BIN_EXE_embed-server"),
        env!("CARGO_BIN_EXE_embed-worker"),
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build reqwest client");

    let query = "what is onnx";
    let docs = vec![
        "onnx is an open format for machine learning models".to_string(),
        "paris is the capital of france".to_string(),
        common::probe_texts()[4].clone(),
    ];
    let results = rerank(&client, &server.base, query, &docs, None);

    assert_eq!(results.len(), 3, "expected 3 results for 3 documents");
    let mut seen: Vec<usize> = results.iter().map(|r| r.0).collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![0, 1, 2],
        "results must cover every document once"
    );
    for (_, score) in &results {
        assert!(score.is_finite(), "relevance_score must be finite: {score}");
    }
    for w in results.windows(2) {
        assert!(w[0].1 >= w[1].1, "scores not descending: {results:?}");
    }

    // Each document scored alone must match its score inside the 3-document
    // call: the pad width differs (the long document sets it in the batch).
    for (idx, batched) in &results {
        let alone = rerank(
            &client,
            &server.base,
            query,
            std::slice::from_ref(&docs[*idx]),
            None,
        );
        assert_eq!(alone.len(), 1);
        let d = (alone[0].1 - batched).abs();
        assert!(
            d < 1e-4,
            "doc {idx}: score depends on its batch-mates ({} alone vs {batched} batched)",
            alone[0].1
        );
    }

    let top = rerank(&client, &server.base, query, &docs, Some(2));
    assert_eq!(top.len(), 2, "top_n=2 must cap the results");
    assert_eq!(top, results[..2], "top_n must keep the best-scored prefix");
}
