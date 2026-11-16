//! End-to-end: `embed-server` in multi-process mode, `POST /embed_sparse`
//! served by a real SPLADE worker loading the tiny masked-LM fixture.
//!
//! Weights are random, so the test checks the response contract and that a
//! text's sparse vector does not depend on its batch-mates.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).
#[path = "common/mod.rs"]
mod common;
use std::collections::BTreeMap;
use std::time::Duration;

type Sparse = BTreeMap<u64, f64>;

fn embed_sparse(
    client: &reqwest::blocking::Client,
    base: &str,
    input: &[&str],
    top_k: Option<usize>,
) -> Vec<Sparse> {
    let mut body = serde_json::json!({ "model": common::SPLADE_NAME, "input": input });
    if let Some(k) = top_k {
        body["top_k"] = k.into();
    }
    let resp = client
        .post(format!("{base}/embed_sparse"))
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
    assert_eq!(data.len(), input.len(), "one sparse vector per text");
    data.iter()
        .enumerate()
        .map(|(i, item)| {
            assert_eq!(item["index"].as_u64().expect("index field"), i as u64);
            let indices = item["indices"].as_array().expect("indices array");
            let values = item["values"].as_array().expect("values array");
            assert_eq!(
                indices.len(),
                values.len(),
                "indices and values must be aligned"
            );
            assert!(!indices.is_empty(), "sparse vector must have a term");
            indices
                .iter()
                .zip(values)
                .map(|(k, v)| {
                    let w = v.as_f64().expect("weight value");
                    assert!(w > 0.0, "weight must be positive, got {w}");
                    (k.as_u64().expect("token id"), w)
                })
                .collect()
        })
        .collect()
}

fn assert_same(what: &str, a: &Sparse, b: &Sparse) {
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "{what}: different term sets"
    );
    for (k, wa) in a {
        let d = (wa - b[k]).abs();
        assert!(d < 1e-4, "{what}: weight of term {k} differs by {d}");
    }
}

#[test]
fn multi_process_splade_e2e() {
    let Some(env) = common::setup() else {
        return;
    };
    let server = common::spawn_server(
        &env,
        "splade-e2e",
        env!("CARGO_BIN_EXE_embed-server"),
        env!("CARGO_BIN_EXE_embed-worker"),
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build reqwest client");

    let a = "onnx is an open format for machine learning models";
    let b = "paris is the capital of france";
    let both = embed_sparse(&client, &server.base, &[a, b], None);
    let only_a = embed_sparse(&client, &server.base, &[a], None);
    let only_b = embed_sparse(&client, &server.base, &[b], None);
    assert_same("text a alone vs batched", &only_a[0], &both[0]);
    assert_same("text b alone vs batched", &only_b[0], &both[1]);

    // The tokenizer must feed the model the actual text: two different texts
    // cannot produce the same sparse vector.
    assert_ne!(
        only_a[0], only_b[0],
        "different texts must give different sparse vectors"
    );

    // top_k keeps the heaviest terms: a prefix of the full vector by weight.
    let k = 3;
    let top = embed_sparse(&client, &server.base, &[a], Some(k));
    assert_eq!(
        top[0].len(),
        k.min(only_a[0].len()),
        "top_k must cap the terms"
    );
    let mut full: Vec<(&u64, &f64)> = only_a[0].iter().collect();
    full.sort_by(|x, y| y.1.total_cmp(x.1));
    for (id, w) in full.iter().take(k) {
        let got = top[0]
            .get(*id)
            .unwrap_or_else(|| panic!("top_k dropped heavy term {id} (weight {w})"));
        assert!(
            (got - **w).abs() < 1e-4,
            "top_k changed weight of term {id}"
        );
    }
}
