//! End-to-end: the optional `<model dir>/calibration.json` reaches the scores on
//! BOTH `/v1/rerank` paths: the in-process session (`EMBED_MULTI_PROCESS=0`) and
//! the worker pool (`EMBED_MULTI_PROCESS=1`).
//!
//! The fixture reranker has random weights, so the tests never assume a score.
//! They read the raw logit (`"normalize":"none"`), install the map `y = x + SHIFT`
//! and require the default sigmoid response to equal `sigmoid(raw + SHIFT)`.
//!
//! Runs on CI; a missing fixture or ONNX Runtime fails there (see `common`).
#[path = "common/mod.rs"]
mod common;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SHIFT: f64 = 3.0;
const CAL_ID: &str = "test-fit-shift3";

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A copy of the fixture reranker dir (symlinked files) with an optional
/// `calibration.json` body.
fn rerank_dir(env: &common::Setup, tag: &str, calibration: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rerank-cal-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create reranker dir");
    for f in ["model_quantized.onnx", "tokenizer.json"] {
        std::os::unix::fs::symlink(env.root.join("rerank").join(f), dir.join(f))
            .expect("symlink fixture");
    }
    if let Some(body) = calibration {
        std::fs::write(dir.join("calibration.json"), body).expect("write calibration.json");
    }
    dir
}

fn shift_file() -> String {
    format!(
        r#"{{"schema":1,"id":"{CAL_ID}","knots_x":[-1000.0,1000.0],"knots_y":[{},{}]}}"#,
        -1000.0 + SHIFT,
        1000.0 + SHIFT
    )
}

fn spawn(
    env: &common::Setup,
    tag: &str,
    multi_process: bool,
    rerank_dir: &Path,
) -> (Child, String) {
    let port = common::free_port();
    let socket_dir = std::env::temp_dir().join(format!("embed-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&socket_dir);
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let child = Command::new(env!("CARGO_BIN_EXE_embed-server"))
        .env("EMBED_MULTI_PROCESS", if multi_process { "1" } else { "0" })
        .env("EMBED_WORKER_BIN", env!("CARGO_BIN_EXE_embed-worker"))
        .env("EMBED_WORKER_SOCKET_DIR", &socket_dir)
        .env("EMBED_PORT", port.to_string())
        .env(
            "EMBED_WORKER_METRICS_PORT_BASE",
            common::free_port().min(65_000).to_string(),
        )
        .env("EMBED_MODELS", env.embed_spec())
        .env("EMBED_DEFAULT_MODEL", common::EMBED_NAME)
        .env(common::MEMORY_PATTERN_ENV.0, common::MEMORY_PATTERN_ENV.1)
        .env("CACHE_MAX_ENTRIES", "0")
        .env(
            "RERANKER_MODELS",
            format!(
                "{}:{}:{}:true",
                common::RERANK_NAME,
                rerank_dir.display(),
                common::MAX_LEN
            ),
        )
        .env("ORT_DYLIB_PATH", &env.ort)
        .env("BATCHING_ENABLED", "false")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .process_group(0)
        .spawn()
        .expect("spawn embed-server");
    (child, format!("http://127.0.0.1:{port}"))
}

fn rerank(base: &str, normalize: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": common::RERANK_NAME,
        "query": "what is onnx",
        "documents": [
            "onnx is an open format for machine learning models",
            "paris is the capital of france",
            "a short one"
        ],
    });
    if let Some(n) = normalize {
        body["normalize"] = n.into();
    }
    let resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
        .post(format!("{base}/v1/rerank"))
        .json(&body)
        .send()
        .expect("POST /v1/rerank");
    assert!(resp.status().is_success(), "status {}", resp.status());
    resp.json().expect("response json")
}

/// `index -> relevance_score`, sorted by index.
fn by_index(v: &serde_json::Value) -> Vec<f64> {
    let mut rows: Vec<(u64, f64)> = v["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|r| {
            (
                r["index"].as_u64().unwrap(),
                r["relevance_score"].as_f64().unwrap(),
            )
        })
        .collect();
    rows.sort_by_key(|r| r.0);
    rows.into_iter().map(|r| r.1).collect()
}

/// Shared body of the two path tests.
fn assert_calibrated(tag: &str, multi_process: bool) {
    let Some(env) = common::setup() else {
        return;
    };
    let dir = rerank_dir(&env, tag, Some(&shift_file()));
    let (child, base) = spawn(&env, tag, multi_process, &dir);
    let _guard = common::ChildGuard {
        child: Some(child),
        socket: dir.join("__noop__"),
    };
    common::wait_healthy(&base, Duration::from_secs(90));

    let raw = rerank(&base, Some("none"));
    assert!(
        raw.get("calibration").is_none(),
        "normalize=none applies no calibration, so it must not claim one: {raw}"
    );
    let raw = by_index(&raw);

    let cal = rerank(&base, None);
    assert_eq!(
        cal["calibration"], CAL_ID,
        "the response must name the applied calibration: {cal}"
    );
    let cal = by_index(&cal);

    for (i, (r, c)) in raw.iter().zip(&cal).enumerate() {
        let want = sigmoid(r + SHIFT);
        assert!(
            (c - want).abs() < 1e-4,
            "doc {i}: raw logit {r} -> expected sigmoid(raw+{SHIFT})={want}, served {c}"
        );
        assert!(
            (want - sigmoid(*r)).abs() > 1e-3,
            "doc {i}: fixture logit {r} too extreme, the shift is not observable"
        );
    }
}

#[test]
fn calibration_applied_in_process_path() {
    assert_calibrated("cal-inproc", false);
}

#[test]
fn calibration_applied_worker_path() {
    assert_calibrated("cal-worker", true);
}

#[test]
fn missing_calibration_file_serves_identity() {
    let Some(env) = common::setup() else {
        return;
    };
    let dir = rerank_dir(&env, "cal-none", None);
    let (child, base) = spawn(&env, "cal-none", true, &dir);
    let _guard = common::ChildGuard {
        child: Some(child),
        socket: dir.join("__noop__"),
    };
    common::wait_healthy(&base, Duration::from_secs(90));

    let raw = by_index(&rerank(&base, Some("none")));
    let served = rerank(&base, None);
    assert!(
        served.get("calibration").is_none(),
        "no file, no calibration field: {served}"
    );
    for (i, (r, c)) in raw.iter().zip(by_index(&served)).enumerate() {
        assert!(
            (c - sigmoid(*r)).abs() < 1e-4,
            "doc {i}: identity expected sigmoid({r}), served {c}"
        );
    }
}

#[test]
fn invalid_calibration_file_fails_startup() {
    let Some(env) = common::setup() else {
        return;
    };
    // Valid JSON, but knots_y decreases: a monotone map is the contract.
    let bad = r#"{"schema":1,"id":"bad","knots_x":[0.0,1.0],"knots_y":[1.0,0.0]}"#;
    for (tag, multi_process) in [("cal-bad-inproc", false), ("cal-bad-worker", true)] {
        let dir = rerank_dir(&env, tag, Some(bad));
        let (mut child, _base) = spawn(&env, tag, multi_process, &dir);
        let deadline = Instant::now() + Duration::from_secs(60);
        let status = loop {
            if let Some(s) = child.try_wait().expect("try_wait") {
                break s;
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("{tag}: server kept running with an invalid calibration.json");
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        assert!(
            !status.success(),
            "{tag}: startup must fail on an invalid calibration.json, got {status}"
        );
    }
}
