//! Shared test helpers — used across integration tests.
//!
//! # Fixtures
//!
//! The model-backed tests run against three tiny BERT encoders committed under
//! `tests/fixtures/tiny/` (see the README there) and a pinned ONNX Runtime
//! library (`scripts/fetch-ort.sh`). The models carry random weights, so tests
//! assert shape, determinism and batch independence, never relevance.
//!
//! A test that cannot find its inputs calls [`setup`], which prints `SKIP` on a
//! developer machine and **panics when `CI=true`**: a missing fixture or
//! runtime on CI is a broken pipeline, not a reason to pass.
#![allow(dead_code)]
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Name the embed fixture is registered under.
pub const EMBED_NAME: &str = "tiny-embed";
/// Hidden size of the embed fixture (its output dimension).
pub const EMBED_DIM: usize = 32;
pub const RERANK_NAME: &str = "tiny-rerank";
pub const SPLADE_NAME: &str = "tiny-splade";
/// Token cap handed to every fixture model. The fixtures allow 128 positions.
pub const MAX_LEN: usize = 64;

/// RAII guard for a spawned child process bound to a UDS socket.
/// On drop: kill child, wait for exit, remove socket file.
pub struct ChildGuard {
    pub child: Option<Child>,
    pub socket: PathBuf,
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// `true` under GitHub Actions and any other runner that exports `CI=true`.
pub fn on_ci() -> bool {
    std::env::var("CI").is_ok_and(|v| v == "true")
}

/// Resolved fixture inputs for one test.
pub struct Setup {
    /// `libonnxruntime` path, already checked to exist.
    pub ort: PathBuf,
    /// Directory holding `embed/`, `rerank/` and `splade/`.
    pub root: PathBuf,
}

/// Locate the ONNX Runtime library and the fixture models.
///
/// `EMBED_TEST_FIXTURES` overrides the fixture root (default: the committed
/// `tests/fixtures/tiny`). Anything missing → `None` plus a `SKIP` line off CI,
/// and a panic on CI.
pub fn setup() -> Option<Setup> {
    let root = std::env::var_os("EMBED_TEST_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny"));
    let ort = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    let mut missing: Vec<String> = Vec::new();
    if ort.is_empty() || !Path::new(&ort).is_file() {
        missing.push(format!(
            "ORT_DYLIB_PATH not set or missing (got {ort:?}); run scripts/fetch-ort.sh"
        ));
    }
    for rel in [
        "embed/model.onnx",
        "embed/tokenizer.json",
        "rerank/model_quantized.onnx",
        "rerank/tokenizer.json",
        "splade/model.onnx",
        "splade/tokenizer.json",
    ] {
        if !root.join(rel).is_file() {
            missing.push(format!("fixture {} missing", root.join(rel).display()));
        }
    }
    if missing.is_empty() {
        return Some(Setup {
            ort: PathBuf::from(ort),
            root,
        });
    }
    let why = missing.join("; ");
    assert!(
        !on_ci(),
        "CI=true and the model-test inputs are unavailable: {why}. These tests must run on CI, not skip."
    );
    eprintln!("SKIP: {why}");
    None
}

impl Setup {
    /// `EMBED_MODELS` value: `name:dir:dim:max_len:pad_id:has_tti:onnx`.
    pub fn embed_spec(&self) -> String {
        format!(
            "{EMBED_NAME}:{}:{EMBED_DIM}:{MAX_LEN}:0:true:model.onnx",
            self.root.join("embed").display()
        )
    }
    /// `RERANKER_MODELS` value: `name:dir:max_len:padded`.
    pub fn rerank_spec(&self) -> String {
        format!(
            "{RERANK_NAME}:{}:{MAX_LEN}:true",
            self.root.join("rerank").display()
        )
    }
    /// `SPLADE_MODELS` value: `name:dir:max_len`.
    pub fn splade_spec(&self) -> String {
        format!(
            "{SPLADE_NAME}:{}:{MAX_LEN}",
            self.root.join("splade").display()
        )
    }
}

/// A free TCP port on loopback. Racy by nature (the port is released before
/// the server binds it) but far less collision-prone than `pid % 10`.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("bind an ephemeral port")
        .port()
}

/// Block until `GET {base}/health` succeeds. Panics on timeout — a server that
/// never becomes healthy is a failure, never a skip.
pub fn wait_healthy(base: &str, timeout: Duration) {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("build reqwest client");
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if client
            .get(format!("{base}/health"))
            .send()
            .is_ok_and(|r| r.status().is_success())
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    panic!("server at {base} not healthy within {timeout:?}");
}

/// Largest absolute element-wise difference.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Six probe texts with token counts (incl. [CLS]/[SEP]) of about 4, 5, 8, 14,
/// 34 and 35. Every word is in the fixture vocabulary, so a word is one token.
///
/// Indices: 0,1 short; 2,3 mid; 4,5 long. With the worker's default sub-batch
/// ratio of 2.0, the pair (2,3) shares one sub-batch whose width is set by the
/// longer text (8 alone vs 16 batched), while shorts and longs always split.
pub fn probe_texts() -> Vec<String> {
    let long =
        "what is the capital of france and what is an open format for machine learning models ";
    vec![
        "hello world".into(),
        "hello world hello".into(),
        "what is the capital of france".into(),
        "what is an open format for machine learning models on onnx runtime".into(),
        long.repeat(2),
        format!("{}onnx", long.repeat(2)),
    ]
}

/// Batches of indices into [`probe_texts`]. Together they cover: a split into
/// two single-text sub-batches in both orders; a shared sub-batch padded wider
/// than one member's own width; and two multi-text sub-batches whose rows must
/// scatter back to the caller's positions.
pub const PROBE_BATCHES: &[&[usize]] = &[
    &[0, 4],
    &[4, 0],
    &[2, 3],
    &[3, 2],
    &[0, 4, 1, 5],
    &[5, 1, 4, 0],
];

/// A running multi-process `embed-server` with all three fixture model kinds
/// loaded. Dropping it kills the supervisor (which takes its workers along)
/// and removes the socket directory.
pub struct Server {
    pub base: String,
    _guard: ChildGuard,
    socket_dir: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.socket_dir);
    }
}

/// Spawn `embed-server` in multi-process mode on a free port and wait for it
/// to become healthy. `tag` keeps the socket directory unique per test binary.
pub fn spawn_server(env: &Setup, tag: &str, server_bin: &str, worker_bin: &str) -> Server {
    let port = free_port();
    let socket_dir = std::env::temp_dir().join(format!("embed-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&socket_dir);
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let child = Command::new(server_bin)
        .env("EMBED_MULTI_PROCESS", "1")
        .env("EMBED_WORKER_BIN", worker_bin)
        .env("EMBED_WORKER_SOCKET_DIR", &socket_dir)
        .env("EMBED_PORT", port.to_string())
        .env("EMBED_MODELS", env.embed_spec())
        .env("EMBED_DEFAULT_MODEL", EMBED_NAME)
        .env("RERANKER_MODELS", env.rerank_spec())
        .env("SPLADE_MODELS", env.splade_spec())
        .env("ORT_DYLIB_PATH", &env.ort)
        // The supervisor's own batcher is off in multi-process mode anyway;
        // keep the worker path the only thing under test.
        .env("BATCHING_ENABLED", "false")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn embed-server");
    let guard = ChildGuard {
        child: Some(child),
        socket: socket_dir.join("__guard_noop__"),
    };
    let base = format!("http://127.0.0.1:{port}");
    wait_healthy(&base, Duration::from_secs(90));
    Server {
        base,
        _guard: guard,
        socket_dir,
    }
}
