//! The real `embed-worker` binary must print its info-level lines.
//!
//! `fmt::init()` with the `env-filter` feature filters at ERROR when `RUST_LOG`
//! is unset, so the `worker starting` line (and every mlock / arena warning)
//! never reached the pipe the supervisor forwards to `docker logs`. No model
//! files are needed: the worker logs the line before it loads anything, then
//! fails on the missing model directory and exits.

use std::process::{Command, Stdio};

#[test]
fn worker_prints_info_lines_without_rust_log() {
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_embed-worker"))
        .env_remove("RUST_LOG")
        .env("EMBED_WORKER_MODEL", "log-probe-model")
        .env("EMBED_WORKER_KIND", "embed")
        .env("EMBED_WORKER_SOCKET", dir.path().join("w.sock"))
        .stdin(Stdio::null())
        .output()
        .expect("run embed-worker");
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        all.contains("worker starting"),
        "worker printed no info-level line without RUST_LOG; output:\n{all}"
    );
}
