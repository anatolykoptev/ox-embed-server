//! Wave 2.4b: ControlMessage removed from protocol (embed/rerank/splade enum
//! supersedes it). This test is kept as a placeholder for when a dedicated
//! control channel is reintroduced (Wave 2.5b heartbeat or similar).

#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

#[tokio::test]
#[ignore = "ControlMessage removed in Wave 2.4b; will be reintroduced in Wave 2.5b heartbeat"]
async fn supervisor_pings_worker() {
    let socket: PathBuf =
        std::env::temp_dir().join(format!("embed-worker-test-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "test-model")
        .env("EMBED_WORKER_KIND", "embed")
        .env("EMBED_WORKER_SOCKET", &socket)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn worker");

    let mut guard = ChildGuard {
        child: Some(child),
        socket: socket.clone(),
    };

    for _ in 0..50 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(socket.exists(), "worker did not create socket");

    // Control channel placeholder — implement when Wave 2.5b heartbeat lands.
    let _conn: UnixStream = UnixStream::connect(&socket).await.expect("connect");
    if let Some(mut c) = guard.child.take() {
        let _ = c.wait();
    }
}
