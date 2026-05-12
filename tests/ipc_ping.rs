#[path = "common/mod.rs"]
mod common;
use common::ChildGuard;

use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::ControlMessage;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

#[tokio::test]
#[ignore = "Wave 2 will reintroduce control channel; worker now speaks InferRequest only"]
async fn supervisor_pings_worker() {
    let socket: PathBuf =
        std::env::temp_dir().join(format!("embed-worker-test-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "test-model")
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

    let mut conn = UnixStream::connect(&socket).await.expect("connect");
    write_frame(&mut conn, &ControlMessage::Ping).await.unwrap();
    let reply: ControlMessage = read_frame(&mut conn).await.unwrap();
    assert_eq!(reply, ControlMessage::Pong);

    write_frame(&mut conn, &ControlMessage::Shutdown)
        .await
        .unwrap();
    if let Some(mut c) = guard.child.take() {
        let _ = c.wait();
    }
    // guard Drop still cleans socket
}
