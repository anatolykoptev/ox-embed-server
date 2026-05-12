use embed_server::ipc::frame::{read_frame, write_frame};
use embed_server::ipc::protocol::ControlMessage;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::UnixStream;

#[tokio::test]
async fn supervisor_pings_worker() {
    let socket = format!("/home/krolik/embed-worker-test-{}.sock", std::process::id());
    let _ = std::fs::remove_file(&socket);

    let worker_bin = env!("CARGO_BIN_EXE_embed-worker");
    let mut child = Command::new(worker_bin)
        .env("EMBED_WORKER_MODEL", "test-model")
        .env("EMBED_WORKER_SOCKET", &socket)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn worker");

    // Wait for socket to appear (worker startup race).
    for _ in 0..50 {
        if std::path::Path::new(&socket).exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        std::path::Path::new(&socket).exists(),
        "worker did not create socket"
    );

    let mut conn = UnixStream::connect(&socket).await.expect("connect");
    write_frame(&mut conn, &ControlMessage::Ping).await.unwrap();
    let reply: ControlMessage = read_frame(&mut conn).await.unwrap();
    assert_eq!(reply, ControlMessage::Pong);

    write_frame(&mut conn, &ControlMessage::Shutdown)
        .await
        .unwrap();
    let _ = child.wait();
    let _ = std::fs::remove_file(&socket);
}
