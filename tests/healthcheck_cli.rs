//! `embed-server healthcheck` end to end: the real binary, the real exit code.
//!
//! The unit tests in `src/healthcheck.rs` cover `probe`; this covers the
//! subcommand wiring (`EMBED_PORT`, exit status), which is what the container
//! `HEALTHCHECK` actually runs.
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;

/// Serve one connection with `reply`, after reading the whole request head.
fn serve_once(reply: &'static str) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = l.accept() {
            let mut head = Vec::new();
            let mut buf = [0u8; 256];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(k) => head.extend_from_slice(&buf[..k]),
                }
            }
            let _ = s.write_all(reply.as_bytes());
        }
    });
    port
}

fn healthcheck_exit(port: u16) -> i32 {
    Command::new(env!("CARGO_BIN_EXE_embed-server"))
        .arg("healthcheck")
        .env("EMBED_PORT", port.to_string())
        .env_remove("EMBED_READY_PROBE_TIMEOUT_MS")
        .status()
        .expect("spawn embed-server healthcheck")
        .code()
        .expect("exit code")
}

#[test]
fn exits_zero_when_ready_answers_200() {
    let port = serve_once("HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n");
    assert_eq!(healthcheck_exit(port), 0);
}

#[test]
fn exits_one_when_ready_answers_503() {
    let port = serve_once("HTTP/1.1 503 Service Unavailable\r\n\r\n");
    assert_eq!(healthcheck_exit(port), 1);
}

#[test]
fn exits_one_when_nothing_listens() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    assert_eq!(healthcheck_exit(port), 1);
}
