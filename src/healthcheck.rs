//! `embed-server healthcheck`: container probe without curl in the image.
//!
//! Asks the local server's `/ready` endpoint and exits 0 on HTTP 200, 1
//! otherwise. `/ready` is the readiness probe: every worker has a live
//! connection and the default model answers a real inference, so a
//! supervisor that is alive but unable to serve shows up as `unhealthy`
//! instead of plain `Up`. Plain std networking, no async runtime and no HTTP
//! client dependency.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

/// Probe budget; the `/ready` inference itself is capped at
/// `EMBED_READY_PROBE_TIMEOUT_MS` (2 s by default).
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// GET `path` on `addr`; `Ok(())` only for a 200.
pub fn probe(addr: SocketAddr, path: &str, timeout: Duration) -> Result<(), String> {
    let mut stream =
        TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("connect {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|()| stream.set_write_timeout(Some(timeout)))
        .map_err(|e| format!("socket timeout: {e}"))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .map_err(|e| format!("write: {e}"))?;
    // The status line is all that matters; 64 bytes is plenty.
    let mut head = [0u8; 64];
    let mut n = 0;
    while n < head.len() {
        match stream.read(&mut head[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) => return Err(format!("read: {e}")),
        }
        if head[..n].contains(&b'\n') {
            break;
        }
    }
    let line = String::from_utf8_lossy(&head[..n]);
    let status = line.split_whitespace().nth(1).unwrap_or("");
    if status == "200" {
        Ok(())
    } else {
        Err(format!(
            "{path} answered {:?}",
            line.lines().next().unwrap_or("")
        ))
    }
}

/// Entry point for the `healthcheck` subcommand; returns the process exit code.
pub fn run() -> i32 {
    let port: u16 = std::env::var("EMBED_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(8082);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    match probe(addr, "/ready", PROBE_TIMEOUT) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("healthcheck failed: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// One-shot server answering `reply` verbatim (or nothing, when `None`,
    /// holding the connection open).
    fn serve(reply: Option<&'static str>) -> SocketAddr {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = l.accept() {
                let mut buf = [0u8; 256];
                let _ = s.read(&mut buf);
                match reply {
                    Some(r) => {
                        let _ = s.write_all(r.as_bytes());
                    }
                    None => std::thread::sleep(Duration::from_secs(3)),
                }
            }
        });
        addr
    }

    #[test]
    fn ok_on_200() {
        let a = serve(Some("HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n"));
        assert!(probe(a, "/ready", Duration::from_secs(2)).is_ok());
    }

    #[test]
    fn fails_on_503() {
        let a = serve(Some("HTTP/1.1 503 Service Unavailable\r\n\r\n"));
        let e = probe(a, "/ready", Duration::from_secs(2)).unwrap_err();
        assert!(e.contains("503"), "{e}");
    }

    #[test]
    fn fails_on_connection_refused() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        assert!(probe(a, "/ready", Duration::from_secs(2)).is_err());
    }

    #[test]
    fn fails_when_server_never_answers() {
        let a = serve(None);
        assert!(probe(a, "/ready", Duration::from_millis(300)).is_err());
    }

    #[test]
    fn fails_on_empty_reply() {
        let a = serve(Some(""));
        assert!(probe(a, "/ready", Duration::from_secs(2)).is_err());
    }
}
