//! `embed-server healthcheck`: container probe without curl in the image.
//!
//! Asks the local server's `/ready` endpoint and exits 0 on HTTP 200, 1
//! otherwise. `/ready` is the readiness probe: every worker has a live
//! connection and the default model answers a real inference, so a
//! supervisor that is alive but unable to serve shows up as `unhealthy`
//! instead of plain `Up`. Plain blocking std networking and no HTTP client
//! dependency; `run()` is a one-shot called from `main` before the server or
//! any logging starts, so blocking inside the async `main` task is harmless.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

/// Server-side `/ready` inference cap when `EMBED_READY_PROBE_TIMEOUT_MS` is
/// unset or invalid (same default as `Config`).
const DEFAULT_READY_PROBE_MS: u64 = 2000;
/// Slack on top of the server's own `/ready` cap: connect, queueing, response.
const PROBE_MARGIN: Duration = Duration::from_secs(3);

/// Healthcheck budget: the server's `/ready` cap plus [`PROBE_MARGIN`], so a
/// healthy server that legitimately uses its whole cap does not fail its own
/// probe. The container's `HEALTHCHECK --timeout` (Dockerfile) must stay above
/// this: with the 2 s default it is 5 s against 8 s; raise it together with
/// `EMBED_READY_PROBE_TIMEOUT_MS` beyond about 5 s.
fn probe_budget(ready_probe_ms: Option<&str>) -> Duration {
    let ms = ready_probe_ms
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .unwrap_or(DEFAULT_READY_PROBE_MS);
    Duration::from_millis(ms).saturating_add(PROBE_MARGIN)
}

/// GET `path` on `addr`; `Ok(())` only for a 200.
pub fn probe(addr: SocketAddr, path: &str, timeout: Duration) -> Result<(), String> {
    let mut stream =
        TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("connect {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|()| stream.set_write_timeout(Some(timeout)))
        .map_err(|e| format!("socket timeout: {e}"))?;
    // One buffer, one write: the request leaves in a single segment.
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    // The status line is all that matters; 64 bytes is plenty.
    let mut head = Vec::new();
    BufReader::new(stream.take(64))
        .read_until(b'\n', &mut head)
        .map_err(|e| format!("read: {e}"))?;
    let line = String::from_utf8_lossy(&head);
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
    // env-raw: the probe runs in a bare subprocess with no metrics; an unusable port falls back to 8082 by design
    let port: u16 = std::env::var("EMBED_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(8082);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let budget = probe_budget(
        // env-raw: parsed and clamped by probe_budget; the probe subprocess has no metrics sink
        std::env::var("EMBED_READY_PROBE_TIMEOUT_MS")
            .ok()
            .as_deref(),
    );
    match probe(addr, "/ready", budget) {
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
                // Read the whole request head before replying: answering and
                // dropping a socket with unread bytes makes the kernel send RST,
                // which can overtake the reply at the client.
                let mut head = Vec::new();
                let mut buf = [0u8; 256];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                    match s.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(k) => head.extend_from_slice(&buf[..k]),
                    }
                }
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
    fn probe_budget_adds_margin_to_the_ready_cap() {
        assert_eq!(
            probe_budget(None),
            Duration::from_millis(2000) + PROBE_MARGIN
        );
        assert_eq!(
            probe_budget(Some("4000")),
            Duration::from_millis(4000) + PROBE_MARGIN
        );
        assert_eq!(
            probe_budget(Some(" 1500 ")),
            Duration::from_millis(1500) + PROBE_MARGIN
        );
        // 0 and garbage fall back to the server's default, like Config does.
        assert_eq!(probe_budget(Some("0")), probe_budget(None));
        assert_eq!(probe_budget(Some("soon")), probe_budget(None));
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
