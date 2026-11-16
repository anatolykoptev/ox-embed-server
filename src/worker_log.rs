//! Worker-process log setup.
//!
//! The supervisor pipes each worker's stdout/stderr into its own streams with
//! a `[<model>] ` prefix (`supervisor::handle::forward_lines`), so what a
//! worker prints is what `docker logs` shows. That only works if the worker
//! prints anything: `tracing_subscriber::fmt::init()` with the `env-filter`
//! feature and no `RUST_LOG` filters at ERROR, so every `worker starting`,
//! mlock and arena-registration `warn!` line was dropped before it reached the
//! pipe. The supervisor's own default is `info`; this gives workers the same
//! baseline.

use tracing_subscriber::EnvFilter;

/// Filter used when `RUST_LOG` is unset, empty or unparsable: `info`, with the
/// ORT bridge target held at `warn` so per-allocation arena chatter cannot
/// flood the pipe (the worker's stdout is a 64 KiB pipe the supervisor drains).
pub const DEFAULT_WORKER_LOG_FILTER: &str = "info,ort::logging=warn";

/// Build the worker's log filter from the raw `RUST_LOG` value.
///
/// Takes the value as an argument (not from the environment) so it can be
/// tested without mutating process-global state.
pub fn worker_env_filter(rust_log: Option<&str>) -> EnvFilter {
    rust_log
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| EnvFilter::try_new(s).ok())
        .unwrap_or_else(|| EnvFilter::new(DEFAULT_WORKER_LOG_FILTER))
}

/// Install the worker's global subscriber: plain text (no ANSI, stdout is a
/// pipe, not a terminal) filtered by [`worker_env_filter`].
pub fn init() {
    tracing_subscriber::fmt()
        .with_env_filter(worker_env_filter(std::env::var("RUST_LOG").ok().as_deref()))
        .with_ansi(false)
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Capture {
        type Writer = Capture;
        fn make_writer(&'a self) -> Capture {
            self.clone()
        }
    }

    /// Emit one info and one error line under the filter built from
    /// `rust_log`, return what the subscriber wrote.
    fn emit(rust_log: Option<&str>) -> String {
        let cap = Capture::default();
        let sub = tracing_subscriber::fmt()
            .with_env_filter(worker_env_filter(rust_log))
            .with_ansi(false)
            .with_writer(cap.clone())
            .finish();
        tracing::subscriber::with_default(sub, || {
            tracing::info!("worker starting");
            tracing::error!("arena registration failed");
        });
        let bytes = cap.0.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }

    /// With no `RUST_LOG`, an INFO line must be emitted (it was dropped when
    /// the worker used `fmt::init()`, whose env-filter default is ERROR).
    #[test]
    fn unset_rust_log_emits_info_lines() {
        let out = emit(None);
        assert!(
            out.contains("worker starting"),
            "info line must pass the default worker filter; got: {out:?}"
        );
        assert!(out.contains("arena registration failed"));
        assert!(!out.contains('\u{1b}'), "no ANSI escapes in piped output");
    }

    #[test]
    fn empty_or_garbage_rust_log_falls_back_to_default() {
        for v in [Some(""), Some("   "), Some("[[not a directive")] {
            let out = emit(v);
            assert!(
                out.contains("worker starting"),
                "RUST_LOG={v:?} must fall back to the info default; got: {out:?}"
            );
        }
    }

    /// An explicit `RUST_LOG` still wins over the default.
    #[test]
    fn explicit_rust_log_overrides_default() {
        let out = emit(Some("error"));
        assert!(!out.contains("worker starting"), "got: {out:?}");
        assert!(out.contains("arena registration failed"));
    }
}
