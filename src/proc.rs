//! `/proc/<pid>/status` helpers — per-worker RSS sampling on Linux.
//!
//! Used by the supervisor to populate `embed_worker_rss_bytes{model}` every 15 s.
//! Gracefully degrades on non-Linux targets: [`read_proc_status_rss`] returns 0
//! and logs a `Debug`-level message once; no crash.

/// Parse a VmRSS line from `/proc/<pid>/status` content and return bytes.
///
/// Accepts the full file content; tolerates any field ordering. Returns
/// `None` if no `VmRSS:` line is found or if the value cannot be parsed.
///
/// Units: the kernel always emits `kB` for VmRSS. We multiply by 1024 to
/// return bytes so callers never have to carry the unit conversion.
pub fn parse_vmrss_bytes(status_content: &str) -> Option<u64> {
    for line in status_content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Read `/proc/<pid>/status` and return the RSS in bytes.
///
/// Linux-only: on other targets returns `Ok(0)` and emits a single
/// `tracing::debug!` to avoid crashing dev/CI environments on macOS/Windows.
///
/// Errors: any `std::io::Error` from the file read (e.g. ENOENT for a
/// dead worker whose PID has been recycled) propagates to the caller.
/// The caller (supervisor RSS-poll loop) logs at `warn` level and skips
/// updating the gauge for that worker until the next tick.
pub fn read_proc_status_rss(pid: u32) -> Result<u64, std::io::Error> {
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string(format!("/proc/{}/status", pid))?;
        parse_vmrss_bytes(&content).ok_or_else(|| {
            std::io::Error::other(format!("VmRSS not found in /proc/{}/status", pid))
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        tracing::debug!(
            pid,
            "read_proc_status_rss: non-Linux host, returning 0 (proc not available)"
        );
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── pure parser tests ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_vmrss_synthetic_known_value() {
        // Synthetic /proc/self/status snippet — kernel always uses `kB` unit.
        let content = "\
Name:\tsleep\n\
VmPeak:\t   4096 kB\n\
VmRSS:\t   3702984 kB\n\
VmSwap:\t        0 kB\n\
";
        let bytes = parse_vmrss_bytes(content).expect("should parse");
        assert_eq!(bytes, 3_702_984 * 1024, "expected kB → bytes conversion");
    }

    #[test]
    fn test_parse_vmrss_missing_field_returns_none() {
        let content = "Name:\tfoo\nVmPeak:\t1024 kB\n";
        assert!(
            parse_vmrss_bytes(content).is_none(),
            "no VmRSS line should yield None"
        );
    }

    #[test]
    fn test_parse_vmrss_malformed_value_returns_none() {
        let content = "VmRSS:\tnotanumber kB\n";
        assert!(
            parse_vmrss_bytes(content).is_none(),
            "non-numeric value should yield None"
        );
    }

    #[test]
    fn test_parse_vmrss_zero_is_valid() {
        // Some zombie / lightweight threads report VmRSS: 0 kB — must not error.
        let content = "VmRSS:\t0 kB\n";
        assert_eq!(parse_vmrss_bytes(content), Some(0));
    }

    #[test]
    fn test_parse_vmrss_tabs_and_spaces() {
        // Kernel may pad with multiple spaces; strip_prefix + trim handles it.
        let content = "VmRSS:         8192 kB\n";
        assert_eq!(parse_vmrss_bytes(content), Some(8192 * 1024));
    }

    // ── IO-path tests ─────────────────────────────────────────────────────────

    /// Non-existent PID → ENOENT (or ESRCH on some kernels). Either way, Err.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_read_proc_status_rss_missing_pid() {
        // PID 999_999_999 will never exist.
        let result = read_proc_status_rss(999_999_999);
        assert!(result.is_err(), "missing PID must return Err");
    }

    /// Self-RSS must be > 0 on a live process.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_parse_vmrss_real_file() {
        let rss = read_proc_status_rss(std::process::id())
            .expect("reading own /proc/self/status must succeed");
        assert!(rss > 0, "own process RSS must be > 0 bytes; got {rss}");
    }
}
