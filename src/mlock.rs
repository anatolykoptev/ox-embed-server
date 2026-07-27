//! mlock helper — pin model weight bytes into RAM.
//!
//! Reads an ONNX file into a `Vec<u8>`, calls `mlock(2)` on the buffer so
//! the kernel cannot swap those pages out, and returns a [`MlockedBuf`] RAII
//! guard that calls `munlock(2)` on drop so pages are properly unlocked
//! before the allocator can reuse them.
//!
//! # Why this matters
//!
//! Under host RAM pressure (rustc, claude-server, other containers competing)
//! the kernel swaps out idle workers' weight pages. The next inference request
//! pays a page-fault storm that can take 60-90 s on a cold 1.4 GiB swap-in.
//! `mlock` keeps 1.8 GiB of ONNX bytes resident so:
//!   1. Those pages are unswappable — the kernel must swap OTHER anonymous
//!      memory first, reducing overall pressure on ORT's runtime tensors.
//!   2. File bytes stay warm in RAM → fast re-loads after idle-eviction.
//!
//! Note: `commit_from_memory` causes ORT to copy weights into its own heap
//! allocation (via `CreateSessionFromArray`). The mlocked buffer is NOT the
//! memory ORT inferences against. The benefit is indirect — see above.
//!
//! # ulimit requirement
//!
//! `mlock(2)` is governed by `RLIMIT_MEMLOCK`. The default container value
//! is 8 MiB — far too small. The compose service must set:
//!
//! ```yaml
//! ulimits:
//!   memlock:
//!     soft: 2147483648   # 2 GiB — covers all workers combined
//!     hard: 2147483648
//! ```
//!
//! When the limit is not raised, `mlock` returns `ENOMEM`. This module logs
//! a warning and continues without locking (the server is still correct but
//! weights may be swapped under pressure).
//!
//! # Opt-out
//!
//! Set `EMBED_MLOCK_WEIGHTS=0` to skip mlock entirely (useful in dev
//! containers or when privileges are unavailable).

use std::path::Path;

use ort::session::Session;

/// RAII guard around a heap buffer that may have been mlocked.
///
/// On drop, calls `munlock(2)` when the lock was successfully acquired so
/// pages are properly unlocked before the allocator can reuse them.
pub struct MlockedBuf {
    buf: Vec<u8>,
    /// `true` when `mlock` returned 0 (success). `false` when mlock was
    /// skipped (disabled via env, non-Linux, or RLIMIT/permission failure).
    locked: bool,
}

impl MlockedBuf {
    /// Returns the bytes as a slice (for `commit_from_memory`).
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Returns the buffer length in bytes.
    // An mlocked buffer is always non-empty by construction (it wraps a sized
    // allocation), so a paired `is_empty()` would be dead weight; suppress the
    // lint rather than add an unused accessor.
    #[allow(dead_code, clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

impl Drop for MlockedBuf {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if self.locked && !self.buf.is_empty() {
            // SAFETY: ptr/len are consistent with the live allocation.
            // `munlock` is idempotent and thread-safe.
            unsafe {
                libc::munlock(self.buf.as_ptr() as *const libc::c_void, self.buf.len());
            }
        }
    }
}

// SAFETY: `MlockedBuf` wraps `Vec<u8>` which is Send.
// The locked flag tracks only our own allocation state.
unsafe impl Send for MlockedBuf {}
unsafe impl Sync for MlockedBuf {}

/// An ORT [`Session`] paired with the mlocked buffer used to build it.
///
/// Keeping `_buf` alive alongside the session ensures that mlock'd pages
/// remain resident for the session's lifetime. When the session is dropped
/// (eviction or shutdown), `_buf` is dropped first, calling `munlock`
/// before the allocator can reuse those pages.
pub struct MlockedSession {
    pub session: Session,
    /// Mlocked weight buffer. Never accessed after construction — held for
    /// RAII drop ordering (buf outlives session within this struct).
    _buf: MlockedBuf,
}

impl MlockedSession {
    pub fn new(session: Session, buf: MlockedBuf) -> Self {
        Self { session, _buf: buf }
    }

    /// Create a session that was loaded via `commit_from_file` (e.g. models
    /// with external-data sibling files). The mlock optimisation does not
    /// apply — ORT holds the weights in its own heap allocation rather than
    /// in our buffer — so `_buf` is left empty.
    pub fn new_without_mlock(session: Session) -> Self {
        Self {
            session,
            _buf: MlockedBuf {
                buf: vec![],
                locked: false,
            },
        }
    }
}

impl std::ops::Deref for MlockedSession {
    type Target = Session;
    fn deref(&self) -> &Session {
        &self.session
    }
}

impl std::ops::DerefMut for MlockedSession {
    fn deref_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

/// Load `path` into a heap-allocated [`MlockedBuf`] and attempt to `mlock`
/// the buffer so the kernel cannot swap those pages out.
///
/// Returns `Ok(buf)` whether or not mlock succeeded — a failed mlock only
/// emits a tracing warning; the caller still receives the bytes and can
/// pass them to ORT via `commit_from_memory`.
///
/// Returns `Err` only when the file cannot be read.
pub fn read_and_mlock(path: &Path) -> Result<MlockedBuf, String> {
    let buf = std::fs::read(path).map_err(|e| format!("read ONNX {}: {e}", path.display()))?;

    if !mlock_enabled() {
        return Ok(MlockedBuf { buf, locked: false });
    }

    let locked = try_mlock(&buf, path);
    Ok(MlockedBuf { buf, locked })
}

/// Attempt `mlock` on `buf`; returns `true` on success, `false` on failure.
/// Logs info on success, warn on failure. Always returns `false` on
/// non-Linux platforms.
fn try_mlock(buf: &[u8], path: &Path) -> bool {
    #[cfg(target_os = "linux")]
    {
        if buf.is_empty() {
            return false;
        }
        // SAFETY: `buf` is a valid heap allocation; ptr and len are consistent.
        // `mlock` is thread-safe (operates on kernel page table entries,
        // not Rust-level shared state).
        let rc = unsafe { libc::mlock(buf.as_ptr() as *const libc::c_void, buf.len()) };
        if rc != 0 {
            let errno = unsafe { *libc::__errno_location() };
            let hint = match errno {
                libc::ENOMEM => "raise ulimits.memlock in compose to ≥2 GiB \
                                  (soft: 2147483648 / hard: 2147483648) — \
                                  set EMBED_MLOCK_WEIGHTS=0 to suppress this warning"
                    .to_string(),
                libc::EPERM => "need CAP_IPC_LOCK or raised RLIMIT_MEMLOCK; \
                                 set EMBED_MLOCK_WEIGHTS=0 to suppress"
                    .to_string(),
                other => format!("errno={other}"),
            };
            tracing::warn!(
                path = %path.display(),
                bytes = buf.len(),
                "EMBED_MLOCK_WEIGHTS: mlock failed — weights may be swapped under pressure: {hint}"
            );
            false
        } else {
            tracing::info!(
                path = %path.display(),
                bytes = buf.len(),
                "EMBED_MLOCK_WEIGHTS: weights mlocked in RAM"
            );
            true
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (buf, path);
        false
    }
}

/// Returns `true` unless `EMBED_MLOCK_WEIGHTS` is set to `"0"` or `"false"`.
///
/// Default is enabled (`true`) — operators opt out rather than opt in.
pub fn mlock_enabled() -> bool {
    match std::env::var("EMBED_MLOCK_WEIGHTS").as_deref() {
        Ok("0") | Ok("false") | Ok("False") | Ok("FALSE") => {
            tracing::debug!("EMBED_MLOCK_WEIGHTS=0: mlock disabled");
            false
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// `read_and_mlock` on a small file must return the correct bytes.
    /// Whether mlock itself succeeds depends on the ulimit in the test
    /// environment — we only assert the data round-trips correctly.
    #[test]
    fn read_and_mlock_returns_correct_bytes() {
        let mut f = NamedTempFile::new().expect("tempfile");
        let payload: &[u8] = b"fake-onnx-weights-payload";
        f.write_all(payload).expect("write");
        f.flush().expect("flush");

        let buf = read_and_mlock(f.path()).expect("read_and_mlock");
        assert_eq!(buf.as_slice(), payload);
        assert_eq!(buf.len(), payload.len());
    }

    /// When `EMBED_MLOCK_WEIGHTS=0`, the function must still return correct
    /// bytes (just skips the mlock syscall, `locked=false`).
    #[test]
    fn read_and_mlock_disabled_still_reads_file() {
        let mut f = NamedTempFile::new().expect("tempfile");
        let payload: &[u8] = b"opt-out-path-payload";
        f.write_all(payload).expect("write");
        f.flush().expect("flush");

        let prev = std::env::var("EMBED_MLOCK_WEIGHTS").ok();
        // SAFETY: single-threaded test; no other thread reads EMBED_MLOCK_WEIGHTS.
        unsafe { std::env::set_var("EMBED_MLOCK_WEIGHTS", "0") };
        let result = read_and_mlock(f.path());
        match prev {
            Some(v) => unsafe { std::env::set_var("EMBED_MLOCK_WEIGHTS", v) },
            None => unsafe { std::env::remove_var("EMBED_MLOCK_WEIGHTS") },
        }

        let buf = result.expect("should succeed even when mlock disabled");
        assert_eq!(buf.as_slice(), payload);
        assert!(
            !buf.locked,
            "locked must be false when EMBED_MLOCK_WEIGHTS=0"
        );
    }

    /// Missing file must return `Err`, not panic.
    #[test]
    fn read_and_mlock_missing_file_returns_err() {
        let result = read_and_mlock(Path::new("/nonexistent/path/model.onnx"));
        assert!(result.is_err(), "expected Err for missing file");
    }

    /// `MlockedBuf::drop` must not panic when `locked=false`.
    #[test]
    fn mlock_buf_drop_unlocked_is_noop() {
        let buf = MlockedBuf {
            buf: b"hello".to_vec(),
            locked: false,
        };
        drop(buf);
    }

    /// `MlockedBuf::drop` must not panic when the buffer is empty
    /// (even if mistakenly flagged as locked).
    #[test]
    fn mlock_buf_drop_empty_buf_is_noop() {
        let buf = MlockedBuf {
            buf: vec![],
            locked: true,
        };
        drop(buf);
    }
}
