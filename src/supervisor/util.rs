//! Shared helpers for supervisor modules.

use std::time::Duration;

/// Parse a u64 seconds value from an environment variable, returning a
/// [`Duration`] on success. Falls back to `default` and emits a `warn`
/// log when the variable is absent, zero, or unparseable.
///
/// Captured at startup; restart the container to pick up a new value.
///
/// `source` is a human-readable label for the default used in log messages
/// (e.g. `"SOCKET_WAIT_SECS (60)"`).
pub(crate) fn resolve_duration_secs_env(key: &str, default: Duration, source: &str) -> Duration {
    match std::env::var(key) {
        Err(_) => default,
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(n) if n > 0 => Duration::from_secs(n),
            Ok(_) => {
                tracing::warn!(
                    env_key = key,
                    value = %raw,
                    fallback = source,
                    "{key}=0 is invalid; using default"
                );
                default
            }
            Err(_) => {
                tracing::warn!(
                    env_key = key,
                    value = %raw,
                    fallback = source,
                    "{key} is not a valid u64; using default"
                );
                default
            }
        },
    }
}

/// Parse a millisecond delay from an environment variable.
///
/// Returns `Some(Duration)` for a valid positive integer (stagger enabled),
/// `None` when the variable is absent or set to `"0"` (stagger disabled),
/// and falls back to `default_ms` with a `warn` log on parse error.
///
/// Unlike [`resolve_duration_secs_env`], zero is explicitly valid here: it
/// means "no stagger" rather than "misconfiguration".
///
/// Captured at startup; restart the container to pick up a new value.
pub(crate) fn resolve_spawn_stagger_ms(key: &str, default_ms: u64) -> Option<std::time::Duration> {
    match std::env::var(key) {
        Err(_) => {
            if default_ms == 0 {
                None
            } else {
                Some(std::time::Duration::from_millis(default_ms))
            }
        }
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(n) => Some(std::time::Duration::from_millis(n)),
            Err(_) => {
                tracing::warn!(
                    env_key = key,
                    value = %raw,
                    fallback_ms = default_ms,
                    "{key} is not a valid u64; using default"
                );
                if default_ms == 0 {
                    None
                } else {
                    Some(std::time::Duration::from_millis(default_ms))
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: set an env var for the duration of one test.
    // Tests here are single-threaded (cfg(test) + serial approach via
    // std::env::set_var), which is safe in Rust < 1.82. For Rust ≥ 1.82
    // use serial_test or run with RUST_TEST_THREADS=1.
    fn with_env<F: FnOnce()>(key: &str, val: &str, f: F) {
        // SAFETY: single-threaded test context; no other thread reads this var.
        #[allow(deprecated)]
        unsafe {
            std::env::set_var(key, val)
        };
        f();
        // SAFETY: same as above.
        #[allow(deprecated)]
        unsafe {
            std::env::remove_var(key)
        };
    }

    // --- resolve_spawn_stagger_ms ---

    #[test]
    fn stagger_absent_uses_default() {
        // Env var unset → default 2000ms
        #[allow(deprecated)]
        let _ = unsafe { std::env::remove_var("EMBED_WORKER_SPAWN_DELAY_MS_TEST1") };
        let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST1", 2000);
        assert_eq!(result, Some(std::time::Duration::from_millis(2000)));
    }

    #[test]
    fn stagger_absent_zero_default_disabled() {
        // Env var unset, default=0 → None (disabled)
        #[allow(deprecated)]
        let _ = unsafe { std::env::remove_var("EMBED_WORKER_SPAWN_DELAY_MS_TEST2") };
        let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST2", 0);
        assert_eq!(result, None);
    }

    #[test]
    fn stagger_zero_disables() {
        // EMBED_WORKER_SPAWN_DELAY_MS=0 → None (disabled, not an error)
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST3", "0", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST3", 2000);
            assert_eq!(result, None);
        });
    }

    #[test]
    fn stagger_custom_override() {
        // EMBED_WORKER_SPAWN_DELAY_MS=3000 → Some(3s)
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST4", "3000", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST4", 2000);
            assert_eq!(result, Some(std::time::Duration::from_millis(3000)));
        });
    }

    #[test]
    fn stagger_invalid_falls_back_to_default() {
        // Non-numeric value → fallback to default, no panic
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST5", "notanumber", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST5", 2000);
            assert_eq!(result, Some(std::time::Duration::from_millis(2000)));
        });
    }

    #[test]
    fn stagger_invalid_zero_default_falls_back_to_none() {
        // Non-numeric + default=0 → None
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST6", "bad", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST6", 0);
            assert_eq!(result, None);
        });
    }
}
