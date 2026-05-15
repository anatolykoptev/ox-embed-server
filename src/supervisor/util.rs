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
