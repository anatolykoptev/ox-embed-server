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
pub(crate) fn resolve_duration_secs_env(
    key: &'static str,
    default: Duration,
    source: &str,
) -> Duration {
    // parsed-by: resolve_duration_secs_env
    let raw = crate::env_parse::raw(key);
    let Some(text) = raw.as_deref().map(str::trim).filter(|t| !t.is_empty()) else {
        return default;
    };
    match text.parse::<u64>() {
        Ok(n) if n > 0 => Duration::from_secs(n),
        _ => {
            crate::env_parse::note_fallback(
                key,
                text,
                "a positive integer number of seconds",
                &source,
            );
            default
        }
    }
}

/// Nice value for the worker of `model`, from `EMBED_WORKER_NICE_<MODEL_KEY>`
/// (key = `config::model_env_key`, e.g. `EMBED_WORKER_NICE_JINA_CODE_V2`).
///
/// Only LOWERING priority is supported (`0..=19`): that never needs a
/// privilege, so applying it cannot fail the spawn. Unset → `0` (the worker
/// inherits the supervisor's priority, as before); negative, above 19 or
/// non-numeric → noted and `0`.
pub(crate) fn resolve_worker_nice(model: &str) -> i32 {
    let var = format!("EMBED_WORKER_NICE_{}", crate::config::model_env_key(model));
    // parsed-by: resolve_worker_nice
    let raw = crate::env_parse::raw(&var);
    crate::env_parse::parse_or(
        &var,
        raw.as_deref(),
        0i32,
        |n| (0..=19).contains(n),
        "an integer 0..=19 (higher = lower priority)",
    )
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
#[allow(dead_code)] // called from src/main.rs (bin) — lib crate can't see cross-crate call
pub(crate) fn resolve_spawn_stagger_ms(key: &str, default_ms: u64) -> Option<std::time::Duration> {
    let default = || (default_ms != 0).then(|| std::time::Duration::from_millis(default_ms));
    // parsed-by: resolve_spawn_stagger_ms
    let raw = crate::env_parse::raw(key);
    let Some(text) = raw.as_deref().map(str::trim).filter(|t| !t.is_empty()) else {
        return default();
    };
    match text.parse::<u64>() {
        Ok(0) => None,
        Ok(n) => Some(std::time::Duration::from_millis(n)),
        Err(_) => {
            crate::env_parse::note_fallback(
                key,
                text,
                "a non-negative integer number of milliseconds",
                &default_ms,
            );
            default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
    #[serial]
    fn stagger_absent_uses_default() {
        // Env var unset → default 2000ms
        #[allow(deprecated)]
        unsafe {
            std::env::remove_var("EMBED_WORKER_SPAWN_DELAY_MS_TEST1")
        };
        let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST1", 2000);
        assert_eq!(result, Some(std::time::Duration::from_millis(2000)));
    }

    #[test]
    #[serial]
    fn stagger_absent_zero_default_disabled() {
        // Env var unset, default=0 → None (disabled)
        #[allow(deprecated)]
        unsafe {
            std::env::remove_var("EMBED_WORKER_SPAWN_DELAY_MS_TEST2")
        };
        let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST2", 0);
        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn stagger_zero_disables() {
        // EMBED_WORKER_SPAWN_DELAY_MS=0 → None (disabled, not an error)
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST3", "0", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST3", 2000);
            assert_eq!(result, None);
        });
    }

    #[test]
    #[serial]
    fn stagger_custom_override() {
        // EMBED_WORKER_SPAWN_DELAY_MS=3000 → Some(3s)
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST4", "3000", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST4", 2000);
            assert_eq!(result, Some(std::time::Duration::from_millis(3000)));
        });
    }

    #[test]
    #[serial]
    fn stagger_invalid_falls_back_to_default() {
        // Non-numeric value → fallback to default, no panic
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST5", "notanumber", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST5", 2000);
            assert_eq!(result, Some(std::time::Duration::from_millis(2000)));
        });
    }

    #[test]
    #[serial]
    fn stagger_invalid_zero_default_falls_back_to_none() {
        // Non-numeric + default=0 → None
        with_env("EMBED_WORKER_SPAWN_DELAY_MS_TEST6", "bad", || {
            let result = resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS_TEST6", 0);
            assert_eq!(result, None);
        });
    }

    // --- resolve_worker_nice ---

    #[test]
    #[serial]
    fn worker_nice_resolution_table() {
        const VAR: &str = "EMBED_WORKER_NICE_NICE_TABLE";
        // (raw value, expected nice, whether the value is noted as unusable)
        let cases: [(Option<&str>, i32, bool); 9] = [
            (None, 0, false),
            (Some("10"), 10, false),
            (Some(" 7 "), 7, false),
            (Some("19"), 19, false),
            (Some("0"), 0, false),
            (Some(""), 0, false),
            (Some("-5"), 0, true),
            (Some("20"), 0, true),
            (Some("abc"), 0, true),
        ];
        for (raw, want, noted) in cases {
            match raw {
                Some(v) => unsafe { std::env::set_var(VAR, v) },
                None => unsafe { std::env::remove_var(VAR) },
            }
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let mut got = -1;
            metrics::with_local_recorder(&recorder, || got = resolve_worker_nice("nice-table"));
            unsafe { std::env::remove_var(VAR) };
            assert_eq!(got, want, "raw {raw:?}");
            let line = format!("embed_config_fallback_total{{env_var=\"{VAR}\"}} 1");
            assert_eq!(
                handle.render().contains(&line),
                noted,
                "raw {raw:?}: an unusable value must be counted and a usable one must not"
            );
        }
    }
}
