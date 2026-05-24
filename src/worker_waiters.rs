//! Per-worker waiter-queue configuration.
//!
//! Extracted from `src/bin/worker.rs` so the resolver is reachable from
//! integration tests under `tests/` without spawning a real worker process.
//!
//! # Env var resolution order (per model key `K`)
//!
//! 1. `EMBED_MAX_WAITERS_<K>` — per-model override (e.g. `EMBED_MAX_WAITERS_JINA_CODE_V2=256`)
//! 2. `EMBED_MAX_WAITERS`     — global override
//! 3. `pool_size × WAITERS_POOL_MULTIPLIER` (floor `WAITERS_FLOOR`) — formula fallback
//!
//! Key transform: `jina-code-v2` → `JINA_CODE_V2` (uppercase, dashes → underscores).
//! This is the same transform used by `EMBED_SESSION_POOL_SIZE_<KEY>`,
//! `EMBED_MEMORY_PATTERN_<KEY>`, and `EMBED_ARENA_MAX_MEM_BYTES_<KEY>` (PR #74).

/// Multiplier applied to `pool_size` when computing the default max-waiters cap.
///
/// 8× gives ample burst headroom while keeping the waiter queue bounded.
pub const WAITERS_POOL_MULTIPLIER: usize = 8;

/// Minimum max-waiters cap regardless of `pool_size`.
///
/// Prevents `pool_size=1` from producing a cap of 8, which is low enough to
/// trigger queue overflow on brief single-connection bursts.
pub const WAITERS_FLOOR: usize = 16;

/// Resolve the max-waiters limit for a specific model worker at startup.
///
/// Checks per-model env `EMBED_MAX_WAITERS_<KEY>` first, falls back to
/// global `EMBED_MAX_WAITERS`, falls back to the formula.
///
/// - Zero values are treated as misconfiguration (would reject every request)
///   and fall through to the next fallback with a warning.
/// - Non-numeric values warn and fall through.
pub fn resolve_max_waiters_for_model(pool_size: usize, model_name: &str) -> usize {
    let formula_default = || {
        pool_size
            .saturating_mul(WAITERS_POOL_MULTIPLIER)
            .max(WAITERS_FLOOR)
    };

    let key = crate::config::model_env_key(model_name);
    let per_model_var = format!("EMBED_MAX_WAITERS_{key}");

    // 1. Per-model override.
    if let Ok(raw) = std::env::var(&per_model_var) {
        match raw.trim().parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!(
                    model = %model_name,
                    env = %per_model_var,
                    max_waiters = n,
                    "per-model EMBED_MAX_WAITERS override applied"
                );
                return n;
            }
            Ok(_) => {
                tracing::warn!(
                    model = %model_name,
                    env = %per_model_var,
                    "per-model EMBED_MAX_WAITERS=0 would reject all requests; falling back to global/formula"
                );
            }
            Err(_) => {
                tracing::warn!(
                    model = %model_name,
                    env = %per_model_var,
                    value = %raw.trim(),
                    "per-model EMBED_MAX_WAITERS is not a valid usize; falling back to global/formula"
                );
            }
        }
    }

    // 2. Global override.
    match std::env::var("EMBED_MAX_WAITERS") {
        Err(_) => formula_default(),
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if n > 0 => n,
            Ok(_) => {
                tracing::warn!(
                    EMBED_MAX_WAITERS = %raw.trim(),
                    fallback = formula_default(),
                    "EMBED_MAX_WAITERS=0 would reject all requests; using formula fallback"
                );
                formula_default()
            }
            Err(_) => {
                tracing::warn!(
                    EMBED_MAX_WAITERS = %raw.trim(),
                    fallback = formula_default(),
                    "EMBED_MAX_WAITERS is not a valid usize; using formula fallback"
                );
                formula_default()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{WAITERS_FLOOR, WAITERS_POOL_MULTIPLIER, resolve_max_waiters_for_model};
    use serial_test::serial;

    // All tests mutate process-global env — serialised via #[serial].

    #[test]
    #[serial]
    fn formula_fallback_default() {
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

        unsafe {
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
            std::env::remove_var("EMBED_MAX_WAITERS");
        }

        let val = resolve_max_waiters_for_model(4, "test-model");

        unsafe {
            match prev_per {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
            match prev_global {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
        }

        assert_eq!(val, 4 * WAITERS_POOL_MULTIPLIER);
    }

    #[test]
    #[serial]
    fn formula_floor() {
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

        unsafe {
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
            std::env::remove_var("EMBED_MAX_WAITERS");
        }

        let val = resolve_max_waiters_for_model(1, "test-model");

        unsafe {
            match prev_per {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
            match prev_global {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
        }

        assert_eq!(val, WAITERS_FLOOR);
    }

    #[test]
    #[serial]
    fn global_override() {
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

        unsafe {
            std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
            std::env::set_var("EMBED_MAX_WAITERS", "50");
        }

        let val = resolve_max_waiters_for_model(1, "test-model");

        unsafe {
            match prev_per {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
            match prev_global {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
        }

        assert_eq!(val, 50);
    }

    #[test]
    #[serial]
    fn per_model_override_takes_precedence() {
        let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
        let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

        unsafe {
            std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", "200");
            std::env::set_var("EMBED_MAX_WAITERS", "10");
        }

        let val = resolve_max_waiters_for_model(1, "test-model");

        unsafe {
            match prev_per {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL"),
            }
            match prev_global {
                Some(v) => std::env::set_var("EMBED_MAX_WAITERS", v),
                None => std::env::remove_var("EMBED_MAX_WAITERS"),
            }
        }

        assert_eq!(val, 200);
    }
}
