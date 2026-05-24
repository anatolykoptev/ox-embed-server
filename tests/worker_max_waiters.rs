//! Tests for per-model `EMBED_MAX_WAITERS_<KEY>` env override and
//! the `embed_worker_queue_depth` gauge.
//!
//! These tests live in `tests/` so they get their own process binary, avoiding
//! the process-global `metrics` recorder clash that would occur in `#[cfg(test)]`
//! blocks inside `src/bin/worker.rs`.
//!
//! All tests mutate process-global env state (`set_var` / `remove_var`).
//! `#[serial]` from the `serial_test` crate serialises them within this binary
//! to avoid UB (Rust 1.82+ `set_var` is unsafe for concurrent use).

use embed_server::worker_waiters::{
    WAITERS_FLOOR, WAITERS_POOL_MULTIPLIER, resolve_max_waiters_for_model,
};
use serial_test::serial;

// ── resolve_max_waiters_for_model ─────────────────────────────────────────────

/// Per-model env (`EMBED_MAX_WAITERS_TEST_MODEL=128`) overrides
/// global (`EMBED_MAX_WAITERS=32`).
#[test]
#[serial]
fn per_model_env_overrides_global() {
    // model key for "test-model" → "TEST_MODEL"
    let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
    let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

    unsafe {
        std::env::set_var("EMBED_MAX_WAITERS_TEST_MODEL", "128");
        std::env::set_var("EMBED_MAX_WAITERS", "32");
    }

    let result = resolve_max_waiters_for_model(1, "test-model");

    // Restore
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

    assert_eq!(result, 128, "per-model env must override global");
}

/// Global `EMBED_MAX_WAITERS` applies when per-model unset.
#[test]
#[serial]
fn global_env_applies_when_per_model_unset() {
    let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
    let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

    unsafe {
        std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        std::env::set_var("EMBED_MAX_WAITERS", "99");
    }

    let result = resolve_max_waiters_for_model(1, "test-model");

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

    assert_eq!(result, 99, "global env must apply when per-model unset");
}

/// Formula fallback: `pool_size × WAITERS_POOL_MULTIPLIER` (floor `WAITERS_FLOOR`)
/// when both env vars are unset.
#[test]
#[serial]
fn formula_fallback_when_both_unset() {
    let prev_per = std::env::var("EMBED_MAX_WAITERS_TEST_MODEL").ok();
    let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

    unsafe {
        std::env::remove_var("EMBED_MAX_WAITERS_TEST_MODEL");
        std::env::remove_var("EMBED_MAX_WAITERS");
    }

    let result_4 = resolve_max_waiters_for_model(4, "test-model");
    let result_1 = resolve_max_waiters_for_model(1, "test-model");

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

    assert_eq!(
        result_4,
        4 * WAITERS_POOL_MULTIPLIER,
        "formula: pool_size=4 × multiplier"
    );
    assert_eq!(result_1, WAITERS_FLOOR, "formula: pool_size=1 → floor");
}

/// Key transform: `jina-code-v2` → `JINA_CODE_V2` (dashes → underscores, uppercase).
#[test]
#[serial]
fn key_transform_dashes_to_underscores() {
    let prev_per = std::env::var("EMBED_MAX_WAITERS_JINA_CODE_V2").ok();
    let prev_global = std::env::var("EMBED_MAX_WAITERS").ok();

    unsafe {
        std::env::set_var("EMBED_MAX_WAITERS_JINA_CODE_V2", "256");
        std::env::set_var("EMBED_MAX_WAITERS", "32");
    }

    let result = resolve_max_waiters_for_model(1, "jina-code-v2");

    unsafe {
        match prev_per {
            Some(v) => std::env::set_var("EMBED_MAX_WAITERS_JINA_CODE_V2", v),
            None => std::env::remove_var("EMBED_MAX_WAITERS_JINA_CODE_V2"),
        }
        match prev_global {
            Some(v) => std::env::set_var("EMBED_MAX_WAITERS", v),
            None => std::env::remove_var("EMBED_MAX_WAITERS"),
        }
    }

    assert_eq!(
        result, 256,
        "jina-code-v2 key must resolve JINA_CODE_V2 var"
    );
}

// ── embed_worker_queue_depth gauge ────────────────────────────────────────────

/// `embed_worker_queue_depth` gauge tracks WAITERS counter value.
/// The gauge must reflect the value set by `set_worker_queue_depth`.
///
/// This test uses a fresh metrics recorder (via `PrometheusBuilder::new()`,
/// which is valid only once per process — running this in the test binary's
/// own process avoids conflict with other test suites).
#[test]
fn worker_queue_depth_gauge_roundtrip() {
    use embed_server::metrics::set_worker_queue_depth;
    use metrics_exporter_prometheus::PrometheusBuilder;

    // Install a fresh recorder for this binary. Panics if called twice, but
    // since this is the only test that installs one, it's safe.
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("install recorder");

    // Gauge starts at zero / absent before first set.
    set_worker_queue_depth("test-model", 1);
    let rendered_1 = handle.render();
    assert!(
        rendered_1.contains("embed_worker_queue_depth"),
        "gauge must appear after first set"
    );
    assert!(
        rendered_1.contains(r#"model="test-model""#),
        "gauge must carry model label"
    );

    // Set to 0 — gauge must still exist (not absent) so operators see the
    // back-to-idle signal.
    set_worker_queue_depth("test-model", 0);
    let rendered_0 = handle.render();
    assert!(
        rendered_0.contains("embed_worker_queue_depth"),
        "gauge must remain after setting to 0"
    );
}
