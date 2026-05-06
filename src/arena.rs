//! Shared CPU arena allocator with `kSameAsRequested` extend strategy.
//!
//! ORT's default `BFCArena` uses `kNextPowerOfTwo` (extend_strategy=0): on every
//! out-of-arena request it rounds up to the next power-of-two and never frees
//! the chunk until process exit. Combined with one BFCArena *per session* (we
//! run dense embedder + cross-encoder × pool_size + SPLADE × pool_size = 5+
//! sessions), each carrying its own arena, peak memory under variable batch
//! sizes hits 8 GB after ~30 minutes of mixed traffic and triggers OOM-restart.
//!
//! Fix has two parts, both required:
//!   1. Register a single CPU arena allocator on the global `OrtEnv` with
//!      `arena_extend_strategy = 1` (`kSameAsRequested`). Extends are exact
//!      to the request, no rounding up.
//!   2. Build every session with `.with_env_allocators()` so all sessions
//!      share that single allocator instead of creating their own.
//!
//! Net effect: one arena across the whole process; extends sized to actual
//! demand; no more 1 GB chunks materializing for a 100 MB activation.
//!
//! This module is deliberately small and `unsafe` — ort 2.0-rc.12 has no
//! high-level `with_arena_extend_strategy` builder, and the C API
//! (`CreateArenaCfg` + `CreateAndRegisterAllocator`) is the canonical
//! configuration path.
//!
//! # Environment variables
//!
//! All four `CreateArenaCfg` parameters are overridable at runtime:
//!
//! | Variable                       | Default (bytes)   | Meaning |
//! |--------------------------------|-------------------|---------|
//! | `EMBED_ARENA_MAX_MEM_BYTES`    | 6 442 450 944 (6 GiB) | Hard ceiling on arena growth |
//! | `EMBED_ARENA_INITIAL_CHUNK_BYTES` | 1 048 576 (1 MiB) | First allocation block |
//! | `EMBED_ARENA_MAX_DEAD_BYTES`   | 33 554 432 (32 MiB) | Dead-bytes threshold for chunk reuse |
//! | `EMBED_ARENA_EXTEND_STRATEGY`  | 1 (`kSameAsRequested`) | 0=kNextPowerOfTwo, 1=kSameAsRequested |
//!
//! The service refuses to start (`panic!`) if `MAX_MEM < INITIAL_CHUNK` or
//! `EXTEND_STRATEGY` is not 0 or 1.

use std::ptr;

use ort::AsPointer;
use ort::environment::Environment;
use ort_sys::{OrtAllocatorType, OrtArenaCfg, OrtEnv, OrtMemType, OrtMemoryInfo};

/// Default arena max memory: 6 GiB.
///
/// Rationale: compose limit is 8 GiB; subtract ~2 GiB e5-large weights +
/// ~2 GiB jina-code-v2 weights + ~0.5 GiB process overhead = ~3.5 GiB
/// available. 6 GiB gives the arena room to hold large jina attention/
/// FusedMatMul scratch tensors (1.0–1.3 GiB single allocations) without
/// exhausting container memory. A BFCArena OOM on a single allocation
/// (HTTP 500 to the caller) is strictly better than a cgroup OOM-kill
/// (whole container restarts, all in-flight requests fail).
const DEFAULT_MAX_MEM_BYTES: usize = 6 * 1024 * 1024 * 1024;

/// Default initial chunk size: 1 MiB.
const DEFAULT_INITIAL_CHUNK_BYTES: usize = 1024 * 1024;

/// Default max dead bytes per chunk: 32 MiB.
///
/// Aggressive vs ORT's default 128 MiB — forces BFCArena to consolidate dead
/// memory into free chunks sooner under variable batch shapes.
const DEFAULT_MAX_DEAD_BYTES: usize = 32 * 1024 * 1024;

/// Default extend strategy: 1 = `kSameAsRequested`.
const DEFAULT_EXTEND_STRATEGY: i32 = 1;

/// Resolved arena configuration values, parsed from environment variables.
/// Exposed as a struct so `init_arena_config` can be unit-tested independently
/// of ORT initialisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaCfg {
    pub max_mem_bytes: usize,
    pub initial_chunk_bytes: usize,
    pub max_dead_bytes: usize,
    pub extend_strategy: i32,
}

/// Parse arena configuration from environment variables, falling back to
/// compiled defaults. Panics on invalid combinations:
///   - `MAX_MEM < INITIAL_CHUNK` (arena would fail its first allocation)
///   - `EXTEND_STRATEGY` not in `{0, 1}`
pub fn init_arena_config() -> ArenaCfg {
    let max_mem_bytes = parse_usize_env("EMBED_ARENA_MAX_MEM_BYTES", DEFAULT_MAX_MEM_BYTES);
    let initial_chunk_bytes =
        parse_usize_env("EMBED_ARENA_INITIAL_CHUNK_BYTES", DEFAULT_INITIAL_CHUNK_BYTES);
    let max_dead_bytes = parse_usize_env("EMBED_ARENA_MAX_DEAD_BYTES", DEFAULT_MAX_DEAD_BYTES);
    let extend_strategy = parse_i32_env("EMBED_ARENA_EXTEND_STRATEGY", DEFAULT_EXTEND_STRATEGY);

    if extend_strategy != 0 && extend_strategy != 1 {
        panic!(
            "EMBED_ARENA_EXTEND_STRATEGY must be 0 (kNextPowerOfTwo) or 1 (kSameAsRequested), got {extend_strategy}"
        );
    }
    if max_mem_bytes < initial_chunk_bytes {
        panic!(
            "EMBED_ARENA_MAX_MEM_BYTES ({max_mem_bytes}) must be >= EMBED_ARENA_INITIAL_CHUNK_BYTES ({initial_chunk_bytes})"
        );
    }

    ArenaCfg {
        max_mem_bytes,
        initial_chunk_bytes,
        max_dead_bytes,
        extend_strategy,
    }
}

fn parse_usize_env(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) => v.trim().parse::<usize>().unwrap_or_else(|_| {
            panic!("{key} must be a non-negative integer, got {v:?}");
        }),
        Err(_) => default,
    }
}

fn parse_i32_env(key: &str, default: i32) -> i32 {
    match std::env::var(key) {
        Ok(v) => v.trim().parse::<i32>().unwrap_or_else(|_| {
            panic!("{key} must be an integer, got {v:?}");
        }),
        Err(_) => default,
    }
}

/// Registers a process-global shared CPU arena allocator with
/// `kSameAsRequested` extend strategy. Idempotent: calling twice will fail
/// the second time on `CreateAndRegisterAllocator`; we treat that as a
/// success and log instead of panicking, since the registration only needs
/// to happen once per env.
///
/// MUST be called after `ort::init().commit()` and BEFORE any
/// `Session::builder()`. Sessions then opt in via `.with_env_allocators()`.
///
/// Reads configuration from `init_arena_config()` (env vars with safe
/// defaults) and records Prometheus gauges for runtime visibility.
pub fn register_shared_cpu_arena() -> Result<(), String> {
    let cfg = init_arena_config();

    tracing::info!(
        max_mem_bytes = cfg.max_mem_bytes,
        initial_chunk_bytes = cfg.initial_chunk_bytes,
        max_dead_bytes = cfg.max_dead_bytes,
        extend_strategy = cfg.extend_strategy,
        "arena config resolved"
    );

    // Publish gauges so operators can verify effective config from /metrics.
    crate::metrics::set_arena_gauges(
        cfg.max_mem_bytes,
        cfg.initial_chunk_bytes,
        cfg.max_dead_bytes,
        cfg.extend_strategy,
    );

    let api = ort::api();

    // 1. Memory info: arena allocator on CPU output memory.
    let mut mem_info: *mut OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        (api.CreateCpuMemoryInfo)(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeCPUOutput,
            &mut mem_info,
        )
    };
    if !status.0.is_null() {
        return Err("CreateCpuMemoryInfo returned non-null status".into());
    }

    // 2. Arena cfg — env-gated since FU-24 (2026-05-05).
    //
    // History: previous config used `max_mem=0` (unlimited) then 3 GiB hard-
    // coded. Neither was sufficient:
    //   - 0: kSameAsRequested removes per-extend rounding but does NOT cap
    //     total arena ownership. Every new (batch, seq_len) shape triggers a
    //     permanent extend. Arena reached 12.3 GiB on an 8 GiB cgroup →
    //     kernel reclaim thrash.
    //   - 3 GiB: jina-code-v2 attention/FusedMatMul scratch tensors request
    //     1.0–1.3 GiB single blocks under prod batch shapes (BATCH_MAX=32,
    //     BATCH_MAX_TOKENS=16384, max_len=512). Three such requests saturate
    //     the 3 GiB cap → 92% error rate observed in prod 2026-05-05.
    //
    // Default is now 6 GiB (safe given 8 GiB compose limit and ~4 GiB model
    // weights + overhead), overridable via EMBED_ARENA_MAX_MEM_BYTES.
    let mut arena_cfg: *mut OrtArenaCfg = ptr::null_mut();
    let status = unsafe {
        (api.CreateArenaCfg)(
            cfg.max_mem_bytes,
            cfg.extend_strategy,
            cfg.initial_chunk_bytes as i32,
            cfg.max_dead_bytes as i32,
            &mut arena_cfg,
        )
    };
    if !status.0.is_null() {
        unsafe { (api.ReleaseMemoryInfo)(mem_info) };
        return Err("CreateArenaCfg returned non-null status".into());
    }

    // 3. Register on the global environment. Environment::current() returns
    //    the singleton committed by ort::init().
    let env = Environment::current().map_err(|e| format!("Environment::current: {e}"))?;
    // SAFETY: ORT C API takes *mut OrtEnv but only mutates internal env state
    // (allocator registry) under its own lock. Casting *const to *mut is safe
    // here per ORT API contract.
    let env_ptr = env.ptr() as *mut OrtEnv;

    let status = unsafe { (api.CreateAndRegisterAllocator)(env_ptr, mem_info, arena_cfg) };

    // Cleanup our cfg + mem_info — env retains its own internal copies.
    unsafe {
        (api.ReleaseArenaCfg)(arena_cfg);
        (api.ReleaseMemoryInfo)(mem_info);
    }

    if !status.0.is_null() {
        // Most common path: allocator already registered (e.g. on process
        // re-init). Treat as success — env already has what we need.
        tracing::warn!(
            "CreateAndRegisterAllocator returned non-null status (likely already registered)"
        );
    }

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Clear arena env vars for the duration of a test and restore on drop.
    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&str]) -> Self {
            let saved = keys
                .iter()
                .map(|&k| {
                    let prev = std::env::var(k).ok();
                    unsafe { std::env::remove_var(k) };
                    (k.to_string(), prev)
                })
                .collect();
            Self { saved }
        }

        fn set(&self, key: &str, value: &str) {
            unsafe { std::env::set_var(key, value) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => unsafe { std::env::set_var(k, val) },
                    None => unsafe { std::env::remove_var(k) },
                }
            }
        }
    }

    const ALL_VARS: &[&str] = &[
        "EMBED_ARENA_MAX_MEM_BYTES",
        "EMBED_ARENA_INITIAL_CHUNK_BYTES",
        "EMBED_ARENA_MAX_DEAD_BYTES",
        "EMBED_ARENA_EXTEND_STRATEGY",
    ];

    #[test]
    fn defaults_when_no_env_vars() {
        let guard = EnvGuard::new(ALL_VARS);
        let cfg = init_arena_config();
        drop(guard);

        assert_eq!(cfg.max_mem_bytes, 6 * 1024 * 1024 * 1024, "default 6 GiB");
        assert_eq!(cfg.initial_chunk_bytes, 1024 * 1024, "default 1 MiB");
        assert_eq!(cfg.max_dead_bytes, 32 * 1024 * 1024, "default 32 MiB");
        assert_eq!(cfg.extend_strategy, 1, "default kSameAsRequested");
    }

    #[test]
    fn env_overrides_all_fields() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "8589934592"); // 8 GiB
        guard.set("EMBED_ARENA_INITIAL_CHUNK_BYTES", "2097152"); // 2 MiB
        guard.set("EMBED_ARENA_MAX_DEAD_BYTES", "67108864"); // 64 MiB
        guard.set("EMBED_ARENA_EXTEND_STRATEGY", "0"); // kNextPowerOfTwo

        let cfg = init_arena_config();
        drop(guard);

        assert_eq!(cfg.max_mem_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(cfg.initial_chunk_bytes, 2 * 1024 * 1024);
        assert_eq!(cfg.max_dead_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.extend_strategy, 0);
    }

    #[test]
    fn partial_override_leaves_others_at_default() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "4294967296"); // 4 GiB
        let cfg = init_arena_config();
        drop(guard);

        assert_eq!(cfg.max_mem_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(cfg.initial_chunk_bytes, 1024 * 1024); // default
        assert_eq!(cfg.max_dead_bytes, 32 * 1024 * 1024); // default
        assert_eq!(cfg.extend_strategy, 1); // default
    }

    #[test]
    fn extend_strategy_0_and_1_are_valid() {
        for strategy in [0i32, 1i32] {
            let guard = EnvGuard::new(ALL_VARS);
            guard.set("EMBED_ARENA_EXTEND_STRATEGY", &strategy.to_string());
            let cfg = init_arena_config();
            drop(guard);
            assert_eq!(cfg.extend_strategy, strategy);
        }
    }

    #[test]
    #[should_panic(expected = "EMBED_ARENA_EXTEND_STRATEGY must be 0")]
    fn extend_strategy_2_panics() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_EXTEND_STRATEGY", "2");
        let _cfg = init_arena_config();
        drop(guard);
    }

    #[test]
    #[should_panic(expected = "EMBED_ARENA_EXTEND_STRATEGY must be 0")]
    fn extend_strategy_negative_panics() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_EXTEND_STRATEGY", "-1");
        let _cfg = init_arena_config();
        drop(guard);
    }

    #[test]
    #[should_panic(expected = "must be >= EMBED_ARENA_INITIAL_CHUNK_BYTES")]
    fn max_mem_less_than_initial_chunk_panics() {
        let guard = EnvGuard::new(ALL_VARS);
        // 512 KiB < default initial_chunk 1 MiB
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "524288");
        let _cfg = init_arena_config();
        drop(guard);
    }

    #[test]
    fn max_mem_equal_to_initial_chunk_is_valid() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "1048576");
        guard.set("EMBED_ARENA_INITIAL_CHUNK_BYTES", "1048576");
        let cfg = init_arena_config();
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, cfg.initial_chunk_bytes);
    }
}
