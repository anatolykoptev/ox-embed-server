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
//! | `EMBED_ARENA_MAX_DEAD_BYTES`   | 67 108 864 (64 MiB) | Dead-bytes threshold for chunk reuse |
//! | `EMBED_ARENA_EXTEND_STRATEGY`  | 1 (`kSameAsRequested`) | 0=kNextPowerOfTwo, 1=kSameAsRequested |
//!
//! Size values (`MAX_MEM_BYTES`, `INITIAL_CHUNK_BYTES`, `MAX_DEAD_BYTES`) accept
//! human-readable suffixes in addition to raw bytes:
//! `B`, `K`/`KiB` (×1024), `M`/`MiB` (×1048576), `G`/`GiB` (×1073741824).
//! Example: `EMBED_ARENA_MAX_MEM_BYTES=2GiB` is equivalent to `=2147483648`.
//!
//! The service refuses to start (`panic!`) if `MAX_MEM < INITIAL_CHUNK` or
//! `EXTEND_STRATEGY` is not 0 or 1.

use std::ffi::{CString, c_char};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::AsPointer;
use ort::environment::Environment;
use ort_sys::{OrtAllocatorType, OrtArenaCfg, OrtEnv, OrtMemType, OrtMemoryInfo};

/// Set to `true` inside `register_shared_cpu_arena()` after a successful
/// (or already-registered) registration. Guards against the silent-bug where
/// sessions are created before the shared arena is registered — in that case
/// each session allocates its own BFCArena and `EMBED_ARENA_*` knobs are
/// silently ignored, leading to monotonic memory growth.
///
/// `pub(crate)` so integration test helpers can call
/// `ARENA_REGISTERED.store(true, ...)` before loading real ONNX models in
/// tests that bypass the normal `main.rs` init sequence.
pub(crate) static ARENA_REGISTERED: AtomicBool = AtomicBool::new(false);

/// Assert that `register_shared_cpu_arena()` has been called before any
/// `Session::builder()`. Call this at the top of every function that creates
/// an ONNX session.
///
/// Panics with a fix hint if the arena has not been registered.
pub fn assert_arena_registered_before_session() {
    if !ARENA_REGISTERED.load(Ordering::Acquire) {
        panic!(
            "BUG: register_shared_cpu_arena() must be called before any \
             Session::builder() — arena knobs (EMBED_ARENA_*) will be ignored. \
             Check main.rs init order."
        );
    }
}

/// Default arena max memory: 6 GiB.
///
/// # WARNING — compose memory limit MUST be ≥12 GiB with this default
///
/// 6 GiB default is *headroom-overcommit*, not a safe ceiling relative to
/// an 8 GiB compose limit. Resident-memory breakdown:
///
/// - ~1.84 GiB model weights resident (e5-large ~650 MiB INT8 +
///   jina-code-v2 ~250 MiB INT8 + SPLADE ~400 MiB +
///   gte-multi-rerank ~540 MiB)
/// - up to cap BFCArena retention — each ML inference batch (B=32,
///   S=512) allocates ~400 MiB attention scratch + matmul buffers that
///   kSameAsRequested never shrinks between calls. Grows monotonically
///   to the cap set here. Compose sets cap = 3 GiB (PR #98,
///   krolik-server).
/// - Note: PR #46 precomputed ALiBi constants, eliminating
///   the 1.258 GiB per-call scratch that previously caused jina-code-v2
///   to require ~1.5 GiB of arena per inference call.
///
/// Total worst-case at this default: ~7.84 GiB (1.84 resident + 6 arena)
///
/// At 8 GiB compose this default **WILL cgroup-OOM-kill** the container
/// under sustained large-batch jina-code-v2 load.
///
/// **Operator action before rolling out this default:**
/// Bump `compose/memdb.yml` embed-server `deploy.resources.limits.memory`
/// to `12288M` (12 GiB = ~2 GiB resident + 6 GiB arena + ~4 GiB safety).
///
/// **Alternative for 8 GiB hosts:**
/// Set `EMBED_ARENA_MAX_MEM_BYTES=3221225472` (3 GiB, prod value per PR #98
/// in krolik-server) — keeps arena + resident under the 8 GiB ceiling with
/// headroom, but caps throughput on large jina-code-v2 batches. Reduce
/// `BATCH_MAX_TOKENS` accordingly if 92% error rate reappears (FU-24).
///
/// A BFCArena OOM on a single allocation (HTTP 500 to caller) is strictly
/// better than a cgroup OOM-kill (whole container restarts, all in-flight
/// requests fail). The default favours the former over the latter; the
/// compose limit controls which actually occurs.
const DEFAULT_MAX_MEM_BYTES: usize = 6 * 1024 * 1024 * 1024;

/// Default initial chunk size: 1 MiB.
const DEFAULT_INITIAL_CHUNK_BYTES: usize = 1024 * 1024;

/// Default max dead bytes per chunk: 64 MiB.
///
/// Aggressive vs ORT's default 128 MiB — forces BFCArena to consolidate dead
/// memory into free chunks sooner under variable batch shapes. We landed on
/// 64 MiB rather than 32 MiB because BERT scratch tensors run ~8 MiB+ each;
/// too small a threshold causes extend cycles to thrash. Live config is
/// observable via `embed_arena_max_dead_bytes` gauge.
const DEFAULT_MAX_DEAD_BYTES: usize = 64 * 1024 * 1024;

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
///
/// Global-only variant — checks no per-model env. Calls
/// `init_arena_config_for_model("")` internally.
// `allow(dead_code)`: production reads per-model config via
// `init_arena_config_for_model()`; this global-only convenience wrapper has
// only `#[cfg(test)]` callers. `arena` is also compiled into main.rs's private
// `mod arena`, whose non-test build therefore sees no caller.
#[allow(dead_code)]
pub fn init_arena_config() -> ArenaCfg {
    init_arena_config_for_model("")
}

/// Parse arena configuration with per-model override support for
/// `EMBED_ARENA_MAX_MEM_BYTES`.
///
/// Lookup order for `max_mem_bytes`:
///   1. `EMBED_ARENA_MAX_MEM_BYTES_<MODEL_KEY_UPPER>` — per-worker override,
///      where `<MODEL_KEY_UPPER>` is `model_key` uppercased with `-` → `_`.
///      Example: `"jina-code-v2"` → `EMBED_ARENA_MAX_MEM_BYTES_JINA_CODE_V2`.
///   2. `EMBED_ARENA_MAX_MEM_BYTES` — global (original behaviour).
///   3. Compiled default: 6 GiB.
///
/// All other arena parameters (`INITIAL_CHUNK_BYTES`, `MAX_DEAD_BYTES`,
/// `EXTEND_STRATEGY`) remain global-only — per-model arena tuning in
/// production only requires adjusting the memory ceiling.
///
/// When `model_key` is empty, the per-model lookup is skipped.
pub fn init_arena_config_for_model(model_key: &str) -> ArenaCfg {
    // Per-model max_mem override: EMBED_ARENA_MAX_MEM_BYTES_<KEY>.
    let max_mem_bytes = if !model_key.is_empty() {
        let suffix = crate::config::model_env_key(model_key);
        let per_model_key = format!("EMBED_ARENA_MAX_MEM_BYTES_{suffix}");
        match std::env::var(&per_model_key) {
            Ok(raw) => {
                let trimmed = raw.trim();
                // Use the same suffix-aware parser as the global path so
                // operators can use "2GiB" in per-model vars too.
                let n = parse_bytes_with_suffix(&per_model_key, trimmed);
                tracing::info!(
                    model = %model_key,
                    env = %per_model_key,
                    max_mem_bytes = n,
                    "per-model EMBED_ARENA_MAX_MEM_BYTES override"
                );
                n
            }
            Err(_) => parse_usize_env("EMBED_ARENA_MAX_MEM_BYTES", DEFAULT_MAX_MEM_BYTES),
        }
    } else {
        parse_usize_env("EMBED_ARENA_MAX_MEM_BYTES", DEFAULT_MAX_MEM_BYTES)
    };

    let initial_chunk_bytes = parse_usize_env(
        "EMBED_ARENA_INITIAL_CHUNK_BYTES",
        DEFAULT_INITIAL_CHUNK_BYTES,
    );
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
    // CreateArenaCfgV2 takes size_t for every value (no i32 truncation),
    // so we no longer reject usize > i32::MAX. The legacy V1 API silently
    // overwrote `max_dead_bytes_per_chunk` to -1L (ORT default 128 MiB) at
    // onnxruntime_c_api.cc:2606-2607 immediately after assignment — we
    // migrated off it. Tests still validate sane ordering against
    // `MAX_MEM >= INITIAL_CHUNK`.

    ArenaCfg {
        max_mem_bytes,
        initial_chunk_bytes,
        max_dead_bytes,
        extend_strategy,
    }
}

/// Parse a `usize` from an environment variable, accepting both raw bytes
/// and human-readable suffixes.
///
/// Accepted suffixes (case-insensitive, binary multipliers):
/// - `B` → ×1
/// - `K` or `KiB` → ×1 024
/// - `M` or `MiB` → ×1 048 576
/// - `G` or `GiB` → ×1 073 741 824
///
/// Examples: `"2GiB"`, `"64M"`, `"1048576"` all accepted.
/// Bare digits with no suffix are treated as raw bytes (backward-compatible).
///
/// Panics if the variable is set but cannot be parsed.
fn parse_usize_env(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) => parse_bytes_with_suffix(key, v.trim()),
        Err(_) => default,
    }
}

/// Parse a byte-count string that may carry a human-readable suffix.
///
/// Called by `parse_usize_env` and by the per-model arena override path
/// so both use identical parsing rules.
///
/// Panics (with `env_key` in the message) on parse failure.
pub(crate) fn parse_bytes_with_suffix(env_key: &str, s: &str) -> usize {
    // Split at the first non-digit character to separate mantissa from suffix.
    let split_pos = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (digits, suffix_raw) = s.split_at(split_pos);
    let suffix = suffix_raw.trim().to_uppercase();

    let mantissa: u64 = digits.parse().unwrap_or_else(|_| {
        panic!("{env_key} must be a non-negative integer with optional suffix (B/K/KiB/M/MiB/G/GiB), got {s:?}");
    });

    let multiplier: u64 = match suffix.as_str() {
        "" | "B" => 1,
        "K" | "KIB" => 1024,
        "M" | "MIB" => 1024 * 1024,
        "G" | "GIB" => 1024 * 1024 * 1024,
        _ => panic!(
            "{env_key}: unrecognised size suffix {suffix_raw:?}; accepted: B, K, KiB, M, MiB, G, GiB"
        ),
    };

    let result = mantissa.checked_mul(multiplier).unwrap_or_else(|| {
        panic!("{env_key}: value {s:?} overflows usize");
    });

    usize::try_from(result).unwrap_or_else(|_| {
        panic!("{env_key}: value {s:?} overflows usize on this platform");
    })
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
/// `model_key` — the model this worker serves (e.g. `"jina-code-v2"`).
/// When non-empty, `EMBED_ARENA_MAX_MEM_BYTES_<MODEL_KEY_UPPER>` is checked
/// before the global. Call from embed-worker after learning `EMBED_WORKER_MODEL`.
/// Pass `""` from the supervisor (global-only, no ONNX sessions in-process).
///
/// Reads configuration from `init_arena_config_for_model()` (env vars with
/// safe defaults) and records Prometheus gauges for runtime visibility.
pub fn register_shared_cpu_arena_for_model(model_key: &str) -> Result<(), String> {
    let cfg = init_arena_config_for_model(model_key);
    register_shared_cpu_arena_with_cfg(cfg)
}

/// Supervisor-process variant: reads only the global `EMBED_ARENA_MAX_MEM_BYTES`.
/// Delegates to `register_shared_cpu_arena_for_model("")`.
pub fn register_shared_cpu_arena() -> Result<(), String> {
    register_shared_cpu_arena_for_model("")
}

fn register_shared_cpu_arena_with_cfg(cfg: ArenaCfg) -> Result<(), String> {
    tracing::info!(
        max_mem_bytes = cfg.max_mem_bytes,
        initial_chunk_bytes = cfg.initial_chunk_bytes,
        max_dead_bytes = cfg.max_dead_bytes,
        extend_strategy = cfg.extend_strategy,
        "arena config resolved"
    );

    let api = ort::api();

    // 1. Memory info: arena allocator on the CPU EP's *default* memory type.
    //
    // Was OrtMemTypeCPUOutput historically; both reach the per-device CPU
    // allocator via the device-keyed lookup in session_state.cc
    // (`UpdateAllocatorsWithEnvAllocators`), but `OrtMemTypeDefault` is the
    // semantically correct key for the CPU EP's working buffers.
    let mut mem_info: *mut OrtMemoryInfo = ptr::null_mut();
    let status = unsafe {
        (api.CreateCpuMemoryInfo)(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault,
            &mut mem_info,
        )
    };
    if !status.0.is_null() {
        return Err("CreateCpuMemoryInfo returned non-null status".into());
    }

    // 2. Arena cfg via the V2 (key/value) API.
    //
    // V1 has an upstream bug at onnxruntime_c_api.cc:2606-2607 that silently
    // overwrites `max_dead_bytes_per_chunk` to -1L (ORT default 128 MiB)
    // immediately after assignment. Live evidence: arena log printed
    // max_dead=128 MiB despite 32 MiB being requested, and after ~50 min
    // uptime jina-code-v2 hit 36 fragmentation errors. V2 takes size_t for
    // every value (no i32 truncation either) and does not have the overwrite.
    //
    // Keys come from ORT's BFCArena: `core/framework/allocator.cc` —
    // ArenaCfgFromConfigOptions reads exactly these names.
    //
    // History on max_mem: 0 → 3 GiB → 6 GiB. Unbounded reached 12.3 GiB on
    // an 8 GiB cgroup. 3 GiB starved jina-code-v2 attention scratch (~1 GiB
    // per request × 3 concurrent → 92% error rate observed 2026-05-05).
    let key_max_mem = CString::new("max_mem").expect("static cstr");
    let key_strategy = CString::new("arena_extend_strategy").expect("static cstr");
    let key_initial = CString::new("initial_chunk_size_bytes").expect("static cstr");
    let key_max_dead = CString::new("max_dead_bytes_per_chunk").expect("static cstr");

    let keys: [*const c_char; 4] = [
        key_max_mem.as_ptr(),
        key_strategy.as_ptr(),
        key_initial.as_ptr(),
        key_max_dead.as_ptr(),
    ];
    // Strategy is 0/1; cast to usize is well-defined since we validated it
    // in `init_arena_config` and ORT itself stores it as size_t in V2.
    let values: [usize; 4] = [
        cfg.max_mem_bytes,
        cfg.extend_strategy as usize,
        cfg.initial_chunk_bytes,
        cfg.max_dead_bytes,
    ];

    let mut arena_cfg: *mut OrtArenaCfg = ptr::null_mut();
    let status = unsafe {
        (api.CreateArenaCfgV2)(keys.as_ptr(), values.as_ptr(), keys.len(), &mut arena_cfg)
    };
    if !status.0.is_null() {
        unsafe { (api.ReleaseMemoryInfo)(mem_info) };
        return Err("CreateArenaCfgV2 returned non-null status".into());
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
        // re-init). Treat as success — env already has what we need from
        // the FIRST registration, but DO NOT publish gauges: they would
        // describe THIS call's config while the live allocator was built
        // from the first config. Bump a counter so the warn-path is
        // observable in /metrics, then return success.
        tracing::warn!(
            "CreateAndRegisterAllocator returned non-null status (likely already registered); skipping gauge publish"
        );
        crate::metrics::record_arena_register_skipped();
        ARENA_REGISTERED.store(true, Ordering::Release);
        return Ok(());
    }

    // 4. Publish gauges only AFTER a successful registration so the gauges
    //    truly describe the live allocator. Idempotency-safe: a second
    //    successful register would imply a brand-new env (impossible with
    //    a singleton OrtEnv), so this branch runs at most once per process.
    crate::metrics::set_arena_gauges(
        cfg.max_mem_bytes,
        cfg.initial_chunk_bytes,
        cfg.max_dead_bytes,
        cfg.extend_strategy,
    );

    ARENA_REGISTERED.store(true, Ordering::Release);

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // ── new guard tests ───────────────────────────────────────────────────────

    #[test]
    #[serial]
    #[should_panic(expected = "register_shared_cpu_arena")]
    fn assert_panics_when_arena_not_registered() {
        ARENA_REGISTERED.store(false, std::sync::atomic::Ordering::Release);
        assert_arena_registered_before_session();
    }

    #[test]
    #[serial]
    fn assert_passes_after_register() {
        ARENA_REGISTERED.store(true, std::sync::atomic::Ordering::Release);
        // must not panic
        assert_arena_registered_before_session();
        // restore to false so subsequent serial tests don't see stale state
        ARENA_REGISTERED.store(false, std::sync::atomic::Ordering::Release);
    }

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
    #[serial]
    fn defaults_when_no_env_vars() {
        let guard = EnvGuard::new(ALL_VARS);
        let cfg = init_arena_config();
        drop(guard);

        assert_eq!(cfg.max_mem_bytes, 6 * 1024 * 1024 * 1024, "default 6 GiB");
        assert_eq!(cfg.initial_chunk_bytes, 1024 * 1024, "default 1 MiB");
        assert_eq!(cfg.max_dead_bytes, 64 * 1024 * 1024, "default 64 MiB");
        assert_eq!(cfg.extend_strategy, 1, "default kSameAsRequested");
    }

    #[test]
    #[serial]
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
    #[serial]
    fn partial_override_leaves_others_at_default() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "4294967296"); // 4 GiB
        let cfg = init_arena_config();
        drop(guard);

        assert_eq!(cfg.max_mem_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(cfg.initial_chunk_bytes, 1024 * 1024); // default
        assert_eq!(cfg.max_dead_bytes, 64 * 1024 * 1024); // default
        assert_eq!(cfg.extend_strategy, 1); // default
    }

    #[test]
    #[serial]
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
    #[serial]
    #[should_panic(expected = "EMBED_ARENA_EXTEND_STRATEGY must be 0")]
    fn extend_strategy_2_panics() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_EXTEND_STRATEGY", "2");
        let _cfg = init_arena_config();
        drop(guard);
    }

    #[test]
    #[serial]
    #[should_panic(expected = "EMBED_ARENA_EXTEND_STRATEGY must be 0")]
    fn extend_strategy_negative_panics() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_EXTEND_STRATEGY", "-1");
        let _cfg = init_arena_config();
        drop(guard);
    }

    #[test]
    #[serial]
    #[should_panic(expected = "must be >= EMBED_ARENA_INITIAL_CHUNK_BYTES")]
    fn max_mem_less_than_initial_chunk_panics() {
        let guard = EnvGuard::new(ALL_VARS);
        // 512 KiB < default initial_chunk 1 MiB
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "524288");
        let _cfg = init_arena_config();
        drop(guard);
    }

    #[test]
    #[serial]
    fn max_mem_equal_to_initial_chunk_is_valid() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "1048576");
        guard.set("EMBED_ARENA_INITIAL_CHUNK_BYTES", "1048576");
        let cfg = init_arena_config();
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, cfg.initial_chunk_bytes);
    }

    #[test]
    #[serial]
    fn values_above_i32_max_are_now_accepted() {
        // CreateArenaCfgV2 takes size_t for every value, so values that
        // would have overflowed the legacy V1 c_int signature are now
        // valid configuration. Documented as a regression-test guard
        // against future revivals of the old i32 panic.
        let guard = EnvGuard::new(ALL_VARS);
        let over = (i32::MAX as usize + 1).to_string();
        guard.set("EMBED_ARENA_INITIAL_CHUNK_BYTES", &over);
        guard.set(
            "EMBED_ARENA_MAX_MEM_BYTES",
            &(i32::MAX as usize + 2).to_string(),
        );
        guard.set("EMBED_ARENA_MAX_DEAD_BYTES", &over);
        let cfg = init_arena_config();
        drop(guard);
        assert_eq!(cfg.initial_chunk_bytes, i32::MAX as usize + 1);
        assert_eq!(cfg.max_dead_bytes, i32::MAX as usize + 1);
    }

    // -----------------------------------------------------------------
    // EMBED_ARENA_MAX_MEM_BYTES_<MODEL_KEY_UPPER> per-model override.
    // -----------------------------------------------------------------

    const ALL_VARS_WITH_PER_MODEL: &[&str] = &[
        "EMBED_ARENA_MAX_MEM_BYTES",
        "EMBED_ARENA_INITIAL_CHUNK_BYTES",
        "EMBED_ARENA_MAX_DEAD_BYTES",
        "EMBED_ARENA_EXTEND_STRATEGY",
        "EMBED_ARENA_MAX_MEM_BYTES_JINA_CODE_V2",
        "EMBED_ARENA_MAX_MEM_BYTES_MULTILINGUAL_E5_LARGE",
    ];

    #[test]
    #[serial]
    fn init_arena_config_for_model_default_when_nothing_set() {
        let guard = EnvGuard::new(ALL_VARS_WITH_PER_MODEL);
        let cfg = init_arena_config_for_model("jina-code-v2");
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, 6 * 1024 * 1024 * 1024, "default 6 GiB");
    }

    #[test]
    #[serial]
    fn init_arena_config_for_model_uses_global_when_no_per_model() {
        let guard = EnvGuard::new(ALL_VARS_WITH_PER_MODEL);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "3221225472"); // 3 GiB global
        let cfg = init_arena_config_for_model("jina-code-v2");
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, 3 * 1024 * 1024 * 1024);
    }

    #[test]
    #[serial]
    fn init_arena_config_for_model_per_model_wins_over_global() {
        // EMBED_ARENA_MAX_MEM_BYTES_JINA_CODE_V2 beats the global.
        let guard = EnvGuard::new(ALL_VARS_WITH_PER_MODEL);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "6442450944"); // 6 GiB global
        guard.set("EMBED_ARENA_MAX_MEM_BYTES_JINA_CODE_V2", "2147483648"); // 2 GiB per-model
        let cfg = init_arena_config_for_model("jina-code-v2");
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    #[serial]
    fn init_arena_config_for_model_per_model_does_not_affect_other_models() {
        // Jina override must not bleed into e5-large.
        let guard = EnvGuard::new(ALL_VARS_WITH_PER_MODEL);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES_JINA_CODE_V2", "2147483648");
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "3221225472"); // global 3 GiB
        let cfg = init_arena_config_for_model("multilingual-e5-large");
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, 3 * 1024 * 1024 * 1024); // e5 sees global
    }

    #[test]
    #[serial]
    fn init_arena_config_empty_model_key_delegates_to_global_path() {
        // Empty model_key = supervisor path, no per-model lookup.
        let guard = EnvGuard::new(ALL_VARS_WITH_PER_MODEL);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "4294967296"); // 4 GiB
        let cfg = init_arena_config_for_model("");
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, 4 * 1024 * 1024 * 1024);
    }

    // -----------------------------------------------------------------
    // parse_usize_env / parse_bytes_with_suffix — suffix parsing.
    // -----------------------------------------------------------------

    #[test]
    fn parse_bytes_with_suffix_raw_bytes() {
        assert_eq!(parse_bytes_with_suffix("TEST", "0"), 0);
        assert_eq!(parse_bytes_with_suffix("TEST", "1024"), 1024);
        assert_eq!(
            parse_bytes_with_suffix("TEST", "2147483648"),
            2 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn parse_bytes_with_suffix_b_suffix() {
        assert_eq!(parse_bytes_with_suffix("TEST", "1B"), 1);
        assert_eq!(parse_bytes_with_suffix("TEST", "1024B"), 1024);
    }

    #[test]
    fn parse_bytes_with_suffix_k_and_kib() {
        assert_eq!(parse_bytes_with_suffix("TEST", "1K"), 1024);
        assert_eq!(parse_bytes_with_suffix("TEST", "1KiB"), 1024);
        assert_eq!(parse_bytes_with_suffix("TEST", "64K"), 64 * 1024);
    }

    #[test]
    fn parse_bytes_with_suffix_m_and_mib() {
        assert_eq!(parse_bytes_with_suffix("TEST", "1M"), 1024 * 1024);
        assert_eq!(parse_bytes_with_suffix("TEST", "1MiB"), 1024 * 1024);
        assert_eq!(parse_bytes_with_suffix("TEST", "64M"), 64 * 1024 * 1024);
    }

    #[test]
    fn parse_bytes_with_suffix_g_and_gib() {
        assert_eq!(
            parse_bytes_with_suffix("TEST", "2G"),
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(
            parse_bytes_with_suffix("TEST", "2GiB"),
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(
            parse_bytes_with_suffix("TEST", "6GiB"),
            6 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn parse_bytes_with_suffix_case_insensitive() {
        assert_eq!(parse_bytes_with_suffix("TEST", "1gib"), 1024 * 1024 * 1024);
        assert_eq!(parse_bytes_with_suffix("TEST", "1Gib"), 1024 * 1024 * 1024);
        assert_eq!(parse_bytes_with_suffix("TEST", "1mib"), 1024 * 1024);
    }

    #[test]
    #[should_panic(expected = "unrecognised size suffix")]
    fn parse_bytes_with_suffix_invalid_suffix_panics() {
        parse_bytes_with_suffix("TEST", "2KB"); // KB not in accepted list
    }

    #[test]
    #[should_panic(expected = "non-negative integer")]
    fn parse_bytes_with_suffix_invalid_digits_panics() {
        parse_bytes_with_suffix("TEST", "notanumber");
    }

    /// Suffix parser also accepted via env var round-trip (parse_usize_env).
    #[test]
    #[serial]
    fn parse_usize_env_accepts_suffix_gib() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_MEM_BYTES", "2GiB");
        let cfg = init_arena_config();
        drop(guard);
        assert_eq!(cfg.max_mem_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    #[serial]
    fn parse_usize_env_accepts_suffix_m() {
        let guard = EnvGuard::new(ALL_VARS);
        guard.set("EMBED_ARENA_MAX_DEAD_BYTES", "64M");
        let cfg = init_arena_config();
        drop(guard);
        assert_eq!(cfg.max_dead_bytes, 64 * 1024 * 1024);
    }
}
