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

use std::ptr;

use ort::AsPointer;
use ort::environment::Environment;
use ort_sys::{OrtAllocatorType, OrtArenaCfg, OrtEnv, OrtMemType, OrtMemoryInfo};

/// Registers a process-global shared CPU arena allocator with
/// `kSameAsRequested` extend strategy. Idempotent: calling twice will fail
/// the second time on `CreateAndRegisterAllocator`; we treat that as a
/// success and log instead of panicking, since the registration only needs
/// to happen once per env.
///
/// MUST be called after `ort::init().commit()` and BEFORE any
/// `Session::builder()`. Sessions then opt in via `.with_env_allocators()`.
pub fn register_shared_cpu_arena() -> Result<(), String> {
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

    // 2. Arena cfg with extend_strategy = kSameAsRequested (=1). The other
    //    fields use ORT defaults (-1 sentinels): initial_chunk_size_bytes
    //    stays at 1 MiB, max_dead_bytes_per_chunk at 128 MiB. We don't cap
    //    max_mem since we still want arena reuse within the requested size;
    //    the OS / cgroup memory limit is the real ceiling.
    let mut arena_cfg: *mut OrtArenaCfg = ptr::null_mut();
    let status = unsafe {
        (api.CreateArenaCfg)(
            0,  // max_mem (0 = unlimited)
            1,  // arena_extend_strategy = kSameAsRequested
            -1, // initial_chunk_size_bytes (default)
            -1, // max_dead_bytes_per_chunk (default)
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
