//! Library facade — exposes modules to binaries (main + worker).
pub mod config;
pub mod ipc;
pub mod model;
pub mod model_reranker;
pub mod model_splade;
pub mod supervisor;

// model.rs imports these internally via `crate::` — they must be pub
// so the library crate compiles when `model` is exposed. Demoting to
// pub(crate) would surface dead-code clippy errors from embed-server's
// metrics helpers that are conditionally used at runtime.
pub mod evictable_pool;
pub mod metrics;
pub mod onnx_cache;
pub mod pool;

// arena is required transitively by model_reranker and model_splade at load time.
pub mod arena;

// mlock helper — pin ONNX weight bytes into RAM to prevent swap-out.
pub mod mlock;

// /proc/<pid>/status helpers for per-worker RSS sampling (Linux-only).
pub mod proc;
