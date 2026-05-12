//! Library facade — exposes modules to binaries (main + worker).
pub mod config;
pub mod ipc;
pub mod model;

// model.rs imports these internally via `crate::` — they must be pub
// so the library crate compiles when `model` is exposed.
pub mod evictable_pool;
pub mod metrics;
pub mod onnx_cache;
pub mod pool;
