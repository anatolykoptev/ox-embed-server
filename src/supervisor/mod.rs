//! Supervisor — manages child worker processes per model.
//!
//! Phase 2 scaffold: spawn workers, route requests through WorkerPool.
//! Auto-restart watchdog lands in Wave 2.5.

pub mod handle;
pub mod pool;

pub use handle::{SpawnSpec, WorkerHandle};
pub use pool::WorkerPool;
