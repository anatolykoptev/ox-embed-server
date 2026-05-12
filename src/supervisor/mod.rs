//! Supervisor — manages child worker processes per model.
//!
//! Wave 2.5: WorkerHandle replaced by WorkerSupervisor (watchdog + auto-restart).

pub mod handle;
pub mod pool;

pub use handle::{SpawnSpec, WorkerSupervisor};
pub use pool::WorkerPool;
