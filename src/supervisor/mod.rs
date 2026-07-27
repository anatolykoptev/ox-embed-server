//! Supervisor — manages child worker processes per model.
//!
//! Wave 2.5: WorkerHandle replaced by WorkerSupervisor (watchdog + auto-restart).

pub mod handle;
pub mod pool;
pub(crate) mod util;

pub use handle::{SpawnSpec, WorkerKind, WorkerSupervisor};
pub use pool::WorkerPool;
