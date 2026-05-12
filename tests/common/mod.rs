//! Shared test helpers — used across integration tests.

use std::path::PathBuf;
use std::process::Child;

/// RAII guard for a spawned child process bound to a UDS socket.
/// On drop: kill child, wait for exit, remove socket file.
pub struct ChildGuard {
    pub child: Option<Child>,
    pub socket: PathBuf,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        let _ = std::fs::remove_file(&self.socket);
    }
}
