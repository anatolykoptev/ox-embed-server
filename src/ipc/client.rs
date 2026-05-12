//! Supervisor-side client to one worker process.
//!
//! Holds N persistent UDS connections (matches worker's pool_size); each
//! connection runs requests serially. `infer()` round-robins across pool.

use crate::ipc::frame::{read_frame, write_frame};
use crate::ipc::protocol::{InferRequest, InferResponse};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

pub struct WorkerClient {
    socket_path: PathBuf,
    pool: Vec<Arc<Mutex<UnixStream>>>,
    next_idx: AtomicU64,
    request_counter: AtomicU64,
}

impl WorkerClient {
    pub async fn connect(socket_path: PathBuf, conns: usize) -> std::io::Result<Self> {
        if conns == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WorkerClient pool size must be >= 1",
            ));
        }
        let mut pool = Vec::with_capacity(conns);
        for _ in 0..conns {
            let stream = UnixStream::connect(&socket_path).await?;
            pool.push(Arc::new(Mutex::new(stream)));
        }
        Ok(Self {
            socket_path,
            pool,
            next_idx: AtomicU64::new(0),
            request_counter: AtomicU64::new(0),
        })
    }

    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    pub async fn infer(
        &self,
        model: String,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> std::io::Result<InferResponse> {
        let idx = (self.next_idx.fetch_add(1, Ordering::Relaxed) as usize) % self.pool.len();
        let req_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let req = InferRequest {
            request_id: req_id,
            model,
            texts,
            max_seq_len,
        };
        let conn = self.pool[idx].clone();
        let mut stream = conn.lock().await;
        write_frame(&mut *stream, &req).await?;
        let resp: InferResponse = read_frame(&mut *stream).await?;
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn connect_rejects_zero_pool_size() {
        let path: PathBuf = "/tmp/does-not-matter".into();
        let result = WorkerClient::connect(path, 0).await;
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
}
