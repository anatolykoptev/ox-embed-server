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
        // next_idx and req_id are independent Relaxed atomics — under concurrent callers
        // the pairing (which slot got which id) is arbitrary, but each callsite gets a
        // consistent (idx, req_id) snapshot. Pool slots serialize via Mutex so a single
        // slot's req/resp interleaving is correct.
        let idx = (self.next_idx.fetch_add(1, Ordering::Relaxed) as usize) % self.pool.len();
        let req_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        // Capture texts_len before moving texts into InferRequest so we can
        // validate response cardinality after the round-trip.
        let texts_len = texts.len();
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
        // NOTE: on read_frame/write_frame error the stream is left in undefined state.
        // This pool slot becomes effectively unusable until WorkerSupervisor (Wave 2.5)
        // detects the worker death and reconnects all slots.
        let resp_id = match &resp {
            InferResponse::Ok { request_id, .. } => *request_id,
            InferResponse::Err { request_id, .. } => *request_id,
        };
        if resp_id != req_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("response id {resp_id} != request id {req_id}"),
            ));
        }
        // Validate that the worker returned exactly one vector per input text.
        // A cardinality mismatch indicates a worker bug; fail loudly rather than
        // silently scattering a wrong result into the response cache.
        if let InferResponse::Ok { ref vectors, .. } = resp
            && vectors.len() != texts_len
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "worker returned {} vectors for {} input texts",
                    vectors.len(),
                    texts_len
                ),
            ));
        }
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
