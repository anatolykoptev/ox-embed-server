//! Supervisor-side client to one worker process.
//!
//! Holds N persistent UDS connections (matches worker's pool_size); each
//! connection runs requests serially. Dispatch methods round-robin across pool.

use crate::ipc::frame::{read_frame, write_frame};
use crate::ipc::protocol::{
    EmbedRequest, RerankRequest, SpladeRequest, WorkerRequest, WorkerResponse,
};
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

    /// Send a request and receive a response, routing to a pool slot via
    /// round-robin. Verifies request_id echo.
    async fn send_request(&self, req: WorkerRequest) -> std::io::Result<WorkerResponse> {
        let idx = (self.next_idx.fetch_add(1, Ordering::Relaxed) as usize) % self.pool.len();
        let req_id = req.request_id();
        let conn = self.pool[idx].clone();
        let mut stream = conn.lock().await;
        write_frame(&mut *stream, &req).await?;
        let resp: WorkerResponse = read_frame(&mut *stream).await?;
        // NOTE: on read_frame/write_frame error the stream is left in undefined state.
        // This pool slot becomes effectively unusable until WorkerSupervisor (Wave 2.5)
        // detects the worker death and reconnects all slots.
        if resp.request_id() != req_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("response id {} != request id {}", resp.request_id(), req_id),
            ));
        }
        Ok(resp)
    }

    /// Dispatch an embed request. Verifies response cardinality matches input texts.
    pub async fn dispatch_embed(
        &self,
        model: String,
        texts: Vec<String>,
        max_seq_len: u32,
    ) -> std::io::Result<WorkerResponse> {
        let req_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let texts_len = texts.len();
        let resp = self
            .send_request(WorkerRequest::Embed(EmbedRequest {
                request_id: req_id,
                model,
                texts,
                max_seq_len,
            }))
            .await?;
        if let WorkerResponse::Embed(ref ok) = resp
            && ok.vectors.len() != texts_len
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "worker returned {} vectors for {} input texts",
                    ok.vectors.len(),
                    texts_len
                ),
            ));
        }
        Ok(resp)
    }

    /// Dispatch a rerank request. Verifies response score count matches document count.
    pub async fn dispatch_rerank(
        &self,
        model: String,
        query: String,
        documents: Vec<String>,
        max_seq_len: u32,
    ) -> std::io::Result<WorkerResponse> {
        let req_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let docs_len = documents.len();
        let resp = self
            .send_request(WorkerRequest::Rerank(RerankRequest {
                request_id: req_id,
                model,
                query,
                documents,
                max_seq_len,
            }))
            .await?;
        if let WorkerResponse::Rerank(ref ok) = resp
            && ok.scores.len() != docs_len
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "worker returned {} scores for {} documents",
                    ok.scores.len(),
                    docs_len
                ),
            ));
        }
        Ok(resp)
    }

    /// Dispatch a splade request. Verifies response sparse vector count matches input texts.
    pub async fn dispatch_splade(
        &self,
        model: String,
        texts: Vec<String>,
        max_seq_len: u32,
        top_k: u32,
        min_weight: f32,
    ) -> std::io::Result<WorkerResponse> {
        let req_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let texts_len = texts.len();
        let resp = self
            .send_request(WorkerRequest::Splade(SpladeRequest {
                request_id: req_id,
                model,
                texts,
                max_seq_len,
                top_k,
                min_weight,
            }))
            .await?;
        if let WorkerResponse::Splade(ref ok) = resp
            && ok.sparse.len() != texts_len
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "worker returned {} sparse vectors for {} texts",
                    ok.sparse.len(),
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
