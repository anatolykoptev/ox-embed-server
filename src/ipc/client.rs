//! Supervisor-side client to one worker process.
//!
//! Per-request connection: each dispatch_* opens a fresh UDS connection,
//! writes the request, reads the response, drops the connection. UDS
//! local-domain connect is ~10-100µs — negligible vs ONNX inference (5-50ms).
//!
//! Why per-request: holding a persistent `Mutex<UnixStream>` across
//! write_frame + read_frame is not cancel-safe. If the caller's future
//! is cancelled (axum timeout, downstream client timeout) between write
//! and read, the response stays buffered. Next caller gets the stale
//! response. Per-request conn drops on cancellation — no stale buffer.

use crate::ipc::frame::{read_frame, write_frame};
use crate::ipc::protocol::{
    EmbedRequest, RerankRequest, SpladeRequest, WorkerRequest, WorkerResponse,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::UnixStream;

pub struct WorkerClient {
    socket_path: PathBuf,
    /// Semaphore caps concurrent in-flight connections. Capacity = `conns` arg to connect().
    semaphore: tokio::sync::Semaphore,
    request_counter: AtomicU64,
}

impl WorkerClient {
    pub async fn connect(socket_path: PathBuf, conns: usize) -> std::io::Result<Self> {
        if conns == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WorkerClient concurrency must be >= 1",
            ));
        }
        // Probe that we can actually connect (one throwaway connection — verifies the
        // socket is bound + the worker is accepting). Drops immediately. Avoids
        // returning an apparently-healthy client whose first real request would 500.
        let probe = UnixStream::connect(&socket_path).await?;
        drop(probe);
        Ok(Self {
            socket_path,
            semaphore: tokio::sync::Semaphore::new(conns),
            request_counter: AtomicU64::new(0),
        })
    }

    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    /// Send a request and receive a response over a fresh UDS connection.
    /// The connection is dropped when this future completes or is cancelled.
    async fn send_request(&self, req: WorkerRequest) -> std::io::Result<WorkerResponse> {
        let _permit = self.semaphore.acquire().await.expect("semaphore closed");
        let req_id = req.request_id();
        let mut stream = UnixStream::connect(&self.socket_path).await?;
        write_frame(&mut stream, &req).await?;
        let resp: WorkerResponse = read_frame(&mut stream).await?;
        if resp.request_id() != req_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("response id {} != request id {}", resp.request_id(), req_id),
            ));
        }
        Ok(resp)
        // stream drops here — closes UDS connection automatically
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
    async fn connect_rejects_zero_concurrency() {
        let path: PathBuf = "/tmp/does-not-matter".into();
        let res = WorkerClient::connect(path, 0).await;
        match res {
            Ok(_) => panic!("must error"),
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput),
        }
    }
}
