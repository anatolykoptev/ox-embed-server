//! HTTP request/response types and shared application state.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::batcher::DynamicBatcher;
use crate::cache::EmbeddingCache;
use crate::model::EmbedModel;
use crate::model_reranker::RerankerModel;
use crate::model_splade::SpladeModel;
use crate::token_cache::TokenCache;

// --- State ---

/// Entry for a single model: its inference handle and optional batcher.
pub struct ModelEntry {
    pub model: Arc<EmbedModel>,
    pub batcher: Option<Arc<DynamicBatcher>>,
}

/// Entry for a single cross-encoder reranker: its inference handle plus
/// the batcher adapter wired in `main.rs`. The batcher's `Vec<Vec<f32>>`
/// contract is bent to fit a scalar-per-pair model by returning
/// 1-element inner vecs — see the adapter comment in `main.rs`.
pub struct RerankerEntry {
    #[allow(dead_code)] // consumed by /v1/rerank handler (E3)
    pub model: Arc<RerankerModel>,
    /// Reranker batching reuses the same `BATCHING_ENABLED` switch as
    /// embeddings: when true the handler dispatches through this batcher,
    /// when false `main.rs` leaves it `None` and the handler (E3) falls
    /// back to a direct `spawn_blocking(score_pairs)` call.
    #[allow(dead_code)] // consumed by /v1/rerank handler (E3)
    pub batcher: Option<Arc<DynamicBatcher>>,
}

/// Entry for a single SPLADE sparse encoder. v1 is batcher-free —
/// `/v1/sparse_embeddings` dispatches one `spawn_blocking` per text
/// rather than queuing into `DynamicBatcher`. The field is reserved
/// here so a follow-up batcher wiring can land without changing
/// `AppState`'s shape.
pub struct SpladeEntry {
    pub model: Arc<SpladeModel>,
    /// Reserved for batcher integration in a follow-up. Always `None`
    /// in v1 — present so the field exists when batching is wired up.
    #[allow(dead_code)]
    pub batcher: Option<Arc<DynamicBatcher>>,
}

/// Shared application state.
pub struct AppState {
    pub models: HashMap<String, ModelEntry>,
    /// Zero-or-more cross-encoder rerankers keyed by model name. An
    /// empty map is valid — the server still serves `/v1/embeddings`;
    /// `/v1/rerank` with any model name will 404/400 (E3).
    #[allow(dead_code)] // consumed by /v1/rerank handler (E3)
    pub rerankers: HashMap<String, RerankerEntry>,
    /// Zero-or-more SPLADE sparse encoders keyed by model name. Empty
    /// map is valid (default — `SPLADE_MODELS` unset); the server still
    /// boots and `/v1/sparse_embeddings` returns 400 for any request.
    pub splades: HashMap<String, SpladeEntry>,
    pub default_model: String,
    /// Cancelled on SIGTERM/SIGINT; handlers check this to reject new requests.
    pub shutdown: CancellationToken,
    /// How long to wait for in-flight requests before axum stops the listener.
    #[allow(dead_code)]
    pub drain_timeout: Duration,
    /// Process-local LRU cache for deterministic embedding lookups.
    /// Disabled (no-op) when constructed with capacity 0.
    pub cache: Arc<EmbeddingCache>,
    /// Per-pair tokenizer cache for the reranker hot path (H.7).
    /// Disabled (no-op) when constructed with capacity 0.
    /// Enabled via `TOKEN_CACHE_MAX_ENTRIES` env var.
    pub token_cache: Arc<TokenCache>,
    /// Global concurrency cap for `/v1/rerank` requests. Prior art: TEI's
    /// `Infer::try_acquire_permit` — failing fast at the HTTP layer with
    /// 429 (instead of letting requests buffer in the batcher's mpsc up
    /// to MAX_QUEUE_SIZE) keeps tokenizer CPU from burning on work that
    /// will be cancelled by upstream timeout. The semaphore tracks
    /// in-flight requests exactly (not approximated by channel depth).
    /// `None` = unlimited (legacy behaviour); set via
    /// `MAX_CONCURRENT_RERANK_REQUESTS`.
    pub rerank_semaphore: Option<Arc<tokio::sync::Semaphore>>,
    /// Maximum number of texts allowed in a single `/v1/embeddings` input
    /// array. Requests exceeding this are rejected with HTTP 400 before
    /// reaching the batcher. Configured via `EMBED_MAX_INPUT_ARRAY`
    /// (default 32). See `Config::embed_max_input_array` for rationale.
    pub embed_max_input_array: usize,
    /// Maximum number of documents allowed in a single `/v1/rerank` request.
    /// Requests exceeding this are rejected with HTTP 400 before tokenization.
    /// Configured via `RERANK_MAX_INPUT_DOCS` (default 32, matching
    /// `embed_max_input_array` so operators memorise one number).
    /// See `Config::rerank_max_input_docs` for rationale.
    pub rerank_max_input_docs: usize,
    /// Active worker pool in multi-process mode (`EMBED_MULTI_PROCESS=1`).
    ///
    /// `None` in legacy single-process mode. When `Some`, each embed model
    /// has a corresponding `WorkerHandle` in the pool; API routing via workers
    /// lands in Wave 2.4. For now the pool is held here so workers are kept
    /// alive (their `Child` drops on `WorkerHandle` drop) for the server lifetime.
    #[allow(dead_code)] // consumed by Wave 2.4 API routing
    pub worker_pool: Option<Arc<crate::supervisor::WorkerPool>>,
}

// --- Request types ---

/// Encoding format for embedding output, matching the OpenAI API spec.
///
/// - `Float` (default): embeddings as JSON arrays of float32.
/// - `Base64`: embeddings as a base64-encoded string of raw little-endian
///   float32 bytes. Reduces HTTP payload by ~33% for bulk ingest paths.
#[derive(Deserialize, Debug, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

/// Document vs. query distinction for asymmetric embedding models
/// (Voyage-style). Our current symmetric models (e5/gte/bge-m3) treat
/// this as a no-op at inference time but it is **included in the cache
/// key** so that future asymmetric-model deploys don't cause cache
/// pollution between document and query vectors.
#[derive(Deserialize, Debug, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    #[default]
    Document,
    Query,
}

#[derive(Deserialize)]
pub struct EmbedRequest {
    pub input: InputField,
    pub model: Option<String>,
    /// Encoding format for the `embedding` field in the response.
    /// Defaults to `float` (JSON array). Use `base64` for bulk ingest
    /// paths to reduce HTTP payload by ~33%.
    pub encoding_format: Option<EncodingFormat>,
    /// Document or query input type. Currently a no-op for our symmetric
    /// models but accepted and included in the cache key for future
    /// asymmetric model support. Defaults to `document`.
    pub input_type: Option<InputType>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum InputField {
    Single(String),
    Batch(Vec<String>),
}

impl InputField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            InputField::Single(s) => vec![s],
            InputField::Batch(v) => v,
        }
    }
}

// --- Response types ---

/// The embedding value in a response: either a JSON float array or a
/// base64-encoded string (little-endian f32 bytes). Serialized as
/// untagged so the field stays `"embedding": [...]` vs `"embedding": "..."`.
#[derive(Serialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Vector(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
pub struct EmbedResponse {
    pub object: &'static str,
    pub data: Vec<EmbedData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct EmbedData {
    pub object: &'static str,
    pub embedding: EmbeddingValue,
    pub index: usize,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
}

/// Richer error body for `input_array_too_large` — includes `cap` and
/// `received` so clients can immediately see what limit they hit and how
/// to split their requests without reading docs.
#[derive(Serialize)]
pub struct InputArrayTooLargeDetail {
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub code: &'static str,
    pub message: String,
    pub cap: usize,
    pub received: usize,
}

#[derive(Serialize)]
pub struct InputArrayTooLargeResponse {
    pub error: InputArrayTooLargeDetail,
}

pub fn error_json(msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.into(),
                error_type: "invalid_request_error",
            },
        }),
    )
}
