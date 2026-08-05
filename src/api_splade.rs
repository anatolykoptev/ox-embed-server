//! HTTP handler for POST /embed_sparse — TEI-style SPLADE sparse-retrieval
//! endpoint.
//!
//! Returns a Qdrant-compatible sparse vector shape per input text:
//!
//! ```json
//! { "model": "splade-v3-distilbert",
//!   "data":  [ { "index": 0,
//!                "indices": [12, 345, 6789],
//!                "values":  [4.2, 3.1, 0.7] }, ... ] }
//! ```
//!
//! # No batcher (v1)
//!
//! Unlike `/v1/embeddings` and `/v1/rerank`, this handler does NOT route
//! through `DynamicBatcher`. SPLADE is single-text-per-call by design,
//! and v1 keeps the dispatch simple: one `spawn_blocking` per text in
//! the request, sequentially. Wiring the dynamic batcher is a follow-up
//! once we observe traffic shapes — sparse retrieval workloads tend to
//! ingest one document at a time, so the batcher amortisation may not
//! be worth the adapter complexity.
//!
//! # No cache
//!
//! Mirrors `/v1/rerank`: SPLADE traffic is dominated by indexing fresh
//! documents (each text seen once), and a process-local LRU would burn
//! RAM for near-zero hit ratio. Embedding cache stays focused on the
//! dense embedding workload.
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::types::{AppState, ErrorDetail, ErrorResponse, error_json};

// ---------------------------------------------------------------------
// Request / response types — colocated with the handler the same way
// `api_rerank.rs` keeps its types.
// ---------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SparseEmbeddingsRequest {
    /// Optional: when the server has exactly one SPLADE model configured
    /// it's implicitly selected. With 2+ SPLADE models this becomes
    /// required (400 otherwise) — same disambiguation contract as
    /// `/v1/rerank` to avoid picking a non-deterministic "first".
    pub model: Option<String>,
    /// Texts to encode. Must be non-empty and contain at least one text.
    /// Each input is encoded independently.
    pub input: Vec<String>,
    /// Maximum sparse entries per output. Defaults to 256 when absent —
    /// covers the typical SPLADE working range without truncating
    /// expansion terms most clients want.
    pub top_k: Option<usize>,
    /// Drop sparse entries with weight `<=` this threshold. Defaults to
    /// 0.0 (drop only exact zeros from the post-ReLU output).
    pub min_weight: Option<f32>,
}

#[derive(Serialize)]
pub struct SparseEmbeddingItem {
    /// Position in the original `input` array (0-based). Preserved across
    /// the response so the client can map results back to texts.
    pub index: usize,
    /// Vocabulary token ids with non-zero weight, sorted by weight desc.
    pub indices: Vec<u32>,
    /// Weights aligned 1:1 with `indices`.
    pub values: Vec<f32>,
}

#[derive(Serialize)]
pub struct SparseEmbeddingsResponse {
    pub model: String,
    pub data: Vec<SparseEmbeddingItem>,
}

const DEFAULT_TOP_K: usize = 256;
const DEFAULT_MIN_WEIGHT: f32 = 0.0;

/// Resolve which SPLADE model to use.
///
///   - Explicit `req.model` → look up by name (400 if not found).
///   - Absent + exactly one SPLADE configured → use it.
///   - Absent + zero or 2+ → 400.
fn resolve_splade_name(state: &AppState, req_model: Option<String>) -> Result<String, String> {
    if let Some(name) = req_model {
        if state.splades.contains_key(&name) {
            Ok(name)
        } else {
            Err(format!("splade model '{name}' not found"))
        }
    } else {
        match state.splades.len() {
            0 => Err("no splade models configured".to_string()),
            1 => Ok(state
                .splades
                .keys()
                .next()
                .expect("len==1 guarantees one key")
                .clone()),
            _ => Err("`model` is required when multiple splade models are configured".to_string()),
        }
    }
}

/// POST /embed_sparse.
pub async fn sparse_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SparseEmbeddingsRequest>,
) -> Response {
    // Shutdown gate — same 503 + retry-after pattern as `/v1/rerank`
    // and `/v1/embeddings`.
    if state.shutdown.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", "5")],
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "server shutting down".to_string(),
                    error_type: "rate_limited",
                },
            }),
        )
            .into_response();
    }

    if req.input.is_empty() {
        return error_json("input must not be empty").into_response();
    }
    // Defensive: an individual blank string can't be tokenised to anything
    // useful and likely indicates a client bug; reject early instead of
    // returning an empty sparse vector. Consistent with `/v1/rerank`'s
    // empty-query rejection.
    if req.input.iter().any(|s| s.trim().is_empty()) {
        return error_json("input texts must not be empty or whitespace-only").into_response();
    }

    let model_name = match resolve_splade_name(&state, req.model) {
        Ok(n) => n,
        Err(msg) => return error_json(msg).into_response(),
    };
    let entry = state
        .splades
        .get(&model_name)
        .expect("resolve_splade_name validated key presence");

    let top_k = req.top_k.unwrap_or(DEFAULT_TOP_K);
    let min_weight = req.min_weight.unwrap_or(DEFAULT_MIN_WEIGHT);

    // Worker pool dispatch (multi-process splade). When enabled, route to
    // worker before the in-process per-text spawn_blocking path.
    if let Some(pool) = state.worker_pool.as_ref() {
        let texts = req.input.clone();
        let resp = pool
            .dispatch_splade(&model_name, texts, 0, top_k as u32, min_weight)
            .await;
        match resp {
            Ok(crate::ipc::protocol::WorkerResponse::Splade(s)) => {
                let data: Vec<SparseEmbeddingItem> = s
                    .sparse
                    .into_iter()
                    .enumerate()
                    .map(|(index, pairs)| {
                        let mut indices = Vec::with_capacity(pairs.len());
                        let mut values = Vec::with_capacity(pairs.len());
                        for (id, w) in pairs {
                            // Apply top_k / min_weight filters that the worker
                            // does not know about (it uses fixed defaults).
                            if w > min_weight {
                                indices.push(id);
                                values.push(w);
                            }
                        }
                        indices.truncate(top_k);
                        values.truncate(top_k);
                        SparseEmbeddingItem {
                            index,
                            indices,
                            values,
                        }
                    })
                    .collect();
                return Json(SparseEmbeddingsResponse {
                    model: model_name,
                    data,
                })
                .into_response();
            }
            Ok(crate::ipc::protocol::WorkerResponse::Err { message, .. }) => {
                let reason = crate::metrics::classify_worker_error(&message);
                tracing::error!(model = %model_name, reason, worker_error = %message, "worker splade returned error");
                crate::metrics::record_inference_failure(&model_name, reason, 0);
                return server_error("splade failed".to_string());
            }
            Ok(_unexpected) => {
                tracing::error!(model = %model_name, "worker returned unexpected variant for splade request");
                return server_error("splade failed: unexpected response kind".to_string());
            }
            Err(e) => {
                use crate::supervisor::pool::DispatchError;
                tracing::error!(model = %model_name, error = ?e, "worker_pool splade dispatch failed");
                // #97 / #150: dispatch timeout → 503 + Retry-After (transient).
                // Distinct error_type from the 429 backpressure path so clients
                // can tell respawn from rate-limit. retry-after=5 (respawn
                // takes 5-15s) vs the 429's 1s backpressure hint.
                let resp = match &e {
                    DispatchError::Timeout { .. } => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        [("retry-after", "5")],
                        Json(ErrorResponse {
                            error: ErrorDetail {
                                message: "splade failed: worker respawning".to_string(),
                                error_type: "worker_unavailable",
                            },
                        }),
                    )
                        .into_response(),
                    _ => server_error("splade failed".to_string()),
                };
                return resp;
            }
        }
    }

    // Sequential per-text dispatch. v1 deliberately skips the dynamic
    // batcher — each call goes through its own `spawn_blocking` so the
    // async runtime stays responsive while ORT does its CPU-bound work.
    // Follow-up: route through DynamicBatcher once SPLADE traffic shape
    // makes the per-batch padding overhead worth amortising.
    let mut data: Vec<SparseEmbeddingItem> = Vec::with_capacity(req.input.len());
    for (index, text) in req.input.into_iter().enumerate() {
        let model = entry
            .model
            .as_ref()
            .expect(
                "in-process SpladeModel session required but not loaded (EMBED_MULTI_PROCESS=1?)",
            )
            .clone();
        let result = tokio::task::spawn_blocking(move || {
            let ids = model.tokenize(&text)?;
            model.encode_sparse(ids, top_k, min_weight)
        })
        .await;

        let entries = match result {
            Ok(Ok(e)) => e,
            Ok(Err(e)) => {
                tracing::error!(error = %e, index, "splade encode failed");
                return server_error(e);
            }
            Err(join_err) => {
                tracing::error!(error = %join_err, index, "splade task panicked");
                return server_error(format!("splade task panicked: {join_err}"));
            }
        };

        // Split (id, weight) pairs into Qdrant's parallel-array shape.
        // Capacity hint avoids the second alloc grow cycle on top_k=256.
        let mut indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        for (id, w) in entries {
            indices.push(id);
            values.push(w);
        }
        data.push(SparseEmbeddingItem {
            index,
            indices,
            values,
        });
    }

    Json(SparseEmbeddingsResponse {
        model: model_name,
        data,
    })
    .into_response()
}

/// 500 wrapper with `server_error` label — same style as `api_rerank`.
fn server_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: "server_error",
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // SparseEmbeddingsRequest deserialization — pins the optional-field
    // contract: a minimal `{ "input": [...] }` request must parse, and
    // explicit `top_k` / `min_weight` / `model` must round-trip.
    // -----------------------------------------------------------------

    #[test]
    fn request_deserializes_with_only_input() {
        let j = r#"{"input":["hello"]}"#;
        let r: SparseEmbeddingsRequest = serde_json::from_str(j).unwrap();
        assert!(r.model.is_none());
        assert!(r.top_k.is_none());
        assert!(r.min_weight.is_none());
        assert_eq!(r.input, vec!["hello".to_string()]);
    }

    #[test]
    fn request_deserializes_with_all_fields() {
        let j = r#"{"model":"m","input":["a","b"],"top_k":50,"min_weight":0.1}"#;
        let r: SparseEmbeddingsRequest = serde_json::from_str(j).unwrap();
        assert_eq!(r.model.as_deref(), Some("m"));
        assert_eq!(r.input.len(), 2);
        assert_eq!(r.top_k, Some(50));
        assert_eq!(r.min_weight, Some(0.1));
    }

    // -----------------------------------------------------------------
    // resolve_splade_name — pure resolution logic, testable without an
    // ONNX session loaded. Mirrors `api_rerank::tests` style.
    // -----------------------------------------------------------------

    fn empty_state() -> AppState {
        use std::collections::HashMap;
        use std::time::Duration;
        use tokio_util::sync::CancellationToken;
        AppState {
            models: HashMap::new(),
            rerankers: HashMap::new(),
            splades: HashMap::new(),
            default_model: "noop".into(),
            shutdown: CancellationToken::new(),
            drain_timeout: Duration::from_secs(1),
            cache: Arc::new(crate::cache::EmbeddingCache::new(0)),
            token_cache: Arc::new(crate::token_cache::TokenCache::new(0)),
            rerank_semaphore: None,
            embed_max_input_array: 32,
            rerank_max_input_docs: 32,
            worker_pool: None,
            ready_probe_timeout_ms: 2000,
        }
    }

    #[test]
    fn resolve_zero_splades_errors() {
        let state = empty_state();
        assert!(resolve_splade_name(&state, None).is_err());
        assert!(resolve_splade_name(&state, Some("anything".into())).is_err());
    }
}
