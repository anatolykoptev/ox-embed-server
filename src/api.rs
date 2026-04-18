//! HTTP handler for POST /v1/embeddings.
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::types::{
    AppState, EmbedData, EmbedRequest, EmbedResponse, ErrorDetail, ErrorResponse, Usage, error_json,
};

/// POST /v1/embeddings — OpenAI-compatible embedding endpoint.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> impl IntoResponse {
    // Reject new requests while draining.
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

    let t0 = std::time::Instant::now();
    let mut status = "error";
    let mut texts_count: usize = 0;

    let model_name = req.model.unwrap_or_else(|| state.default_model.clone());

    let entry = match state.models.get(&model_name) {
        Some(e) => e,
        None => {
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return error_json(format!("model '{model_name}' not found")).into_response();
        }
    };

    let texts = req.input.into_vec();
    if texts.is_empty() {
        crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
        return error_json("input must not be empty").into_response();
    }
    texts_count = texts.len();

    // Tokenize before the batch path so the batcher can account for
    // token counts (Phase B token-budget accounting) and to keep the
    // blocking ONNX thread focused purely on inference.
    //
    // Run on spawn_blocking: tokenization is CPU-bound and was the last
    // sync path left on the tokio reactor. Under concurrent load, workers
    // contended for CPU here, the batcher's 10ms window closed before
    // stragglers arrived, and batches collapsed to single-item.
    let model = entry.model.clone();
    let token_ids = match tokio::task::spawn_blocking(move || model.tokenize(&texts)).await {
        Ok(Ok(ids)) => ids,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "tokenize failed");
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e,
                        error_type: "server_error",
                    },
                }),
            )
                .into_response();
        }
        Err(join_err) => {
            tracing::error!(error = %join_err, "tokenize task panicked");
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("tokenize task panicked: {join_err}"),
                        error_type: "server_error",
                    },
                }),
            )
                .into_response();
        }
    };

    // Run inference via batcher (if enabled) or legacy spawn_blocking path.
    // Note: batcher already calls record_inference inside dispatch_batch — do not call it here.
    let vectors = if let Some(b) = &entry.batcher {
        match b.embed_tokens(token_ids).await {
            Ok(v) => v,
            Err(crate::batcher::BatchError::QueueFull(e)) => {
                tracing::warn!(error = %e, "queue full");
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [("retry-after", "1")],
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: e.to_string(),
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
            Err(crate::batcher::BatchError::Inference(msg)) => {
                tracing::error!(error = %msg, "embed failed");
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: msg,
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
            Err(crate::batcher::BatchError::Shutdown) => {
                tracing::error!("batcher shut down");
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "batcher shut down".to_string(),
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
        }
    } else {
        // Legacy path: run in spawn_blocking to avoid holding the async executor on sync ort call.
        let model = entry.model.clone();
        let infer_start = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || model.embed_tokens(&token_ids))
            .await
            .map_err(|e| format!("spawn: {e}"));
        let infer_elapsed = infer_start.elapsed();

        match result {
            Ok(Ok(v)) => {
                crate::metrics::record_inference(&model_name, infer_elapsed, v.len());
                v
            }
            Ok(Err(e)) => {
                tracing::error!(error = %e, "embed failed");
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: e,
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: e,
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
        }
    };

    status = "ok";

    let data: Vec<EmbedData> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, emb)| EmbedData {
            object: "embedding",
            embedding: emb,
            index: i,
        })
        .collect();

    crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
    Json(EmbedResponse {
        object: "list",
        data,
        model: model_name,
        usage: Usage {
            prompt_tokens: 0,
            total_tokens: 0,
        },
    })
    .into_response()
}
