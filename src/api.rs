//! HTTP handler for POST /v1/embeddings.
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::cache_flow::partition_hits_and_misses;
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
    // Capture the original request size — this is what clients asked for
    // and what `embed_texts_per_request` must reflect, regardless of how
    // many turn into cache hits vs. misses.
    texts_count = texts.len();

    // -- Cache partition pass --
    //
    // Probe the response cache for each text and split into:
    //   cached[i]  : Some(vec) for hits, None for misses.
    //   pending    : text → positions-to-scatter, deduplicated across
    //                the request (same text at multiple positions → one
    //                entry with many positions).
    //
    // Hit/miss metrics are recorded per-original-position (duplicates
    // count as multiple misses) so the hit ratio is per-text-in-request.
    let (mut cached, pending) = partition_hits_and_misses(&state.cache, &model_name, &texts);

    let hit_count = cached.iter().filter(|c| c.is_some()).count();
    for _ in 0..hit_count {
        crate::metrics::record_cache_hit(&model_name);
    }
    let miss_positions_total: usize = pending.values().map(Vec::len).sum();
    for _ in 0..miss_positions_total {
        crate::metrics::record_cache_miss(&model_name);
    }

    // All-hit short-circuit: every position served from cache, no
    // tokenize or inference needed. Still records request-level metrics.
    if pending.is_empty() {
        let vectors: Vec<Vec<f32>> = cached.into_iter().map(|o| o.expect("all hits")).collect();
        status = "ok";
        crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
        return Json(EmbedResponse {
            object: "list",
            data: vectors
                .into_iter()
                .enumerate()
                .map(|(i, emb)| EmbedData {
                    object: "embedding",
                    embedding: emb,
                    index: i,
                })
                .collect(),
            model: model_name,
            usage: Usage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        })
        .into_response();
    }

    // Only the unique miss texts are tokenized + embedded. `pending_texts`
    // is the aligned key order we'll use to zip miss vectors back to
    // original positions.
    let pending_texts: Vec<String> = pending.keys().cloned().collect();
    let tokenize_input = pending_texts.clone();

    // Tokenize only the unique miss texts. Runs on spawn_blocking because
    // tokenization is CPU-bound and used to contend with the async runtime.
    let model = entry.model.clone();
    let token_ids = match tokio::task::spawn_blocking(move || model.tokenize(&tokenize_input)).await
    {
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
    // Only the miss set flows through here.
    //
    // Cache inserts happen AFTER successful inference — if any error path
    // below is taken, we never populate the cache with a partial result.
    //
    // Note: batcher already calls record_inference inside dispatch_batch — do not call it here.
    let miss_vectors = if let Some(b) = &entry.batcher {
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

    // Sanity: inference returned exactly one vector per unique miss text.
    // A length mismatch would indicate a batcher bug; fail loudly rather
    // than silently producing a wrong response.
    if miss_vectors.len() != pending_texts.len() {
        tracing::error!(
            expected = pending_texts.len(),
            got = miss_vectors.len(),
            "miss vector count mismatch"
        );
        crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "internal error: vector count mismatch".to_string(),
                    error_type: "server_error",
                },
            }),
        )
            .into_response();
    }

    // Scatter each miss vector into every original position it fills,
    // and insert into the cache for future requests.
    for (pending_text, vec) in pending_texts.iter().zip(miss_vectors.into_iter()) {
        state.cache.insert(&model_name, pending_text, vec.clone());
        let positions = pending
            .get(pending_text)
            .expect("pending_text came from pending.keys()");
        for &pos in positions {
            cached[pos] = Some(vec.clone());
        }
    }
    // Refresh the cache-size gauge once per request after the insert
    // batch settles. Single call after all inserts avoids N lock+read
    // cycles — LRU size is monotone within a request (only grows).
    crate::metrics::set_cache_size(state.cache.len());

    let vectors: Vec<Vec<f32>> = cached
        .into_iter()
        .map(|o| o.expect("every position filled"))
        .collect();

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
