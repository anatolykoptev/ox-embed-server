//! HTTP handler for POST /v1/embeddings.
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use base64::Engine as _;

use crate::cache_flow::partition_hits_and_misses;
use crate::types::{
    AppState, EmbedData, EmbedRequest, EmbedResponse, EmbeddingValue, EncodingFormat, ErrorDetail,
    ErrorResponse, InputArrayTooLargeDetail, InputArrayTooLargeResponse, InputType, Usage,
    error_json,
};

/// Encode a float32 vector as base64 (little-endian f32 bytes).
fn encode_base64(vec: &[f32]) -> String {
    let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

/// Build the cache-model key that partitions entries by both model name and
/// input_type. This future-proofs the cache against asymmetric models (e.g.
/// Voyage) where "document" and "query" inputs produce different vectors for
/// the same text. For current symmetric models this is a no-op distinction,
/// but baking it in now prevents cache pollution on a model swap.
fn cache_model_key(model_name: &str, input_type: InputType) -> String {
    match input_type {
        InputType::Document => model_name.to_string(),
        InputType::Query => format!("{model_name}:query"),
    }
}

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
    let encoding_format = req.encoding_format.unwrap_or_default();
    let input_type = req.input_type.unwrap_or_default();

    // Parse input early so we can check length and record the histogram
    // before incurring any model-lookup or batcher cost.
    let texts = req.input.into_vec();
    if texts.is_empty() {
        crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
        return error_json("input must not be empty").into_response();
    }

    // Always record the input array size — both accepted and rejected paths —
    // so operators can see the natural distribution and tune EMBED_MAX_INPUT_ARRAY.
    crate::metrics::record_input_array_size(&model_name, texts.len());

    // Server-side cap: reject oversized input arrays BEFORE the batcher.
    // B=100 × H=12 × S=512² × 4 = 1.258 GiB attention scratch per inference
    // caused BFCArena OOM at ~1/min in prod. Default cap=32 keeps scratch
    // under 402 MiB. This is a permanent client misuse (HTTP 400), not a
    // transient overload (503), so Retry-After is intentionally absent.
    let cap = state.embed_max_input_array;
    if texts.len() > cap {
        let input_len = texts.len();
        tracing::warn!(
            input_len,
            cap,
            model = %model_name,
            "input array exceeds cap; rejecting with 400"
        );
        crate::metrics::record_input_array_rejected(&model_name, "size_cap");
        crate::metrics::record_request(&model_name, status, t0.elapsed(), input_len);
        return (
            StatusCode::BAD_REQUEST,
            Json(InputArrayTooLargeResponse {
                error: InputArrayTooLargeDetail {
                    error_type: "invalid_request_error",
                    code: "input_array_too_large",
                    message: format!(
                        "input array contains {input_len} items, server cap is {cap}; \
                         split into multiple requests"
                    ),
                    cap,
                    received: input_len,
                },
            }),
        )
            .into_response();
    }

    let entry = match state.models.get(&model_name) {
        Some(e) => e,
        None => {
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return error_json(format!("model '{model_name}' not found")).into_response();
        }
    };

    // Cache key includes input_type to prevent pollution between document/query
    // vectors if an asymmetric model is deployed in the future.
    let cache_key = cache_model_key(&model_name, input_type);

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
    let (mut cached, pending) = partition_hits_and_misses(&state.cache, &cache_key, &texts);

    let hit_count = cached.iter().filter(|c| c.is_some()).count();
    crate::metrics::record_cache_hit_n(&model_name, hit_count as u64);
    let miss_positions_total: usize = pending.values().map(Vec::len).sum();
    crate::metrics::record_cache_miss_n(&model_name, miss_positions_total as u64);

    // Cache-hit path: skip backend entirely, report 0 tokens (consistent
    // with miss path's "hits report 0, only missed texts charged").
    if pending.is_empty() {
        let vectors: Vec<Vec<f32>> = cached.into_iter().map(|o| o.expect("all hits")).collect();
        // Full cache hit — no backend work performed, no tokens charged.
        // Matches OpenAI billing semantics: tokens are billed for compute we
        // actually did. The same request on a cold cache will report
        // total_tokens > 0; on a warm cache the caller already paid for those
        // tokens in a prior request.
        let total_tokens: u32 = 0;
        status = "ok";
        crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
        return Json(EmbedResponse {
            object: "list",
            data: build_data(vectors, encoding_format),
            model: model_name,
            usage: Usage {
                prompt_tokens: total_tokens,
                total_tokens,
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
            tracing::warn!(error = %e, "tokenize failed in /v1/embeddings");
            crate::metrics::record_tokenize_fallback();
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
            tracing::warn!(error = %join_err, "tokenize task panicked in /v1/embeddings");
            crate::metrics::record_tokenize_fallback();
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

    // Count tokens for the fresh (miss) set only. Cache hits report 0 tokens
    // (they were accounted in a prior request). Token count is the sum of
    // non-padded token lengths across all unique miss texts.
    let total_tokens: u32 = token_ids.iter().map(|v| v.len() as u32).sum();

    // Run inference via worker pool (multi-process), batcher, or legacy spawn_blocking path.
    // Only the miss set flows through here.
    //
    // Worker pool path: dispatch raw pending_texts to the worker process; the
    // worker handles its own tokenization + inference internally.
    //
    // Cache inserts happen AFTER successful inference — if any error path
    // below is taken, we never populate the cache with a partial result.
    //
    // Note: batcher already calls record_inference inside dispatch_batch — do not call it here.
    let max_seq_len: u32 = token_ids.iter().map(|v| v.len() as u32).max().unwrap_or(1);
    let miss_vectors = if let Some(pool) = state.worker_pool.as_ref() {
        // Multi-process path: send raw texts to the worker; token_ids were
        // already computed above for the total_tokens billing count only.
        let infer_start = std::time::Instant::now();
        let resp = pool
            .dispatch_embed(&model_name, pending_texts.clone(), max_seq_len)
            .await;
        let infer_elapsed = infer_start.elapsed();
        match resp {
            Ok(crate::ipc::protocol::WorkerResponse::Embed(ok)) => {
                crate::metrics::record_inference(&model_name, infer_elapsed, ok.vectors.len());
                ok.vectors
            }
            Ok(crate::ipc::protocol::WorkerResponse::Err {
                request_id,
                message,
            }) => {
                tracing::error!(
                    model = %model_name,
                    request_id,
                    worker_error = %message,
                    "worker returned inference error",
                );
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "inference failed".to_string(),
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
            Ok(unexpected) => {
                tracing::error!(
                    model = %model_name,
                    kind = %unexpected.kind(),
                    request_id = unexpected.request_id(),
                    "worker returned unexpected response variant for embed request",
                );
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "inference failed: unexpected response kind".to_string(),
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                tracing::error!(model = %model_name, error = ?e, "worker_pool dispatch failed");
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "inference failed".to_string(),
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response();
            }
        }
    } else if let Some(b) = &entry.batcher {
        match b.embed_tokens(token_ids).await {
            Ok(v) => v,
            Err(crate::batcher::BatchError::QueueFull(e)) => {
                // E2: queue near capacity (≥80%) → fast-fail with 429
                // Too Many Requests + Retry-After: 1. Clients (memdb-go
                // commit 90b964f1) retry with exp backoff — closed
                // loop. Previously returned 503, which conflated "queue
                // full" (retryable) with shutdown (also retryable but
                // usually transient differently).
                tracing::warn!(error = %e, "queue full — returning 429");
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", "1")],
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: e.to_string(),
                            error_type: "rate_limited",
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
    for (pending_text, vec) in pending_texts.iter().zip(miss_vectors) {
        state.cache.insert(&cache_key, pending_text, vec.clone());
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

    crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
    Json(EmbedResponse {
        object: "list",
        data: build_data(vectors, encoding_format),
        model: model_name,
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    })
    .into_response()
}

/// Build the `data` field of an EmbedResponse from raw vectors and the
/// requested encoding format. This function is the single place that converts
/// `Vec<f32>` → `EmbeddingValue` so both the cache-hit and miss paths stay DRY.
fn build_data(vectors: Vec<Vec<f32>>, format: EncodingFormat) -> Vec<EmbedData> {
    vectors
        .into_iter()
        .enumerate()
        .map(|(i, vec)| {
            let embedding = match format {
                EncodingFormat::Float => EmbeddingValue::Vector(vec),
                EncodingFormat::Base64 => EmbeddingValue::Base64(encode_base64(&vec)),
            };
            EmbedData {
                object: "embedding",
                embedding,
                index: i,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use crate::cache::EmbeddingCache;
    use crate::types::{EmbeddingValue, EncodingFormat, InputType};

    use super::{build_data, cache_model_key, encode_base64};

    // ------------------------------------------------------------------
    // Feature C: encoding_format deserialization + base64 round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_encoding_format_float_default() {
        // Missing `encoding_format` field → Float (serde default).
        let json = r#"{"input": "hello", "model": "test"}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.encoding_format, None);
        // Unwrap default → Float.
        assert_eq!(
            req.encoding_format.unwrap_or_default(),
            EncodingFormat::Float
        );
    }

    #[test]
    fn test_encoding_format_base64_deserializes() {
        let json = r#"{"input": "hello", "encoding_format": "base64"}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.encoding_format, Some(EncodingFormat::Base64));
    }

    #[test]
    fn test_encoding_format_unknown_rejected() {
        // "hex" is not a valid EncodingFormat variant.
        let json = r#"{"input": "hello", "encoding_format": "hex"}"#;
        let result: Result<crate::types::EmbedRequest, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "unknown encoding_format should be rejected by serde"
        );
    }

    #[test]
    fn test_encoding_format_base64_decodes_to_original_floats() {
        // encode_base64 produces little-endian f32 bytes; decode and verify round-trip.
        let original = vec![1.0_f32, -0.5_f32, 0.25_f32, 0.0_f32];
        let encoded = encode_base64(&original);

        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .expect("valid base64");
        assert_eq!(decoded_bytes.len(), original.len() * 4, "4 bytes per f32");

        let decoded_floats: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_floats, original, "round-trip must be lossless");
    }

    #[test]
    fn test_build_data_float_format() {
        let vecs = vec![vec![1.0_f32, 2.0_f32], vec![3.0_f32]];
        let data = build_data(vecs, EncodingFormat::Float);
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].index, 0);
        assert_eq!(data[1].index, 1);
        assert_eq!(data[0].embedding, EmbeddingValue::Vector(vec![1.0, 2.0]));
        assert_eq!(data[1].embedding, EmbeddingValue::Vector(vec![3.0]));
    }

    #[test]
    fn test_build_data_base64_format() {
        let v = vec![1.0_f32, 0.0_f32];
        let data = build_data(vec![v.clone()], EncodingFormat::Base64);
        assert_eq!(data.len(), 1);
        match &data[0].embedding {
            EmbeddingValue::Base64(s) => {
                // Decode and verify.
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(s)
                    .expect("valid base64");
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert_eq!(floats, v);
            }
            other => panic!("expected Base64, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Feature D: input_type deserialization + cache key isolation
    // ------------------------------------------------------------------

    #[test]
    fn test_input_type_default_document() {
        let json = r#"{"input": "hello"}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.input_type, None);
        assert_eq!(req.input_type.unwrap_or_default(), InputType::Document);
    }

    #[test]
    fn test_input_type_query_deserializes() {
        let json = r#"{"input": "hello", "input_type": "query"}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.input_type, Some(InputType::Query));
    }

    #[test]
    fn test_input_type_unknown_rejected() {
        // "passage" is not a valid InputType variant.
        let json = r#"{"input": "hello", "input_type": "passage"}"#;
        let result: Result<crate::types::EmbedRequest, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "unknown input_type should be rejected by serde"
        );
    }

    #[test]
    fn test_cache_key_differs_per_input_type() {
        // Same model name but different input_type must produce different cache keys
        // so asymmetric models don't collide in the process-local LRU cache.
        let doc_key = cache_model_key("e5", InputType::Document);
        let query_key = cache_model_key("e5", InputType::Query);
        assert_ne!(
            doc_key, query_key,
            "document and query must use separate cache namespaces"
        );
        // Verify the document key is the plain model name (no suffix) for backward
        // compat: existing cache entries stored before this feature landed remain valid.
        assert_eq!(doc_key, "e5");
        assert!(query_key.contains(":query"));
    }

    #[test]
    fn test_cache_key_document_is_model_name() {
        // Document (default) must equal plain model name so pre-existing cache entries
        // written before input_type support landed remain valid on upgrade.
        assert_eq!(
            cache_model_key("multilingual-e5-large", InputType::Document),
            "multilingual-e5-large"
        );
    }

    // ------------------------------------------------------------------
    // Backward compat: legacy {input, model}-only request deserializes
    // ------------------------------------------------------------------

    #[test]
    fn test_legacy_request_only_input_model_works() {
        let json = r#"{"input": "hello world", "model": "multilingual-e5-large"}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("legacy request must deserialize cleanly");
        // All new fields default to None — no encoding_format, no input_type.
        assert_eq!(req.encoding_format, None);
        assert_eq!(req.input_type, None);
        // Defaults unwrap to the expected values.
        assert_eq!(
            req.encoding_format.unwrap_or_default(),
            EncodingFormat::Float
        );
        assert_eq!(req.input_type.unwrap_or_default(), InputType::Document);
    }

    #[test]
    fn test_legacy_single_string_input() {
        let json = r#"{"input": "single"}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("single-string input must work");
        let texts = req.input.into_vec();
        assert_eq!(texts, vec!["single".to_string()]);
    }

    #[test]
    fn test_legacy_batch_input() {
        let json = r#"{"input": ["a", "b", "c"]}"#;
        let req: crate::types::EmbedRequest =
            serde_json::from_str(json).expect("batch input must work");
        let texts = req.input.into_vec();
        assert_eq!(texts, vec!["a", "b", "c"]);
    }

    // ------------------------------------------------------------------
    // Feature A: token count logic (pure, no ONNX needed)
    // ------------------------------------------------------------------

    #[test]
    fn test_token_sum_from_ids() {
        // Verify the token-counting formula used in the handler is correct.
        // The handler does: token_ids.iter().map(|v| v.len() as u32).sum()
        let token_ids: Vec<Vec<u32>> = vec![
            vec![101, 1234, 5678, 102], // 4 tokens
            vec![101, 9999, 102],       // 3 tokens
        ];
        let total: u32 = token_ids.iter().map(|v| v.len() as u32).sum();
        assert_eq!(
            total, 7,
            "sum of token sequence lengths must equal total tokens"
        );
    }

    // ------------------------------------------------------------------
    // Feature C + cache: cache stores full-dim vectors (no truncation)
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_stores_raw_vectors() {
        // The cache key for a document input is the bare model name.
        let cache = EmbeddingCache::new(10);
        let model = "e5";
        let text = "hello";
        let vec = vec![0.1_f32, 0.2_f32, 0.3_f32];

        cache.insert(model, text, vec.clone());
        let retrieved = cache.get(model, text).expect("cache hit expected");
        assert_eq!(retrieved, vec);
    }

    // ------------------------------------------------------------------
    // Input-array cap (EMBED_MAX_INPUT_ARRAY)
    // ------------------------------------------------------------------
    //
    // These tests drive the embeddings handler through axum's tower stack
    // using a minimal AppState (no real model entries needed — the cap
    // check fires before model lookup).

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use serde_json::Value;
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt as _;

    use crate::token_cache::TokenCache;
    use crate::types::AppState;

    /// Delegate to the shared test recorder so only one `install_recorder()`
    /// call happens per process regardless of which test module runs first.
    fn test_prom() -> &'static metrics_exporter_prometheus::PrometheusHandle {
        crate::metrics::test_prometheus_handle()
    }

    /// Build a minimal `AppState` with the given `embed_max_input_array` cap.
    /// All model maps are empty — tests targeting the cap check fire before
    /// the model-lookup path.
    fn make_state(cap: usize) -> Arc<AppState> {
        Arc::new(AppState {
            models: HashMap::new(),
            rerankers: HashMap::new(),
            splades: HashMap::new(),
            default_model: "test-model".to_string(),
            shutdown: CancellationToken::new(),
            drain_timeout: Duration::from_secs(5),
            cache: Arc::new(EmbeddingCache::new(0)),
            token_cache: Arc::new(TokenCache::new(0)),
            rerank_semaphore: None,
            embed_max_input_array: cap,
            rerank_max_input_docs: 32,
            worker_pool: None,
        })
    }

    /// Build a POST /v1/embeddings request with an input array of `n` strings.
    fn make_request(n: usize) -> Request<Body> {
        let inputs: Vec<String> = (0..n).map(|i| format!("text {i}")).collect();
        let body = serde_json::json!({
            "input": inputs,
            "model": "test-model"
        });
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn make_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/v1/embeddings", post(crate::api::embeddings))
            .with_state(state)
    }

    // ---- test: oversized input → 400 with correct JSON body ----

    #[tokio::test]
    async fn embeddings_handler_rejects_oversized_input_array() {
        let handle = test_prom();
        let state = make_state(32);
        let app = make_app(state);

        // 100 texts, cap=32 → must reject with 400.
        let resp = app.oneshot(make_request(100)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "oversized input array must return HTTP 400"
        );

        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(
            json["error"]["code"], "input_array_too_large",
            "error.code must be input_array_too_large"
        );
        assert_eq!(
            json["error"]["type"], "invalid_request_error",
            "error.type must be invalid_request_error"
        );
        assert_eq!(json["error"]["cap"], 32, "cap must equal configured cap");
        assert_eq!(
            json["error"]["received"], 100,
            "received must equal actual input length"
        );
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("split into multiple requests"),
            "message must suggest splitting"
        );

        // Counter must have been incremented.
        let counter_text = handle.render();
        assert!(
            counter_text.contains("embed_input_array_rejected_total"),
            "embed_input_array_rejected_total counter must appear in metrics after rejection"
        );
    }

    // ---- test: exactly at cap → cap check passes (no input_array_too_large) ----

    #[tokio::test]
    async fn embeddings_handler_accepts_at_cap() {
        let state = make_state(32);
        let app = make_app(state);

        // 32 texts, cap=32 → cap check passes (len == cap, NOT > cap).
        // The request then fails with "model not found" (empty models map),
        // but that's a different 400 — verifying it does NOT have code=input_array_too_large
        // proves the cap check passed.
        let resp = app.oneshot(make_request(32)).await.unwrap();
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body_bytes).unwrap();

        // Must NOT be an input_array_too_large rejection.
        assert_ne!(
            json["error"]["code"], "input_array_too_large",
            "request at exactly cap=32 must not trigger the array-size rejection"
        );
    }

    // ---- test: histogram records both accept and reject paths ----

    #[tokio::test]
    async fn embed_input_array_size_histogram_records_both_accept_and_reject() {
        let handle = test_prom();

        // Rejected path: 100 texts, cap=32.
        {
            let app = make_app(make_state(32));
            let _resp = app.oneshot(make_request(100)).await.unwrap();
        }

        // Accepted path: 5 texts, cap=32.
        {
            let app = make_app(make_state(32));
            let _resp = app.oneshot(make_request(5)).await.unwrap();
        }

        let metrics_text = handle.render();
        assert!(
            metrics_text.contains("embed_input_array_size"),
            "embed_input_array_size histogram must appear in /metrics output \
             for both accepted and rejected requests"
        );
    }
}
