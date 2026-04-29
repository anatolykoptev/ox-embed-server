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
    ErrorResponse, InputType, Usage, error_json,
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
}
