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
/// input_type. For a model with `EMBED_INPUT_PREFIX_<MODEL>` set, "document"
/// and "query" inputs produce different vectors for the same text; for the
/// rest the split only costs cache capacity.
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

    // Cache key includes input_type so document and query vectors of an
    // asymmetric (prefixed) model never share an entry.
    let cache_key = cache_model_key(&model_name, input_type);

    // Capture the original request size — this is what clients asked for
    // and what `embed_texts_per_request` must reflect, regardless of how
    // many turn into cache hits vs. misses.
    texts_count = texts.len();

    // Per-model embed concurrency cap — the same TEI `try_acquire_permit`
    // load-shed pattern as `state.rerank_semaphore` (api_rerank.rs),
    // applied per model so one saturated model cannot starve the others.
    // A model absent from `embed_semaphores` is uncapped. The permit is
    // held for the whole post-lookup request lifetime (cache + inference)
    // and released on return; over-cap fails fast with 429 +
    // `Retry-After: 1` rather than queueing on the worker-side gate.
    let _permit = match state.embed_semaphores.get(&model_name) {
        Some(sem) => match sem.clone().try_acquire_owned() {
            Ok(p) => Some(p),
            Err(_) => {
                tracing::warn!(model = %model_name, "embed concurrency cap reached — returning 429");
                status = "rate_limited";
                crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", "1")],
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "embed concurrency cap reached".to_string(),
                            error_type: "rate_limited",
                        },
                    }),
                )
                    .into_response();
            }
        },
        None => None,
    };

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

    // Only the unique miss texts are embedded. `pending_texts` is the aligned
    // key order we'll use to zip miss vectors back to original positions.
    let pending_texts: Vec<String> = pending.keys().cloned().collect();
    let n_pending = pending_texts.len();
    // The model sees the prefixed text (when the model opts in); the cache and
    // the scatter below stay keyed by the caller's raw text.
    let infer_texts = entry
        .input_prefix
        .apply(input_type == InputType::Query, &pending_texts);

    // Run inference via worker pool (multi-process), batcher, or legacy spawn_blocking path.
    // Only the miss set flows through here.
    //
    // Cache inserts happen AFTER successful inference — if any error path
    // below is taken, we never populate the cache with a partial result.
    //
    // Note: batcher already calls record_inference inside dispatch_batch — do not call it here.
    let (miss_vectors, total_tokens) = if let Some(pool) = state.worker_pool.as_ref() {
        // Worker-path billing uses a char/4 estimate instead of running the tokenizer
        // on the supervisor — the worker re-tokenizes authoritatively for inference,
        // so the supervisor-side pass was pure waste (~1.5-3ms for batch=32).
        // Off-by-one OK for billing — exact count requires tokenizer in supervisor.
        let estimated_total_tokens: u32 = infer_texts
            .iter()
            .map(|t| ((t.chars().count() as f32 / 4.0).ceil() as u32).max(1))
            .sum();
        let estimated_max_seq_len: u32 = infer_texts
            .iter()
            .map(|t| ((t.chars().count() as f32 / 4.0).ceil() as u32).max(1))
            .max()
            .unwrap_or(1);

        // Multi-process path: send the texts to the worker; the worker handles
        // tokenization + inference internally. `pending_texts` stays here for
        // the scatter zip (`pending_texts.iter().zip(miss_vectors)`).
        let infer_start = std::time::Instant::now();
        let resp = pool
            .dispatch_embed(&model_name, infer_texts, estimated_max_seq_len)
            .await;
        let infer_elapsed = infer_start.elapsed();
        let vectors = match resp {
            Ok(crate::ipc::protocol::WorkerResponse::Embed(ok)) => {
                crate::metrics::record_inference(&model_name, infer_elapsed, ok.vectors.len());
                ok.vectors
            }
            Ok(crate::ipc::protocol::WorkerResponse::Err {
                request_id,
                message,
            }) => {
                let reason = crate::metrics::classify_worker_error(&message);
                tracing::error!(
                    model = %model_name,
                    request_id,
                    reason,
                    worker_error = %message,
                    "worker returned inference error",
                );
                crate::metrics::record_inference_failure(&model_name, reason, 0);
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
                crate::metrics::record_inference_failure(&model_name, "unexpected_response", 0);
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
                crate::metrics::record_inference_failure(&model_name, e.failure_reason(), 0);
                // #97 / #150: dispatch timeout (worker respawning) is transient →
                // 503 + Retry-After so clients retry instead of treating it
                // as a hard 500 failure. The status label and error_type are
                // DISTINCT from the 500 path so operators and clients can
                // tell "worker respawning, retry in 5s" apart from "request
                // broken" and from 429 backpressure ("slow down, retry in 1s").
                // `Ipc` (worker died mid-request; the supervisor respawns it) is
                // equally transient, so it takes the same 503 branch.
                return match &e {
                    e if e.is_transient() => {
                        status = "worker_unavailable";
                        crate::metrics::record_request(
                            &model_name,
                            status,
                            t0.elapsed(),
                            texts_count,
                        );
                        (
                            StatusCode::SERVICE_UNAVAILABLE,
                            [("retry-after", "5")],
                            Json(ErrorResponse {
                                error: ErrorDetail {
                                    message: if matches!(&e, crate::supervisor::pool::DispatchError::Ipc(io) if io.kind() == std::io::ErrorKind::TimedOut) {
                                        "inference failed: worker did not answer in time"
                                    } else {
                                        "inference failed: worker respawning"
                                    }
                                    .to_string(),
                                    error_type: "worker_unavailable",
                                },
                            }),
                        )
                            .into_response()
                    }
                    _ => {
                        crate::metrics::record_request(
                            &model_name,
                            status,
                            t0.elapsed(),
                            texts_count,
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: ErrorDetail {
                                    message: "inference failed".to_string(),
                                    error_type: "server_error",
                                },
                            }),
                        )
                            .into_response()
                    }
                };
            }
        };
        (vectors, estimated_total_tokens)
    } else {
        // In-process paths (batcher and legacy spawn_blocking) need real token_ids
        // for inference — tokenize the miss set here.
        //
        // Tokenize only the unique miss texts. Runs on spawn_blocking because
        // tokenization is CPU-bound and would contend with the async runtime.
        let model = entry
            .model
            .as_ref()
            .expect(
                "in-process EmbedModel session required but not loaded (EMBED_MULTI_PROCESS=1?)",
            )
            .clone();
        let tokenize_input = infer_texts;
        let token_ids =
            match tokio::task::spawn_blocking(move || model.tokenize(&tokenize_input)).await {
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

        let vectors = if let Some(b) = &entry.batcher {
            match b.embed_tokens(token_ids).await {
                Ok(v) => v,
                Err(crate::batcher::BatchError::QueueFull(e)) => {
                    // E2: queue near capacity (≥80%) → fast-fail with 429
                    // Too Many Requests + Retry-After: 1. Clients (the downstream consumer
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
            let model = entry
                .model
                .as_ref()
                .expect(
                    "in-process EmbedModel session required but not loaded (EMBED_MULTI_PROCESS=1?)",
                )
                .clone();
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
        (vectors, total_tokens)
    };

    // Sanity: inference returned exactly one vector per unique miss text.
    // A length mismatch would indicate a batcher bug; fail loudly rather
    // than silently producing a wrong response.
    if miss_vectors.len() != n_pending {
        tracing::error!(
            expected = n_pending,
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
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
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
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
            embed_semaphores: HashMap::new(),
            embed_max_input_array: cap,
            rerank_max_input_docs: 32,
            worker_pool: None,
            ready_probe_timeout_ms: 2000,
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

    // ------------------------------------------------------------------
    // #150 — 503 worker-respawn must be distinguishable from 500 in BOTH
    // the `embed_requests_total{status}` metric label AND the client-facing
    // `error_type` field. The 429 queue-full path keeps `error_type=
    // "rate_limited"`; the 503 worker-respawn path gets a distinct value.
    //
    // These tests drive the REAL handler through axum with a WorkerPool
    // whose supervisor has no live client (`for_test_no_client`) and a
    // 1 ms dispatch timeout, so `dispatch_embed` returns
    // `DispatchError::Timeout` → the 503 branch fires end-to-end.
    // ------------------------------------------------------------------

    use crate::supervisor::{SpawnSpec, WorkerKind, WorkerPool, WorkerSupervisor};
    use std::path::PathBuf;

    /// Build an AppState wired for the 503 worker-respawn path: a
    /// `WorkerPool` with a no-client supervisor + 1 ms timeout, and a
    /// metadata-only `ModelEntry` (multi-process shape: `model: None`).
    async fn make_state_worker_respawning() -> Arc<AppState> {
        let spec = SpawnSpec {
            model: "test-model".to_string(),
            kind: WorkerKind::Embed,
            worker_bin: PathBuf::from("/bin/true"),
            socket_dir: PathBuf::from("/tmp"),
            pool_size: 1,
            intra_threads: 1,
            env_extra: vec![],
        };
        let sup = WorkerSupervisor::for_test_no_client(spec);
        let pool = WorkerPool::with_dispatch_timeout(std::time::Duration::from_millis(1));
        pool.add(sup).await;
        Arc::new(AppState {
            models: HashMap::from([(
                "test-model".to_string(),
                crate::types::ModelEntry {
                    model: None,
                    batcher: None,
                    input_prefix: Default::default(),
                },
            )]),
            rerankers: HashMap::new(),
            splades: HashMap::new(),
            default_model: "test-model".to_string(),
            shutdown: CancellationToken::new(),
            drain_timeout: Duration::from_secs(5),
            cache: Arc::new(EmbeddingCache::new(0)),
            token_cache: Arc::new(crate::token_cache::TokenCache::new(0)),
            rerank_semaphore: None,
            embed_semaphores: HashMap::new(),
            embed_max_input_array: 32,
            rerank_max_input_docs: 32,
            worker_pool: Some(Arc::new(pool)),
            ready_probe_timeout_ms: 2000,
        })
    }

    /// F1 — the 503 worker-respawn path must emit a `status` label on
    /// `embed_requests_total` that is DISTINCT from the 500 path's
    /// `"server_error"`. Reverting the 503 branch's status back to
    /// `"server_error"` makes this test RED (the regex matches the 500
    /// line, not a distinct 503 line).
    ///
    /// Anchored regex on the rendered /metrics text — a substring check
    /// would go green on broken structured output (the defect that
    /// shipped a blocker on this fleet before).
    #[tokio::test]
    async fn f1_embed_503_status_label_distinct_from_500() {
        let handle = test_prom();
        let state = make_state_worker_respawning().await;
        let app = make_app(state);

        // Send a request that reaches the worker-pool path. The 1 ms
        // dispatch timeout against a no-client supervisor fires
        // DispatchError::Timeout → the 503 branch.
        let resp = app.oneshot(make_request(1)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "worker-respawn path must return HTTP 503"
        );

        let metrics_text = handle.render();
        // Anchored line check: a line starting with `embed_requests_total{`
        // that carries `status="worker_unavailable"` and ends with a number.
        // A substring check would go green on broken structured output (the
        // defect that shipped a blocker on this fleet before).
        let has_503_label = metrics_text.lines().any(|line| {
            line.starts_with("embed_requests_total{")
                && line.contains("status=\"worker_unavailable\"")
                && line
                    .rsplit('}')
                    .next()
                    .map(|s| s.trim().parse::<u64>().is_ok())
                    == Some(true)
        });
        assert!(
            has_503_label,
            "metrics must contain a distinct `worker_unavailable` status line for the 503 path; \
             got:\n{metrics_text}"
        );
        // The 500 path must NOT carry this label — proves the two paths
        // emit distinct status values.
        let has_500_with_same_label = metrics_text.lines().any(|line| {
            line.starts_with("embed_requests_total{")
                && line.contains("status=\"server_error\"")
                && line.contains("worker_unavailable")
        });
        assert!(
            !has_500_with_same_label,
            "500 and 503 paths must not share the same status label"
        );
    }

    /// F3 — the 503 worker-respawn path must carry an `error_type` value
    /// DISTINCT from the 429 queue-full path's `"rate_limited"`. Reverting
    /// the 503 branch's `error_type` back to `"rate_limited"` makes this
    /// test RED (the assertion expects a different value).
    #[tokio::test]
    async fn f3_embed_503_error_type_distinct_from_429() {
        let state = make_state_worker_respawning().await;
        let app = make_app(state);

        let resp = app.oneshot(make_request(1)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        // Capture Retry-Before before consuming the body.
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body_bytes).unwrap();
        // ErrorDetail serializes `error_type` as `"type"` (OpenAI shape).
        let et = json["error"]["type"].as_str().unwrap_or("");
        assert_ne!(
            et, "rate_limited",
            "503 worker-respawn must NOT reuse the 429 queue-full error_type"
        );
        assert!(
            !et.is_empty(),
            "503 worker-respawn must carry a non-empty error_type"
        );
        // Retry-After should differ from the 429's "1" — a respawn wants a
        // longer backoff than backpressure.
        assert_ne!(
            retry_after, "1",
            "503 worker-respawn Retry-After should differ from the 429 backpressure hint"
        );
    }

    // ------------------------------------------------------------------
    // Every 5xx on the worker path must be counted in
    // `embed_inference_failures_total`, and a dead worker must be retryable.
    // ------------------------------------------------------------------

    /// Current value of `embed_inference_failures_total{model,reason}`
    /// (0 when the series is absent).
    fn failures(model: &str, reason: &str) -> f64 {
        let needle = format!(
            "embed_inference_failures_total{{model=\"{model}\",reason=\"{reason}\",bin_num=\"0\"}} "
        );
        test_prom()
            .render()
            .lines()
            .find_map(|l| l.strip_prefix(needle.as_str()))
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    /// Current value of `embed_requests_total{model,status}` (0 when the
    /// series is absent).
    fn request_count(model: &str, status: &str) -> f64 {
        let needle = format!("embed_requests_total{{model=\"{model}\",status=\"{status}\"}} ");
        test_prom()
            .render()
            .lines()
            .find_map(|l| l.strip_prefix(needle.as_str()))
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    fn request_for(model: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"input": ["a"], "model": model}).to_string(),
            ))
            .unwrap()
    }

    fn state_with_pool(
        model: &str,
        pool: WorkerPool,
        embed_semaphores: HashMap<String, Arc<tokio::sync::Semaphore>>,
    ) -> Arc<AppState> {
        state_with_pool_prefix(model, pool, embed_semaphores, Default::default())
    }
    fn state_with_pool_prefix(
        model: &str,
        pool: WorkerPool,
        embed_semaphores: HashMap<String, Arc<tokio::sync::Semaphore>>,
        input_prefix: crate::input_prefix::InputPrefix,
    ) -> Arc<AppState> {
        Arc::new(AppState {
            models: HashMap::from([(
                model.to_string(),
                crate::types::ModelEntry {
                    model: None,
                    batcher: None,
                    input_prefix,
                },
            )]),
            rerankers: HashMap::new(),
            splades: HashMap::new(),
            default_model: model.to_string(),
            shutdown: CancellationToken::new(),
            drain_timeout: Duration::from_secs(5),
            cache: Arc::new(EmbeddingCache::new(0)),
            token_cache: Arc::new(crate::token_cache::TokenCache::new(0)),
            rerank_semaphore: None,
            embed_semaphores,
            embed_max_input_array: 32,
            rerank_max_input_docs: 32,
            worker_pool: Some(Arc::new(pool)),
            ready_probe_timeout_ms: 2000,
        })
    }

    fn spec_for(model: &str) -> SpawnSpec {
        SpawnSpec {
            model: model.to_string(),
            kind: WorkerKind::Embed,
            worker_bin: PathBuf::from("/bin/true"),
            socket_dir: PathBuf::from("/tmp"),
            pool_size: 1,
            intra_threads: 1,
            env_extra: vec![],
        }
    }

    /// A worker that died mid-request: the client connected once (so the
    /// supervisor holds a live `WorkerClient`), then the socket went away.
    /// `dispatch_embed` returns `DispatchError::Ipc`.
    ///
    /// Before: 500 and no failure counter. Now: 503 + Retry-After and
    /// `reason="worker_died"` counted once.
    #[tokio::test]
    async fn embed_ipc_failure_is_counted_and_retryable() {
        let model = "ipc-dead-model";
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("w.sock");
        let listener = tokio::net::UnixListener::bind(&sock).unwrap();
        let client =
            crate::ipc::client::WorkerClient::connect(sock.clone(), 1, Duration::from_secs(5))
                .await
                .unwrap();
        drop(listener);
        std::fs::remove_file(&sock).unwrap();

        let pool = WorkerPool::with_dispatch_timeout(Duration::from_millis(50));
        pool.add(WorkerSupervisor::for_test(
            spec_for(model),
            Arc::new(client),
        ))
        .await;
        let before = failures(model, "worker_died");

        let resp = make_app(state_with_pool(model, pool, HashMap::new()))
            .oneshot(request_for(model))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        assert_eq!(
            failures(model, "worker_died") - before,
            1.0,
            "an Ipc failure must increment embed_inference_failures_total{{reason=\"worker_died\"}}"
        );
    }

    /// A worker on `sock` that records the `max_seq_len` of the Embed request
    /// it last received and answers with one 2-d vector per text. No model.
    fn spawn_recording_worker(sock: &std::path::Path) -> Arc<std::sync::atomic::AtomicU32> {
        use crate::ipc::frame::{read_frame, write_frame};
        use crate::ipc::protocol::{EmbedResponseOk, WorkerRequest, WorkerResponse};
        let seen = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let listener = tokio::net::UnixListener::bind(sock).unwrap();
        let rec = Arc::clone(&seen);
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let rec = Arc::clone(&rec);
                tokio::spawn(async move {
                    // The connect probe (connect + drop) fails to decode: ignore it.
                    let Ok(WorkerRequest::Embed(r)) =
                        read_frame::<_, WorkerRequest>(&mut stream).await
                    else {
                        return;
                    };
                    rec.store(r.max_seq_len, std::sync::atomic::Ordering::Relaxed);
                    let resp = WorkerResponse::Embed(EmbedResponseOk {
                        request_id: r.request_id,
                        vectors: vec![vec![0.5, 1.5]; r.texts.len()],
                        dims: 2,
                    });
                    let _ = write_frame(&mut stream, &resp).await;
                });
            }
        });
        seen
    }

    /// The SUCCESS path through the worker pool, end to end: 200 with one
    /// vector per input, the char/4 billing estimate in `usage`, and the same
    /// estimate handed to the worker as `max_seq_len`. Every other test in
    /// this module drives an error branch, so a handler that turned every
    /// good answer into a 500 (the vector-count sanity check inverted) or
    /// billed garbage went unnoticed.
    ///
    /// Found by cargo-mutants on `src/api.rs`: `/ 4.0` replaced by `% 4.0` or
    /// `* 4.0` (both estimates) and `!= n_pending` replaced by `==` all
    /// survived before this test.
    #[tokio::test]
    async fn embed_worker_success_returns_vectors_and_billing_estimate() {
        let model = "success-model";
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("w.sock");
        let seen = spawn_recording_worker(&sock);
        let client = crate::ipc::client::WorkerClient::connect(sock, 1, Duration::from_secs(5))
            .await
            .unwrap();
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_secs(5));
        pool.add(WorkerSupervisor::for_test(
            spec_for(model),
            Arc::new(client),
        ))
        .await;

        // 9 chars -> ceil(9 / 4) = 3 estimated tokens.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"input": ["abcdefghi"], "model": model}).to_string(),
            ))
            .unwrap();
        let resp = make_app(state_with_pool(model, pool, HashMap::new()))
            .oneshot(req)
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a good worker answer is a 200"
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["data"].as_array().map(Vec::len),
            Some(1),
            "one embedding per input: {json}"
        );
        assert_eq!(
            json["data"][0]["embedding"].as_array().map(Vec::len),
            Some(2),
            "the worker's 2-d vector must be returned: {json}"
        );
        assert_eq!(json["usage"]["prompt_tokens"], 3, "char/4 estimate: {json}");
        assert_eq!(json["usage"]["total_tokens"], 3, "char/4 estimate: {json}");
        assert_eq!(
            seen.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "the worker must be told the estimated max sequence length"
        );
    }

    /// A dispatch timeout (worker respawning) is a 503 and was uncounted too.
    #[tokio::test]
    async fn embed_dispatch_timeout_is_counted() {
        let model = "timeout-counted-model";
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_millis(1));
        pool.add(WorkerSupervisor::for_test_no_client(spec_for(model)))
            .await;
        let before = failures(model, "worker_unavailable");

        let resp = make_app(state_with_pool(model, pool, HashMap::new()))
            .oneshot(request_for(model))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(failures(model, "worker_unavailable") - before, 1.0);
    }

    /// No registered worker is a config error: stays 500 (a retry cannot
    /// fix it) but is counted.
    #[tokio::test]
    async fn embed_no_worker_is_500_and_counted() {
        let model = "no-worker-model";
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_millis(1));
        let before = failures(model, "no_worker");

        let resp = make_app(state_with_pool(model, pool, HashMap::new()))
            .oneshot(request_for(model))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(failures(model, "no_worker") - before, 1.0);
    }

    /// F1 — a worker that accepts the connection and never answers must
    /// surface as a bounded, retryable 503 (not a hang): the per-request
    /// IPC deadline in `WorkerClient::send_request` fires and flows through
    /// `DispatchError::Ipc(TimedOut)` → `is_transient` → 503 +
    /// `Retry-After: 5` + `worker_unavailable`, counted
    /// `reason="ipc_timeout"`. Removing the `tokio::time::timeout` wrapper
    /// in `send_request` makes this test RED — the request hangs until the
    /// outer guard fires.
    #[tokio::test]
    async fn embed_wedged_worker_is_retryable_503_within_deadline() {
        let model = "wedged-model";
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("w.sock");
        let listener = tokio::net::UnixListener::bind(&sock).unwrap();
        tokio::spawn(async move {
            // Accept and hold every connection open forever — the exact
            // signature of a wedged worker (listener alive, never reads).
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let _stream = stream;
                    std::future::pending::<()>().await;
                });
            }
        });
        let client = crate::ipc::client::WorkerClient::connect(sock, 1, Duration::from_millis(300))
            .await
            .unwrap();
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_secs(5));
        pool.add(WorkerSupervisor::for_test(
            spec_for(model),
            Arc::new(client),
        ))
        .await;
        let before = failures(model, "ipc_timeout");

        // Outer guard far above the 300 ms IPC deadline: without the
        // send_request deadline the request hangs until this fires.
        let resp = tokio::time::timeout(
            Duration::from_secs(10),
            make_app(state_with_pool(model, pool, HashMap::new())).oneshot(request_for(model)),
        )
        .await
        .expect("request must return within the IPC deadline, not hang")
        .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a wedged worker must surface as a retryable 503, not a hang or 500"
        );
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        assert_eq!(
            failures(model, "ipc_timeout") - before,
            1.0,
            "IPC deadline expiry must be counted as reason=\"ipc_timeout\""
        );
    }

    /// F2 — a request over the per-model embed concurrency cap is shed
    /// with 429 + `Retry-After: 1` and counted as `status="rate_limited"`
    /// in `embed_requests_total` — the same contract as the rerank
    /// semaphore. Weakening the cap check at its call site in `embeddings`
    /// (e.g. to an impossible condition) makes this test RED.
    #[tokio::test]
    async fn embed_over_concurrency_cap_is_429_rate_limited() {
        let model = "cap-model";
        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        // Occupy the single permit so the request below is over the cap.
        let _held = sem.clone().try_acquire_owned().unwrap();
        let sems = HashMap::from([(model.to_string(), sem)]);
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_millis(1));
        let before = request_count(model, "rate_limited");

        let resp = make_app(state_with_pool(model, pool, sems))
            .oneshot(request_for(model))
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "over the per-model embed concurrency cap must be HTTP 429"
        );
        assert_eq!(resp.headers().get("retry-after").unwrap(), "1");
        assert_eq!(
            request_count(model, "rate_limited") - before,
            1.0,
            "an over-cap request must be counted as status=\"rate_limited\""
        );
    }

    /// F3 — a worker that accepts and then closes the socket mid-request is
    /// a transient `DispatchError::Ipc` (EOF inside `read_frame`): the
    /// handler must answer 503 + `Retry-After` with
    /// `status="worker_unavailable"`, not a 500. Mapping
    /// `DispatchError::Ipc` back to 500 in `embeddings` makes this RED.
    #[tokio::test]
    async fn embed_ipc_mid_request_close_is_retryable_503() {
        let model = "midclose-model";
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("w.sock");
        let listener = tokio::net::UnixListener::bind(&sock).unwrap();
        tokio::spawn(async move {
            // Accept, read the request frame so the client is parked in
            // read_frame waiting for a response, then drop the connection.
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let _ =
                        crate::ipc::frame::read_frame::<_, crate::ipc::protocol::WorkerRequest>(
                            &mut stream,
                        )
                        .await;
                });
            }
        });
        let client = crate::ipc::client::WorkerClient::connect(sock, 1, Duration::from_secs(5))
            .await
            .unwrap();
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_secs(5));
        pool.add(WorkerSupervisor::for_test(
            spec_for(model),
            Arc::new(client),
        ))
        .await;
        let before_died = failures(model, "worker_died");
        let before_unavail = request_count(model, "worker_unavailable");

        let resp = make_app(state_with_pool(model, pool, HashMap::new()))
            .oneshot(request_for(model))
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a worker closing the socket mid-request must surface as 503, not 500"
        );
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        assert_eq!(
            failures(model, "worker_died") - before_died,
            1.0,
            "a mid-request socket close must count as reason=\"worker_died\""
        );
        assert_eq!(
            request_count(model, "worker_unavailable") - before_unavail,
            1.0,
            "a mid-request socket close must count as status=\"worker_unavailable\""
        );
    }

    // ---- fake-worker tests: the worker answers, but wrongly ----

    use crate::ipc::protocol::{EmbedResponseOk, RerankResponseOk, WorkerResponse};

    /// Drive /v1/embeddings against a fake worker that replies with `reply`.
    async fn embed_against_fake(
        model: &str,
        reply: fn(u64) -> WorkerResponse,
    ) -> axum::response::Response {
        let dir = tempfile::tempdir().unwrap();
        let client = crate::ipc::client::test_support::fake_worker_client(
            dir.path().join("w.sock"),
            reply,
            Duration::from_secs(5),
        )
        .await;
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_millis(50));
        pool.add(WorkerSupervisor::for_test(
            spec_for(model),
            Arc::new(client),
        ))
        .await;
        make_app(state_with_pool(model, pool, HashMap::new()))
            .oneshot(request_for(model))
            .await
            .unwrap()
    }

    /// A worker answering an embed request with a rerank response is a
    /// 500 (`unexpected_response`), and must be counted.
    #[tokio::test]
    async fn embed_unexpected_variant_is_500_and_counted() {
        let model = "wrong-variant-model";
        let before = failures(model, "unexpected_response");
        let resp = embed_against_fake(model, |id| {
            WorkerResponse::Rerank(RerankResponseOk {
                request_id: id,
                scores: vec![],
            })
        })
        .await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(failures(model, "unexpected_response") - before, 1.0);
    }

    /// A live worker returning the wrong number of vectors broke the IPC
    /// contract: deterministic, so 500 `ipc_protocol` (not a retryable 503
    /// `worker_died`).
    #[tokio::test]
    async fn embed_wrong_vector_count_is_500_ipc_protocol() {
        let model = "wrong-count-model";
        let before = failures(model, "ipc_protocol");
        let died = failures(model, "worker_died");
        let resp = embed_against_fake(model, |id| {
            WorkerResponse::Embed(EmbedResponseOk {
                request_id: id,
                vectors: vec![],
                dims: 4,
            })
        })
        .await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(resp.headers().get("retry-after").is_none());
        assert_eq!(failures(model, "ipc_protocol") - before, 1.0);
        assert_eq!(failures(model, "worker_died"), died);
    }

    // ---- input prefix: what the worker is actually asked to embed ----

    /// A worker that records the texts of every Embed request it receives and
    /// answers with one 2-d vector per text.
    async fn spy_worker_pool(
        model: &str,
        dir: &std::path::Path,
        seen: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    ) -> WorkerPool {
        use crate::ipc::frame::{read_frame, write_frame};
        use crate::ipc::protocol::WorkerRequest;
        let sock = dir.join("spy.sock");
        let listener = tokio::net::UnixListener::bind(&sock).unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let seen = seen.clone();
                tokio::spawn(async move {
                    let Ok(WorkerRequest::Embed(r)) =
                        read_frame::<_, WorkerRequest>(&mut stream).await
                    else {
                        return;
                    };
                    seen.lock().unwrap().push(r.texts.clone());
                    let resp = WorkerResponse::Embed(EmbedResponseOk {
                        request_id: r.request_id,
                        vectors: vec![vec![0.0, 1.0]; r.texts.len()],
                        dims: 2,
                    });
                    let _ = write_frame(&mut stream, &resp).await;
                });
            }
        });
        let client = crate::ipc::client::WorkerClient::connect(sock, 1, Duration::from_secs(60))
            .await
            .unwrap();
        let pool = WorkerPool::with_dispatch_timeout(Duration::from_millis(500));
        pool.add(WorkerSupervisor::for_test(
            spec_for(model),
            Arc::new(client),
        ))
        .await;
        pool
    }

    /// POST one embed request and return the texts the worker received.
    async fn texts_seen_by_worker(
        prefix: crate::input_prefix::InputPrefix,
        body: serde_json::Value,
    ) -> Vec<String> {
        let dir = tempfile::tempdir().unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let pool = spy_worker_pool("prefix-model", dir.path(), seen.clone()).await;
        let resp = make_app(state_with_pool_prefix(
            "prefix-model",
            pool,
            HashMap::new(),
            prefix,
        ))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 1, "one worker request expected");
        calls.remove(0)
    }

    #[tokio::test]
    async fn embed_without_prefix_sends_raw_text() {
        use crate::input_prefix::InputPrefix;
        let got = texts_seen_by_worker(
            InputPrefix::Off,
            serde_json::json!({"input": ["hello"], "model": "prefix-model", "input_type": "query"}),
        )
        .await;
        assert_eq!(got, vec!["hello"], "prefix off must not touch the text");
    }

    #[tokio::test]
    async fn embed_with_e5_prefix_follows_input_type() {
        use crate::input_prefix::InputPrefix;
        let query = texts_seen_by_worker(
            InputPrefix::E5,
            serde_json::json!({"input": ["hello"], "model": "prefix-model", "input_type": "query"}),
        )
        .await;
        assert_eq!(
            query,
            vec!["query: hello"],
            "input_type=query must get the query prefix"
        );
        // The worker sees the cache misses in hash-map order: compare sorted.
        let mut doc = texts_seen_by_worker(
            InputPrefix::E5,
            serde_json::json!({"input": ["hello", "passage: kept"], "model": "prefix-model"}),
        )
        .await;
        doc.sort();
        assert_eq!(
            doc,
            vec!["passage: hello", "passage: kept"],
            "default input_type is document; an already-prefixed text is not prefixed twice"
        );
    }
}
