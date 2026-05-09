//! HTTP handler for POST /v1/rerank — Cohere-compatible cross-encoder
//! reranker endpoint.
//!
//! # Cache bypass
//!
//! Unlike `/v1/embeddings`, this handler intentionally does NOT consult
//! the response cache. The cache key would need to cover the full
//! `(query, doc)` pair and rerank traffic is almost always unique:
//!
//!   - a typical request has one query against N candidate documents;
//!   - the next request uses a different query, so the previous docs
//!     will re-hit the model under a new composite key;
//!   - even when a doc repeats across requests, its query rarely does.
//!
//! Caching would therefore burn RAM for near-zero hit ratio while
//! lengthening the hot path with a probe+insert pass. The embed cache
//! stays focused on its actual wins (reused `(model, single_text)`
//! lookups for embedding workloads).
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::metrics;
use crate::types::{AppState, ErrorDetail, ErrorResponse, error_json};

/// Richer error body for `documents_too_many` — includes `cap` and `received`
/// so clients can immediately see what limit they hit and how to split their
/// requests without reading docs. Mirrors `InputArrayTooLargeResponse` in
/// `types.rs` for the embeddings endpoint.
#[derive(Serialize)]
struct DocumentsTooManyDetail {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    message: String,
    cap: usize,
    received: usize,
}

#[derive(Serialize)]
struct DocumentsTooManyResponse {
    error: DocumentsTooManyDetail,
}

// ---------------------------------------------------------------------
// G2-server: optional server-side score normalization.
// ---------------------------------------------------------------------

/// Sigmoid input clamp threshold. f32::exp() overflows around x=88.7
/// (yielding +∞), and at x=-88.7 the denominator 1+exp(-x) ≈ exp(89)
/// also overflows. Clamping to ±50 keeps exp() arg in a comfortable
/// range while still giving sigmoid(±50) ≈ 1 / 0 within f32 precision
/// (the difference between sigmoid(50) and sigmoid(88) is ~1e-22,
/// far below f32 ULP at that magnitude).
const SIGMOID_CLAMP: f32 = 50.0;

/// Optional server-side normalization applied to cross-encoder logits
/// before sort. Default `Sigmoid` returns [0,1] scores compatible with
/// quality-floor consumers (memdb-go, Mem0, etc). Send `"normalize":"none"`
/// in the request body for raw cross-encoder logits (Cohere/Jina
/// convention) when the downstream consumer expects them.
///
/// 2026-05-02 — flipped default from `None` to `Sigmoid` after Run #12
/// LoCoMo F1 regression (-39% vs Run #8). Root cause: memdb-go consumer
/// never sent `"normalize"` → received raw logits (often negative for
/// top-1) → `MEMDB_CE_QUALITY_FLOOR=0.05` floor dropped 99% of CE
/// results into math fallback (63% low_quality + 36% degraded in prod).
/// BREAKING for callers that expected raw logits without sending the
/// field — Cohere SDK consumers must now opt in via `"normalize":"none"`.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum NormalizeMode {
    /// Identity — return raw cross-encoder logits. Opt-in for Cohere SDK
    /// compat: send `"normalize":"none"` in request body.
    None,
    /// Apply 1/(1+exp(-x)) per score → [0,1]. Default since 2026-05-02.
    #[default]
    Sigmoid,
}

/// Pure score normalization — applied after `score_pairs` and before
/// `build_sorted_results`. Sigmoid is monotonic so sort order is preserved.
pub(crate) fn apply_normalize(scores: &mut [f32], mode: NormalizeMode) {
    match mode {
        NormalizeMode::None => {}
        NormalizeMode::Sigmoid => {
            for s in scores.iter_mut() {
                // Clamp inputs to ±50 to avoid (-x).exp() overflowing f32
                // (~1e22 at x=-50). Values outside ±50 are effectively
                // saturated (sigmoid(-50) ≈ 0, sigmoid(50) ≈ 1).
                let clamped = s.clamp(-SIGMOID_CLAMP, SIGMOID_CLAMP);
                *s = 1.0 / (1.0 + (-clamped).exp());
            }
        }
    }
}

// ---------------------------------------------------------------------
// Request / response types. Kept local to this module (not `types.rs`)
// because they are only consumed by this one handler — colocating
// request/response with their handler keeps the API surface area near
// its usage, mirroring how many axum codebases evolve once they sprout
// a second endpoint.
// ---------------------------------------------------------------------

/// Document input. Cohere accepts EITHER a plain string OR an object
/// `{"text": "..."}`. Many SDKs (Jina/Voyage/Mixedbread/Cohere SDK) send
/// the object form — supporting both is required for drop-in compat.
/// `serde(untagged)` picks the variant by shape.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object { text: String },
}

impl RerankDocument {
    fn into_text(self) -> String {
        match self {
            Self::Text(s) => s,
            Self::Object { text } => text,
        }
    }
}

#[derive(Deserialize)]
pub struct RerankRequest {
    /// Optional: if the server has exactly one reranker configured,
    /// it's implicitly selected. With 2+ rerankers this field becomes
    /// required (400 otherwise) — we avoid picking a non-deterministic
    /// "first" from the HashMap's unspecified iteration order.
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<RerankDocument>,
    /// Optional: if absent, all scored results are returned in sorted
    /// order. If `0`, an empty `results` array is returned. If greater
    /// than `documents.len()`, we silently cap at `documents.len()`.
    pub top_n: Option<usize>,
    /// Cohere-compat: when true, each result includes a `document` field
    /// with the original text. Default false to keep responses small —
    /// most clients already have the source text and only need scores.
    #[serde(default)]
    pub return_documents: bool,
    /// Optional server-side score normalization. When absent the server
    /// applies the `NormalizeMode` default (`Sigmoid` since 2026-05-02 →
    /// [0,1] scores). Send `"none"` for raw cross-encoder logits
    /// (Cohere/Jina convention). `"sigmoid"` is also accepted as an
    /// explicit opt-in. Other values rejected with 400.
    #[serde(default)]
    pub normalize: Option<NormalizeMode>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct RerankResult {
    /// Position of the document in the *original request's* `documents`
    /// array (0-based). Preserved across sort so clients can map scores
    /// back to the text they sent.
    pub index: usize,
    /// Cross-encoder relevance score. By default sigmoid-normalized to
    /// [0,1] (default since 2026-05-02; higher = more relevant). Send
    /// `"normalize":"none"` in the request for raw logits / Cohere SDK
    /// compat (unbounded, often negative for low-relevance pairs).
    pub relevance_score: f32,
    /// Set only when the request had `return_documents=true`. Cohere shape:
    /// `{"text": "..."}` object, NOT a bare string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocumentEcho>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct RerankDocumentEcho {
    pub text: String,
}

#[derive(Serialize)]
pub struct RerankResponse {
    /// Cohere-compat: opaque request id (UUID). Clients sometimes log it
    /// for tracing; we generate fresh per request so failures can be
    /// correlated across distributed traces.
    pub id: String,
    pub model: String,
    pub results: Vec<RerankResult>,
}

/// Pure sort+truncate: zip scores with positional indices, sort by
/// score descending, apply optional `top_n` cap. Extracted as a
/// `pub(crate)` free function so it's unit-testable without an
/// `AppState` harness (see the E3 tests below).
pub(crate) fn build_sorted_results(
    scores: &[f32],
    top_n: Option<usize>,
    documents: Option<&[String]>,
) -> Vec<RerankResult> {
    let mut scored: Vec<RerankResult> = scores
        .iter()
        .enumerate()
        .map(|(index, &relevance_score)| RerankResult {
            index,
            relevance_score,
            document: documents.map(|d| RerankDocumentEcho {
                text: d[index].clone(),
            }),
        })
        .collect();
    // `total_cmp` gives a total order over f32 (handles NaN / -0.0),
    // unlike `partial_cmp` which returns `None` for NaN and would
    // panic via `.unwrap()`. Reranker logits from the ONNX graph
    // should never be NaN, but defense-in-depth is cheap.
    scored.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(n) = top_n {
        scored.truncate(n);
    }
    scored
}

/// Resolve which reranker to use for this request.
///
///   - Explicit `req.model` → look up by name (400 if not found).
///   - Absent `req.model` + exactly one configured reranker → use it.
///   - Absent `req.model` + zero or 2+ rerankers → 400.
///
/// We deliberately don't fall back to `state.default_model`: that name
/// belongs to the embedding-model namespace and almost certainly isn't
/// present in `state.rerankers`.
fn resolve_reranker_name(state: &AppState, req_model: Option<String>) -> Result<String, String> {
    if let Some(name) = req_model {
        if state.rerankers.contains_key(&name) {
            Ok(name)
        } else {
            Err(format!("reranker '{name}' not found"))
        }
    } else {
        match state.rerankers.len() {
            0 => Err("no reranker models configured".to_string()),
            1 => Ok(state
                .rerankers
                .keys()
                .next()
                .expect("len==1 guarantees one key")
                .clone()),
            _ => Err("`model` is required when multiple rerankers are configured".to_string()),
        }
    }
}

/// POST /v1/rerank — Cohere-compatible cross-encoder rerank endpoint.
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RerankRequest>,
) -> Response {
    // Phase 1A: end-to-end request timer. Started before any work so
    // semaphore-rejected (429) and shutdown (503) requests are also
    // attributed in `embed_rerank_request_duration_seconds`. Pre-model
    // exits use label `model="unknown"` because we have not yet resolved
    // which reranker would have served the request.
    let request_start = Instant::now();

    // Global rerank concurrency cap (load-shed before tokenize). Prior
    // art: TEI's `Infer::try_acquire_permit`. When MAX_CONCURRENT_RERANK_REQUESTS
    // is unset, `state.rerank_semaphore` is None and this is a no-op.
    // The acquired permit is held for the entire request lifetime
    // (including async wait on batcher) and released when `_permit`
    // drops at function return.
    let _permit = match &state.rerank_semaphore {
        Some(sem) => match sem.clone().try_acquire_owned() {
            Ok(p) => Some(p),
            Err(_) => {
                tracing::warn!("rerank concurrency cap reached — returning 429");
                metrics::record_rerank_request(
                    "unknown",
                    "rate_limited",
                    request_start.elapsed(),
                    0,
                );
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", "1")],
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "rerank concurrency cap reached".to_string(),
                            error_type: "rate_limited",
                        },
                    }),
                )
                    .into_response();
            }
        },
        None => None,
    };

    // Shutdown gate — matches `api::embeddings`. Reject new requests
    // with 503 once SIGTERM has flipped the token, giving clients a
    // Retry-After hint to back off while drain completes.
    if state.shutdown.is_cancelled() {
        metrics::record_rerank_request("unknown", "shutdown", request_start.elapsed(), 0);
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

    // Validate inputs BEFORE resolving a model — a malformed request
    // shouldn't need reranker configuration to reach a 400.
    if req.query.trim().is_empty() {
        metrics::record_rerank_request("unknown", "bad_request", request_start.elapsed(), 0);
        return error_json("query must not be empty").into_response();
    }
    if req.documents.is_empty() {
        metrics::record_rerank_request("unknown", "bad_request", request_start.elapsed(), 0);
        return error_json("documents must not be empty").into_response();
    }

    // Always record the documents array size — both accepted and rejected
    // paths — so operators can see the natural distribution and tune
    // RERANK_MAX_INPUT_DOCS.
    let input_len = req.documents.len();
    metrics::record_rerank_input_docs_size(input_len);

    // Server-side cap: reject oversized documents arrays BEFORE tokenization.
    // At cap=32, gte-multi-rerank (max_len=256) scratch ≈ 32×1×256²×4 ≈ 8 MiB
    // per slot — well under the arena. Quadratic cost is 4× lower than jina
    // (S=512). This is permanent client misuse (HTTP 400), not transient load.
    let cap = state.rerank_max_input_docs;
    if input_len > cap {
        tracing::warn!(
            input_len,
            cap,
            "rerank documents exceed cap; rejecting with 400"
        );
        metrics::record_rerank_input_docs_rejected("size_cap");
        metrics::record_rerank_request(
            "unknown",
            "bad_request",
            request_start.elapsed(),
            input_len,
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(DocumentsTooManyResponse {
                error: DocumentsTooManyDetail {
                    error_type: "invalid_request_error",
                    code: "documents_too_many",
                    message: format!(
                        "documents array contains {input_len} items, server cap is {cap}; \
                         split into multiple requests"
                    ),
                    cap,
                    received: input_len,
                },
            }),
        )
            .into_response();
    }

    let model_name = match resolve_reranker_name(&state, req.model) {
        Ok(n) => n,
        Err(msg) => {
            metrics::record_rerank_request("unknown", "bad_request", request_start.elapsed(), 0);
            return error_json(msg).into_response();
        }
    };
    // `expect` is safe: `resolve_reranker_name` guarantees presence.
    let entry = state
        .rerankers
        .get(&model_name)
        .expect("resolve_reranker_name validated key presence");

    // Phase 1A: in-flight gauge. RAII increment/decrement across the
    // entire remaining handler lifetime, including async waits on
    // tokenize and batcher dispatch. Drop runs on every return path —
    // success, error, panic.
    let _in_flight = metrics::RerankInFlightGuard::new(&model_name);

    let query = req.query;
    // Normalise mixed string/object form into a flat Vec<String>. Cohere
    // accepts either; clients rarely care which they send.
    let documents: Vec<String> = req.documents.into_iter().map(|d| d.into_text()).collect();
    let doc_count = documents.len();
    let return_documents = req.return_documents;
    let normalize = req.normalize;

    // Phase 1A helper: deduplicates the terminal record_rerank_request
    // call across the ~6 post-model-resolution exit points. Required
    // because each branch chooses its own status_code + body, so a single
    // `?`-friendly Result<Response, _> shape doesn't fit. The macro is
    // hygienic and only exists in this fn's scope.
    #[allow(unused_macros)]
    macro_rules! finish {
        ($status:literal, $resp:expr) => {{
            metrics::record_rerank_request(
                &model_name,
                $status,
                request_start.elapsed(),
                doc_count,
            );
            return $resp;
        }};
    }

    // ---- Phase: tokenize (with H.7 cache + timing) -----------------------
    let token_ids =
        match tokenize_with_cache(&state, &model_name, &entry.model, &query, &documents).await {
            Ok(ids) => ids,
            Err(TokenizeError::Failed(msg)) => {
                tracing::error!(error = %msg, "rerank tokenize failed");
                finish!("server_error", server_error(msg));
            }
            Err(TokenizeError::Panic(msg)) => {
                tracing::error!(error = %msg, "rerank tokenize task panicked");
                finish!("server_error", server_error(msg));
            }
        };

    // ---- Phase: inference dispatch (batcher or direct) -------------------
    let scores = match dispatch_inference(entry, token_ids, doc_count).await {
        Ok(s) => s,
        Err(InferenceError::QueueFull(msg)) => {
            tracing::warn!(error = %msg, "rerank queue full — returning 429");
            finish!("rate_limited", rate_limited_response(msg));
        }
        Err(InferenceError::Inference(msg)) => {
            tracing::error!(error = %msg, "rerank inference failed");
            finish!("server_error", server_error(msg));
        }
        Err(InferenceError::Shutdown) => {
            tracing::error!("rerank batcher shut down");
            finish!(
                "shutdown",
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: "batcher shut down".to_string(),
                            error_type: "server_error",
                        },
                    }),
                )
                    .into_response()
            );
        }
        Err(InferenceError::ShapeMismatch(msg)) => {
            tracing::error!(error = %msg, "rerank batcher output shape invalid");
            finish!("server_error", server_error(msg));
        }
        Err(InferenceError::Panic(msg)) => {
            tracing::error!(error = %msg, "rerank score task panicked");
            finish!("server_error", server_error(msg));
        }
    };

    // ---- Phase: validate shape, normalize, sort, respond -----------------
    if scores.len() != doc_count {
        tracing::error!(
            expected = doc_count,
            got = scores.len(),
            "rerank score count mismatch"
        );
        finish!(
            "server_error",
            server_error("internal error: score count mismatch".to_string())
        );
    }

    // G2-server: apply optional score normalization before sort. Sigmoid
    // is monotonic so sort order is preserved — safe to run pre-sort.
    let mut scores = scores;
    apply_normalize(&mut scores, normalize.unwrap_or_default());

    let echo_docs: Option<&[String]> = if return_documents {
        Some(&documents)
    } else {
        None
    };
    let results = build_sorted_results(&scores, req.top_n, echo_docs);

    let response = Json(RerankResponse {
        id: opaque_request_id(),
        model: model_name.clone(),
        results,
    })
    .into_response();
    finish!("success", response);
}

// ---------------------------------------------------------------------
// Phase decomposition: tokenize + inference are extracted as private
// helpers so the handler reads as a clear sequence of phases (validate →
// resolve model → tokenize → infer → respond) rather than a 250-line wall
// of inline branches. Each helper owns its own metric emissions for the
// observable it controls (tokenizer duration, inference duration, cache
// hit/miss counts) — the handler only emits terminal-state series
// (`embed_rerank_request_*`) it owns end-to-end.
// ---------------------------------------------------------------------

/// Failure modes returned by `tokenize_with_cache`. Keeping them as a
/// 2-variant enum (vs a flat `String` error) lets the handler attach
/// distinct log fields and, in future, distinct status labels.
enum TokenizeError {
    Failed(String),
    Panic(String),
}

/// Tokenize a `(query, documents[])` set with H.7 cache lookup, falling
/// back to one batched `tokenize_pairs` call for misses. Records cache
/// hit/miss counters and `embed_rerank_tokenizer_duration_seconds`
/// (miss-path wall time only — cache-only requests skip the histogram
/// since their tokenizer time is zero).
///
/// Returns token IDs in original document order so the caller can use
/// `documents[i]` and the corresponding `token_ids[i]` interchangeably.
async fn tokenize_with_cache(
    state: &Arc<AppState>,
    model_name: &str,
    model: &Arc<crate::model_reranker::RerankerModel>,
    query: &str,
    documents: &[String],
) -> Result<Vec<Vec<u32>>, TokenizeError> {
    // Probe the cache for every (query, doc) pair. None = miss.
    let mut result_slots: Vec<(usize, Option<std::sync::Arc<Vec<u32>>>)> = documents
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let hit = state.token_cache.get(model_name, query, doc);
            (i, hit)
        })
        .collect();

    let hit_count = result_slots.iter().filter(|(_, h)| h.is_some()).count();
    let miss_count = result_slots.len() - hit_count;
    metrics::record_token_cache_hit(model_name, hit_count as u64);
    metrics::record_token_cache_miss(model_name, miss_count as u64);

    if miss_count == 0 {
        // Fast path: every pair was in cache. No tokenizer wall time to
        // record (would be a 0-bucket spike that distorts p50).
        return Ok(result_slots
            .into_iter()
            .map(|(_, arc)| arc.expect("hit").as_ref().clone())
            .collect());
    }

    // Miss path: collect docs needing tokenization, preserving original
    // indices so the splice back is in-order.
    let miss_indices: Vec<usize> = result_slots
        .iter()
        .filter(|(_, h)| h.is_none())
        .map(|(i, _)| *i)
        .collect();
    let miss_docs: Vec<String> = miss_indices.iter().map(|&i| documents[i].clone()).collect();

    let model_for_tokenize = model.clone();
    let tokenize_query = query.to_string();
    let tokenize_start = Instant::now();
    let tokenized_misses = match tokio::task::spawn_blocking(move || {
        model_for_tokenize.tokenize_pairs(&tokenize_query, &miss_docs)
    })
    .await
    {
        Ok(Ok(ids)) => ids,
        Ok(Err(e)) => return Err(TokenizeError::Failed(e)),
        Err(join_err) => {
            return Err(TokenizeError::Panic(format!(
                "tokenize task panicked: {join_err}"
            )));
        }
    };
    metrics::record_rerank_tokenizer(model_name, tokenize_start.elapsed());

    // Splice tokenized misses back into result_slots in original order
    // and populate the cache as we go.
    let mut miss_iter = tokenized_misses.into_iter().zip(miss_indices.iter());
    for slot in &mut result_slots {
        if slot.1.is_none()
            && let Some((ids, &orig_idx)) = miss_iter.next()
        {
            let arc = std::sync::Arc::new(ids);
            state
                .token_cache
                .insert(model_name, query, &documents[orig_idx], arc.clone());
            slot.1 = Some(arc);
        }
    }

    Ok(result_slots
        .into_iter()
        .map(|(_, arc)| {
            arc.expect("all slots filled after tokenize")
                .as_ref()
                .clone()
        })
        .collect())
}

/// Failure modes returned by `dispatch_inference`. Each variant maps to a
/// distinct HTTP response shape (429 / 500 / 503) and a distinct
/// `embed_rerank_requests_total{status}` label, so they cannot collapse
/// into one error type without losing observability.
enum InferenceError {
    QueueFull(String),
    Inference(String),
    Shutdown,
    ShapeMismatch(String),
    Panic(String),
}

/// Dispatch a pre-tokenized batch to the configured backend: the
/// per-model `DynamicBatcher` if `BATCHING_ENABLED=true` (E2 wires one
/// for every reranker entry at startup), otherwise a single
/// `spawn_blocking(score_pairs)` legacy path. The legacy path is
/// preserved for environments that explicitly disable batching for
/// debugging.
async fn dispatch_inference(
    entry: &crate::types::RerankerEntry,
    token_ids: Vec<Vec<u32>>,
    doc_count: usize,
) -> Result<Vec<f32>, InferenceError> {
    if let Some(b) = &entry.batcher {
        match b.embed_tokens(token_ids).await {
            Ok(v) => unwrap_scalar_batch(v, doc_count).map_err(InferenceError::ShapeMismatch),
            Err(crate::batcher::BatchError::QueueFull(e)) => {
                Err(InferenceError::QueueFull(e.to_string()))
            }
            Err(crate::batcher::BatchError::Inference(msg)) => Err(InferenceError::Inference(msg)),
            Err(crate::batcher::BatchError::Shutdown) => Err(InferenceError::Shutdown),
        }
    } else {
        let model_for_score = entry.model.clone();
        match tokio::task::spawn_blocking(move || model_for_score.score_pairs(&token_ids)).await {
            Ok(Ok(s)) => Ok(s),
            Ok(Err(e)) => Err(InferenceError::Inference(e)),
            Err(join_err) => Err(InferenceError::Panic(format!(
                "score task panicked: {join_err}"
            ))),
        }
    }
}

/// 429 Too Many Requests with `Retry-After: 1`. Both load-shed layers
/// (semaphore cap, batcher queue full) share the 1-second backoff hint —
/// neither failure mode is a meaningful steady state on healthy infra,
/// and a fixed short retry matches the `/v1/embeddings` contract.
fn rate_limited_response(message: String) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [("retry-after", "1")],
        Json(ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: "rate_limited",
            },
        }),
    )
        .into_response()
}

/// Cohere-shape opaque request id. Clients use it for log correlation
/// only — the contract is "unique-ish, not cryptographic".
fn opaque_request_id() -> String {
    format!(
        "rrk_{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    )
}

/// Unwrap the batcher's `Vec<Vec<f32>>` (one 1-element inner per pair)
/// into a flat `Vec<f32>` of scores. Validates both the outer length
/// and each inner length, so a shape regression in the E2 adapter
/// surfaces here with a clear error instead of an opaque panic at the
/// `[0]` index.
fn unwrap_scalar_batch(batched: Vec<Vec<f32>>, expected: usize) -> Result<Vec<f32>, String> {
    if batched.len() != expected {
        return Err(format!(
            "batcher returned {} rows, expected {}",
            batched.len(),
            expected
        ));
    }
    let mut out = Vec::with_capacity(expected);
    for (i, inner) in batched.into_iter().enumerate() {
        if inner.len() != 1 {
            return Err(format!(
                "rerank row {i}: expected 1-element score vec, got {}",
                inner.len()
            ));
        }
        out.push(inner[0]);
    }
    Ok(out)
}

/// 500-wrapper with the `server_error` error-type label matching
/// `api::embeddings`. Kept as a free function to deduplicate the
/// several internal-error paths above.
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
    // RerankRequest deserialization — exercises the two optional-field
    // paths (no model / no top_n, and both present) so a client omitting
    // those fields doesn't get a 400 from serde.
    // -----------------------------------------------------------------

    fn doc_text(d: &RerankDocument) -> &str {
        match d {
            RerankDocument::Text(s) => s.as_str(),
            RerankDocument::Object { text } => text.as_str(),
        }
    }

    #[test]
    fn rerank_request_deserializes_with_optional_fields() {
        let j1 = r#"{"query":"q","documents":["a"]}"#;
        let r: RerankRequest = serde_json::from_str(j1).unwrap();
        assert!(r.model.is_none() && r.top_n.is_none());
        assert_eq!(r.query, "q");
        assert_eq!(r.documents.len(), 1);
        assert_eq!(doc_text(&r.documents[0]), "a");
        assert!(!r.return_documents);

        let j2 = r#"{"model":"m","query":"q","documents":["a","b"],"top_n":1}"#;
        let r: RerankRequest = serde_json::from_str(j2).unwrap();
        assert_eq!(r.model.as_deref(), Some("m"));
        assert_eq!(r.top_n, Some(1));
        assert_eq!(r.documents.len(), 2);
    }

    /// Cohere SDK + Jina/Voyage shape: documents as `[{"text": "..."}]`.
    /// Covers the "object form" half of the Cohere `documents` contract.
    #[test]
    fn rerank_request_deserializes_object_documents() {
        let j = r#"{"query":"q","documents":[{"text":"a"},{"text":"b"}]}"#;
        let r: RerankRequest = serde_json::from_str(j).unwrap();
        assert_eq!(r.documents.len(), 2);
        assert_eq!(doc_text(&r.documents[0]), "a");
        assert_eq!(doc_text(&r.documents[1]), "b");
    }

    /// Mixed string + object — defensive test for clients that mix forms.
    /// `serde(untagged)` handles each entry independently.
    #[test]
    fn rerank_request_deserializes_mixed_documents() {
        let j = r#"{"query":"q","documents":["plain",{"text":"obj"}]}"#;
        let r: RerankRequest = serde_json::from_str(j).unwrap();
        assert_eq!(r.documents.len(), 2);
        assert_eq!(doc_text(&r.documents[0]), "plain");
        assert_eq!(doc_text(&r.documents[1]), "obj");
    }

    /// `return_documents=true` populates the `document` echo field on each
    /// result. Default false — `document` stays None and is omitted from
    /// JSON via `skip_serializing_if`.
    #[test]
    fn build_sorted_results_echoes_documents_when_requested() {
        let scores = vec![0.5_f32, 1.0];
        let docs = vec!["first".to_string(), "second".to_string()];
        let results = build_sorted_results(&scores, None, Some(&docs));
        assert_eq!(results.len(), 2);
        // Top result is index 1 ("second", score 1.0).
        assert_eq!(results[0].index, 1);
        assert_eq!(
            results[0].document.as_ref().map(|d| d.text.as_str()),
            Some("second"),
        );
        assert_eq!(results[1].index, 0);
        assert_eq!(
            results[1].document.as_ref().map(|d| d.text.as_str()),
            Some("first"),
        );

        // None case: no echo, all `document` fields stay None.
        let results = build_sorted_results(&scores, None, None);
        assert!(results.iter().all(|r| r.document.is_none()));
    }

    // -----------------------------------------------------------------
    // build_sorted_results — pure sort+truncate. The handler delegates
    // ordering to this function, so the guarantee clients rely on
    // (descending scores, original indices preserved, top_n applied
    // AFTER sort) is proved here without an AppState harness.
    // -----------------------------------------------------------------

    #[test]
    fn sort_and_truncate_picks_top_n_desc() {
        let scores = vec![1.0f32, 5.0, 3.0, 2.0];
        let results = build_sorted_results(&scores, Some(2), None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 1); // score 5.0
        assert_eq!(results[0].relevance_score, 5.0);
        assert_eq!(results[1].index, 2); // score 3.0
        assert_eq!(results[1].relevance_score, 3.0);
    }

    #[test]
    fn sort_without_top_n_returns_all_desc() {
        // No truncation: every input position appears exactly once in
        // the output, sorted by score descending. Proves that
        // `top_n=None` doesn't silently clip the tail.
        let scores = vec![0.1f32, 0.9, 0.5];
        let results = build_sorted_results(&scores, None, None);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 1); // 0.9
        assert_eq!(results[1].index, 2); // 0.5
        assert_eq!(results[2].index, 0); // 0.1
    }

    #[test]
    fn sort_with_top_n_zero_returns_empty() {
        // Spec leaves `top_n=0` unspecified; we chose "empty results"
        // rather than 400 (the sort+truncate is a pure function and
        // clients may legitimately want to probe a model's
        // availability without materialising scores).
        let scores = vec![1.0f32, 2.0, 3.0];
        assert!(build_sorted_results(&scores, Some(0), None).is_empty());
    }

    #[test]
    fn sort_with_top_n_greater_than_len_caps_at_len() {
        // `Vec::truncate` is saturating, so an over-sized top_n is
        // naturally benign — test pins that behaviour so future
        // refactors can't tighten it into a 400 without noticing.
        let scores = vec![1.0f32, 2.0];
        let results = build_sorted_results(&scores, Some(10), None);
        assert_eq!(results.len(), 2);
    }

    // -----------------------------------------------------------------
    // G2-server: apply_normalize — unit tests for NormalizeMode variants.
    // -----------------------------------------------------------------

    #[test]
    fn apply_normalize_none_is_identity() {
        let mut scores = vec![-15.0, -1.0, 0.0, 1.0, 15.0];
        apply_normalize(&mut scores, NormalizeMode::None);
        assert_eq!(scores, vec![-15.0, -1.0, 0.0, 1.0, 15.0]);
    }

    #[test]
    fn apply_normalize_sigmoid_known_inputs() {
        let mut scores = vec![-1.0, 0.0, 1.0];
        apply_normalize(&mut scores, NormalizeMode::Sigmoid);
        // sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5, sigmoid(1) ≈ 0.7311
        assert!((scores[0] - 0.268_9).abs() < 1e-3);
        assert!((scores[1] - 0.5).abs() < 1e-6);
        assert!((scores[2] - 0.731_1).abs() < 1e-3);
    }

    #[test]
    fn apply_normalize_sigmoid_extremes_no_inf() {
        let mut scores = vec![-100.0, 100.0];
        apply_normalize(&mut scores, NormalizeMode::Sigmoid);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
        assert!(scores[0] >= 0.0 && scores[0] <= 1.0);
        assert!(scores[1] >= 0.0 && scores[1] <= 1.0);
    }

    #[test]
    fn apply_normalize_sigmoid_preserves_sort_order() {
        let original = vec![-3.0, -1.0, 0.5, 2.0, 5.0];
        let mut sigmoid = original.clone();
        apply_normalize(&mut sigmoid, NormalizeMode::Sigmoid);
        // sigmoid is monotonic: original[i] < original[j] → sigmoid[i] < sigmoid[j]
        for i in 0..original.len() {
            for j in (i + 1)..original.len() {
                if original[i] < original[j] {
                    assert!(sigmoid[i] < sigmoid[j], "sort order broken at i={i}, j={j}");
                }
            }
        }
    }

    #[test]
    fn rerank_request_deserialize_normalize_field() {
        // Without field — defaults to None.
        let r1: RerankRequest = serde_json::from_str(r#"{"query":"q","documents":["a"]}"#).unwrap();
        assert_eq!(r1.normalize, None);

        // Explicit "none".
        let r2: RerankRequest =
            serde_json::from_str(r#"{"query":"q","documents":["a"],"normalize":"none"}"#).unwrap();
        assert_eq!(r2.normalize, Some(NormalizeMode::None));

        // Explicit "sigmoid".
        let r3: RerankRequest =
            serde_json::from_str(r#"{"query":"q","documents":["a"],"normalize":"sigmoid"}"#)
                .unwrap();
        assert_eq!(r3.normalize, Some(NormalizeMode::Sigmoid));
    }

    #[test]
    fn normalize_mode_default_is_sigmoid() {
        // Pins the 2026-05-02 default flip: callers that omit `normalize`
        // get sigmoid-normalized [0,1] scores, not raw logits. Regression
        // guard against accidental revert that would re-trigger the
        // memdb-go quality-floor cliff (Run #12 LoCoMo F1 -39%).
        assert_eq!(NormalizeMode::default(), NormalizeMode::Sigmoid);

        // End-to-end: omitted field → Option::None → handler's
        // `unwrap_or_default()` → Sigmoid. Verify the apply_normalize
        // fallback transforms a known logit to its sigmoid value.
        let mut scores = vec![0.0_f32];
        apply_normalize(&mut scores, NormalizeMode::default());
        assert!(
            (scores[0] - 0.5).abs() < 1e-6,
            "default must be sigmoid (sigmoid(0)=0.5), got {}",
            scores[0]
        );
    }

    #[test]
    fn rerank_request_invalid_normalize_rejected() {
        // serde with rename_all="lowercase" rejects unknown values.
        let r: Result<RerankRequest, _> =
            serde_json::from_str(r#"{"query":"q","documents":["a"],"normalize":"softmax"}"#);
        assert!(r.is_err(), "unknown normalize value must error");
    }

    // -----------------------------------------------------------------
    // RERANK_MAX_INPUT_DOCS cap — mirrors the EMBED_MAX_INPUT_ARRAY tests
    // in api.rs. The cap check fires before any model lookup so an empty
    // rerankers map is sufficient.
    // -----------------------------------------------------------------

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

    use crate::cache::EmbeddingCache;
    use crate::token_cache::TokenCache;
    use crate::types::AppState;

    fn test_prom() -> &'static metrics_exporter_prometheus::PrometheusHandle {
        crate::metrics::test_prometheus_handle()
    }

    /// Minimal AppState for cap tests: no models, no rerankers. The cap
    /// check fires before model resolution so missing models are fine.
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
            embed_max_input_array: 32,
            rerank_max_input_docs: cap,
        })
    }

    /// Build a POST /v1/rerank request with `n` documents.
    fn make_rerank_request(n: usize) -> Request<Body> {
        let docs: Vec<String> = (0..n).map(|i| format!("doc {i}")).collect();
        let body = serde_json::json!({
            "query": "test query",
            "documents": docs
        });
        Request::builder()
            .method("POST")
            .uri("/v1/rerank")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn make_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/v1/rerank", post(crate::api_rerank::rerank))
            .with_state(state)
    }

    // ---- test: oversized documents → 400 with documents_too_many body ----

    #[tokio::test]
    async fn rerank_handler_rejects_oversized_documents_array() {
        let handle = test_prom();
        let state = make_state(32);
        let app = make_app(state);

        // 33 documents, cap=32 → must reject with 400.
        let resp = app.oneshot(make_rerank_request(33)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "oversized documents array must return HTTP 400"
        );

        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(
            json["error"]["code"], "documents_too_many",
            "error.code must be documents_too_many"
        );
        assert_eq!(
            json["error"]["type"], "invalid_request_error",
            "error.type must be invalid_request_error"
        );
        assert_eq!(json["error"]["cap"], 32, "cap must equal configured cap");
        assert_eq!(
            json["error"]["received"], 33,
            "received must equal actual documents count"
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
            counter_text.contains("embed_rerank_input_docs_rejected_total"),
            "embed_rerank_input_docs_rejected_total counter must appear in metrics after rejection"
        );
    }

    // ---- test: exactly at cap → cap check passes (not documents_too_many) ----

    #[tokio::test]
    async fn rerank_handler_accepts_at_cap() {
        let state = make_state(32);
        let app = make_app(state);

        // 32 documents, cap=32 → cap check passes (len == cap, NOT > cap).
        // Request then fails with "no reranker models configured" (empty map),
        // which is a different 400 — verify it does NOT have code=documents_too_many.
        let resp = app.oneshot(make_rerank_request(32)).await.unwrap();
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body_bytes).unwrap();

        assert_ne!(
            json["error"]["code"], "documents_too_many",
            "request at exactly cap=32 must not trigger the documents-size rejection"
        );
    }

    // ---- test: histogram records both accept and reject paths ----

    #[tokio::test]
    async fn embed_rerank_input_docs_size_histogram_records_both_paths() {
        let handle = test_prom();

        // Rejected path: 33 documents, cap=32.
        {
            let app = make_app(make_state(32));
            let _resp = app.oneshot(make_rerank_request(33)).await.unwrap();
        }

        // Accepted path: 5 documents, cap=32.
        {
            let app = make_app(make_state(32));
            let _resp = app.oneshot(make_rerank_request(5)).await.unwrap();
        }

        let metrics_text = handle.render();
        assert!(
            metrics_text.contains("embed_rerank_input_docs_size"),
            "embed_rerank_input_docs_size histogram must appear in /metrics output \
             for both accepted and rejected requests"
        );
    }
}
