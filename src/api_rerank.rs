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

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::types::{AppState, ErrorDetail, ErrorResponse, error_json};

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
/// before sort. Default `None` returns raw logits (Cohere/Jina convention).
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum NormalizeMode {
    /// Identity — return raw cross-encoder logits (Cohere compat default).
    #[default]
    None,
    /// Apply 1/(1+exp(-x)) per score → [0,1].
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
    /// G2: optional server-side score normalization. None (default) returns
    /// raw logits — preserves Cohere convention. "sigmoid" applies
    /// 1/(1+exp(-x)) to each score after `score_pairs` and before sort.
    /// Other values rejected with 400.
    #[serde(default)]
    pub normalize: Option<NormalizeMode>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct RerankResult {
    /// Position of the document in the *original request's* `documents`
    /// array (0-based). Preserved across sort so clients can map scores
    /// back to the text they sent.
    pub index: usize,
    /// Cross-encoder relevance score. By default raw logit (matches
    /// Cohere/Jina convention; higher = more relevant, unbounded). If the
    /// request had `"normalize":"sigmoid"`, this is sigmoid-normalized to [0,1].
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
    // Shutdown gate — matches `api::embeddings`. Reject new requests
    // with 503 once SIGTERM has flipped the token, giving clients a
    // Retry-After hint to back off while drain completes.
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

    // Validate inputs BEFORE resolving a model — a malformed request
    // shouldn't need reranker configuration to reach a 400.
    if req.query.trim().is_empty() {
        return error_json("query must not be empty").into_response();
    }
    if req.documents.is_empty() {
        return error_json("documents must not be empty").into_response();
    }

    let model_name = match resolve_reranker_name(&state, req.model) {
        Ok(n) => n,
        Err(msg) => return error_json(msg).into_response(),
    };
    // `expect` is safe: `resolve_reranker_name` guarantees presence.
    let entry = state
        .rerankers
        .get(&model_name)
        .expect("resolve_reranker_name validated key presence");

    let query = req.query;
    // Normalise mixed string/object form into a flat Vec<String>. Cohere
    // accepts either; clients rarely care which they send.
    let documents: Vec<String> = req.documents.into_iter().map(|d| d.into_text()).collect();
    let doc_count = documents.len();
    let return_documents = req.return_documents;
    let normalize = req.normalize;

    // Tokenize pairs in spawn_blocking — same reasoning as `api::embeddings`:
    // tokenization is CPU-bound and blocking it on the async runtime
    // starves other futures under concurrent load.
    let model_for_tokenize = entry.model.clone();
    let tokenize_query = query.clone();
    let tokenize_docs = documents.clone();
    let token_ids = match tokio::task::spawn_blocking(move || {
        model_for_tokenize.tokenize_pairs(&tokenize_query, &tokenize_docs)
    })
    .await
    {
        Ok(Ok(ids)) => ids,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "rerank tokenize failed");
            return server_error(e);
        }
        Err(join_err) => {
            tracing::error!(error = %join_err, "rerank tokenize task panicked");
            return server_error(format!("tokenize task panicked: {join_err}"));
        }
    };

    // Inference: prefer the configured batcher (E2 wires every reranker
    // with one when `BATCHING_ENABLED=true`); fall back to a direct
    // `spawn_blocking(score_pairs)` when batching is off.
    let scores: Vec<f32> = if let Some(b) = &entry.batcher {
        // The E2 adapter closure wraps each scalar score as a 1-element
        // `Vec<f32>` to fit the batcher's `Vec<Vec<f32>>` contract;
        // here we unwrap that, validating the shape so a batcher bug
        // surfaces as a 500 instead of a panic.
        match b.embed_tokens(token_ids).await {
            Ok(v) => match unwrap_scalar_batch(v, doc_count) {
                Ok(s) => s,
                Err(msg) => {
                    tracing::error!(error = %msg, "rerank batcher output shape invalid");
                    return server_error(msg);
                }
            },
            Err(crate::batcher::BatchError::QueueFull(e)) => {
                // E2: 429 Too Many Requests on backpressure (≥80% full).
                // Matches `/v1/embeddings` semantics — see api.rs for
                // the full rationale.
                tracing::warn!(error = %e, "rerank queue full — returning 429");
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
                tracing::error!(error = %msg, "rerank inference failed");
                return server_error(msg);
            }
            Err(crate::batcher::BatchError::Shutdown) => {
                tracing::error!("rerank batcher shut down");
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
        // Legacy path: one `spawn_blocking` for the entire batch (NOT
        // per-pair — score_pairs already does a single vectorised
        // forward pass over all pre-tokenised pairs).
        let model_for_score = entry.model.clone();
        match tokio::task::spawn_blocking(move || model_for_score.score_pairs(&token_ids)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                tracing::error!(error = %e, "rerank score_pairs failed");
                return server_error(e);
            }
            Err(join_err) => {
                tracing::error!(error = %join_err, "rerank score task panicked");
                return server_error(format!("score task panicked: {join_err}"));
            }
        }
    };

    // Sanity: the inference layer must return exactly one score per
    // document. A mismatch is a batcher/model bug; surface it as 500.
    if scores.len() != doc_count {
        tracing::error!(
            expected = doc_count,
            got = scores.len(),
            "rerank score count mismatch"
        );
        return server_error("internal error: score count mismatch".to_string());
    }

    // G2-server: apply optional score normalization before sort.
    // Sigmoid is monotonic so sort order is preserved — safe to run here.
    let mut scores = scores;
    apply_normalize(&mut scores, normalize.unwrap_or_default());

    let echo_docs: Option<&[String]> = if return_documents {
        Some(&documents)
    } else {
        None
    };
    let results = build_sorted_results(&scores, req.top_n, echo_docs);

    Json(RerankResponse {
        // Cohere generates an opaque request id for tracing; we use a
        // simple time-based hex (no extra crate dep). Clients use it for
        // log correlation only — semantics are "opaque, unique-ish".
        id: format!(
            "rrk_{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ),
        model: model_name,
        results,
    })
    .into_response()
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
    fn rerank_request_invalid_normalize_rejected() {
        // serde with rename_all="lowercase" rejects unknown values.
        let r: Result<RerankRequest, _> =
            serde_json::from_str(r#"{"query":"q","documents":["a"],"normalize":"softmax"}"#);
        assert!(r.is_err(), "unknown normalize value must error");
    }
}
