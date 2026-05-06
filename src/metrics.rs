//! Prometheus metrics helpers for embed-server.
use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Install the Prometheus recorder and return its rendering handle.
///
/// Sets sensible histogram buckets for latency (_duration_seconds) and batch
/// size metrics. Stamps `embed_build_info{version}` to 1.
pub fn init(version: &str) -> PrometheusHandle {
    let duration_matcher =
        metrics_exporter_prometheus::Matcher::Suffix("_duration_seconds".to_string());
    let batch_matcher = metrics_exporter_prometheus::Matcher::Suffix("batch_size".to_string());
    // ms-scale buckets for `embed_batch_wait_ms` — the queue-to-dispatch
    // wall time. Bounded on the high end by `BATCH_WAIT_MS=30` under
    // normal load; a p95 > 100 ms means the worker is saturated before
    // the coalescing window closes.
    let wait_ms_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_batch_wait_ms".to_string());
    // Padding-waste ratio matcher: 0.0 (no waste) → 1.0 (all padding). Linear
    // 0.1 buckets give resolution at the operational decision boundary
    // (median > 0.4 → length-bucketing payoff per Phase 3C plan).
    let waste_matcher =
        metrics_exporter_prometheus::Matcher::Suffix("padding_waste_ratio".to_string());
    // Token count per batcher batch: [batch_size × max_seq_len] (padded-model formula).
    // Max for e5-large: 8 × 256 = 2048; for jina: 8 × 512 = 4096; BATCH_MAX_TOKENS cap = 8192.
    let batch_tokens_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_batch_tokens".to_string());
    // Texts per HTTP request: go-search sends 1-50, bulk ingest up to ~1500.
    let texts_per_req_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_texts_per_request".to_string());
    // Pairs per /v1/rerank request: typically CROSS_ENCODER_MAX_DOCS = 15.
    let rerank_pairs_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_rerank_pairs_per_request".to_string());

    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            duration_matcher,
            &[
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            ],
        )
        .expect("set duration buckets")
        .set_buckets_for_metric(
            batch_matcher,
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        )
        .expect("set batch buckets")
        .set_buckets_for_metric(
            wait_ms_matcher,
            &[
                1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0,
            ],
        )
        .expect("set batch wait buckets")
        .set_buckets_for_metric(
            waste_matcher,
            &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        .expect("set padding waste buckets")
        .set_buckets_for_metric(
            batch_tokens_matcher,
            &[128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0],
        )
        .expect("set batch tokens buckets")
        .set_buckets_for_metric(
            texts_per_req_matcher,
            &[
                1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
            ],
        )
        .expect("set texts per request buckets")
        .set_buckets_for_metric(
            rerank_pairs_matcher,
            &[1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0],
        )
        .expect("set rerank pairs buckets")
        .install_recorder()
        .expect("install Prometheus recorder");

    metrics::gauge!("embed_build_info", "version" => version.to_string()).set(1.0);
    handle
}

/// Record a completed request.
pub fn record_request(model: &str, status: &str, duration: Duration, texts: usize) {
    metrics::counter!(
        "embed_requests_total",
        "model" => model.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
    metrics::histogram!(
        "embed_request_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
    metrics::histogram!(
        "embed_texts_per_request",
        "model" => model.to_string()
    )
    .record(texts as f64);
}

/// Record one ONNX inference call.
pub fn record_inference(model: &str, duration: Duration, batch_size: usize) {
    metrics::histogram!(
        "embed_inference_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
    metrics::histogram!(
        "embed_batch_size",
        "model" => model.to_string()
    )
    .record(batch_size as f64);
}

/// Increment rejected-due-to-backpressure counter. Called from the
/// `DynamicBatcher` 80%-of-capacity gate and its belt-and-suspenders
/// `try_send` fallthrough — so every item that produces a 429 on the
/// wire increments this exactly once.
pub fn record_queue_rejected(model: &str) {
    metrics::counter!(
        "embed_queue_full_rejected_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Increment count of batcher items skipped because the caller's reply
/// channel was already closed (client disconnected before dispatch).
///
/// Call once per skipped item — both at the inner coalesce check and the
/// pre-dispatch `retain` pass in `batcher::run_worker`.
pub fn record_cancelled(model: &str) {
    metrics::counter!(
        "embed_batcher_cancelled_items_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Set current queue depth for a model's batcher. Emitted on every
/// `embed_tokens` producer call (before the 80% gate) — so scrapes see
/// the most recent producer-side view. Computed as
/// `max_queue − sender.capacity()`.
pub fn set_queue_depth(model: &str, depth: usize) {
    metrics::gauge!(
        "embed_queue_depth_current",
        "model" => model.to_string()
    )
    .set(depth as f64);
}

/// Record the wall-clock time in milliseconds from the moment an Item
/// was handed to `tx.try_send` to the moment its batch was cut and
/// passed to `dispatch_batch`. Sampled once per live Item in each
/// dispatched batch. Rising p95 indicates producers outrunning dispatch
/// capacity — usually a prelude to 429s from the 80% gate.
pub fn record_batch_wait_ms(model: &str, wait_ms: f64) {
    metrics::histogram!(
        "embed_batch_wait_ms",
        "model" => model.to_string()
    )
    .record(wait_ms);
}

/// Record padded-compute tokens for one dispatched batch.
///
/// For padded models this is `max_len * items` (the actual compute unit
/// count); for non-padded models it's `total_tokens` (raw sum). The
/// caller decides which value to pass based on the `padded_model` flag.
pub fn record_batch_tokens(model: &str, tokens: usize) {
    metrics::histogram!(
        "embed_batch_tokens",
        "model" => model.to_string()
    )
    .record(tokens as f64);
}

/// Record padding-waste ratio for one dispatched batch.
///
/// Ratio is `(padded - raw) / padded`, clamped to [0.0, 1.0]. 0 means
/// every sequence in the batch was the same length (no padding); → 1
/// means the batch was mostly padding. Caller passes `padded == raw`
/// for non-padded models so the ratio is always 0 there.
pub fn record_padding_waste(model: &str, padded: usize, raw: usize) {
    let ratio = if padded == 0 {
        0.0
    } else {
        ((padded.saturating_sub(raw)) as f64 / padded as f64).clamp(0.0, 1.0)
    };
    metrics::histogram!(
        "embed_batch_padding_waste_ratio",
        "model" => model.to_string()
    )
    .record(ratio);
}

/// Increment the carry-events counter (token-budget overflow deferred
/// an item into the next batch). A rising rate here is a signal that
/// clients send heterogeneous sizes — consider length-bucketing.
pub fn record_carry(model: &str) {
    metrics::counter!(
        "embed_carry_events_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Increment embedding-cache hit counter by `n` in a single call.
///
/// Positions-in-request semantics: the same text at N positions counts
/// as N hits (matches miss accounting so hit ratio is per-text-in-request,
/// not per-unique-text). Use the `_n` form from hot paths that know the
/// batched count; it compiles to one `.increment(n)` call instead of N
/// singleton increments.
pub fn record_cache_hit_n(model: &str, n: u64) {
    metrics::counter!(
        "embed_cache_hit_total",
        "model" => model.to_string()
    )
    .increment(n);
}

/// Increment embedding-cache miss counter by `n` in a single call.
/// See `record_cache_hit_n` for the positions-in-request semantic.
pub fn record_cache_miss_n(model: &str, n: u64) {
    metrics::counter!(
        "embed_cache_miss_total",
        "model" => model.to_string()
    )
    .increment(n);
}

/// Increment embedding-cache hit counter by 1. Back-compat wrapper that
/// delegates to `record_cache_hit_n`; prefer the `_n` form from hot
/// paths that already know the count. (`allow(dead_code)` because the
/// production call site in api.rs now uses the `_n` form; the singular
/// wrapper is retained for test call sites and future callers that only
/// need a single increment.)
#[allow(dead_code)]
pub fn record_cache_hit(model: &str) {
    record_cache_hit_n(model, 1);
}

/// Increment embedding-cache miss counter by 1. Back-compat wrapper that
/// delegates to `record_cache_miss_n`.
#[allow(dead_code)]
pub fn record_cache_miss(model: &str) {
    record_cache_miss_n(model, 1);
}

/// Set the current embedding-cache entry count.
///
/// Intentionally unlabelled (global gauge): the cache is keyed by
/// `(model, text)` so a per-model size would require iterating the LRU
/// each update. A single global counter updated on insert is cheap and
/// sufficient for capacity monitoring.
pub fn set_cache_size(size: usize) {
    metrics::gauge!("embed_cache_size").set(size as f64);
}

/// Increment the tokenize-fallback counter. Called whenever a tokenize
/// call on the miss path fails (Err result or task panic). Allows ops to
/// detect tokenizer regressions instead of seeing silent total_tokens=0.
pub fn record_tokenize_fallback() {
    metrics::counter!("embed_tokenize_fallback_total").increment(1);
}

/// Increment the token-cache hit counter by `n`.
///
/// Called from the reranker hot path (H.7) once per batch hit. Using an
/// `_n` form mirrors the embedding-cache pattern so the caller can record
/// all hits in a batch with a single atomic increment. Pre-warming with
/// `n=0` at startup ensures the series appears in `/metrics` before the
/// first request arrives.
pub fn record_token_cache_hit(model: &str, n: u64) {
    metrics::counter!(
        "embed_token_cache_total",
        "model" => model.to_string(),
        "outcome" => "hit"
    )
    .increment(n);
}

/// Publish arena configuration as Prometheus gauges. Called once at startup
/// from `arena::register_shared_cpu_arena` so operators can verify effective
/// config from `/metrics` without reading logs.
///
/// `embed_arena_extend_strategy` uses a bounded-cardinality label
/// (`strategy="kSameAsRequested"` or `strategy="kNextPowerOfTwo"`) instead of
/// a raw integer — this keeps the series human-readable in Grafana without
/// growing unbounded cardinality.
pub fn set_arena_gauges(
    max_mem_bytes: usize,
    initial_chunk_bytes: usize,
    max_dead_bytes: usize,
    extend_strategy: i32,
) {
    metrics::gauge!("embed_arena_max_mem_bytes").set(max_mem_bytes as f64);
    metrics::gauge!("embed_arena_initial_chunk_bytes").set(initial_chunk_bytes as f64);
    metrics::gauge!("embed_arena_max_dead_bytes").set(max_dead_bytes as f64);
    let strategy_label = match extend_strategy {
        1 => "kSameAsRequested",
        _ => "kNextPowerOfTwo",
    };
    metrics::gauge!(
        "embed_arena_extend_strategy",
        "strategy" => strategy_label
    )
    .set(extend_strategy as f64);
}

/// Increment the token-cache miss counter by `n`.
///
/// See `record_token_cache_hit` for the batch-increment semantic and
/// startup pre-warm contract.
pub fn record_token_cache_miss(model: &str, n: u64) {
    metrics::counter!(
        "embed_token_cache_total",
        "model" => model.to_string(),
        "outcome" => "miss"
    )
    .increment(n);
}

// ---------------------------------------------------------------------
// Rerank-specific instrumentation (Phase 1A — 2026-05-01).
//
// Series prefix `embed_rerank_*` (kept under the global `embed_` namespace
// so existing scrape configs and Grafana dashboards pick them up without
// label-relabeling). Until this phase the /v1/rerank path emitted only
// the batcher-level series (queue depth, batch tokens) and `embed_token_cache_total` —
// no per-request counters, no inference timing, no tokenizer split.
// Without these series, Phase 2's head-to-head bench (gte-multi vs
// gte-modernbert) would only be measurable as black-box wall-clock,
// with no way to attribute regressions to inference vs tokenize vs queue.
// ---------------------------------------------------------------------

/// Record a completed /v1/rerank request — counter + end-to-end latency
/// histogram + pairs-per-request distribution.
///
/// Call once per request, on every exit path (success and error). Use the
/// `_RerankInFlightGuard_` for the in-flight gauge; this function only
/// records terminal-state series.
pub fn record_rerank_request(model: &str, status: &str, duration: Duration, doc_count: usize) {
    metrics::counter!(
        "embed_rerank_requests_total",
        "model" => model.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
    metrics::histogram!(
        "embed_rerank_request_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
    metrics::histogram!(
        "embed_rerank_pairs_per_request",
        "model" => model.to_string()
    )
    .record(doc_count as f64);
}

/// Record one cross-encoder inference — wraps the `session.run()` call only,
/// not the tokenize or pool-acquire phases (those have their own series).
/// `batch_size` is the number of (query, doc) pairs in the forward pass —
/// i.e. the tensor's batch dim, which is the actual unit of cross-encoder
/// compute cost.
pub fn record_rerank_inference(model: &str, duration: Duration, batch_size: usize) {
    metrics::histogram!(
        "embed_rerank_inference_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
    metrics::histogram!(
        "embed_rerank_batch_size",
        "model" => model.to_string()
    )
    .record(batch_size as f64);
}

/// Record HuggingFace tokenizer wall time for one rerank request's miss
/// batch. Excludes cache-hit pairs (those bypass the tokenizer entirely).
/// On cache-only requests this series is not recorded — track miss
/// fraction via `embed_token_cache_total{outcome}`.
#[allow(dead_code)]
pub fn record_rerank_tokenizer(model: &str, duration: Duration) {
    metrics::histogram!(
        "embed_rerank_tokenizer_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
}

/// Record session-pool mutex acquire wall time. Sub-ms under no contention;
/// rises to the inference-duration tail under saturation. Drives Phase 3A
/// `RERANKER_SESSION_POOL_SIZE` decisions.
pub fn record_rerank_pool_acquire(model: &str, duration: Duration) {
    metrics::histogram!(
        "embed_rerank_pool_acquire_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
}

/// Record padding-waste ratio for one cross-encoder forward pass. Computed
/// as `(padded_tokens - real_tokens) / padded_tokens` clamped to `[0, 1]`.
/// Drives Phase 3C (length-bucketing) decisions: median > 0.4 = payoff.
pub fn record_rerank_padding_waste(model: &str, padded: usize, raw: usize) {
    let ratio = if padded == 0 {
        0.0
    } else {
        ((padded.saturating_sub(raw)) as f64 / padded as f64).clamp(0.0, 1.0)
    };
    metrics::histogram!(
        "embed_rerank_padding_waste_ratio",
        "model" => model.to_string()
    )
    .record(ratio);
}

/// RAII guard that increments `embed_rerank_in_flight{model}` on construction
/// and decrements it on `Drop`. Holding it across the entire request
/// lifetime — including async waits on tokenize and batcher — is what makes
/// the gauge a faithful "currently processing" count rather than a
/// peak-counter approximation.
///
/// Drop semantics are panic-safe: even if the handler unwinds (it shouldn't,
/// but axum catches panics anyway), the gauge decrement still fires.
pub struct RerankInFlightGuard {
    model: String,
}

impl RerankInFlightGuard {
    pub fn new(model: &str) -> Self {
        metrics::gauge!(
            "embed_rerank_in_flight",
            "model" => model.to_string()
        )
        .increment(1.0);
        Self {
            model: model.to_string(),
        }
    }
}

impl Drop for RerankInFlightGuard {
    fn drop(&mut self) {
        metrics::gauge!(
            "embed_rerank_in_flight",
            "model" => self.model.clone()
        )
        .decrement(1.0);
    }
}
