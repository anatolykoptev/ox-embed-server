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
