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

/// Increment rejected-due-to-backpressure counter (will be used in R6).
#[allow(dead_code)]
pub fn record_queue_rejected(model: &str) {
    metrics::counter!(
        "embed_queue_rejected_total",
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

/// Set current queue depth (will be used in R5/R6).
#[allow(dead_code)]
pub fn set_queue_depth(model: &str, depth: usize) {
    metrics::gauge!(
        "embed_queue_depth",
        "model" => model.to_string()
    )
    .set(depth as f64);
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

/// Increment embedding-cache hit counter (one per request position; same
/// text at N positions counts as N hits, matching the "positions-in-request"
/// semantic used for miss accounting).
pub fn record_cache_hit(model: &str) {
    metrics::counter!(
        "embed_cache_hit_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Increment embedding-cache miss counter (one per request position;
/// duplicates-within-request count as multiple misses so hit ratio is
/// measured per-text-in-request, not per-unique-text).
pub fn record_cache_miss(model: &str) {
    metrics::counter!(
        "embed_cache_miss_total",
        "model" => model.to_string()
    )
    .increment(1);
}
