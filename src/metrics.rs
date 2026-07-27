//! Prometheus metrics helpers for embed-server.
use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

// ── constant numerics used only as metric bucket boundaries ───────────────────
const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * MIB;

/// Apply every embed-server histogram bucket configuration to a
/// `PrometheusBuilder`.
///
/// This is the SINGLE authority for bucket boundaries. Both the supervisor
/// recorder (`init`) and the per-worker recorder
/// (`bin/worker.rs::install_worker_metrics`) call it, so a histogram emitted
/// from a worker process lands in the same buckets as the same series emitted
/// from the supervisor. Before this was extracted, the worker installed a
/// bare `PrometheusBuilder::new()` with NO bucket config — every worker-side
/// histogram (e.g. `embed_worker_queue_wait_duration_seconds`,
/// `embed_inference_duration_seconds`) fell back to library defaults, so the
/// same metric had different buckets depending on which process emitted it.
///
/// The `_duration_seconds` matcher is a Suffix match, so any new latency
/// histogram ending in `_duration_seconds` (supervisor or worker) inherits the
/// 5 ms → 30 s ladder automatically.
pub fn apply_histogram_buckets(builder: PrometheusBuilder) -> PrometheusBuilder {
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
    // Texts per HTTP request: the downstream consumer sends 1-50, bulk ingest up to ~1500.
    let texts_per_req_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_texts_per_request".to_string());
    // Pairs per /v1/rerank request: typically CROSS_ENCODER_MAX_DOCS = 15.
    let rerank_pairs_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_rerank_pairs_per_request".to_string());
    // Effective max seq len per dispatched batch (post-round_up). Powers of
    // two from 1 to 512 + 512 cap. This matches the static ONNX tensor shapes
    // used by e5-large (max_len=256) and jina-code-v2 (max_len=512).
    let max_eff_seq_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_batch_max_effective_seq".to_string());

    // Token-budget per batch: [batch_size × effective_seq_len].
    // Covers e5 (max 8×256=2048) through jina worst-case (32×512=16384)
    // up to full BATCH_MAX_TOKENS (16384) and a deliberate 4M ceiling to
    // capture runaway single allocations.
    let batch_token_budget_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_batch_token_budget".to_string());

    // Attention scratch: B×H×S²×4 bytes. Scaled to bytes (MiB/GiB bins).
    // 1 GiB bin is the smoking-gun boundary for the 1.258 GiB OOM tensor.
    let attention_scratch_matcher = metrics_exporter_prometheus::Matcher::Full(
        "embed_inference_attention_scratch_bytes".to_string(),
    );

    // Peak allocated bytes per inference (procfs RSS delta).
    let peak_bytes_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_inference_peak_bytes".to_string());

    // Input array length per /v1/embeddings request (both accepted + rejected).
    // Buckets capture happy path (1-32) through pathological 100-text batches.
    let input_array_size_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_input_array_size".to_string());

    // Documents per /v1/rerank request (both accepted + rejected). Buckets
    // mirror embed_input_array_size so operators can compare the two caps.
    // At cap=32 (gte-multi-rerank, max_len=256): 32×1×256²×4 ≈ 8 MiB/slot.
    let rerank_input_docs_size_matcher =
        metrics_exporter_prometheus::Matcher::Full("embed_rerank_input_docs_size".to_string());

    builder
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
        .set_buckets_for_metric(
            batch_token_budget_matcher,
            &[
                1_000.0,
                4_000.0,
                16_000.0,
                64_000.0,
                128_000.0,
                256_000.0,
                1_000_000.0,
                4_000_000.0,
            ],
        )
        .expect("set batch token budget buckets")
        .set_buckets_for_metric(
            attention_scratch_matcher,
            &[MIB, 16.0 * MIB, 64.0 * MIB, 256.0 * MIB, GIB, 4.0 * GIB],
        )
        .expect("set attention scratch buckets")
        .set_buckets_for_metric(
            peak_bytes_matcher,
            &[
                16.0 * MIB,
                64.0 * MIB,
                256.0 * MIB,
                512.0 * MIB,
                GIB,
                2.0 * GIB,
                4.0 * GIB,
                8.0 * GIB,
            ],
        )
        .expect("set peak bytes buckets")
        .set_buckets_for_metric(
            max_eff_seq_matcher,
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        )
        .expect("set max effective seq buckets")
        .set_buckets_for_metric(
            input_array_size_matcher,
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        )
        .expect("set input array size buckets")
        .set_buckets_for_metric(
            rerank_input_docs_size_matcher,
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        )
        .expect("set rerank input docs size buckets")
}

/// Install the supervisor Prometheus recorder and return its rendering handle.
///
/// Delegates bucket configuration to [`apply_histogram_buckets`] (the single
/// authority) so the supervisor and worker recorders never drift. Stamps
/// `embed_build_info{version}` to 1.
pub fn init(version: &str) -> PrometheusHandle {
    let handle = apply_histogram_buckets(PrometheusBuilder::new())
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
///
/// Emitted from the SUPERVISOR recorder (`:8082`). Here `duration` is the
/// `dispatch_embed` round-trip the supervisor measured — UDS connect + worker
/// queue wait + ONNX forward pass — NOT the pure forward pass. The worker
/// records its own pure-inference split via [`record_worker_inference`] under a
/// distinct name; do NOT emit `embed_inference_duration_seconds` from the
/// worker recorder or the two distributions merge under Prometheus
/// `sum by (model, le)` (both scrape jobs carry `service="embed-server"`) and
/// the per-model latency alerts read a meaningless quantile.
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

/// Record the pure ONNX forward-pass time on the WORKER recorder.
///
/// Distinct name from the supervisor's [`record_inference`]
/// (`embed_inference_duration_seconds`) ON PURPOSE — both processes are scraped
/// with `service="embed-server"`, so emitting the same series name from both
/// would make Prometheus `sum by (model, le)` merge the supervisor round-trip
/// histogram (UDS + queue + ONNX) with the worker pure-inference histogram into
/// one quantile (two observations per request, different distributions),
/// silently breaking the per-model latency alerts (e.g. `EmbedHighLatencyJina`).
///
/// With the supervisor round-trip and this worker pure-inference series both
/// available, queue wait is recoverable two ways: subtract
/// (`embed_inference_duration_seconds` round-trip − `embed_worker_inference_duration_seconds`
/// pure) or read [`record_worker_queue_wait`]'s
/// `embed_worker_queue_wait_duration_seconds` directly. Scoped to the embed
/// path only (no rerank/splade) so it never lands without a supervisor
/// counterpart to subtract against — those paths keep their existing
/// `embed_rerank_*` namespace.
//
// `allow(dead_code)`: `metrics` is compiled twice (lib `pub mod metrics` +
// `mod metrics` in main.rs). The only caller is `src/bin/worker.rs` via the
// lib path (`embed_server::metrics::`), so main.rs's private copy sees no
// caller and the lib's public copy is exempt. Same masking applied to the
// sibling worker-only recorders below.
#[allow(dead_code)]
pub fn record_worker_inference(model: &str, duration: Duration, batch_size: usize) {
    metrics::histogram!(
        "embed_worker_inference_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
    metrics::histogram!(
        "embed_worker_batch_size",
        "model" => model.to_string()
    )
    .record(batch_size as f64);
}

/// Record the time a worker request spent waiting for the per-worker
/// inference permit (UDS frame read → semaphore acquired), i.e. the
/// head-of-line queue wait BEFORE the ONNX forward pass starts.
///
/// Why this exists (jina-code-v2 backpressure incident, 2026-05-27/06-02):
/// the supervisor's `embed_inference_duration_seconds` on `:8082` measures the
/// full `dispatch_embed` round-trip — UDS connect + worker queue wait + ONNX
/// inference. When `jina-code-v2` (pool_size=1, ~13 s/inference on
/// Neoverse-N1) is backlogged by the fleet auto-index, that single histogram
/// shows a multi-second p95 that is INDISTINGUISHABLE between "the model is
/// slow" and "the queue is deep". This worker-side split lets operators
/// subtract: supervisor round-trip `embed_inference_duration_seconds` − worker
/// `embed_worker_inference_duration_seconds`
/// = `embed_worker_queue_wait_duration_seconds`. A rising queue-wait with flat
/// inference time means the producer is outrunning the drain rate — the
/// signal that previously required a manual `docker restart` to surface.
///
/// Emitted from the worker recorder, so it appears on the per-worker
/// `EMBED_WORKER_METRICS_PORT` scrape, not on supervisor `:8082`. The metric
/// name ends in `_duration_seconds` ON PURPOSE so it matches the Suffix
/// matcher in [`apply_histogram_buckets`] and is rendered as a HISTOGRAM with
/// the 5 ms → 30 s latency ladder — a name ending in only `_seconds` would
/// miss the suffix and fall back to a summary (no buckets), which the
/// regression test in `tests/worker_inference_observability.rs` guards against.
// `allow(dead_code)`: worker-only recorder, see [`record_worker_inference`].
#[allow(dead_code)]
pub fn record_worker_queue_wait(model: &str, duration: Duration) {
    metrics::histogram!(
        "embed_worker_queue_wait_duration_seconds",
        "model" => model.to_string()
    )
    .record(duration.as_secs_f64());
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

/// Record one `malloc_trim(0)` invocation by the Linux-only background
/// task in `main.rs`.
///
/// glibc keeps freed pages in arena, doesn't return them to the OS;
/// `malloc_trim(0)` forces release of all unused trailing pages. TEI runs
/// this every 100 ms (`router/src/main.rs:222-229`) — observable here as
/// `embed_malloc_trim_invocations_total`. The task is no-op on macOS/Windows.
///
/// `released` is the libc return value (`1` = some memory was released,
/// `0` = nothing to release); when non-zero the released-events counter
/// also ticks so dashboards can chart "actual work" vs raw cadence.
pub fn record_malloc_trim(released: bool) {
    metrics::counter!("embed_malloc_trim_invocations_total").increment(1);
    if released {
        metrics::counter!("embed_malloc_trim_released_events_total").increment(1);
    }
}

/// Increment the seq-capped counter — fired when a batch is split
/// because admitting another item would push `max(seq_len)` strictly
/// above `BATCH_MAX_SEQ`. Distinct from the legacy `embed_carry_events_total`
/// (which fires on ANY overflow reason): rising
/// `embed_batch_seq_capped_total` specifically indicates long-doc
/// outliers being isolated into their own batches — the desired
/// behaviour. Operators expect non-zero values whenever real prod
/// traffic mixes long and short documents.
pub fn record_seq_capped(model: &str) {
    metrics::counter!(
        "embed_batch_seq_capped_total",
        "model" => model.to_string(),
        "reason" => "seq_overflow"
    )
    .increment(1);
}

/// Increment the length-ratio carry counter — fired when a batch is
/// split because the candidate item's `max_seq_len` exceeds
/// `accum.max_len * BATCH_LENGTH_RATIO_THRESHOLD` (and the threshold
/// is > 0.0). Shares the `embed_batch_seq_capped_total` counter family
/// with `reason="length_ratio"` so dashboards can distinguish the two
/// carry causes. Only fires when the ratio gate is explicitly enabled
/// via env — default 0.0 means this counter stays at 0.
pub fn record_length_ratio_carry(model: &str) {
    metrics::counter!(
        "embed_batch_seq_capped_total",
        "model" => model.to_string(),
        "reason" => "length_ratio"
    )
    .increment(1);
}

/// Increment the solo-seq-overflow counter — fired when the FIRST item
/// of an EMPTY batch exceeds `max_batch_seq`.
///
/// The empty-batch path always admits the item (long-doc starvation
/// guard: a single request must always make forward progress regardless
/// of seq_len). This counter is pure observability — it does NOT change
/// whether the item is admitted.
///
/// Rising `embed_batch_seq_capped_total{reason="solo_seq_overflow"}` means
/// concurrent requests at seq_len > `BATCH_MAX_SEQ_{MODEL}` are arriving.
/// With `pool_size=2` each such request allocates its own attention-scratch
/// tensor from the shared arena — monitor alongside
/// `embed_inference_attention_scratch_bytes` to detect arena pressure.
pub fn record_solo_seq_overflow(model: &str) {
    metrics::counter!(
        "embed_batch_seq_capped_total",
        "model" => model.to_string(),
        "reason" => "solo_seq_overflow"
    )
    .increment(1);
}

/// Record the effective max sequence length for a dispatched batch.
///
/// `effective_seq` is the post-`round_up_seq_len` value — the actual
/// tensor dimension that attention scratch was allocated for. This is
/// what determines memory pressure in the BFCArena:
///   attention_scratch ≈ 4 × batch_size × heads × effective_seq²
///
/// Histogram buckets mirror the static tensor shapes: powers of two
/// from 1 to `max_len` plus the model `max_len` cap. Operators watching
/// this series can verify that warmup at `Some(max_len)` anchors the
/// distribution at the rightmost bucket rather than causing a replan
/// spike on the first long prod request.
pub fn record_batch_max_effective_seq(model: &str, effective_seq: usize) {
    metrics::histogram!(
        "embed_batch_max_effective_seq",
        "model" => model.to_string()
    )
    .record(effective_seq as f64);
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

/// Increment the carry-lost counter — fired when a carry item's reply
/// is resolved with `BatchError::Shutdown` because the worker exited
/// (channel closed) before the carry could be dispatched.
///
/// This is the observability path for the carry-drop-on-abort fix.
/// Called from `DynamicBatcher::shutdown`'s timeout branch, right before
/// `worker.abort()` — the reachable path where a stuck worker is aborted
/// and its in-flight carry item's reply sender is dropped via stack
/// unwind. A non-zero rate here means shutdown timed out while a carry
/// item was pending, which operators should never see in healthy
/// operation (it implies a `dispatch_batch` call was stuck in a long
/// ONNX inference at shutdown time).
pub fn record_carry_lost(model: &str) {
    metrics::counter!(
        "embed_batcher_carry_lost_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Pre-touch the carry-lost counter to 0 at batcher construction.
///
/// Prometheus counters only appear in `/metrics` after the first
/// `increment`. Without this, "0 carry items lost since boot" is
/// indistinguishable from "metric not wired" — operators get a false
/// absence-of-signal. Follows the same pattern as
/// [`worker_restart_touch`].
///
/// Called from `DynamicBatcher::with_tokens_and_max_len` (the single
/// construction path) so every batcher has its carry-lost counter
/// visible at value 0 from startup.
pub fn carry_lost_touch(model: &str) {
    metrics::counter!(
        "embed_batcher_carry_lost_total",
        "model" => model.to_string()
    )
    .absolute(0);
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

/// Increment the counter tracking how many times arena allocator
/// registration was skipped because the env already had one registered.
///
/// Expected steady-state value is 0 in a healthy single-process service —
/// a non-zero value means a second `register_shared_cpu_arena` call hit
/// the warn-path, in which case the live allocator was built from the
/// FIRST call's config and `embed_arena_*` gauges describe THAT config,
/// not whatever the latest call requested.
pub fn record_arena_register_skipped() {
    metrics::counter!("embed_arena_register_skipped_total").increment(1);
}

/// Increment the counter tracking how many times arena allocator
/// registration FAILED, causing worker startup to abort.
///
/// Expected steady-state value is 0 — a non-zero value means a worker
/// process exited immediately because `register_shared_cpu_arena_for_model`
/// returned `Err`. Without the shared arena, the embed path silently falls
/// back to per-session BFCArena (unbounded memory growth), so startup MUST
/// fail rather than warn-and-continue (matching the reranker/splade pattern
/// which panics via `assert_arena_registered_before_session`).
///
/// Pre-touched to 0 at worker startup via [`arena_registration_failed_touch`]
/// so the series is visible in `/metrics` even when no failure occurs.
// `allow(dead_code)`: worker-only recorder, see [`record_worker_inference`].
// The only caller is `src/bin/worker.rs` (embed-worker binary).
#[allow(dead_code)]
pub fn record_arena_registration_failed() {
    metrics::counter!("embed_arena_registration_failed_total").increment(1);
}

/// Pre-touch the arena-registration-failed counter to 0 at worker startup.
///
/// Prometheus counters only appear in `/metrics` after the first
/// `increment`. Without this, "0 registration failures since boot" is
/// indistinguishable from "metric not wired" — operators get a false
/// absence-of-signal. Follows the same pattern as [`carry_lost_touch`] and
/// [`worker_restart_touch`].
///
/// Called from `register_arena_for_worker` in `src/bin/worker.rs` before
/// the registration attempt, so every worker has the counter visible at
/// value 0 from startup.
// `allow(dead_code)`: worker-only recorder, see [`record_worker_inference`].
// The only caller is `src/bin/worker.rs` (embed-worker binary).
#[allow(dead_code)]
pub fn arena_registration_failed_touch() {
    metrics::counter!("embed_arena_registration_failed_total").absolute(0);
}

// ── #96: ONNX cache silently disabled ─────────────────────────────────────
/// Increment when `ONNX_OPT_CACHE_DIR` is explicitly set but unusable
/// (mkdir failed, not writable). Without this counter, the cache silently
/// disables and every startup pays full Level3 optimization cost with no
/// metric visibility — operators can't distinguish "cache working" from
/// "cache disabled" (issue #96).
pub fn record_onnx_cache_disabled(reason: &'static str) {
    metrics::counter!(
        "embed_onnx_cache_disabled_total",
        "reason" => reason
    )
    .increment(1);
}

// ── #100: warmup failure ──────────────────────────────────────────────────
/// Increment when a warmup shape fails. Without this, the first prod
/// request at that shape pays cold-start cost with no metric visibility
/// (issue #100).
pub fn record_warmup_failed(model: &str, phase: &'static str) {
    metrics::counter!(
        "embed_warmup_failed_total",
        "model" => model.to_string(),
        "phase" => phase
    )
    .increment(1);
}

// ── #101: config fallback ─────────────────────────────────────────────────
/// Increment when an explicitly-set env var has an invalid value (zero,
/// unparseable) and the system falls back to the default. Without this,
/// operators can't detect config drift from typos (issue #101).
pub fn record_config_fallback(env_var: &'static str) {
    metrics::counter!(
        "embed_config_fallback_total",
        "env_var" => env_var
    )
    .increment(1);
}

/// Increment the token-cache miss counter by `n`.
///
/// See `record_token_cache_hit` for the batch-increment semantic and
/// startup pre-warm contract.
///
/// ## Computing hit ratio in Grafana / PromQL
///
/// The hit ratio is exposed as two split counters (`outcome="hit"` /
/// `outcome="miss"`), not a precomputed gauge — gauges of ratios are a
/// Prometheus anti-pattern (loses rate information). Compute in PromQL:
///
/// ```promql
/// sum by (model) (rate(embed_token_cache_total{outcome="hit"}[5m]))
///   /
/// sum by (model) (rate(embed_token_cache_total[5m]))
/// ```
///
/// Returns the fraction of token-cache lookups that hit, per model, averaged
/// over the last 5 minutes. Useful for tuning `TOKEN_CACHE_MAX_ENTRIES` —
/// ratio < 0.3 means the cache is too small for the workload, > 0.95 means
/// memory could be reclaimed by shrinking it.
pub fn record_token_cache_miss(model: &str, n: u64) {
    metrics::counter!(
        "embed_token_cache_total",
        "model" => model.to_string(),
        "outcome" => "miss"
    )
    .increment(n);
}

// ── Deep forensic metrics (enabled when EMBED_DEEP_METRICS=1) ────────────────
//
// Added 2026-05-06 to localise the 1.258 GiB BFCArena OOM in jina-code-v2.
// The five series below pinpoint the exact (model, batch_size, seq_len) tuple
// that overflows the arena.  Cheap counters/histograms — no locks on the hot
// path.

/// Record token-budget per dispatched batch: `batch_size × effective_seq_len`.
///
/// For padded models (all BERT-family dense embedders) effective_seq_len is
/// the power-of-two-rounded `max(seq_len_in_batch)` that ends up in the actual
/// tensor.  This is the direct predictor of per-inference scratch memory.
///
/// Buckets: 1k, 4k, 16k, 64k, 128k, 256k, 1M, 4M.  The 1.258 GiB tensor
/// maps to a product ≈ 330M — clearly in the 256k→1M or 1M→4M bin.
pub fn record_batch_token_budget(model: &str, batch_size: usize, seq_len: usize) {
    metrics::histogram!(
        "embed_batch_token_budget",
        "model" => model.to_string()
    )
    .record((batch_size * seq_len) as f64);
}

/// Increment the 2D (batch_size_bucket, seq_len_bucket) counter for one
/// dispatched batch.  Gives the full (B, S) distribution without unbounded
/// cardinality — six batch buckets × five seq buckets = 30 label combinations
/// maximum per model.
///
/// batch_size_bucket: "1" | "2" | "4" | "8" | "16" | "32+"
/// seq_len_bucket:    "64" | "128" | "256" | "384" | "512+"
pub fn record_batch_dimensions(model: &str, batch_size: usize, seq_len: usize) {
    let bs_bucket = match batch_size {
        1 => "1",
        2 => "2",
        3..=4 => "4",
        5..=8 => "8",
        9..=16 => "16",
        _ => "32+",
    };
    let sl_bucket = match seq_len {
        0..=64 => "64",
        65..=128 => "128",
        129..=256 => "256",
        257..=384 => "384",
        _ => "512+",
    };
    metrics::counter!(
        "embed_batch_dimensions_total",
        "model" => model.to_string(),
        "batch_size_bucket" => bs_bucket,
        "seq_len_bucket" => sl_bucket,
    )
    .increment(1);
}

/// Record estimated self-attention scratch bytes for one inference:
///   `batch_size × num_heads × seq_len² × 4`
///
/// For jina-code-v2 (12 heads, max_len=512):
///   B=1, S=512  → 1 × 12 × 512² × 4 = 12 MiB
///   B=8, S=512  → 96 MiB
///   B=32, S=512 → 384 MiB   ← still under 512 MiB
/// The 1.258 GiB tensor requires B×H×S²×4 ≈ 1.258 GiB:
///   1.258G / (12 × 4) = 26.8M tokens² → S²×B ≈ 26.8M → B=32, S=915 (impossible),
///   or multi-head intermediate Q/K/V stacks (3×) → B=1, S=512 × 3 layers = 1.258 GiB
///   Actual: memory_pattern allocates the full forward pass graph at once.
///
/// Buckets: 1 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB, 4 GiB+.
pub fn record_attention_scratch(model: &str, batch_size: usize, num_heads: usize, seq_len: usize) {
    // Use f64 throughout to avoid usize overflow on large shapes.
    let bytes = (batch_size as f64) * (num_heads as f64) * (seq_len as f64).powi(2) * 4.0;
    metrics::histogram!(
        "embed_inference_attention_scratch_bytes",
        "model" => model.to_string()
    )
    .record(bytes);
}

/// Record peak resident-set delta for one inference (bytes).
///
/// Uses `/proc/self/statm` resident-page count (Linux only).  On non-Linux
/// the call is a no-op — the metric simply never appears.  On Linux, the
/// delta is the change in RSS from just before `session.run()` to just after;
/// negative deltas (freed pages returned to OS concurrently) are recorded as 0.
///
/// This is NOT a per-thread allocator delta (no jemalloc dependency) — it is
/// a process-wide RSS snapshot.  Under concurrent inference it can be noisy,
/// but a spike to >1 GiB is unambiguous even with noise.
pub fn record_inference_peak_bytes(model: &str, delta_bytes: u64) {
    metrics::histogram!(
        "embed_inference_peak_bytes",
        "model" => model.to_string()
    )
    .record(delta_bytes as f64);
}

/// Increment the BFCArena extend counter for one parsed ORT log event.
///
/// Called by the tracing `OrtLogLayer` when it intercepts an ORT INFO event
/// matching `Extending BFCArena for Cpu. bin_num:N num_bytes:M`.
///
/// `bin_num` is encoded as a string label so operators can immediately see
/// which bin is responsible (bin 20 ≈ 1.25 GiB per extension).
pub fn record_arena_extend(model: &str, bin_num: u32) {
    metrics::counter!(
        "embed_arena_extend_total",
        "model" => model.to_string(),
        "bin_num" => bin_num.to_string(),
    )
    .increment(1);
}

/// Increment the inference-failure counter with an OOM reason tag.
///
/// `reason`: "arena_oom" when the error message contains
/// "Available memory of X is smaller than requested bytes of Y";
/// "other" for everything else. Use `classify_worker_error` to derive
/// the reason label from a raw error string.
///
/// `bin_num`: always 0 on the worker-pool path (post-Wave-2.4b) — the
/// worker process surfaces a final error string, not the BFCArena bin.
/// The legacy in-process path (EMBED_MULTI_PROCESS=0) may set non-zero
/// when the OrtLogLayer intercepts a BFCArena extend event. Pass 0
/// when calling from worker-pool error handling.
pub fn record_inference_failure(model: &str, reason: &str, bin_num: u32) {
    metrics::counter!(
        "embed_inference_failures_total",
        "model" => model.to_string(),
        "reason" => reason.to_string(),
        "bin_num" => bin_num.to_string(),
    )
    .increment(1);
}

/// Classify a worker-side error message into a bounded-cardinality `reason`
/// label for `embed_inference_failures_total`. Avoids letting raw worker
/// strings into Prometheus labels (would explode cardinality and leak
/// internal paths).
///
/// Categories:
/// - `queue_overflow` — per-worker waiter cap exceeded (AtomicUsize gate)
/// - `semaphore_closed` — semaphore explicitly closed (post-PR-#70 path)
/// - `worker_saturated` — legacy bucket (pre-PR-#70 `try_acquire` path);
///   kept for backward-compat during rolling deploys / image rollbacks
/// - `arena_oom` — BFCArena allocation failed
/// - `kind_mismatch` — supervisor sent wrong message variant to worker
/// - `tokenize` — tokenizer error
/// - `other` — fallback bucket for uncategorized errors (track via logs)
///
/// Ordering: longer/more-specific substrings first to prevent `"saturated"`
/// matching inside longer messages. `"queue overflow"` before `"semaphore
/// closed"` before `"saturated"` preserves most-specific-first semantics.
pub fn classify_worker_error(message: &str) -> &'static str {
    let m = message.to_ascii_lowercase();
    if m.contains("queue overflow") {
        "queue_overflow"
    } else if m.contains("semaphore closed") {
        "semaphore_closed"
    } else if m.contains("saturated") {
        "worker_saturated"
    } else if m.contains("bfcarena") || m.contains("arena") || m.contains("oom") {
        "arena_oom"
    } else if m.contains("kind mismatch") {
        "kind_mismatch"
    } else if m.contains("tokeniz") {
        "tokenize"
    } else {
        "other"
    }
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

/// Increment the input-array-rejected counter. Called when a `/v1/embeddings`
/// request carries more texts than `embed_max_input_array`. The request is
/// rejected with HTTP 400 before reaching the batcher — the counter lets
/// operators correlate this guard with BFCArena pressure relief.
///
/// `reason` is always `"size_cap"` today; the label exists for future
/// extension (e.g. a token-count-based pre-reject).
pub fn record_input_array_rejected(model: &str, reason: &str) {
    metrics::counter!(
        "embed_input_array_rejected_total",
        "model" => model.to_string(),
        "reason" => reason.to_string(),
    )
    .increment(1);
}

/// Record the input array length for one `/v1/embeddings` request — both
/// accepted and rejected requests are recorded so operators can see the
/// natural distribution and tune `EMBED_MAX_INPUT_ARRAY` accordingly.
///
/// Histogram buckets: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, +Inf]`.
/// These capture both the happy path (1-32 texts from the downstream consumer) and the
/// pathological 100-text batches from the downstream consumer that caused the BFCArena OOM.
pub fn record_input_array_size(model: &str, n: usize) {
    metrics::histogram!(
        "embed_input_array_size",
        "model" => model.to_string()
    )
    .record(n as f64);
}

/// Increment the batcher first-item-oversize counter. Called from
/// `run_worker` when the very first item in an empty batch would have
/// failed `fits()` against a hypothetical non-empty accumulator —
/// i.e., it was admitted only because empty-batch starvation guard
/// bypasses normal `fits()` checks.
///
/// `reason` is one of:
/// - `"exceeds_batch_max_items"` — `item.n_texts() > cfg.max_batch_items`
/// - `"exceeds_max_batch_tokens"` — token-budget would overflow even alone
///
/// This counter is pure observability; it does NOT gate admission or change
/// any behaviour. A rising rate combined with large `embed_input_array_size`
/// buckets indicates the server-side cap is doing its job.
pub fn record_batcher_first_item_oversize(model: &str, reason: &str) {
    metrics::counter!(
        "embed_batcher_first_item_oversize_total",
        "model" => model.to_string(),
        "reason" => reason.to_string(),
    )
    .increment(1);
}

/// Set the `embed_arena_shrink_enabled` gauge (1.0 = on, 0.0 = off).
///
/// Called once at model load time from `EmbedModel::load` so operators can
/// verify from `/metrics` whether per-run BFCArena shrinkage is active for
/// each model without reading logs.
pub fn set_arena_shrink_enabled(model: &str, enabled: bool) {
    metrics::gauge!(
        "embed_arena_shrink_enabled",
        "model" => model.to_string()
    )
    .set(if enabled { 1.0 } else { 0.0 });
}

/// Increment the per-run arena shrinkage call counter.
///
/// Called ONLY from `embed_tokens` when shrinkage is enabled — immediately
/// before or after `session.run_with_options` that carries the
/// `"memory.enable_memory_arena_shrinkage"` config entry. Warmup
/// (`warmup_at_shape`) intentionally does NOT increment this counter so the
/// rate reflects only production inference, not startup warmup passes.
///
/// For models with shrinkage enabled, the counter rate should match
/// `embed_requests_total{model="<that-model>"}` rate (filter to the same
/// model label — comparing against the all-models aggregate would
/// undercount because e5-large + reranker + splade don't increment this
/// counter when shrinkage is auto-disabled for them). A rate lower than
/// the per-model `embed_requests_total` means shrinkage is disabled
/// mid-run or the gate is misfiring; a higher rate indicates
/// double-counting (a bug).
pub fn record_arena_shrink_call(model: &str) {
    metrics::counter!(
        "embed_arena_shrink_calls_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Increment the worker-restart counter on successful supervisor respawn.
///
/// Called by `watchdog_loop` in `supervisor::handle` after each successful
/// `spawn_one`. Label `model` matches the `SpawnSpec::model` string so
/// operators can correlate restarts with request-error spikes on the same
/// model series.
pub fn worker_restart_inc(model: &str) {
    metrics::counter!(
        "embed_worker_restart_total",
        "model" => model.to_string()
    )
    .increment(1);
}

/// Pre-touch the worker-restart counter to 0 on supervisor startup.
///
/// Prometheus counters only appear in `/metrics` after the first
/// `increment`. Without this, "0 restarts since boot" is indistinguishable
/// from "metric not wired" — operators get a false absence-of-signal.
///
/// Called by `WorkerSupervisor::launch` after the initial spawn succeeds,
/// so every healthy worker has its restart counter visible at value 0.
pub fn worker_restart_touch(model: &str) {
    metrics::counter!(
        "embed_worker_restart_total",
        "model" => model.to_string()
    )
    .absolute(0);
}
/// Set the `embed_worker_queue_depth` gauge for a worker process.
///
/// Called from `src/bin/worker.rs` after each `WAITERS.fetch_add` /
/// `WAITERS.fetch_sub` mutation to keep the gauge in sync with the in-flight
/// waiter count.
///
/// Gauge semantics: this is per-worker-process. Each worker exposes its own
/// `/metrics` HTTP server (port assigned by `EMBED_WORKER_METRICS_PORT`);
/// the supervisor's `/metrics` endpoint does NOT include this series.
///
/// Race note: the gauge is updated with a relaxed atomic load — it is an
/// observation, not a control signal. A brief lag of one request cycle is
/// acceptable for a gauge vs. introducing unnecessary acquire/release fences.
// `allow(dead_code)`: worker-only recorder, see [`record_worker_inference`].
#[allow(dead_code)]
pub fn set_worker_queue_depth(model: &str, depth: usize) {
    metrics::gauge!(
        "embed_worker_queue_depth",
        "model" => model.to_string()
    )
    .set(depth as f64);
}

/// Set the per-worker RSS gauge.
///
/// Called by the supervisor RSS-poll loop every 15 s. Value is bytes
/// read from `/proc/<pid>/status` VmRSS field.
///
/// `model` matches `SpawnSpec::model` (e.g. `multilingual-e5-large`) so
/// operators can alert per-model: `embed_worker_rss_bytes{model} > 4 GiB`.
pub fn worker_rss_set(model: &str, bytes: f64) {
    metrics::gauge!(
        "embed_worker_rss_bytes",
        "model" => model.to_string()
    )
    .set(bytes);
}

/// Pre-touch the per-worker RSS gauge to 0 at supervisor launch.
///
/// Without this, Prometheus reads "no data" as absent until the first
/// 15 s poll fires — indistinguishable from "metric not wired". Calling
/// this immediately after initial spawn makes the 0-byte baseline visible.
pub fn worker_rss_touch(model: &str) {
    metrics::gauge!(
        "embed_worker_rss_bytes",
        "model" => model.to_string()
    )
    .set(0.0);
}

/// Increment the rerank-documents-rejected counter. Called when a
/// `/v1/rerank` request carries more documents than `rerank_max_input_docs`.
/// The request is rejected with HTTP 400 before tokenization — the counter
/// lets operators correlate this guard with BFCArena pressure relief.
///
/// `reason` is always `"size_cap"` today; the label exists for future
/// extension (e.g. a token-count-based pre-reject).
///
/// **Schema note**: `model="unknown"` is emitted as a placeholder because
/// the cap fires BEFORE model resolution from the request body — the
/// reranker model is selected per-request via `model` field, not yet
/// known at the rejection point. The label exists for schema symmetry
/// with `embed_input_array_rejected_total{model,reason}` (PR #49) so
/// dashboards can join both series under the same label set.
pub fn record_rerank_input_docs_rejected(reason: &str) {
    metrics::counter!(
        "embed_rerank_input_docs_rejected_total",
        "model" => "unknown",
        "reason" => reason.to_string(),
    )
    .increment(1);
}

/// Record the documents array length for one `/v1/rerank` request — both
/// accepted and rejected requests are recorded so operators can see the
/// natural distribution and tune `RERANK_MAX_INPUT_DOCS`.
///
/// Histogram buckets: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, +Inf]`.
///
/// **Schema note**: `model="unknown"` is emitted as a placeholder because
/// the size is recorded BEFORE model resolution from the request body —
/// the reranker model is selected per-request via `model` field. The label
/// exists for schema symmetry with `embed_input_array_size{model}` (PR #49)
/// so dashboards can join both series under the same label set.
pub fn record_rerank_input_docs_size(n: usize) {
    metrics::histogram!(
        "embed_rerank_input_docs_size",
        "model" => "unknown",
    )
    .record(n as f64);
}

// ── shared test helper ────────────────────────────────────────────────────────

/// Install a Prometheus recorder the first time it's needed and return its
/// handle. Subsequent calls from any module return the same handle; the global
/// `metrics` recorder can only be installed once per process.
///
/// Used by `batcher::tests`, `api::tests`, and any other in-process test
/// module that needs to assert on metric values. A single shared function
/// ensures only one `install_recorder()` call happens regardless of test
/// parallelism.
#[cfg(test)]
pub fn test_prometheus_handle() -> &'static PrometheusHandle {
    use std::sync::OnceLock;
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("install Prometheus recorder for tests")
    })
}

// ── unit tests for bucket-label logic ────────────────────────────────────────

#[cfg(test)]
mod classify_worker_error_tests {
    use super::classify_worker_error;

    #[test]
    fn queue_overflow_detected() {
        assert_eq!(
            classify_worker_error("worker queue overflow"),
            "queue_overflow"
        );
        assert_eq!(
            classify_worker_error("WORKER QUEUE OVERFLOW"),
            "queue_overflow"
        );
    }

    #[test]
    fn semaphore_closed_detected() {
        assert_eq!(
            classify_worker_error("worker semaphore closed"),
            "semaphore_closed"
        );
        assert_eq!(
            classify_worker_error("Semaphore Closed"),
            "semaphore_closed"
        );
    }

    #[test]
    fn saturated_backward_compat() {
        assert_eq!(
            classify_worker_error("worker saturated"),
            "worker_saturated"
        );
    }

    #[test]
    fn arena_oom_variants() {
        assert_eq!(classify_worker_error("bfcarena out of memory"), "arena_oom");
        assert_eq!(classify_worker_error("arena alloc failed"), "arena_oom");
        assert_eq!(classify_worker_error("oom kill"), "arena_oom");
    }

    #[test]
    fn kind_mismatch() {
        assert_eq!(
            classify_worker_error("kind mismatch: model is embed"),
            "kind_mismatch"
        );
    }

    #[test]
    fn tokenize_error() {
        assert_eq!(classify_worker_error("tokenizer failed"), "tokenize");
    }

    #[test]
    fn other_fallback() {
        assert_eq!(classify_worker_error("unexpected runtime panic"), "other");
    }
}

#[cfg(test)]
mod bucket_label_tests {
    /// Replicate the batch_size and seq_len bucketing logic from
    /// `record_batch_dimensions` to verify boundary conditions without
    /// requiring a live Prometheus recorder.
    fn bs_bucket(n: usize) -> &'static str {
        match n {
            1 => "1",
            2 => "2",
            3..=4 => "4",
            5..=8 => "8",
            9..=16 => "16",
            _ => "32+",
        }
    }

    fn sl_bucket(n: usize) -> &'static str {
        match n {
            0..=64 => "64",
            65..=128 => "128",
            129..=256 => "256",
            257..=384 => "384",
            _ => "512+",
        }
    }

    #[test]
    fn batch_size_boundaries() {
        assert_eq!(bs_bucket(1), "1");
        assert_eq!(bs_bucket(2), "2");
        assert_eq!(bs_bucket(3), "4");
        assert_eq!(bs_bucket(4), "4");
        assert_eq!(bs_bucket(5), "8");
        assert_eq!(bs_bucket(8), "8");
        assert_eq!(bs_bucket(9), "16");
        assert_eq!(bs_bucket(16), "16");
        assert_eq!(bs_bucket(17), "32+");
        assert_eq!(bs_bucket(32), "32+");
        assert_eq!(bs_bucket(100), "32+");
    }

    #[test]
    fn seq_len_boundaries() {
        assert_eq!(sl_bucket(0), "64");
        assert_eq!(sl_bucket(64), "64");
        assert_eq!(sl_bucket(65), "128");
        assert_eq!(sl_bucket(128), "128");
        assert_eq!(sl_bucket(129), "256");
        assert_eq!(sl_bucket(256), "256");
        assert_eq!(sl_bucket(257), "384");
        assert_eq!(sl_bucket(384), "384");
        assert_eq!(sl_bucket(385), "512+");
        assert_eq!(sl_bucket(512), "512+");
        assert_eq!(sl_bucket(1024), "512+");
    }

    #[test]
    fn attention_scratch_formula() {
        // jina-code-v2: B=1, H=12, S=512 → 1×12×512²×4 = 12_582_912 bytes (12 MiB)
        let bytes = (1_f64) * (12_f64) * (512_f64).powi(2) * 4.0;
        assert_eq!(bytes as u64, 12_582_912);

        // B=8, H=12, S=512 → 100_663_296 bytes (96 MiB)
        let bytes_b8 = (8_f64) * (12_f64) * (512_f64).powi(2) * 4.0;
        assert_eq!(bytes_b8 as u64, 100_663_296);
    }
}

// ── readiness probe ───────────────────────────────────────────────────────────

/// Record the outcome of a `/ready` probe.
///
/// `result` is one of `"ok"`, `"timeout"`, `"error"`, or `"shutdown"`.
/// Operators alert on `rate(embed_ready_probe_total{result="timeout"}[5m]) > 0`
/// to catch a wedged worker before downstream consumers notice.
pub fn record_ready_probe(result: &str) {
    metrics::counter!(
        "embed_ready_probe_total",
        "result" => result.to_string()
    )
    .increment(1);
}

// ── worker heartbeat (issue #90: wedged-worker detection) ──────────────────────

/// Record a heartbeat probe outcome. Labels:
/// - `ok`     — probe completed, worker responsive.
/// - `timeout`— probe timed out (worker may be wedged).
/// - `error`  — probe returned an error (worker error or dispatch failure).
/// - `kill`   — N consecutive fails reached, worker SIGKILL'd for respawn.
///
/// Operators alert on `rate(embed_worker_heartbeat_total{result="timeout"}[5m]) > 0`
/// to catch a wedged worker before the heartbeat kills it, and on
/// `rate(embed_worker_heartbeat_total{result="kill"}[5m]) > 0` to detect
/// chronic wedging (a worker that keeps getting killed is a deeper problem).
pub fn record_worker_heartbeat(result: &str) {
    metrics::counter!(
        "embed_worker_heartbeat_total",
        "result" => result.to_string()
    )
    .increment(1);
}

/// Pre-touch the ready-probe counter to 0 so "no probes yet" is visible
/// in Prometheus as a present-but-zero series, not absent.
pub fn ready_probe_touch() {
    metrics::counter!(
        "embed_ready_probe_total",
        "result" => "ok".to_string()
    )
    .absolute(0);
    metrics::counter!(
        "embed_ready_probe_total",
        "result" => "timeout".to_string()
    )
    .absolute(0);
    metrics::counter!(
        "embed_ready_probe_total",
        "result" => "error".to_string()
    )
    .absolute(0);
    metrics::counter!(
        "embed_ready_probe_total",
        "result" => "shutdown".to_string()
    )
    .absolute(0);
}

/// Pre-touch the heartbeat counter to 0 for all result labels so "no
/// heartbeats yet" is visible in Prometheus as present-but-zero, not absent.
pub fn worker_heartbeat_touch() {
    for result in ["ok", "timeout", "error", "kill"] {
        metrics::counter!(
            "embed_worker_heartbeat_total",
            "result" => result.to_string()
        )
        .absolute(0);
    }
}
