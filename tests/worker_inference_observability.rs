//! Regression guard for the jina-code-v2 backpressure observability fix
//! (2026-06-03).
//!
//! Before this fix the supervisor's `embed_inference_duration_seconds` on
//! `:8082` conflated three costs into one histogram: UDS connect + per-worker
//! queue wait + ONNX forward pass. When `jina-code-v2` (pool_size=1, ~13 s per
//! inference on Neoverse-N1) was backlogged by the fleet auto-index, that
//! single metric showed a multi-second p95 that could NOT distinguish "the
//! model is slow" from "the queue is deep" — the operator had to `docker
//! restart` to find out.
//!
//! The fix records, on the WORKER recorder:
//!   - `embed_worker_queue_wait_duration_seconds{model}` — head-of-line queue wait
//!     (frame read → inference permit acquired)
//!   - `embed_inference_duration_seconds{model}` — pure ONNX forward pass
//!
//! and routes the worker recorder through `metrics::apply_histogram_buckets`
//! (the single bucket authority) so worker-side latency histograms land in the
//! same 5 ms → 30 s ladder as the supervisor's.
//!
//! Each test installs a process-global Prometheus recorder, so they live in
//! their own test binary (one `install_recorder()` per process). They run in a
//! single process serially via the harness; the FIRST test to call
//! `install_recorder` wins, so we install once in a shared helper and let both
//! assertions share the handle.

use embed_server::metrics::{apply_histogram_buckets, record_inference, record_worker_queue_wait};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
use std::time::Duration;

/// Process-global handle: a Prometheus recorder can be installed only once per
/// process. Both metric families are exercised against the same recorder.
fn handle() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        apply_histogram_buckets(PrometheusBuilder::new())
            .install_recorder()
            .expect("install recorder with shared buckets")
    })
}

#[test]
fn worker_queue_wait_and_inference_render_with_shared_buckets() {
    let h = handle();

    // Exercise the queue-wait histogram and the pure-inference histogram the
    // way the worker does — both labelled by model.
    record_worker_queue_wait("jina-code-v2", Duration::from_millis(1500));
    record_inference("jina-code-v2", Duration::from_millis(13_000), 1);

    let rendered = h.render();

    // 1. Both series exist, model-labelled.
    assert!(
        rendered.contains("embed_worker_queue_wait_duration_seconds"),
        "queue-wait histogram must be exported so operators can separate \
         queue depth from model speed:\n{rendered}"
    );
    assert!(
        rendered.contains("embed_inference_duration_seconds"),
        "pure-inference histogram must be exported from the worker recorder:\n{rendered}"
    );
    assert!(
        rendered.contains(r#"embed_worker_queue_wait_duration_seconds_bucket{model="jina-code-v2""#),
        "queue-wait series must carry the model label:\n{rendered}"
    );

    // 2. The shared bucket authority applied to BOTH histograms — the
    //    `_duration_seconds` Suffix matcher gives every latency histogram the
    //    30 s top bucket. A bare `PrometheusBuilder::new()` (the pre-fix worker
    //    recorder) would NOT produce a `le="30"` bucket. This guards the
    //    bucket-drift regression: if a future edit reverts the worker to a bare
    //    builder, this assertion fails.
    assert!(
        rendered.contains(r#"embed_worker_queue_wait_duration_seconds_bucket{model="jina-code-v2",le="30""#),
        "queue-wait histogram must inherit the shared 30 s top bucket \
         (proves apply_histogram_buckets ran on the worker recorder):\n{rendered}"
    );
    assert!(
        rendered
            .contains(r#"embed_inference_duration_seconds_bucket{model="jina-code-v2",le="30""#),
        "inference histogram must inherit the shared 30 s top bucket:\n{rendered}"
    );

    // 3. The 1.5 s queue wait lands at/above the 2.5 s cumulative bucket but
    //    NOT in the 1 s bucket — confirms the value (not just the series name)
    //    is recorded. Cumulative histogram semantics: le="2.5" count == 1,
    //    le="1" count == 0.
    assert!(
        rendered.contains(r#"embed_worker_queue_wait_duration_seconds_bucket{model="jina-code-v2",le="2.5"} 1"#),
        "1.5 s wait must fall in the le=2.5 cumulative bucket:\n{rendered}"
    );
}
