//! OpenTelemetry tracing integration. Phase H.18 (2026-05-01).
//!
//! ## Why
//!
//! embed-server already exposes Prometheus aggregates (Phase 1A —
//! `embed_rerank_*` histograms etc.) but those are post-hoc averages.
//! When debugging a slow chat turn the operator wants to see the EXACT
//! request that took 12 s, not "p99 was 8 s in the last 5 min". OTEL
//! traces give per-request span trees correlated with the downstream consumer's
//! existing OTEL spans via the W3C `traceparent` header — so a single
//! Jaeger query on a chat trace_id surfaces the embed-server span tree
//! inline with the upstream search, retrieval, and LLM-extract spans.
//!
//! ## What's plumbed
//!
//! - `tracing::info_span!` / `tracing::instrument` attributes that the
//!   handlers and model code already emit are routed via
//!   `tracing-opentelemetry` into an OTLP gRPC exporter.
//! - `extract_remote_context` reads the W3C `traceparent` (and optional
//!   `tracestate`) headers off an axum `HeaderMap`, returning a
//!   `tracing::Span` that should be set as the parent of the request
//!   span. This is the link between the downstream consumer's outgoing trace and our
//!   span tree.
//!
//! ## What stays disabled
//!
//! - When `OTEL_EXPORTER_OTLP_ENDPOINT` is unset the exporter is NOT
//!   installed — `init` is a no-op and the program runs identically to
//!   the pre-Phase-H.18 baseline. CI / local dev / non-Jaeger envs see
//!   no change.
//! - Sampling defaults to `parentbased_traceidratio=0.05` (5 %) when
//!   the env var is unset, matching the downstream consumer's setting. Override via
//!   `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG` per the spec.

use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::Subscriber;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

/// Service name advertised on every span. Visible in Jaeger's service
/// dropdown and as the `service.name` resource attribute.
const SERVICE_NAME: &str = "embed-server";

// ── ORT log interception layer ────────────────────────────────────────────────

/// A `tracing_subscriber::Layer` that intercepts ORT log events from the
/// `ort::logging` target and parses them into Prometheus counters.
///
/// ORT emits arena events at INFO level (target = `ort::logging`) with
/// messages like:
///   "Extending BFCArena for Cpu. bin_num:20 num_bytes:1258291200"
///
/// We parse those into `embed_arena_extend_total{model,bin_num}`.
///
/// The layer is zero-cost when there are no matching events — the visit
/// closure is only called for events that match the `ort::logging` target.
pub struct OrtLogLayer;

impl<S> Layer<S> for OrtLogLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        // Only process events from ORT's logging bridge.
        if event.metadata().target() != "ort::logging" {
            return;
        }
        // Collect the message field via a visitor.
        let mut visitor = OrtMessageVisitor::default();
        event.record(&mut visitor);
        let msg = visitor.message;

        // Parse: "Extending BFCArena for Cpu. bin_num:N num_bytes:M"
        if let Some(bin_num) = parse_bfc_extend(&msg) {
            // The ORT log doesn't carry a model name; use "unknown" so the
            // metric still appears. Once we can disambiguate sessions
            // (future ort API), we can pass the real name.
            crate::metrics::record_arena_extend("unknown", bin_num);
        }
    }
}

/// Visitor that extracts the `message` field from a tracing event.
#[derive(Default)]
struct OrtMessageVisitor {
    message: String,
}

impl tracing::field::Visit for OrtMessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }
}

/// Parse a BFCArena extend message.
///
/// Expected format: `"Extending BFCArena for Cpu. bin_num:N num_bytes:M"`
/// Returns `Some(bin_num)` on success, `None` if the message doesn't match.
fn parse_bfc_extend(msg: &str) -> Option<u32> {
    if !msg.contains("Extending BFCArena") {
        return None;
    }
    // Find "bin_num:" and parse the integer that follows.
    let bin_start = msg.find("bin_num:")?;
    let after = &msg[bin_start + "bin_num:".len()..];
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    after[..end].parse::<u32>().ok()
}

/// Initialise tracing with OTLP exporter when configured, falling back
/// to plain stdout-JSON when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset.
///
/// Returns `Some(provider)` so `main` can hold the handle for graceful
/// shutdown (`provider.shutdown()` flushes the batch exporter on SIGTERM
/// — without it spans for the last few requests get dropped).
///
/// Idempotency: must only be called once per process. Subsequent calls
/// will panic in `tracing_subscriber::set_global_default`. The contract
/// matches `tracing_subscriber::fmt::init` for that reason.
pub fn init() -> Option<SdkTracerProvider> {
    // Always set the W3C propagator regardless of exporter — cheap, and
    // lets `extract_remote_context` work even in the no-exporter path
    // (the resulting context just has no upstream parent to attach to).
    global::set_text_map_propagator(TraceContextPropagator::new());

    let endpoint = match std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        Ok(e) if !e.trim().is_empty() => e,
        _ => {
            // No exporter configured — fall back to plain JSON logs
            // exactly as the pre-Phase-H.18 setup did. This is the path
            // CI takes (no Jaeger present) and is also the safe default
            // if the env var is dropped from compose.
            // Apply RUST_LOG filter to fmt output, defaulting to the same
            // baseline the pre-Phase-H.18 main.rs used: info + ort::logging=warn.
            // Without this, the global subscriber falls back to TRACE and
            // ort::logging floods stdout with per-allocation messages.
            // ORT arena logging: allow INFO for the ort::logging target so
            // OrtLogLayer can parse "Extending BFCArena" events.  The fmt
            // layer still suppresses them via its own filter — OrtLogLayer
            // intercepts at the registry level before fmt filtering.
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort::logging=info".parse().unwrap());
            tracing_subscriber::registry()
                .with(OrtLogLayer)
                .with(tracing_subscriber::fmt::layer().json().with_filter(filter))
                .init();
            tracing::info!("OTEL disabled (OTEL_EXPORTER_OTLP_ENDPOINT unset)");
            return None;
        }
    };

    let exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_timeout(Duration::from_secs(3))
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            // Don't fail the server if Jaeger is unreachable at boot —
            // serve traffic with stdout-JSON only and let an alert pick
            // up the missing trace flow downstream.
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort::logging=info".parse().unwrap());
            tracing_subscriber::registry()
                .with(OrtLogLayer)
                .with(tracing_subscriber::fmt::layer().json().with_filter(filter))
                .init();
            tracing::error!(error = %err, endpoint = %endpoint, "OTLP exporter init failed — running without traces");
            return None;
        }
    };

    let resource = Resource::builder()
        .with_attribute(KeyValue::new("service.name", SERVICE_NAME))
        .with_attribute(KeyValue::new(
            "service.version",
            std::env::var("EMBED_VERSION").unwrap_or_else(|_| "dev".to_string()),
        ))
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer(SERVICE_NAME);
    // Each subscriber layer needs its OWN EnvFilter — they are NOT
    // global filters. Without one, the layer defaults to TRACE and
    // ort::logging floods stdout with per-allocation messages
    // (incident 2026-05-02 H.18 deploy: log volume + perf hit).
    let otel_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,ort::logging=info".parse().unwrap());
    let fmt_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,ort::logging=info".parse().unwrap());
    let otel_layer = OpenTelemetryLayer::new(tracer).with_filter(otel_filter);

    tracing_subscriber::registry()
        .with(OrtLogLayer)
        .with(otel_layer)
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_filter(fmt_filter),
        )
        .init();

    // Set the provider globally so `global::tracer(...)` calls (and
    // shutdown) pick it up. The local `provider` clone returned to main
    // is for explicit shutdown only.
    global::set_tracer_provider(provider.clone());

    tracing::info!(endpoint = %endpoint, "OTEL exporter installed");
    Some(provider)
}

/// Shut the batch exporter down cleanly on SIGTERM. Called from main's
/// drain handler so spans for in-flight / just-finished requests get
/// flushed instead of dropped.
pub fn shutdown(provider: SdkTracerProvider) {
    if let Err(e) = provider.shutdown() {
        tracing::warn!(error = ?e, "OTEL provider shutdown error");
    }
}

/// Axum middleware that creates a root span per request, links it to
/// the upstream caller's trace via the W3C `traceparent` header, and
/// awaits the handler inside the span scope. Wire on the router via
/// `.layer(axum::middleware::from_fn(otel::trace_request))`.
///
/// Span name is `HTTP {method} {path}` (matches OTEL semantic
/// conventions), and the span carries `http.method`, `http.route`,
/// `http.status_code` attributes for Jaeger filtering.
///
/// When no `traceparent` header is present (curl smoke, healthcheck,
/// Prometheus scrape) the span becomes a new trace root — still useful
/// in Jaeger, just no upstream link to follow.
pub async fn trace_request(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use opentelemetry::propagation::Extractor;
    use tracing::Instrument;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // The handler will populate http.status_code on the response after
    // it runs; we initialise the field as `0` so it shows up in Jaeger
    // even before being set (otherwise the attribute is missing and
    // the operator can't tell "0" from "the handler crashed before
    // setting it").
    let span = tracing::info_span!(
        "http_request",
        otel.name = %format_args!("HTTP {} {}", method, path),
        http.method = %method,
        http.route = %path,
        http.status_code = tracing::field::Empty,
    );

    // Extract upstream traceparent and stamp it as the span's parent.
    struct HeaderExtractor<'a>(&'a axum::http::HeaderMap);
    impl<'a> Extractor for HeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key)?.to_str().ok()
        }
        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }
    let parent_cx =
        global::get_text_map_propagator(|prop| prop.extract(&HeaderExtractor(req.headers())));
    // tracing-opentelemetry 0.33 made `set_parent` fallible (it was infallible
    // in 0.31). It fails when the span has no OpenTelemetry layer attached —
    // i.e. when OTEL is disabled — which is a normal configuration here, not an
    // error worth a warn on every request. Losing the parent link only breaks
    // trace correlation for this request, so debug-log it and continue serving.
    if let Err(e) = span.set_parent(parent_cx) {
        tracing::debug!(error = %e, "otel: could not attach remote parent context to request span");
    }

    let resp = next.run(req).instrument(span.clone()).await;
    span.record("http.status_code", resp.status().as_u16());
    resp
}

// ── unit tests for ORT log parsing ───────────────────────────────────────────

#[cfg(test)]
mod ort_log_tests {
    use super::parse_bfc_extend;

    #[test]
    fn parses_standard_bfc_extend_message() {
        let msg = "Extending BFCArena for Cpu. bin_num:20 num_bytes:1258291200";
        assert_eq!(parse_bfc_extend(msg), Some(20));
    }

    #[test]
    fn parses_bin_num_zero() {
        let msg = "Extending BFCArena for Cpu. bin_num:0 num_bytes:1048576";
        assert_eq!(parse_bfc_extend(msg), Some(0));
    }

    #[test]
    fn parses_large_bin_num() {
        let msg = "Extending BFCArena for Cpu. bin_num:255 num_bytes:9999";
        assert_eq!(parse_bfc_extend(msg), Some(255));
    }

    #[test]
    fn returns_none_for_unrelated_message() {
        let msg = "ONNX inference completed in 123ms";
        assert_eq!(parse_bfc_extend(msg), None);
    }

    #[test]
    fn returns_none_when_bin_num_missing() {
        let msg = "Extending BFCArena for Cpu. num_bytes:1048576";
        assert_eq!(parse_bfc_extend(msg), None);
    }

    #[test]
    fn returns_none_for_empty_string() {
        assert_eq!(parse_bfc_extend(""), None);
    }

    #[test]
    fn bin_num_not_followed_by_digits_is_none() {
        // "bin_num:" with no digits after it — parse should return None.
        let msg = "Extending BFCArena for Cpu. bin_num: num_bytes:1048576";
        assert_eq!(parse_bfc_extend(msg), None);
    }
}
