//! Health and readiness endpoints.
//!
//! `/health` is a constant liveness probe — it answers `ok` as long as the
//! process is up and the listener is bound. It touches no model, no worker,
//! no queue. Container orchestrators use it for restart-on-crash.
//!
//! `/ready` is a readiness probe — it dispatches a single 1-word embed to
//! the default model and waits up to `EMBED_READY_PROBE_TIMEOUT_MS` (default
//! 2 s). If the worker is wedged (busy-spin with zero throughput), the
//! dispatch hangs, the probe times out, and `/ready` returns 503. This lets
//! downstream consumers and load balancers stop sending traffic before
//! users notice.
//!
//! Why real inference, not a lightweight ping: the worker runs ONNX in
//! `spawn_blocking`, so its async runtime stays responsive while ONNX
//! spins. A ping would answer `ok` while the worker is wedged. Only a real
//! inference that queues behind the stuck `spawn_blocking` call can detect
//! the wedge.

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Serialize;

use crate::types::AppState;

#[derive(Serialize)]
struct ReadyResponse {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Liveness probe — constant `ok`. Touches nothing.
pub async fn health() -> &'static str {
    "ok"
}

/// Readiness probe — dispatches a 1-word embed to the default model with
/// a tight timeout. Returns 200 on success, 503 on timeout/error/shutdown.
pub async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.shutdown.is_cancelled() {
        crate::metrics::record_ready_probe("shutdown");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyResponse {
                status: "not ready",
                model: None,
                error: Some("shutting down".into()),
            }),
        );
    }

    let model_name = state.default_model.clone();
    let timeout = Duration::from_millis(state.ready_probe_timeout_ms);

    // Multi-process mode: dispatch via worker pool.
    if let Some(pool) = state.worker_pool.as_ref() {
        let result = tokio::time::timeout(
            timeout,
            pool.dispatch_embed(&model_name, vec!["test".to_string()], 8),
        )
        .await;

        return match result {
            Ok(Ok(crate::ipc::protocol::WorkerResponse::Embed(_))) => {
                crate::metrics::record_ready_probe("ok");
                (
                    StatusCode::OK,
                    Json(ReadyResponse {
                        status: "ready",
                        model: Some(model_name),
                        error: None,
                    }),
                )
            }
            Ok(Ok(crate::ipc::protocol::WorkerResponse::Err { message, .. })) => {
                tracing::warn!(model = %model_name, error = %message, "ready probe: worker error");
                crate::metrics::record_ready_probe("error");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ReadyResponse {
                        status: "not ready",
                        model: Some(model_name),
                        error: Some(format!("worker error: {message}")),
                    }),
                )
            }
            Ok(Ok(unexpected)) => {
                tracing::warn!(
                    model = %model_name,
                    kind = %unexpected.kind(),
                    "ready probe: unexpected response kind"
                );
                crate::metrics::record_ready_probe("error");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ReadyResponse {
                        status: "not ready",
                        model: Some(model_name),
                        error: Some("unexpected response kind".into()),
                    }),
                )
            }
            Ok(Err(e)) => {
                tracing::warn!(model = %model_name, error = ?e, "ready probe: dispatch failed");
                crate::metrics::record_ready_probe("error");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ReadyResponse {
                        status: "not ready",
                        model: Some(model_name),
                        error: Some(format!("dispatch failed: {e}")),
                    }),
                )
            }
            Err(_) => {
                tracing::warn!(
                    model = %model_name,
                    timeout_ms = timeout.as_millis(),
                    "ready probe: timed out (worker may be wedged)"
                );
                crate::metrics::record_ready_probe("timeout");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ReadyResponse {
                        status: "not ready",
                        model: Some(model_name),
                        error: Some(format!(
                            "probe timed out after {}ms (worker may be wedged)",
                            timeout.as_millis()
                        )),
                    }),
                )
            }
        };
    }

    // Single-process mode: run inference directly via the in-process model.
    let entry = match state.models.get(&model_name) {
        Some(e) => e,
        None => {
            // No model registered — nothing to probe. Return ready=true
            // so a misconfigured-but-booted server doesn't fail health
            // forever; the /v1/embeddings handler will 400 on real requests.
            crate::metrics::record_ready_probe("ok");
            return (
                StatusCode::OK,
                Json(ReadyResponse {
                    status: "ready",
                    model: None,
                    error: Some("no models configured".into()),
                }),
            );
        }
    };

    let model = match entry.model.as_ref() {
        Some(m) => m.clone(),
        None => {
            // multi_process=true but worker_pool is None — shouldn't happen
            // (main.rs wires them together), but handle gracefully.
            crate::metrics::record_ready_probe("error");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not ready",
                    model: Some(model_name),
                    error: Some("no in-process model and no worker pool".into()),
                }),
            );
        }
    };

    let probe_result = tokio::time::timeout(
        timeout,
        tokio::task::spawn_blocking(move || {
            let token_ids = model.tokenize(&["test".to_string()])?;
            model.embed_tokens(&token_ids)
        }),
    )
    .await;

    // Three nested Results, not two. The spawn_blocking closure ends in
    // `model.embed_tokens(&token_ids)` -> Result<_, String>, so the awaited
    // expression is Result<Result<Result<_, String>, JoinError>, Elapsed>.
    //
    // The previous match had three arms over two levels, so `Ok(Ok(_))` bound
    // the INFERENCE Result — Err included — and answered 200 "ready" while
    // every inference was failing, incrementing embed_ready_probe_total{ok}.
    // That is the "wedged while /health stayed 200" incident this endpoint was
    // added to close, reproduced one branch over: the multi-process arm above
    // handles WorkerResponse::Err correctly, and only this single-process path
    // (EMBED_MULTI_PROCESS=0, the documented monolith rollback) was wrong.
    match probe_result {
        Ok(Ok(Ok(_))) => {
            crate::metrics::record_ready_probe("ok");
            (
                StatusCode::OK,
                Json(ReadyResponse {
                    status: "ready",
                    model: Some(model_name),
                    error: None,
                }),
            )
        }
        Ok(Ok(Err(e))) => {
            tracing::warn!(model = %model_name, error = %e, "ready probe: inference failed");
            crate::metrics::record_ready_probe("error");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not ready",
                    model: Some(model_name),
                    error: Some(format!("inference failed: {e}")),
                }),
            )
        }
        Ok(Err(e)) => {
            // JoinError — the blocking task panicked or was cancelled.
            tracing::warn!(model = %model_name, error = ?e, "ready probe: probe task panicked");
            crate::metrics::record_ready_probe("error");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not ready",
                    model: Some(model_name),
                    error: Some(format!("probe task panicked: {e}")),
                }),
            )
        }
        Err(_) => {
            tracing::warn!(
                model = %model_name,
                timeout_ms = timeout.as_millis(),
                "ready probe: timed out (model may be wedged)"
            );
            crate::metrics::record_ready_probe("timeout");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not ready",
                    model: Some(model_name),
                    error: Some(format!(
                        "probe timed out after {}ms (model may be wedged)",
                        timeout.as_millis()
                    )),
                }),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::EmbeddingCache;
    use crate::token_cache::TokenCache;
    use crate::types::AppState;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
    use std::collections::HashMap;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt as _;

    fn make_state(
        shutdown: bool,
        models: HashMap<String, crate::types::ModelEntry>,
    ) -> Arc<AppState> {
        let token = CancellationToken::new();
        if shutdown {
            token.cancel();
        }
        Arc::new(AppState {
            models,
            rerankers: HashMap::new(),
            splades: HashMap::new(),
            default_model: "test-model".to_string(),
            shutdown: token,
            drain_timeout: Duration::from_secs(5),
            cache: Arc::new(EmbeddingCache::new(0)),
            token_cache: Arc::new(TokenCache::new(0)),
            rerank_semaphore: None,
            embed_max_input_array: 32,
            rerank_max_input_docs: 32,
            worker_pool: None,
            ready_probe_timeout_ms: 2000,
        })
    }

    fn make_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .with_state(state)
    }

    async fn call_ready(state: Arc<AppState>) -> (StatusCode, String) {
        let app = make_app(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&body).to_string())
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = make_app(make_state(false, HashMap::new()));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn ready_returns_503_when_shutting_down() {
        let state = make_state(true, HashMap::new());
        let (status, body) = call_ready(state).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("shutting down"));
    }

    #[tokio::test]
    async fn ready_returns_200_when_no_models_and_no_worker_pool() {
        // No models configured, no worker pool — nothing to probe.
        // Returns 200 so a misconfigured-but-booted server doesn't fail
        // health forever; /v1/embeddings will 400 on real requests.
        let state = make_state(false, HashMap::new());
        let (status, body) = call_ready(state).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("ready"));
    }

    // ── hard test: wedged worker → /ready 503 ──────────────────────────────
    //
    // This is the test that proves the fix actually works. The core claim of
    // #89 is: a wedged worker (accepts UDS connections but never responds,
    // simulating 220% CPU / zero throughput) must cause /ready to return 503,
    // not 200. Without this test, the probe could be silently broken (e.g.
    // timeout not wired, wrong branch taken) and the happy-path tests above
    // would still pass.
    //
    // Fixture: mock UDS server that accepts connections but never writes a
    // response frame. WorkerClient::dispatch_embed connects, writes the
    // request, then blocks forever on read_frame. The tokio::time::timeout
    // in /ready fires → 503.

    #[tokio::test]
    async fn ready_returns_503_when_worker_is_wedged() {
        use crate::ipc::client::WorkerClient;
        use crate::supervisor::WorkerPool;
        use crate::supervisor::handle::{SpawnSpec, WorkerKind, WorkerSupervisor};
        use std::path::PathBuf;
        use tokio::net::UnixListener;

        // Unique socket path per test to avoid collisions with parallel tests.
        let socket_path: PathBuf = std::env::temp_dir().join(format!(
            "embed-ready-test-wedged-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&socket_path);

        // Mock server: accept connections but never respond.
        // This is exactly what a wedged worker looks like from the socket
        // layer — the OS listener is alive (accept succeeds), but the worker
        // process is spinning on ONNX and never reads/writes the socket.
        // Each accepted connection is held open forever (spawned task sleeps
        // indefinitely) so the client's read_frame blocks — matching the
        // real wedge signature.
        let listener = UnixListener::bind(&socket_path).expect("bind mock socket");
        let server_task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                // Hold the stream open forever — never read or write.
                // The client blocks on read_frame, simulating a wedged
                // worker that accepted the connection but never
                // processes the request.
                tokio::spawn(async move {
                    std::future::pending::<()>().await;
                    drop(stream);
                });
            }
        });

        // Connect a real WorkerClient to the mock server.
        let client = Arc::new(
            WorkerClient::connect(socket_path.clone(), 1)
                .await
                .expect("connect to mock server"),
        );

        // Build a WorkerSupervisor fixture with the pre-connected client.
        let spec = SpawnSpec {
            model: "test-model".to_string(),
            kind: WorkerKind::Embed,
            worker_bin: PathBuf::from("/bin/true"),
            socket_dir: socket_path.parent().unwrap().to_path_buf(),
            pool_size: 1,
            intra_threads: 1,
            env_extra: vec![],
        };
        let supervisor = WorkerSupervisor::for_test(spec, client);

        // Build a real WorkerPool and add the supervisor.
        let pool = Arc::new(WorkerPool::new());
        pool.add(supervisor).await;

        // Build AppState with the pool and a tight probe timeout (100ms
        // — short enough for a fast test, long enough to not flake on a
        // slow CI runner).
        let token = CancellationToken::new();
        let state = Arc::new(AppState {
            models: HashMap::new(),
            rerankers: HashMap::new(),
            splades: HashMap::new(),
            default_model: "test-model".to_string(),
            shutdown: token,
            drain_timeout: Duration::from_secs(5),
            cache: Arc::new(EmbeddingCache::new(0)),
            token_cache: Arc::new(TokenCache::new(0)),
            rerank_semaphore: None,
            embed_max_input_array: 32,
            rerank_max_input_docs: 32,
            worker_pool: Some(pool),
            ready_probe_timeout_ms: 100,
        });

        let (status, body) = call_ready(state).await;

        // Cleanup: stop the mock server.
        server_task.abort();
        let _ = std::fs::remove_file(&socket_path);

        // The assertion that matters: wedged worker → 503, not 200.
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "wedged worker must produce 503, got {status}: {body}"
        );
        assert!(
            body.contains("timed out"),
            "response must mention timeout, got: {body}"
        );
        assert!(
            body.contains("wedged"),
            "response must mention 'wedged', got: {body}"
        );
    }
}
