mod api;
mod batcher;
mod config;
mod metrics;
mod model;
mod pool;
mod types;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::model::EmbedModel;
use crate::types::{AppState, ModelEntry};

/// Waits for SIGTERM or SIGINT, then cancels the token and sleeps for drain_timeout
/// to allow in-flight HTTP requests to complete before axum closes the listener.
async fn shutdown_signal(token: CancellationToken, drain_timeout: Duration) {
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = term.recv() => tracing::info!("SIGTERM received, starting graceful shutdown"),
        _ = int.recv()  => tracing::info!("SIGINT received, starting graceful shutdown"),
    }
    token.cancel();
    // Give in-flight HTTP requests drain_timeout to complete naturally.
    // After this future returns, axum stops accepting new connections; when
    // the last handler finishes, Arc<AppState> drops → batcher workers exit.
    tracing::info!(
        secs = drain_timeout.as_secs(),
        "draining in-flight requests"
    );
    tokio::time::sleep(drain_timeout).await;
    tracing::info!("drain complete");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort::logging=warn".parse().unwrap()),
        )
        .init();

    // Initialize ort runtime explicitly (required for load-dynamic).
    if !ort::init().commit() {
        eprintln!("ort init failed (environment already configured?)");
    }
    tracing::info!("ort runtime initialized");

    let cfg = Config::from_env().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });

    let version = std::env::var("EMBED_VERSION").unwrap_or_else(|_| "dev".into());
    let prom_handle = std::sync::Arc::new(metrics::init(&version));

    let mut raw_models: HashMap<String, Arc<EmbedModel>> = HashMap::new();
    for def in &cfg.models {
        tracing::info!(model = %def.name, dir = %def.dir, "loading model");
        let m = EmbedModel::load(def, cfg.intra_threads, cfg.auto_truncate).unwrap_or_else(|e| {
            eprintln!("failed to load model '{}': {e}", def.name);
            std::process::exit(1);
        });
        raw_models.insert(def.name.clone(), Arc::new(m));
    }

    tracing::info!(
        models = raw_models.len(),
        default = %cfg.default_model,
        "all models loaded"
    );

    let mut model_entries: HashMap<String, ModelEntry> = HashMap::new();
    for (name, model_arc) in raw_models {
        let batcher = if cfg.batching_enabled {
            let m = model_arc.clone();
            let b = batcher::DynamicBatcher::with_name(
                &name,
                move |texts| m.embed(&texts),
                cfg.batch_max,
                cfg.batch_wait_ms,
                cfg.max_queue_size,
            );
            Some(Arc::new(b))
        } else {
            None
        };
        model_entries.insert(
            name,
            ModelEntry {
                model: model_arc,
                batcher,
            },
        );
    }

    tracing::info!(
        batching_enabled = cfg.batching_enabled,
        models = model_entries.len(),
        "app state ready"
    );

    let drain_timeout = Duration::from_secs(cfg.drain_timeout_s);
    let shutdown_token = CancellationToken::new();

    let state = Arc::new(AppState {
        models: model_entries,
        default_model: cfg.default_model,
        shutdown: shutdown_token.clone(),
        drain_timeout,
    });

    let metrics_handle = prom_handle.clone();
    let app = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/metrics",
            axum::routing::get(move || {
                let h = metrics_handle.clone();
                async move {
                    (
                        [(
                            axum::http::header::CONTENT_TYPE,
                            "text/plain; version=0.0.4",
                        )],
                        h.render(),
                    )
                }
            }),
        )
        .route("/v1/embeddings", axum::routing::post(api::embeddings))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!(addr = %addr, "embed-server listening");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_token, drain_timeout))
        .await
        .unwrap();
}
