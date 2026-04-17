mod api;
mod config;
mod model;
mod pool;

use std::collections::HashMap;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::api::AppState;
use crate::config::Config;
use crate::model::EmbedModel;

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

    let mut models = HashMap::new();
    for def in &cfg.models {
        tracing::info!(model = %def.name, dir = %def.dir, "loading model");
        let m = EmbedModel::load(def, cfg.intra_threads).unwrap_or_else(|e| {
            eprintln!("failed to load model '{}': {e}", def.name);
            std::process::exit(1);
        });
        models.insert(def.name.clone(), Arc::new(m));
    }

    tracing::info!(
        models = models.len(),
        default = %cfg.default_model,
        "all models loaded"
    );

    let state = Arc::new(AppState {
        models,
        default_model: cfg.default_model,
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/embeddings", post(api::embeddings))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!(addr = %addr, "embed-server listening");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
