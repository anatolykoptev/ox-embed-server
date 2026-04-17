use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::model::EmbedModel;

/// Shared application state.
pub struct AppState {
    pub models: HashMap<String, Arc<EmbedModel>>,
    pub default_model: String,
}

// --- Request types ---

#[derive(Deserialize)]
pub struct EmbedRequest {
    pub input: InputField,
    pub model: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum InputField {
    Single(String),
    Batch(Vec<String>),
}

impl InputField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            InputField::Single(s) => vec![s],
            InputField::Batch(v) => v,
        }
    }
}

// --- Response types ---

#[derive(Serialize)]
pub struct EmbedResponse {
    pub object: &'static str,
    pub data: Vec<EmbedData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct EmbedData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
}

fn error_json(msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.into(),
                error_type: "invalid_request_error",
            },
        }),
    )
}

/// POST /v1/embeddings — OpenAI-compatible embedding endpoint.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> impl IntoResponse {
    let t0 = std::time::Instant::now();
    let mut status = "error";
    let mut texts_count: usize = 0;

    let model_name = req
        .model
        .unwrap_or_else(|| state.default_model.clone());

    let model = match state.models.get(&model_name) {
        Some(m) => Arc::clone(m),
        None => {
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return error_json(format!("model '{model_name}' not found")).into_response();
        }
    };

    let texts = req.input.into_vec();
    if texts.is_empty() {
        crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
        return error_json("input must not be empty").into_response();
    }
    texts_count = texts.len();

    // Run inference in blocking task to avoid starving tokio.
    let infer_start = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || model.embed(&texts))
        .await
        .map_err(|e| format!("spawn: {e}"));
    let infer_elapsed = infer_start.elapsed();

    let vectors = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "embed failed");
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e,
                        error_type: "invalid_request_error",
                    },
                }),
            )
                .into_response();
        }
        Err(e) => {
            crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e,
                        error_type: "invalid_request_error",
                    },
                }),
            )
                .into_response();
        }
    };

    crate::metrics::record_inference(&model_name, infer_elapsed, vectors.len());
    status = "ok";

    let data: Vec<EmbedData> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, emb)| EmbedData {
            object: "embedding",
            embedding: emb,
            index: i,
        })
        .collect();

    crate::metrics::record_request(&model_name, status, t0.elapsed(), texts_count);
    Json(EmbedResponse {
        object: "list",
        data,
        model: model_name,
        usage: Usage {
            prompt_tokens: 0,
            total_tokens: 0,
        },
    })
    .into_response()
}
