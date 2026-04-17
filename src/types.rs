//! HTTP request/response types and shared application state.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::batcher::DynamicBatcher;
use crate::model::EmbedModel;

// --- State ---

/// Entry for a single model: its inference handle and optional batcher.
pub struct ModelEntry {
    pub model: Arc<EmbedModel>,
    pub batcher: Option<Arc<DynamicBatcher>>,
}

/// Shared application state.
pub struct AppState {
    pub models: HashMap<String, ModelEntry>,
    pub default_model: String,
    /// Cancelled on SIGTERM/SIGINT; handlers check this to reject new requests.
    pub shutdown: CancellationToken,
    /// How long to wait for in-flight requests before axum stops the listener.
    #[allow(dead_code)]
    pub drain_timeout: Duration,
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

pub fn error_json(msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
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
