use std::env;

/// Definition of a single model to load.
pub struct ModelDef {
    pub name: String,
    pub dir: String,
    pub dim: usize,
    pub max_len: usize,
    pub pad_id: u32,
    pub has_token_type_ids: bool,
}

/// Server configuration parsed from environment variables.
pub struct Config {
    pub port: u16,
    pub models: Vec<ModelDef>,
    pub default_model: String,
    pub intra_threads: usize,
    pub batching_enabled: bool,
    pub batch_max: usize,
    pub batch_wait_ms: u64,
    pub max_queue_size: usize,
    /// Graceful drain timeout for future shutdown support.
    #[allow(dead_code)]
    pub drain_timeout_s: u64,
    /// When true (default, TEI-compat), tokenizer silently truncates
    /// overlong inputs to model `max_len`. Set `AUTO_TRUNCATE=false`
    /// to disable and keep the old strict behaviour.
    pub auto_truncate: bool,
}

impl Config {
    /// Parse configuration from environment variables.
    ///
    /// - `EMBED_PORT`: listen port (default 8082)
    /// - `EMBED_MODELS`: comma-separated model specs
    ///   Format: `name:dir:dim:max_len:pad_id:has_tti`
    /// - `EMBED_DEFAULT_MODEL`: default model name (default: first)
    pub fn from_env() -> Result<Self, String> {
        let port = env::var("EMBED_PORT")
            .unwrap_or_else(|_| "8082".into())
            .parse::<u16>()
            .map_err(|e| format!("invalid EMBED_PORT: {e}"))?;

        let models_str =
            env::var("EMBED_MODELS").map_err(|_| "EMBED_MODELS env var is required")?;

        let models = parse_models(&models_str)?;
        if models.is_empty() {
            return Err("EMBED_MODELS must define at least one model".into());
        }

        let default_model =
            env::var("EMBED_DEFAULT_MODEL").unwrap_or_else(|_| models[0].name.clone());

        if !models.iter().any(|m| m.name == default_model) {
            return Err(format!(
                "EMBED_DEFAULT_MODEL '{default_model}' not found in models"
            ));
        }

        let intra_threads = env::var("EMBED_INTRA_THREADS")
            .unwrap_or_else(|_| "4".into())
            .parse::<usize>()
            .map_err(|e| format!("invalid EMBED_INTRA_THREADS: {e}"))?;

        let batching_enabled = env::var("BATCHING_ENABLED")
            .ok()
            .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
            .unwrap_or(false);

        let batch_max = env::var("BATCH_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32usize);

        let batch_wait_ms = env::var("BATCH_WAIT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10u64);

        let max_queue_size = env::var("MAX_QUEUE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256usize);

        let drain_timeout_s = env::var("DRAIN_TIMEOUT_S")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10u64);

        // AUTO_TRUNCATE defaults to true (TEI-compat). Only the literal
        // string "false" (case-insensitive) disables it; anything else
        // keeps the safe default.
        let auto_truncate = env::var("AUTO_TRUNCATE")
            .ok()
            .map(|s| !s.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        Ok(Config {
            port,
            models,
            default_model,
            intra_threads,
            batching_enabled,
            batch_max,
            batch_wait_ms,
            max_queue_size,
            drain_timeout_s,
            auto_truncate,
        })
    }
}

/// Parse comma-separated model definitions.
/// Each entry: `name:dir:dim:max_len:pad_id:has_tti`
fn parse_models(s: &str) -> Result<Vec<ModelDef>, String> {
    s.split(',')
        .filter(|e| !e.trim().is_empty())
        .map(parse_one_model)
        .collect()
}

fn parse_one_model(entry: &str) -> Result<ModelDef, String> {
    let parts: Vec<&str> = entry.trim().split(':').collect();
    if parts.len() != 6 {
        return Err(format!(
            "model entry must have 6 colon-separated fields, got {}: '{entry}'",
            parts.len()
        ));
    }

    let dim = parts[2]
        .parse::<usize>()
        .map_err(|e| format!("invalid dim '{}': {e}", parts[2]))?;
    let max_len = parts[3]
        .parse::<usize>()
        .map_err(|e| format!("invalid max_len '{}': {e}", parts[3]))?;
    let pad_id = parts[4]
        .parse::<u32>()
        .map_err(|e| format!("invalid pad_id '{}': {e}", parts[4]))?;
    let has_tti = match parts[5] {
        "true" | "1" => true,
        "false" | "0" => false,
        v => return Err(format!("invalid has_token_type_ids '{v}'")),
    };

    Ok(ModelDef {
        name: parts[0].to_string(),
        dir: parts[1].to_string(),
        dim,
        max_len,
        pad_id,
        has_token_type_ids: has_tti,
    })
}
