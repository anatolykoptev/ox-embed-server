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

/// Definition of a single cross-encoder reranker to load.
///
/// Far fewer fields than `ModelDef` because:
///   - no `dim` — rerankers emit a scalar, not a vector;
///   - no `pad_id` — discovered from the tokenizer at load time
///     (see `RerankerModel::load`); different reranker families use
///     different pad ids and we don't want to hand-maintain a table;
///   - no `has_token_type_ids` — the production target (BGE, Jina,
///     mxbai rerankers) all use XLM-RoBERTa which has none, and
///     BERT-based rerankers mask `token_type_ids=0` anyway; ONNX graph
///     introspection at session time would be cleaner still, but the
///     ort 2.0-rc API doesn't expose it ergonomically.
#[derive(Debug, PartialEq, Eq)]
pub struct RerankerModelDef {
    pub name: String,
    pub dir: String,
    pub max_len: usize,
    /// True for BERT-style padded models (which is every reranker we
    /// ship). Kept as a config knob rather than hard-coded so tests and
    /// future model families can flip it without a code change.
    pub padded_model: bool,
}

/// Definition of a single SPLADE sparse encoder to load.
///
/// Even leaner than `RerankerModelDef`:
///   - no `padded_model` — v1 doesn't route SPLADE through the dynamic
///     batcher (one `spawn_blocking` per text), so the padding flag has
///     no consumer yet. Add when batcher integration lands.
///   - no `dim` — SPLADE's output dim is the BERT vocab size, discovered
///     by `SpladeModel::load` from the ONNX graph (no hard-coded 30522).
///   - no `pad_id` — single-text inputs need no padding inside the
///     sequence, and the tokenizer-driven truncation runs at load time.
#[derive(Debug, PartialEq, Eq)]
pub struct SpladeModelDef {
    pub name: String,
    pub dir: String,
    pub max_len: usize,
}

/// Server configuration parsed from environment variables.
pub struct Config {
    pub port: u16,
    pub models: Vec<ModelDef>,
    /// Zero-or-more cross-encoder rerankers. Unlike `models`, an empty
    /// list is valid (and the default when `RERANKER_MODELS` is unset):
    /// the server still boots serving `/v1/embeddings` alone.
    pub rerankers: Vec<RerankerModelDef>,
    pub default_model: String,
    pub intra_threads: usize,
    /// Number of ONNX `Session` instances loaded per reranker model.
    /// Each session can run inference independently, so requests scoring
    /// pairs against the same reranker can run in parallel up to
    /// `reranker_pool_size` at a time. `1` (the default) preserves the
    /// pre-pool behaviour exactly: a single Mutex-guarded session.
    ///
    /// IMPORTANT: when raising this above 1, the operator should also
    /// lower `EMBED_INTRA_THREADS` so `pool_size * intra_threads` stays
    /// at or below the available CPU cores. The model side does NOT
    /// auto-divide the per-session intra threads — caller controls the
    /// math so the config is honest about what's being requested.
    pub reranker_pool_size: usize,
    /// Per-session intra-op threads for reranker ONNX sessions. Defaults
    /// to `intra_threads` (so unset = same as today's shared budget). Set
    /// independently from `EMBED_INTRA_THREADS` so the embedder is not
    /// affected when raising `reranker_pool_size`. Recommended: keep
    /// `pool_size * reranker_intra_threads ≤ EMBED_INTRA_THREADS` so the
    /// reranker doesn't steal threads from the embedder when both run
    /// concurrently.
    pub reranker_intra_threads: usize,
    /// Zero-or-more SPLADE sparse encoders. Empty when `SPLADE_MODELS`
    /// is unset (the default) — server boots without `/v1/sparse_embeddings`
    /// active. Same fail-loud parse contract as `RERANKER_MODELS`.
    pub splades: Vec<SpladeModelDef>,
    /// Number of ONNX `Session` instances loaded per SPLADE model.
    /// Same semantics as `reranker_pool_size`: `1` (default) preserves
    /// single-session behaviour; values >1 enable concurrent inference
    /// at N× per-session memory. SPLADE-v3-distilbert sessions are
    /// ~360 MB fp32 each, so pool sizes >2 are usually overkill on the
    /// current production box.
    pub splade_pool_size: usize,
    /// Per-session intra-op threads for SPLADE ONNX sessions. Defaults
    /// to `intra_threads` when unset, mirroring `reranker_intra_threads`.
    /// Caller should keep `splade_pool_size * splade_intra_threads`
    /// under the cores reserved for SPLADE.
    pub splade_intra_threads: usize,
    pub batching_enabled: bool,
    /// Soft cap on items (texts) per batch — retained for fairness, so
    /// one giant multi-text request can't monopolise a single dispatch.
    /// The primary budget in Phase B is `batch_max_tokens`.
    pub batch_max: usize,
    /// Reranker-specific item cap. Overrides `batch_max` for reranker
    /// dispatch only — enables larger coalescing for `/v1/rerank` (each
    /// call is multi-doc → quickly hits global `batch_max=8` and falls
    /// back to single-call batches, killing concurrent throughput) while
    /// keeping the conservative embed-side cap for ARM cache-thrash
    /// avoidance. Default = 4× `batch_max` to give 4 concurrent rerank
    /// calls of 5-doc room without changing the embed knob.
    pub reranker_batch_max: usize,
    /// Primary batch budget: maximum total tokens per dispatched batch.
    /// Counted with padded-model accounting — see `DynamicBatcher::with_tokens`.
    /// Default 16384 (TEI).
    pub batch_max_tokens: usize,
    pub batch_wait_ms: u64,
    pub max_queue_size: usize,
    /// Graceful drain timeout for future shutdown support.
    #[allow(dead_code)]
    pub drain_timeout_s: u64,
    /// When true (default, TEI-compat), tokenizer silently truncates
    /// overlong inputs to model `max_len`.
    ///
    /// Only the literal string `"false"` (case-insensitive) disables
    /// this; values like `"0"`, `"no"`, `"off"`, or `""` LEAVE truncation
    /// enabled. This matches Hugging Face `text-embeddings-inference`
    /// convention — `AUTO_TRUNCATE=false` is the one documented escape
    /// hatch, and we refuse to silently interpret other "falsy"
    /// strings the same way to avoid surprise disables.
    pub auto_truncate: bool,
    /// Maximum entries in the process-local response cache.
    ///
    /// `0` disables caching (EmbeddingCache::new(0) returns a no-op
    /// shell); use this as the runtime kill-switch without needing a
    /// separate boolean flag. Default `10_000` — a modest memory
    /// footprint (~40 MB for 1024-dim f32 vectors) that comfortably
    /// covers MemDB's recurring search strings.
    pub cache_max_entries: usize,
    /// Maximum entries in the per-pair tokenizer cache (H.7).
    ///
    /// `0` disables the token cache (TokenCache::new(0) returns a no-op
    /// shell that always misses — identical behaviour to the pre-H.7 code
    /// path). Default `0` (disabled) when `TOKEN_CACHE_MAX_ENTRIES` is
    /// unset, so existing deployments are not affected. An explicit
    /// positive value enables caching.
    ///
    /// Memory estimate: each entry holds `Arc<Vec<u32>>` (~2 KB per 512-
    /// token pair) + 40 B key ≈ ~2 KB. At 20 000 entries that's ~40 MB
    /// — negligible next to the ONNX session memory (~300–550 MB each).
    pub token_cache_max_entries: usize,
    /// Per-shape warmup batch sizes for cross-encoder rerankers.
    ///
    /// Each entry is a batch size at which every reranker session runs
    /// one dummy inference at boot — pre-paying the ORT kernel-binding /
    /// memory-pattern / arena-allocation cost so the FIRST production
    /// request at that shape doesn't see the cold-path spike. Default
    /// `[1, 5]` covers the two prod-traffic shapes: batch=1 (the static
    /// fast-path single-pair calls) and batch=5 (memdb-go's D7 sub-query
    /// fanout default). Operators set `RERANK_WARMUP_BATCH_SIZES` to
    /// override (e.g. `1,2,5,10` for boxes serving wider batch ranges).
    pub rerank_warmup_batch_sizes: Vec<usize>,
    /// Per-shape warmup batch sizes for dense (bi-encoder) embedders.
    ///
    /// Default `[1, 8]`: batch=1 covers the trivial `/v1/embeddings`
    /// caller, batch=8 matches the `texts_per_req=8` default the typical
    /// memdb-go embedder client uses. Override via
    /// `EMBED_WARMUP_BATCH_SIZES`.
    pub embed_warmup_batch_sizes: Vec<usize>,
    /// Per-shape warmup batch sizes for SPLADE sparse encoders.
    ///
    /// SPLADE's `encode_sparse` is intrinsically single-text (batch=1
    /// hard-coded inside the model — see `model_splade.rs`), so this
    /// list typically has one entry. Defaults to `[1]`. Operators
    /// who only ever call SPLADE with batch=1 traffic (the v1 norm)
    /// should leave this unset. Override via
    /// `SPLADE_WARMUP_BATCH_SIZES` if a future SPLADE batched API
    /// lands and shape pre-warming becomes useful.
    pub splade_warmup_batch_sizes: Vec<usize>,
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

        let reranker_pool_size =
            parse_reranker_pool_size(env::var("RERANKER_SESSION_POOL_SIZE").ok().as_deref());

        // `RERANKER_INTRA_THREADS` defaults to `intra_threads` so unset
        // means "share the embedder budget" (today's behaviour). Set
        // explicitly when raising pool_size to keep total reranker
        // threads under control without changing embedder threads.
        let reranker_intra_threads = env::var("RERANKER_INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(intra_threads);

        let batching_enabled = env::var("BATCHING_ENABLED")
            .ok()
            .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
            .unwrap_or(false);

        let batch_max = env::var("BATCH_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32usize);

        // Reranker-specific item cap. Defaults to 4× embed batch_max so a
        // single env (`BATCH_MAX=8`) can keep embed-side cache-thrash
        // mitigation while the reranker still coalesces 4 concurrent
        // multi-doc calls. Operators can override with RERANKER_BATCH_MAX
        // when they want a different ratio.
        let reranker_batch_max = env::var("RERANKER_BATCH_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(batch_max.saturating_mul(4));

        let batch_max_tokens = parse_batch_max_tokens(env::var("BATCH_MAX_TOKENS").ok().as_deref());

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

        let cache_max_entries =
            parse_cache_max_entries(env::var("CACHE_MAX_ENTRIES").ok().as_deref());

        let token_cache_max_entries =
            parse_token_cache_max_entries(env::var("TOKEN_CACHE_MAX_ENTRIES").ok().as_deref());

        // `RERANKER_MODELS` is optional: unset or empty → no rerankers,
        // server boots serving only `/v1/embeddings`. `/v1/rerank` with
        // any model name will 400. Errors here only on malformed entries
        // (bad integer fields, wrong colon count) — a strict
        // fail-at-boot contract matching `EMBED_MODELS`.
        let rerankers = env::var("RERANKER_MODELS")
            .ok()
            .map(|s| parse_rerankers(&s))
            .transpose()?
            .unwrap_or_default();

        // `SPLADE_MODELS` follows the same contract as `RERANKER_MODELS`:
        // unset/empty → no SPLADE endpoints; malformed → fail boot.
        let splades = env::var("SPLADE_MODELS")
            .ok()
            .map(|s| parse_splades(&s))
            .transpose()?
            .unwrap_or_default();

        let splade_pool_size =
            parse_splade_pool_size(env::var("SPLADE_SESSION_POOL_SIZE").ok().as_deref());

        // SPLADE_INTRA_THREADS defaults to `intra_threads` (share embedder
        // budget when unset), same fallback as RERANKER_INTRA_THREADS.
        let splade_intra_threads = env::var("SPLADE_INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(intra_threads);

        // Per-kind warmup shape lists. Defaults are tuned to the prod
        // traffic shape we already know about (memdb's batch=5 fanout,
        // texts_per_req=8 embed default, single-text splade) — the
        // unset path MUST get the cold-start fix, hence non-empty
        // defaults rather than `[]` when the env var is absent.
        let rerank_warmup_batch_sizes =
            parse_warmup_batch_sizes(env::var("RERANK_WARMUP_BATCH_SIZES").ok().as_deref(), &[1, 5]);
        let embed_warmup_batch_sizes =
            parse_warmup_batch_sizes(env::var("EMBED_WARMUP_BATCH_SIZES").ok().as_deref(), &[1, 8]);
        let splade_warmup_batch_sizes =
            parse_warmup_batch_sizes(env::var("SPLADE_WARMUP_BATCH_SIZES").ok().as_deref(), &[1]);

        Ok(Config {
            port,
            models,
            rerankers,
            default_model,
            intra_threads,
            reranker_pool_size,
            reranker_intra_threads,
            splades,
            splade_pool_size,
            splade_intra_threads,
            batching_enabled,
            batch_max,
            reranker_batch_max,
            batch_max_tokens,
            batch_wait_ms,
            max_queue_size,
            drain_timeout_s,
            auto_truncate,
            cache_max_entries,
            token_cache_max_entries,
            rerank_warmup_batch_sizes,
            embed_warmup_batch_sizes,
            splade_warmup_batch_sizes,
        })
    }
}

/// Parse `CACHE_MAX_ENTRIES` env value. Unset, empty, or unparseable →
/// 10_000 (sensible default). An explicit `0` is honoured as the
/// documented disable signal (EmbeddingCache becomes a no-op shell).
/// Exposed for testing; env lookup stays in `from_env`.
fn parse_cache_max_entries(raw: Option<&str>) -> usize {
    const DEFAULT: usize = 10_000;
    match raw {
        None => DEFAULT,
        Some(s) => s.trim().parse::<usize>().unwrap_or(DEFAULT),
    }
}

/// Parse `TOKEN_CACHE_MAX_ENTRIES` env value.
///
/// Unset or empty → `0` (disabled). This differs from `CACHE_MAX_ENTRIES`
/// which defaults to 10_000 — the token cache is a new opt-in feature and
/// we want zero surprise behaviour for existing deployments. Operators
/// who want the speedup set an explicit positive value.
///
/// An explicit `0` is honoured as the documented disable signal (same as
/// `CACHE_MAX_ENTRIES=0` for the embedding cache). Garbage → `0` (disabled,
/// not a hard error — production operators who don't set this var should
/// never see a boot failure from an env quoting mistake in an adjacent line).
///
/// Exposed for testing; env lookup stays in `from_env`.
fn parse_token_cache_max_entries(raw: Option<&str>) -> usize {
    const DEFAULT: usize = 0; // disabled by default
    match raw {
        None => DEFAULT,
        Some(s) => s.trim().parse::<usize>().unwrap_or(DEFAULT),
    }
}

/// Parse `BATCH_MAX_TOKENS` env value. Unset, empty, unparseable, or `0` →
/// 16384 (TEI default). `0` would degenerate the batcher to one item per
/// dispatch (strict `<` gate never admits a 2nd item), so it's rejected
/// with a warn rather than silently accepted. Exposed for testing; env
/// lookup stays in `from_env`.
fn parse_batch_max_tokens(raw: Option<&str>) -> usize {
    const DEFAULT: usize = 16384;
    match raw {
        None => DEFAULT,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(0) => {
                tracing::warn!("BATCH_MAX_TOKENS=0 is invalid; falling back to default {DEFAULT}");
                DEFAULT
            }
            Ok(n) => n,
            Err(_) => DEFAULT,
        },
    }
}

/// Parse `RERANKER_SESSION_POOL_SIZE` env value. Unset, empty, or
/// unparseable → 1 (single-session, mirrors the pre-pool behaviour
/// exactly). `0` is rejected with a warn rather than silently accepted —
/// it would `% 0` panic at request time, and follows the same
/// "0 is invalid, fall back" stance as `BATCH_MAX_TOKENS=0`. Exposed for
/// testing; env lookup stays in `from_env`.
fn parse_reranker_pool_size(raw: Option<&str>) -> usize {
    const DEFAULT: usize = 1;
    match raw {
        None => DEFAULT,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(0) => {
                tracing::warn!(
                    "RERANKER_SESSION_POOL_SIZE=0 is invalid; falling back to default {DEFAULT}"
                );
                DEFAULT
            }
            Ok(n) => n,
            Err(_) => DEFAULT,
        },
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

/// Parse `RERANKER_MODELS` into zero-or-more `RerankerModelDef`.
///
/// Format: `name:dir:max_len:padded`, comma-separated. Empty or
/// whitespace-only input returns `Ok(vec![])` — the "unset → no
/// rerankers" contract. Malformed entries return `Err`, aborting boot
/// (same fail-loud stance as `EMBED_MODELS`).
pub fn parse_rerankers(s: &str) -> Result<Vec<RerankerModelDef>, String> {
    s.split(',')
        .filter(|e| !e.trim().is_empty())
        .map(parse_one_reranker)
        .collect()
}

fn parse_one_reranker(entry: &str) -> Result<RerankerModelDef, String> {
    let parts: Vec<&str> = entry.trim().split(':').collect();
    if parts.len() != 4 {
        return Err(format!(
            "reranker entry must have 4 colon-separated fields (name:dir:max_len:padded), got {}: '{entry}'",
            parts.len()
        ));
    }
    let max_len = parts[2]
        .parse::<usize>()
        .map_err(|e| format!("invalid reranker max_len '{}': {e}", parts[2]))?;
    let padded_model = match parts[3] {
        "true" | "1" => true,
        "false" | "0" => false,
        v => {
            return Err(format!(
                "invalid reranker padded '{v}' (expected true|false|1|0)"
            ));
        }
    };
    Ok(RerankerModelDef {
        name: parts[0].to_string(),
        dir: parts[1].to_string(),
        max_len,
        padded_model,
    })
}

/// Parse `SPLADE_MODELS` into zero-or-more `SpladeModelDef`.
///
/// Format: `name:dir:max_len`, comma-separated. 3 fields (no `padded`
/// switch — v1 SPLADE bypasses the dynamic batcher entirely). Empty
/// string returns `Ok(vec![])` — the unset path. Malformed entries
/// fail boot, same as `parse_rerankers` / `parse_models`.
pub fn parse_splades(s: &str) -> Result<Vec<SpladeModelDef>, String> {
    s.split(',')
        .filter(|e| !e.trim().is_empty())
        .map(parse_one_splade)
        .collect()
}

fn parse_one_splade(entry: &str) -> Result<SpladeModelDef, String> {
    let parts: Vec<&str> = entry.trim().split(':').collect();
    if parts.len() != 3 {
        return Err(format!(
            "splade entry must have 3 colon-separated fields (name:dir:max_len), got {}: '{entry}'",
            parts.len()
        ));
    }
    let max_len = parts[2]
        .parse::<usize>()
        .map_err(|e| format!("invalid splade max_len '{}': {e}", parts[2]))?;
    Ok(SpladeModelDef {
        name: parts[0].to_string(),
        dir: parts[1].to_string(),
        max_len,
    })
}

/// Parse a `*_WARMUP_BATCH_SIZES` env value into a deduped, order-
/// preserving list of positive batch sizes.
///
/// Shared by `RERANK_WARMUP_BATCH_SIZES`, `EMBED_WARMUP_BATCH_SIZES`,
/// and `SPLADE_WARMUP_BATCH_SIZES` — each has different defaults but the
/// parsing rules are identical, so the per-env-var helpers are thin
/// wrappers in `from_env` that pass their own defaults in.
///
/// Behaviour:
///   - `None` → return `defaults` verbatim. The unset path MUST get the
///     pre-warm benefit; otherwise the feature is off-by-default and
///     prod still hits the cold-path latency spike on first batch=N.
///   - empty / whitespace-only / all-comma → defaults (env quoting
///     mishaps shouldn't accidentally disable warmup).
///   - "1,5" → `[1, 5]`. Order is preserved — operator decides which
///     shape to compile first; we don't sort.
///   - "3,5,3" → `[3, 5]`. Duplicates collapse to first-occurrence so
///     total warmup time stays bounded under operator paste mistakes.
///   - "0,1,5" → `[1, 5]`. `0` would `% 0` panic in any per-shape
///     dispatch loop and is meaningless as a batch size; we drop it
///     silently rather than fail boot.
///   - "1,nope,5" → `[1, 5]`. Single-token typo doesn't kill the warmup
///     for valid neighbours; we log a warn but keep going.
///   - "garbage" or "nope,oops" (no valid entries at all) → defaults,
///     with a warn — same fall-back stance as `parse_batch_max_tokens`
///     when zero or unparseable. Better to over-warm than under-warm
///     production.
///
/// Exposed for testing; env lookup stays in `from_env`.
fn parse_warmup_batch_sizes(raw: Option<&str>, defaults: &[usize]) -> Vec<usize> {
    let Some(s) = raw else {
        return defaults.to_vec();
    };

    // Track which valid sizes we've already emitted so dupes collapse to
    // the first occurrence (preserves order — sorting would change the
    // operator's intended compile-first shape).
    let mut seen = std::collections::HashSet::<usize>::new();
    // First-occurrence-wins, order-preserving. Capacity hint = comma
    // count + 1, which over-allocates by at most a few slots — cheap.
    let mut out: Vec<usize> = Vec::with_capacity(s.matches(',').count() + 1);
    let mut bad_tokens: Vec<&str> = Vec::new();

    for piece in s.split(',') {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            // Skip empty splits (handles trailing comma / `,,` / pure
            // whitespace input). Not "bad" — operators legitimately
            // produce these via env-quoting.
            continue;
        }
        match trimmed.parse::<usize>() {
            Ok(0) => {
                // 0 is invalid as a batch size (would `% 0` panic
                // downstream and means nothing semantically). Treat as
                // a bad token so the warn fires if it's the only entry.
                bad_tokens.push(trimmed);
            }
            Ok(n) => {
                if seen.insert(n) {
                    out.push(n);
                }
            }
            Err(_) => bad_tokens.push(trimmed),
        }
    }

    if out.is_empty() {
        // Nothing valid — fall back to defaults so prod still warms.
        // Only emit a warn when the input WAS something (vs the empty/
        // whitespace path, which is silent because it's a common
        // env-quoting artefact, not a configuration mistake).
        let trimmed_full = s.trim();
        if !trimmed_full.is_empty() && trimmed_full.chars().any(|c| c != ',') {
            tracing::warn!(
                raw = %s,
                bad_tokens = ?bad_tokens,
                defaults = ?defaults,
                "*_WARMUP_BATCH_SIZES had no valid positive integers; falling back to defaults"
            );
        }
        return defaults.to_vec();
    }

    if !bad_tokens.is_empty() {
        tracing::warn!(
            raw = %s,
            bad_tokens = ?bad_tokens,
            kept = ?out,
            "*_WARMUP_BATCH_SIZES had unparseable or zero entries; warming with the rest"
        );
    }

    out
}

/// Parse `SPLADE_SESSION_POOL_SIZE`. Same shape as
/// `parse_reranker_pool_size`: default 1, `0` → fall back with a warn,
/// garbage → fall back. Kept as a separate function so the warn
/// message names the right env var (operators grep for the literal).
fn parse_splade_pool_size(raw: Option<&str>) -> usize {
    const DEFAULT: usize = 1;
    match raw {
        None => DEFAULT,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(0) => {
                tracing::warn!(
                    "SPLADE_SESSION_POOL_SIZE=0 is invalid; falling back to default {DEFAULT}"
                );
                DEFAULT
            }
            Ok(n) => n,
            Err(_) => DEFAULT,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_max_tokens_default_is_16384_when_unset() {
        assert_eq!(parse_batch_max_tokens(None), 16384);
    }

    #[test]
    fn batch_max_tokens_parses_valid_positive_integer() {
        assert_eq!(parse_batch_max_tokens(Some("8192")), 8192);
        assert_eq!(parse_batch_max_tokens(Some("32768")), 32768);
        // Surrounding whitespace is tolerated — env vars sometimes pick it
        // up from shell quoting mistakes.
        assert_eq!(parse_batch_max_tokens(Some("  4096  ")), 4096);
    }

    #[test]
    fn batch_max_tokens_falls_back_on_garbage() {
        // Non-numeric → default (TEI behaviour: don't crash on typos).
        assert_eq!(parse_batch_max_tokens(Some("nope")), 16384);
        assert_eq!(parse_batch_max_tokens(Some("-1")), 16384);
        assert_eq!(parse_batch_max_tokens(Some("")), 16384);
    }

    #[test]
    fn batch_max_tokens_rejects_zero() {
        // `0` parses as a valid usize but would starve the batcher (strict `<`
        // budget gate means no 2nd item ever joins a batch); fall back to default.
        assert_eq!(parse_batch_max_tokens(Some("0")), 16384);
        assert_eq!(parse_batch_max_tokens(Some("  0  ")), 16384);
    }

    #[test]
    fn cache_max_entries_default_when_unset() {
        assert_eq!(parse_cache_max_entries(None), 10_000);
    }

    #[test]
    fn cache_max_entries_parses_valid_values() {
        assert_eq!(parse_cache_max_entries(Some("500")), 500);
        assert_eq!(parse_cache_max_entries(Some("50000")), 50_000);
        // Surrounding whitespace tolerated (env quoting mishaps).
        assert_eq!(parse_cache_max_entries(Some("  200  ")), 200);
    }

    #[test]
    fn cache_max_entries_zero_is_explicit_disable() {
        // 0 is THE documented disable signal — must round-trip, not fall
        // back to the default like batch_max_tokens does.
        assert_eq!(parse_cache_max_entries(Some("0")), 0);
    }

    #[test]
    fn cache_max_entries_falls_back_on_garbage() {
        assert_eq!(parse_cache_max_entries(Some("nope")), 10_000);
        assert_eq!(parse_cache_max_entries(Some("")), 10_000);
        assert_eq!(parse_cache_max_entries(Some("-1")), 10_000);
    }

    // -----------------------------------------------------------------
    // E2: RERANKER_MODELS parser. Mirrors the EMBED_MODELS parse style
    // but with 4 fields instead of 6 and an empty-list-is-valid contract.
    // -----------------------------------------------------------------

    #[test]
    fn parse_rerankers_empty_string_is_empty_list() {
        // The "unset" path in `from_env` turns None → Ok(vec![]); this
        // test covers the "set to empty string" edge (env quoting quirk).
        assert_eq!(parse_rerankers("").unwrap(), vec![]);
        assert_eq!(parse_rerankers("   ").unwrap(), vec![]);
        // Trailing comma variants — `filter` drops empty splits.
        assert_eq!(parse_rerankers(",,").unwrap(), vec![]);
    }

    #[test]
    fn parse_rerankers_single_entry_round_trips() {
        let got = parse_rerankers("gte-multi-rerank:/models-gte-rerank:256:true").unwrap();
        assert_eq!(
            got,
            vec![RerankerModelDef {
                name: "gte-multi-rerank".into(),
                dir: "/models-gte-rerank".into(),
                max_len: 256,
                padded_model: true,
            }]
        );
    }

    #[test]
    fn parse_rerankers_multiple_entries_parse_in_order() {
        let got = parse_rerankers("bge:/a:256:true,jina:/b:512:false").unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "bge");
        assert_eq!(got[0].max_len, 256);
        assert!(got[0].padded_model);
        assert_eq!(got[1].name, "jina");
        assert_eq!(got[1].max_len, 512);
        assert!(!got[1].padded_model);
        // Accept both spellings of the boolean — matches EMBED_MODELS'
        // has_tti parser for consistency.
        let got = parse_rerankers("a:/x:128:1,b:/y:64:0").unwrap();
        assert!(got[0].padded_model);
        assert!(!got[1].padded_model);
    }

    #[test]
    fn parse_rerankers_garbage_errors() {
        // Wrong field count.
        assert!(parse_rerankers("toofew:/a:512").is_err());
        assert!(parse_rerankers("way:too:many:colons:here:oops").is_err());
        // Unparseable max_len.
        assert!(parse_rerankers("bad:/a:notanumber:true").is_err());
        // Invalid padded boolean.
        let err = parse_rerankers("bad:/a:512:maybe").unwrap_err();
        assert!(err.contains("padded"), "unexpected err: {err}");
    }

    // -----------------------------------------------------------------
    // RERANKER_SESSION_POOL_SIZE parser. Default is 1 (single-session,
    // exactly the pre-pool behaviour). Mirrors the cache/batch parser
    // shape: helper takes Option<&str>, env lookup stays in `from_env`.
    // -----------------------------------------------------------------

    #[test]
    fn reranker_pool_size_default_is_1_when_unset() {
        // Unset env — preserves the legacy single-Mutex<Session> path.
        assert_eq!(parse_reranker_pool_size(None), 1);
    }

    #[test]
    fn reranker_pool_size_parses_valid_positive_integer() {
        assert_eq!(parse_reranker_pool_size(Some("1")), 1);
        assert_eq!(parse_reranker_pool_size(Some("2")), 2);
        assert_eq!(parse_reranker_pool_size(Some("4")), 4);
        // Surrounding whitespace tolerated (env quoting mishaps).
        assert_eq!(parse_reranker_pool_size(Some("  3  ")), 3);
    }

    #[test]
    fn reranker_pool_size_rejects_zero() {
        // 0 would `% 0` panic in the round-robin selector; fall back
        // rather than silently accept (matches BATCH_MAX_TOKENS=0 stance).
        assert_eq!(parse_reranker_pool_size(Some("0")), 1);
        assert_eq!(parse_reranker_pool_size(Some("  0  ")), 1);
    }

    #[test]
    fn reranker_pool_size_falls_back_on_garbage() {
        assert_eq!(parse_reranker_pool_size(Some("nope")), 1);
        assert_eq!(parse_reranker_pool_size(Some("")), 1);
        assert_eq!(parse_reranker_pool_size(Some("-1")), 1);
    }

    // -----------------------------------------------------------------
    // SPLADE_MODELS parser. Mirrors the RERANKER_MODELS test set but
    // with 3 fields (name:dir:max_len — no `padded` switch).
    // -----------------------------------------------------------------

    #[test]
    fn parse_splades_empty_string_is_empty_list() {
        assert_eq!(parse_splades("").unwrap(), vec![]);
        assert_eq!(parse_splades("   ").unwrap(), vec![]);
        assert_eq!(parse_splades(",,").unwrap(), vec![]);
    }

    #[test]
    fn parse_splades_single_entry_round_trips() {
        let got = parse_splades("splade-v3-distilbert:/models-splade:512").unwrap();
        assert_eq!(
            got,
            vec![SpladeModelDef {
                name: "splade-v3-distilbert".into(),
                dir: "/models-splade".into(),
                max_len: 512,
            }]
        );
    }

    #[test]
    fn parse_splades_multiple_entries_parse_in_order() {
        let got = parse_splades("a:/x:128,b:/y:256").unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "a");
        assert_eq!(got[0].max_len, 128);
        assert_eq!(got[1].name, "b");
        assert_eq!(got[1].max_len, 256);
    }

    #[test]
    fn parse_splades_garbage_errors() {
        // Wrong field count — too few.
        assert!(parse_splades("toofew:/a").is_err());
        // Wrong field count — too many. (4-field reranker entry won't
        // parse as a splade entry, demonstrating the strict 3-field
        // contract guards against env-var copy-paste mistakes.)
        assert!(parse_splades("name:/dir:512:true").is_err());
        // Unparseable max_len.
        let err = parse_splades("bad:/a:notanumber").unwrap_err();
        assert!(err.contains("max_len"), "unexpected err: {err}");
    }

    // -----------------------------------------------------------------
    // SPLADE_SESSION_POOL_SIZE parser — same shape as the reranker one.
    // -----------------------------------------------------------------

    #[test]
    fn splade_pool_size_default_is_1_when_unset() {
        assert_eq!(parse_splade_pool_size(None), 1);
    }

    #[test]
    fn splade_pool_size_parses_valid_positive_integer() {
        assert_eq!(parse_splade_pool_size(Some("1")), 1);
        assert_eq!(parse_splade_pool_size(Some("2")), 2);
        assert_eq!(parse_splade_pool_size(Some("  3  ")), 3);
    }

    #[test]
    fn splade_pool_size_rejects_zero() {
        assert_eq!(parse_splade_pool_size(Some("0")), 1);
        assert_eq!(parse_splade_pool_size(Some("  0  ")), 1);
    }

    #[test]
    fn splade_pool_size_falls_back_on_garbage() {
        assert_eq!(parse_splade_pool_size(Some("nope")), 1);
        assert_eq!(parse_splade_pool_size(Some("")), 1);
        assert_eq!(parse_splade_pool_size(Some("-1")), 1);
    }

    // -----------------------------------------------------------------
    // TOKEN_CACHE_MAX_ENTRIES parser.
    //
    // Differs from CACHE_MAX_ENTRIES: default is 0 (disabled) so
    // existing deployments see no behaviour change without opting in.
    // -----------------------------------------------------------------

    #[test]
    fn token_cache_max_entries_default_disabled_when_unset() {
        assert_eq!(parse_token_cache_max_entries(None), 0);
    }

    #[test]
    fn token_cache_max_entries_parses_valid_values() {
        assert_eq!(parse_token_cache_max_entries(Some("20000")), 20_000);
        assert_eq!(parse_token_cache_max_entries(Some("1")), 1);
        // Surrounding whitespace tolerated.
        assert_eq!(parse_token_cache_max_entries(Some("  500  ")), 500);
    }

    #[test]
    fn token_cache_max_entries_zero_is_explicit_disable() {
        assert_eq!(parse_token_cache_max_entries(Some("0")), 0);
    }

    #[test]
    fn token_cache_max_entries_falls_back_to_disabled_on_garbage() {
        assert_eq!(parse_token_cache_max_entries(Some("nope")), 0);
        assert_eq!(parse_token_cache_max_entries(Some("")), 0);
        assert_eq!(parse_token_cache_max_entries(Some("-1")), 0);
    }

    // -----------------------------------------------------------------
    // {RERANK,EMBED,SPLADE}_WARMUP_BATCH_SIZES parser.
    //
    // Single shared parser used by all three env vars (per-kind defaults
    // passed in by the caller — `parse_warmup_batch_sizes` is unaware of
    // which env var produced `raw`, so it cannot bake the defaults in).
    // Contract:
    //   - None or empty → return defaults verbatim
    //   - "1,5" → [1, 5] (preserve order)
    //   - "5,1" → [5, 1] (preserve order — operator decides which shape
    //     to compile first; we don't second-guess)
    //   - "3,5,3" → [3, 5] (dedup, first-occurrence wins, no resort)
    //   - "0,1,5" → [1, 5] (drop 0 — would `% 0` panic in the dispatch
    //     loop and is meaningless as a batch size)
    //   - "garbage" or "1,nope,5" → [1, 5] (skip unparseable, keep
    //     valid neighbours so a single typo doesn't kill the warmup)
    //   - "nope,oops" (all garbage) → defaults (nothing valid → fall back
    //     so prod still gets the pre-warm benefit on env-var typos)
    // -----------------------------------------------------------------

    #[test]
    fn warmup_batch_sizes_default_when_unset() {
        // None → defaults verbatim. This is the load-bearing path: when
        // no env override is set, we must still pre-warm the shapes that
        // production traffic uses (1 + memdb's batch=5, or embed's
        // texts_per_req=8). Otherwise the feature is off-by-default and
        // the cold-path latency spike persists.
        assert_eq!(parse_warmup_batch_sizes(None, &[1, 5]), vec![1, 5]);
        assert_eq!(parse_warmup_batch_sizes(None, &[1]), vec![1]);
        assert_eq!(parse_warmup_batch_sizes(None, &[1, 8]), vec![1, 8]);
    }

    #[test]
    fn warmup_batch_sizes_parses_csv_in_order() {
        // Order matters — operator picks "compile this first" by ordering
        // the list. We do not sort.
        assert_eq!(
            parse_warmup_batch_sizes(Some("1,5"), &[1, 5]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("5,1"), &[1, 5]),
            vec![5, 1]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("1,2,5,10"), &[1]),
            vec![1, 2, 5, 10]
        );
        // Whitespace tolerance — env quoting in compose files is fragile.
        assert_eq!(
            parse_warmup_batch_sizes(Some("  1 , 5 "), &[1]),
            vec![1, 5]
        );
    }

    #[test]
    fn warmup_batch_sizes_dedups_preserving_order() {
        // Duplicates collapse to first occurrence — keeps total warmup
        // time bounded when an operator pastes "1,5,1" by accident.
        assert_eq!(
            parse_warmup_batch_sizes(Some("3,5,3"), &[1]),
            vec![3, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("1,1,1"), &[1]),
            vec![1]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("1,5,1,5,8"), &[1]),
            vec![1, 5, 8]
        );
    }

    #[test]
    fn warmup_batch_sizes_drops_zero_and_negatives() {
        // 0 would `% 0` panic in any per-batch dispatch loop. Negatives
        // can't parse as usize. Both get dropped without falling back —
        // the rest of the list is still useful.
        assert_eq!(
            parse_warmup_batch_sizes(Some("0,1,5"), &[1]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("1,-3,5"), &[1]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("0"), &[1, 5]),
            vec![1, 5],
            "all-zero CSV must fall back to defaults, not return empty"
        );
    }

    #[test]
    fn warmup_batch_sizes_skips_unparseable_keeps_valid() {
        // Single typo in the middle should not kill warmup for the other
        // shapes — drop the bad token, keep the rest.
        assert_eq!(
            parse_warmup_batch_sizes(Some("1,nope,5"), &[1]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes(Some("garbage"), &[1, 5]),
            vec![1, 5],
            "all-garbage must fall back to defaults so prod still warms"
        );
    }

    #[test]
    fn warmup_batch_sizes_empty_string_falls_back() {
        // Set-to-empty (not unset — those are different) still gets the
        // defaults. An operator who unsets warmup has to write something
        // explicit ("the warmup feature has no documented disable knob"
        // — same stance as the reranker pool size's `0` rejection).
        assert_eq!(parse_warmup_batch_sizes(Some(""), &[1, 5]), vec![1, 5]);
        assert_eq!(
            parse_warmup_batch_sizes(Some("   "), &[1, 5]),
            vec![1, 5]
        );
        assert_eq!(parse_warmup_batch_sizes(Some(",,"), &[1, 5]), vec![1, 5]);
    }
}
