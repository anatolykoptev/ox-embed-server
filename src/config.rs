use std::env;

/// Definition of a single model to load.
pub struct ModelDef {
    pub name: String,
    pub dir: String,
    /// Filename of the ONNX model inside `dir`.
    ///
    /// Defaults to `"model_quantized.onnx"` when the 7th segment of the
    /// `EMBED_MODELS` spec is absent, preserving byte-identical behaviour for
    /// all existing 6-segment entries.
    ///
    /// Set to e.g. `"model_int8.onnx"` for models whose file does not follow
    /// the default naming convention. Validation: non-empty, no path separator
    /// characters (`/`, `\`) and no `..` component — the value is joined onto
    /// `dir` as a plain filename.
    pub onnx_filename: String,
    pub dim: usize,
    pub max_len: usize,
    pub pad_id: u32,
    pub has_token_type_ids: bool,
    /// Per-model cap on the longest sequence in a batch.
    ///
    /// Defaults to the global `BATCH_MAX_SEQ` value (256). Override per
    /// model via `BATCH_MAX_SEQ_<MODEL_UPPER>=N` (uppercase name, `-`→`_`).
    /// Example: `BATCH_MAX_SEQ_JINA_CODE_V2=384`. This lets operators
    /// set a tighter cap for short-context models (e5 max_len=256) while
    /// allowing higher seq values for long-context ones (jina max_len=512)
    /// without forcing a single global compromise.
    pub batch_max_seq: usize,
    /// Per-model warmup sequence length.
    ///
    /// `Some(n)` → warmup pads tensors to `min(n, max_len)`.
    /// `None`    → warmup pads to `max_len` (legacy "max" keyword).
    ///
    /// Defaults to `Some(max_len)` — warmup at the model's full context
    /// window so `memory_pattern` plans against worst-case scratch and
    /// the first prod request at max length does NOT trigger a replan.
    /// This is the correct fix for the jina-code-v2 OOM: warm at 512,
    /// not at the old global default of 128.
    ///
    /// Override per model via `EMBED_WARMUP_SEQ_LEN_<MODEL_UPPER>=N`
    /// (uppercase name, `-`→`_`). Setting `=max` gives `None`.
    /// Example: `EMBED_WARMUP_SEQ_LEN_MULTILINGUAL_E5_LARGE=128`.
    ///
    /// Trade-off: larger warmup = larger startup memory commitment but
    /// stable steady-state inference latency.
    pub warmup_seq_len: Option<usize>,
    /// Whether to enable ORT's memory-pattern optimisation for this model.
    ///
    /// `true` (default) — ORT pre-allocates the entire forward-pass plan as
    /// a single BFCArena block. Amortises per-shape scratch allocation for
    /// models with stable input shapes (e5-large: max_len=256, fixed prod
    /// traffic → 0 errors in prod).
    ///
    /// `false` — per-op allocation on every inference call. Real peak working
    /// memory jina S=512 B=1 ≈ 80 MiB (vs 1.258 GiB plan-block with
    /// `true`). Required for jina-code-v2 (max_len=512, variable seq,
    /// BFCArena fills monotonically → arena OOM with `true`).
    ///
    /// Override per model via `EMBED_MEMORY_PATTERN_<MODEL_UPPER>=false`
    /// (uppercase name, `-`→`_`). Example: `EMBED_MEMORY_PATTERN_JINA_CODE_V2=false`.
    /// Invalid values fall back to `true` with a warn.
    pub memory_pattern: bool,
    /// Instruction-prefix scheme applied to `/v1/embeddings` inputs before
    /// inference (`query: ` / `passage: ` for e5 models).
    ///
    /// Defaults to off: turning it on changes every vector the model returns,
    /// so every store built from this server's vectors must be rebuilt first.
    /// Set per model via `EMBED_INPUT_PREFIX_<MODEL_UPPER>=e5` (uppercase
    /// name, `-`→`_`). An unknown value fails config parsing.
    pub input_prefix: crate::input_prefix::InputPrefix,
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
    /// Number of ONNX `Session` instances loaded per embedding model.
    /// Each session can run inference independently, so concurrent
    /// `/v1/embeddings` calls hitting the same model can run in parallel
    /// up to `embed_pool_size` at a time. `1` (the default) preserves
    /// the legacy single-Mutex<Session> behaviour byte-for-byte.
    ///
    /// IMPORTANT: ort 2.0-rc has no shared-weights mode — each pool
    /// member pays its own ~400 MiB (e5-large) / ~250 MiB (jina-code-v2)
    /// weight buffer. When raising above 1, lower
    /// `EMBED_INTRA_THREADS` so `pool_size * intra_threads` stays at or
    /// below the available CPU cores. See CLAUDE.md "Environment" table.
    pub embed_pool_size: usize,
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
    /// Per-batch cap on `max_seq` (longest token sequence in the batch).
    ///
    /// When admitting another item would push `max(current_max_seq,
    /// item.seq_len)` strictly above this value AND the batch is non-
    /// empty, the worker flushes the current batch and carries the
    /// outlier into a new one. Long docs end up in B=1 batches at full
    /// `max_len`; short docs stay packed and small.
    ///
    /// Architectural waste fix: one 500-token doc in a batch of 7×50-
    /// token docs forces all 8 to pad to 500 → tensor `[8, 500]` ≈ 10×
    /// the honest token volume. With this cap (default `256`), the
    /// 500-token doc is split into its own batch.
    ///
    /// Default `256`. Override via `BATCH_MAX_SEQ`. Operators should
    /// keep this ≤ the smallest model `max_len` they serve to avoid
    /// the gate becoming a no-op for long traffic on long-context
    /// models. Empty batches always admit their first item regardless
    /// of seq_len so single-long-doc requests never starve.
    pub batch_max_seq: usize,
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
    /// fast-path single-pair calls) and batch=5 (the downstream consumer's D7 sub-query
    /// fanout default). Operators set `RERANK_WARMUP_BATCH_SIZES` to
    /// override (e.g. `1,2,5,10` for boxes serving wider batch ranges).
    pub rerank_warmup_batch_sizes: Vec<usize>,
    /// Per-shape warmup batch sizes for dense (bi-encoder) embedders.
    ///
    /// Default `[1, 8]`: batch=1 covers the trivial `/v1/embeddings`
    /// caller, batch=8 matches the `texts_per_req=8` default the typical
    /// the downstream consumer embedder client uses. Override via
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
    /// Cap on the per-shape warmup `max_seq` dimension (in tokens).
    ///
    /// `None` (env unset OR set to literal `"max"`) → pad warmup tensors
    /// to the model's `max_len` (legacy behaviour: pre-commits worst-
    /// case scratch slabs at startup).
    ///
    /// `Some(n)` → pad warmup tensors to `min(n, max_len)`. With
    /// `memory_pattern=true`, ORT plans on the first prod request to
    /// re-bind kernels for any longer shape — one-time cost is
    /// acceptable. Saves 200-400 MiB resident memory after startup.
    /// Default 128 — a sane median between e5's 256 max and jina's 512
    /// max that covers most prod traffic without committing worst-case
    /// scratch up front.
    ///
    /// Applies to dense embedders and rerankers. SPLADE warmup is
    /// already token-bounded (uses tokenizer output directly) and is
    /// not affected.
    pub embed_warmup_seq_len: Option<usize>,
    /// Idle-eviction threshold for ONNX sessions in all pools.
    ///
    /// `0` (default) — eviction disabled. The pool behaves like a simple
    /// pre-allocated pool: sessions are never evicted and memory is held
    /// for the lifetime of the process.
    ///
    /// Positive value — sessions idle longer than this many seconds are
    /// evicted (freed). The next acquire triggers a cold start (~5–10s
    /// at ONNX reload time; no ONNX_OPT_CACHE_DIR — see Phase H.21
    /// incident). Use with `EMBED_SESSION_POOL_SIZE=2` to trade latency
    /// tail (rare cold starts) for memory savings (~250 MiB per session
    /// slot freed during extended idle periods).
    ///
    /// Set via `EMBED_IDLE_EVICT_SECS`. Recommended minimum: 300
    /// (5 minutes) to avoid cold-start churn under intermittent traffic.
    /// The eviction background tick runs at `max(idle_secs/4, 5s)` so
    /// the worst-case overshoot is one tick interval past the threshold.
    pub idle_evict_secs: u64,
    /// Maximum number of texts allowed in a single `/v1/embeddings` input
    /// array. Requests that exceed this cap are rejected with HTTP 400
    /// **before** they reach the batcher — protecting the BFCArena from
    /// oversized single-call attention-scratch allocations.
    ///
    /// **Why 32**: jina-code-v2 (12 heads, max_len=512) attention scratch
    /// per inference is `B × H × S² × 4`. At the cap:
    ///   32 × 12 × 512² × 4 ≈ 402 MiB (under the 512 MiB safe threshold).
    /// At 100 texts (the downstream consumer client's former default):
    ///   100 × 12 × 512² × 4 ≈ 1.258 GiB → BFCArena OOM (~1/min in prod).
    ///
    /// **Override**: set `EMBED_MAX_INPUT_ARRAY` to a positive integer.
    /// Operators on high-memory hosts can raise this; operators on
    /// constrained boxes should lower it further.
    ///
    /// Rejected requests get HTTP 400 (not 503): this is a permanent client
    /// misuse (array too large), not a transient overload.
    pub embed_max_input_array: usize,
    /// Maximum number of documents allowed in a single `/v1/rerank` request.
    /// Requests exceeding this are rejected with HTTP 400 before tokenization
    /// — protecting the BFCArena from oversized cross-encoder scratch allocs.
    ///
    /// **Why 32** (same as `embed_max_input_array`): gte-multi-rerank uses
    /// `max_len=256`. Reranker attention scratch shape is `B × pairs(1) ×
    /// S² × 4`; at cap=32, docs=32:
    ///   32 × 1 × 256² × 4 ≈ 8 MiB per slot — well under the arena even
    ///   with concurrent rerank sessions. Quadratic cost is 4× lower than
    ///   jina (S=512) because `256² vs 512²`. Keeping the cap aligned with
    ///   `embed_max_input_array` means operators only memorise one number.
    ///
    /// **Override**: set `RERANK_MAX_INPUT_DOCS` to a positive integer.
    pub rerank_max_input_docs: usize,
    /// Enable multi-process mode — each embedding model runs in a separate
    /// `embed-worker` child process communicating over Unix domain sockets.
    ///
    /// When `false` (default), the server runs all models in-process (legacy
    /// behaviour, unchanged). When `true`, `WorkerSupervisor::launch` is called
    /// for each model at startup; HTTP routing via workers landed in Wave 2.4.
    ///
    /// Set via `EMBED_MULTI_PROCESS=1` or `EMBED_MULTI_PROCESS=true`.
    pub multi_process: bool,
    /// Path to the `embed-worker` binary to spawn in multi-process mode.
    ///
    /// Set via `EMBED_WORKER_BIN` (default `/usr/local/bin/embed-worker`).
    /// Ignored when `multi_process` is `false`.
    pub worker_bin_path: std::path::PathBuf,
    /// Directory for worker Unix domain sockets in multi-process mode.
    ///
    /// Each model gets `<dir>/<model_name>.sock`. The directory is created
    /// by `WorkerSupervisor::launch` if it does not exist.
    ///
    /// Set via `EMBED_WORKER_SOCKET_DIR` (default `/tmp/embed-workers`).
    /// Ignored when `multi_process` is `false`.
    pub worker_socket_dir: std::path::PathBuf,
    /// Timeout for the `/ready` probe inference, in milliseconds.
    ///
    /// The readiness probe dispatches a single 1-word embed to the default
    /// model and waits up to this long. If the worker is wedged (busy-spin
    /// with zero throughput), the probe times out and `/ready` returns 503,
    /// letting the container orchestrator or downstream consumer react.
    ///
    /// Set via `EMBED_READY_PROBE_TIMEOUT_MS` (default 2000). Must be > 0;
    /// invalid values fall back to the default.
    pub ready_probe_timeout_ms: u64,
}

impl Config {
    /// Parse configuration from environment variables.
    ///
    /// - `EMBED_PORT`: listen port (default 8082)
    /// - `EMBED_MODELS`: comma-separated model specs
    ///   Format: `name:dir:dim:max_len:pad_id:has_tti[:onnx_filename]`
    ///   The 7th segment is optional; omitting it defaults to `model_quantized.onnx`.
    /// - `EMBED_DEFAULT_MODEL`: default model name (default: first)
    pub fn from_env() -> Result<Self, String> {
        let port = env::var("EMBED_PORT") // env-strict: a bad port fails startup below
            .unwrap_or_else(|_| "8082".into())
            .parse::<u16>()
            .map_err(|e| format!("invalid EMBED_PORT: {e}"))?;

        let models_str = env::var("EMBED_MODELS") // env-strict: required, parse errors abort startup
            .map_err(|_| "EMBED_MODELS env var is required")?;

        // Resolve global batch_max_seq and warmup_seq_len BEFORE parsing
        // models so per-model env overrides can fall back to them.
        //
        // Note on warmup_seq_len sentinel: we pass `None` when the env var
        // is UNSET so that parse_one_model defaults each model to
        // `Some(max_len)` (the per-model "warmup at full context window"
        // default). We pass `Some(n)` when the operator explicitly set
        // EMBED_WARMUP_SEQ_LEN, and we pass `None` again when they wrote
        // "max" (both cases are `None` from `parse_embed_warmup_seq_len`,
        // which is intentional — "max" and "unset" have the same effect on
        // models that haven't set a per-model override).
        let global_batch_max_seq_for_models =
            // parsed-by: parse_batch_max_seq
            parse_batch_max_seq(crate::env_parse::raw("BATCH_MAX_SEQ").as_deref());
        // Parsed ONCE: parse_embed_warmup_seq_len notes a bad value, and two
        // parses of the same variable would count it twice.
        // parsed-by: parse_embed_warmup_seq_len
        let warmup_raw = crate::env_parse::raw("EMBED_WARMUP_SEQ_LEN");
        let embed_warmup_seq_len = parse_embed_warmup_seq_len(warmup_raw.as_deref());
        let global_warmup_for_models: Option<usize> = match warmup_raw.as_deref() {
            // unset → each model defaults to Some(max_len)
            None | Some("") => None,
            // "max" → None (pad to max_len); a number or garbage → the parsed
            // value, which parse_one_model uses as the global fallback.
            Some(_) => embed_warmup_seq_len,
        };
        let models = parse_models_with_globals(
            &models_str,
            global_batch_max_seq_for_models,
            global_warmup_for_models,
        )?;
        if models.is_empty() {
            return Err("EMBED_MODELS must define at least one model".into());
        }

        let default_model = env::var("EMBED_DEFAULT_MODEL") // env-strict: unknown name aborts startup below
            .unwrap_or_else(|_| models[0].name.clone());

        if !models.iter().any(|m| m.name == default_model) {
            return Err(format!(
                "EMBED_DEFAULT_MODEL '{default_model}' not found in models"
            ));
        }

        // Default lowered 4 → 2 (2026-05-06): kernel-level perf audit
        // measured 5× ORT thread oversubscription on a 4-core ARM
        // Neoverse-N1 host with 14 intra_op threads in flight (4×e5
        // sessions × 4 threads + reranker pool overhead). DynamicQuantize-
        // MatMul + MatMulIntegerToFloat are 73.8 % of inference time
        // (NEON asimddp INT8 GEMM at the hardware ceiling — IPC=2.46,
        // cache_miss=1.3 %), so the bottleneck is contention, not
        // kernels. Two threads/inference + EMBED_SESSION_POOL_SIZE=2
        // yields 4 concurrent slots × 2 threads = 8 ORT threads on 4
        // cores = 2× oversub instead of 5×, ~2× throughput under load
        // at the cost of ~10–20 % solo-inference latency. Operators on
        // dedicated-CPU hosts can override back to 4.
        let intra_threads = {
            let raw = env::var("EMBED_INTRA_THREADS") // env-strict: garbage aborts startup
                .unwrap_or_else(|_| "2".into())
                .parse::<usize>()
                .map_err(|e| format!("invalid EMBED_INTRA_THREADS: {e}"))?;
            if raw == 0 {
                crate::env_parse::note_fallback(
                    "EMBED_INTRA_THREADS",
                    "0",
                    "a positive integer (0 would mean ORT auto-select, but the session pool expects an explicit count)",
                    &2,
                );
                2
            } else {
                raw
            }
        };

        // parsed-by: parse_embed_pool_size
        let embed_pool_size = parse_embed_pool_size(
            crate::env_parse::raw("EMBED_SESSION_POOL_SIZE").as_deref(),
            None,
        );
        // parsed-by: parse_reranker_pool_size
        let reranker_pool_size = parse_reranker_pool_size(
            crate::env_parse::raw("RERANKER_SESSION_POOL_SIZE").as_deref(),
        );

        // `RERANKER_INTRA_THREADS` defaults to `intra_threads` so unset
        // means "share the embedder budget" (today's behaviour). Set
        // explicitly when raising pool_size to keep total reranker
        // threads under control without changing embedder threads.
        let reranker_intra_threads = crate::env_parse::positive(
            "RERANKER_INTRA_THREADS",
            crate::env_parse::raw("RERANKER_INTRA_THREADS").as_deref(), // parsed-by: positive
            intra_threads,
        );
        let batching_enabled = crate::env_parse::flag(
            "BATCHING_ENABLED",
            crate::env_parse::raw("BATCHING_ENABLED").as_deref(), // parsed-by: flag
            false,
            &["true", "1"],
            &["false", "0"],
        );
        let batch_max = crate::env_parse::any(
            "BATCH_MAX",
            crate::env_parse::raw("BATCH_MAX").as_deref(), // parsed-by: any
            32usize,
            "a non-negative integer",
        );

        // Reranker-specific item cap. Defaults to 4× embed batch_max so a
        // single env (`BATCH_MAX=8`) can keep embed-side cache-thrash
        // mitigation while the reranker still coalesces 4 concurrent
        // multi-doc calls. Operators can override with RERANKER_BATCH_MAX
        // when they want a different ratio.
        let reranker_batch_max = crate::env_parse::any(
            "RERANKER_BATCH_MAX",
            crate::env_parse::raw("RERANKER_BATCH_MAX").as_deref(), // parsed-by: any
            batch_max.saturating_mul(4),
            "a non-negative integer",
        );
        // parsed-by: parse_batch_max_tokens
        let batch_max_tokens =
            parse_batch_max_tokens(crate::env_parse::raw("BATCH_MAX_TOKENS").as_deref());
        let batch_max_seq = global_batch_max_seq_for_models; // parsed once, above
        let batch_wait_ms = crate::env_parse::any(
            "BATCH_WAIT_MS",
            crate::env_parse::raw("BATCH_WAIT_MS").as_deref(), // parsed-by: any
            10u64,
            "a non-negative integer",
        );
        let max_queue_size = crate::env_parse::any(
            "MAX_QUEUE_SIZE",
            crate::env_parse::raw("MAX_QUEUE_SIZE").as_deref(), // parsed-by: any
            256usize,
            "a non-negative integer",
        );
        let drain_timeout_s = crate::env_parse::any(
            "DRAIN_TIMEOUT_S",
            crate::env_parse::raw("DRAIN_TIMEOUT_S").as_deref(), // parsed-by: any
            10u64,
            "a non-negative integer",
        );

        // AUTO_TRUNCATE defaults to true (TEI-compat). Only the literal
        // string "false" (case-insensitive) disables it; anything else
        // keeps the safe default.
        //
        // `0` / `no` / a typo keep truncation ON (as before) but are now noted:
        // only `false` turns it off.
        let auto_truncate = crate::env_parse::flag(
            "AUTO_TRUNCATE",
            crate::env_parse::raw("AUTO_TRUNCATE").as_deref(), // parsed-by: flag
            true,
            &["true", "1"],
            &["false"],
        );
        // parsed-by: parse_cache_max_entries
        let cache_max_entries =
            parse_cache_max_entries(crate::env_parse::raw("CACHE_MAX_ENTRIES").as_deref());
        // parsed-by: parse_token_cache_max_entries
        let token_cache_max_entries = parse_token_cache_max_entries(
            crate::env_parse::raw("TOKEN_CACHE_MAX_ENTRIES").as_deref(),
        );

        // `RERANKER_MODELS` is optional: unset or empty → no rerankers,
        // server boots serving only `/v1/embeddings`. `/v1/rerank` with
        // any model name will 400. Errors here only on malformed entries
        // (bad integer fields, wrong colon count) — a strict
        // fail-at-boot contract matching `EMBED_MODELS`.
        let rerankers = env::var("RERANKER_MODELS") // env-strict: malformed entries abort startup
            .ok()
            .map(|s| parse_rerankers(&s))
            .transpose()?
            .unwrap_or_default();

        // `SPLADE_MODELS` follows the same contract as `RERANKER_MODELS`:
        // unset/empty → no SPLADE endpoints; malformed → fail boot.
        let splades = env::var("SPLADE_MODELS") // env-strict: malformed entries abort startup
            .ok()
            .map(|s| parse_splades(&s))
            .transpose()?
            .unwrap_or_default();

        // parsed-by: parse_splade_pool_size
        let splade_pool_size =
            parse_splade_pool_size(crate::env_parse::raw("SPLADE_SESSION_POOL_SIZE").as_deref());

        // SPLADE_INTRA_THREADS defaults to `intra_threads` (share embedder
        // budget when unset), same fallback as RERANKER_INTRA_THREADS.
        let splade_intra_threads = crate::env_parse::positive(
            "SPLADE_INTRA_THREADS",
            crate::env_parse::raw("SPLADE_INTRA_THREADS").as_deref(), // parsed-by: positive
            intra_threads,
        );

        // Per-kind warmup shape lists. Defaults are tuned to the prod
        // traffic shape we already know about (memdb's batch=5 fanout,
        // texts_per_req=8 embed default, single-text splade) — the
        // unset path MUST get the cold-start fix, hence non-empty
        // defaults rather than `[]` when the env var is absent.
        // parsed-by: parse_warmup_batch_sizes
        let rerank_warmup_batch_sizes = parse_warmup_batch_sizes(
            "RERANK_WARMUP_BATCH_SIZES",
            crate::env_parse::raw("RERANK_WARMUP_BATCH_SIZES").as_deref(),
            &[1, 5],
        );
        // parsed-by: parse_warmup_batch_sizes
        let embed_warmup_batch_sizes = parse_warmup_batch_sizes(
            "EMBED_WARMUP_BATCH_SIZES",
            crate::env_parse::raw("EMBED_WARMUP_BATCH_SIZES").as_deref(),
            &[1, 8],
        );
        // parsed-by: parse_warmup_batch_sizes
        let splade_warmup_batch_sizes = parse_warmup_batch_sizes(
            "SPLADE_WARMUP_BATCH_SIZES",
            crate::env_parse::raw("SPLADE_WARMUP_BATCH_SIZES").as_deref(),
            &[1],
        );
        let idle_evict_secs = crate::env_parse::any(
            "EMBED_IDLE_EVICT_SECS",
            crate::env_parse::raw("EMBED_IDLE_EVICT_SECS").as_deref(), // parsed-by: any
            0u64,
            "a non-negative integer (0 disables eviction)",
        );
        let embed_max_input_array = crate::env_parse::positive(
            "EMBED_MAX_INPUT_ARRAY",
            crate::env_parse::raw("EMBED_MAX_INPUT_ARRAY").as_deref(), // parsed-by: positive
            32usize,
        );
        let rerank_max_input_docs = crate::env_parse::positive(
            "RERANK_MAX_INPUT_DOCS",
            crate::env_parse::raw("RERANK_MAX_INPUT_DOCS").as_deref(), // parsed-by: positive
            32usize,
        );
        // parsed-by: parse_multi_process_flag
        let multi_process =
            parse_multi_process_flag(crate::env_parse::raw("EMBED_MULTI_PROCESS").as_deref());
        let worker_bin_path = env::var("EMBED_WORKER_BIN") // env-raw: a path, no parse
            .unwrap_or_else(|_| "/usr/local/bin/embed-worker".into())
            .into();
        let worker_socket_dir = env::var("EMBED_WORKER_SOCKET_DIR") // env-raw: a path, no parse
            .unwrap_or_else(|_| "/tmp/embed-workers".into())
            .into();
        let ready_probe_timeout_ms = crate::env_parse::positive(
            "EMBED_READY_PROBE_TIMEOUT_MS",
            crate::env_parse::raw("EMBED_READY_PROBE_TIMEOUT_MS").as_deref(), // parsed-by: positive
            2000u64,
        );
        Ok(Config {
            port,
            models,
            rerankers,
            default_model,
            intra_threads,
            embed_pool_size,
            reranker_pool_size,
            reranker_intra_threads,
            splades,
            splade_pool_size,
            splade_intra_threads,
            batching_enabled,
            batch_max,
            reranker_batch_max,
            batch_max_tokens,
            batch_max_seq,
            batch_wait_ms,
            max_queue_size,
            drain_timeout_s,
            auto_truncate,
            cache_max_entries,
            token_cache_max_entries,
            rerank_warmup_batch_sizes,
            embed_warmup_batch_sizes,
            splade_warmup_batch_sizes,
            embed_warmup_seq_len,
            idle_evict_secs,
            embed_max_input_array,
            rerank_max_input_docs,
            multi_process,
            worker_bin_path,
            worker_socket_dir,
            ready_probe_timeout_ms,
        })
    }
}

/// Parse `CACHE_MAX_ENTRIES` env value. Unset, empty, or unparseable →
/// 10_000 (sensible default). An explicit `0` is honoured as the
/// documented disable signal (EmbeddingCache becomes a no-op shell).
/// Exposed for testing; env lookup stays in `from_env`.
fn parse_cache_max_entries(raw: Option<&str>) -> usize {
    const DEFAULT: usize = 10_000;
    crate::env_parse::any("CACHE_MAX_ENTRIES", raw, DEFAULT, "a non-negative integer")
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
    crate::env_parse::any(
        "TOKEN_CACHE_MAX_ENTRIES",
        raw,
        DEFAULT,
        "a non-negative integer",
    )
}

/// Parse `BATCH_MAX_TOKENS` env value. Unset, empty, unparseable, or `0` →
/// 16384 (TEI default). `0` would degenerate the batcher to one item per
/// dispatch (strict `<` gate never admits a 2nd item), so it's rejected
/// with a warn rather than silently accepted. Exposed for testing; env
/// lookup stays in `from_env`.
fn parse_batch_max_tokens(raw: Option<&str>) -> usize {
    crate::env_parse::positive("BATCH_MAX_TOKENS", raw, 16384usize)
}

/// Parse `BATCH_MAX_SEQ` env value.
///
/// Unset, empty, unparseable, or `0` → 256 (sane default — covers most
/// short-document traffic while keeping long-doc outliers in their own
/// B=1 batches). `0` would degenerate the admission gate to "no item
/// ever fits" because the strict `>` check on the *new* max_seq would
/// trip on every non-empty token sequence; treated as garbage and a
/// warn is logged so operators notice typos.
///
/// Exposed for testing; env lookup stays in `from_env`.
fn parse_batch_max_seq(raw: Option<&str>) -> usize {
    crate::env_parse::positive("BATCH_MAX_SEQ", raw, 256usize)
}

/// Parse `EMBED_WARMUP_SEQ_LEN` env value.
///
/// - Unset, empty, or `"max"` (case-insensitive) → `None`: warmup pads
///   tensors to the model's `max_len` (legacy behaviour).
/// - Positive integer → `Some(n)`: warmup pads tensors to `min(n, max_len)`.
/// - `0`, negatives, or unparseable → `Some(128)`: sane median default
///   with a warn so operators notice typos in the env file.
///
/// The default of 128 covers most prod traffic between e5 (max_len=256)
/// and jina (max_len=512) without committing worst-case scratch slabs
/// at startup. Operators wanting the legacy max-len warmup set
/// `EMBED_WARMUP_SEQ_LEN=max`.
///
/// Exposed for testing; env lookup stays in `from_env`.
fn parse_embed_warmup_seq_len(raw: Option<&str>) -> Option<usize> {
    const DEFAULT: usize = 128;
    match raw {
        None => Some(DEFAULT),
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Some(DEFAULT);
            }
            if trimmed.eq_ignore_ascii_case("max") {
                return None;
            }
            Some(crate::env_parse::positive(
                "EMBED_WARMUP_SEQ_LEN",
                Some(trimmed),
                DEFAULT,
            ))
        }
    }
}

/// Parse `EMBED_SESSION_POOL_SIZE`. Same contract as
/// `parse_reranker_pool_size`: default 1 (single-session, byte-for-byte
/// equivalent to the pre-pool path), `0` rejected with a warn (would
/// `% 0` panic at request time), garbage falls back silently. ort 2.0-rc
/// has no shared-weights mode, so each pool member duplicates the
/// ~400 MiB weight buffer — operators opt in only when they have the
/// memory headroom AND want concurrency. Exposed for testing.
///
/// `source_key` is the exact env var name that provided `raw`, used in
/// warn messages so operators see `EMBED_SESSION_POOL_SIZE_JINA_CODE_V2=0`
/// rather than the generic `EMBED_SESSION_POOL_SIZE=0`. Pass `None` when
/// the key is not known (falls back to the generic message).
fn parse_embed_pool_size(raw: Option<&str>, source_key: Option<&str>) -> usize {
    const DEFAULT: usize = 1;
    let key = source_key.unwrap_or("EMBED_SESSION_POOL_SIZE");
    crate::env_parse::positive(key, raw, DEFAULT)
}

/// Convert a model key (e.g. `"jina-code-v2"`) to the SCREAMING_SNAKE_CASE
/// suffix used in per-model env var names (e.g. `"JINA_CODE_V2"`).
///
/// Rule: uppercase + replace `-` with `_`. This is the same transform used
/// by `ONNX_OPT_CACHE_DIR_<KEY>` and `EMBED_MEMORY_PATTERN_<KEY>`.
pub fn model_env_key(model_key: &str) -> String {
    model_key.to_uppercase().replace('-', "_")
}

/// Resolve the embed session pool size for a specific model.
///
/// Lookup order:
///   1. `EMBED_SESSION_POOL_SIZE_<MODEL_KEY_UPPER>` — per-model override.
///      Example: `"jina-code-v2"` → `EMBED_SESSION_POOL_SIZE_JINA_CODE_V2`.
///   2. `global_raw` — the value of the global `EMBED_SESSION_POOL_SIZE` env
///      (pass `env::var("EMBED_SESSION_POOL_SIZE").ok().as_deref()`).
///   3. Default: 1.
///
/// Same rejection contract as `parse_embed_pool_size`: 0 warns + falls back,
/// garbage falls back silently.
// `config` is compiled into both the lib (`pub mod config`) and the binary
// (`mod config` in main.rs). The only non-test caller lives in the binary
// (main.rs:~673), so the lib-target dead-code pass sees no caller; suppress
// rather than delete — deleting breaks the worker-spawn pool-size resolution.
#[allow(dead_code)]
pub(crate) fn resolve_embed_pool_size_for_model(
    model_key: &str,
    global_raw: Option<&str>,
) -> usize {
    let (size, per_model_key) = pool_size_for_model(model_key, global_raw);
    if let Some(key) = per_model_key {
        tracing::info!(
            model = %model_key,
            env = %key,
            pool_size = size,
            "per-model EMBED_SESSION_POOL_SIZE override"
        );
    }
    size
}

/// Pool-size resolution without the override info line, for callers that
/// only need the number (the concurrency-cap sizing) so the line is not
/// logged twice per model. Returns the per-model env key when it applied.
fn pool_size_for_model(model_key: &str, global_raw: Option<&str>) -> (usize, Option<String>) {
    let suffix = model_env_key(model_key);
    let per_model_key = format!("EMBED_SESSION_POOL_SIZE_{suffix}");
    // parsed-by: parse_embed_pool_size
    if let Some(raw) = crate::env_parse::raw(&per_model_key) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let parsed = parse_embed_pool_size(Some(trimmed), Some(per_model_key.as_str()));
            return (parsed, Some(per_model_key));
        }
    }
    (parse_embed_pool_size(global_raw, None), None)
}

/// Default IPC deadline (UDS connect + write + read, started after the
/// gate permit is granted — queueing is not bounded) applied per request
/// by `WorkerClient::send_request`.
///
/// 60 s sits above the worst measured pass — e5 batches up to ~28 s,
/// rerank up to 15 s, code-rank mean 5.2 s — so a healthy inference never
/// trips it, while a wedged or saturated worker surfaces as a bounded,
/// retryable failure (503 + Retry-After) instead of an infinite hang.
pub const DEFAULT_IPC_TIMEOUT_SECS: u64 = 60;

/// Resolve the per-model IPC request deadline in seconds.
///
/// Lookup order (same shape as `EMBED_MAX_WAITERS` / `EMBED_SESSION_POOL_SIZE`):
///   1. `EMBED_IPC_TIMEOUT_SECS_<MODEL_KEY>` — per-model override.
///   2. `EMBED_IPC_TIMEOUT_SECS` — global.
///   3. [`DEFAULT_IPC_TIMEOUT_SECS`].
///
/// `0` or an unparseable value at either level warns and falls through to
/// the next — a zero deadline would fail every request instantly, which is
/// never the intent. Resolved once at `WorkerSupervisor::launch`; restart
/// the container to change.
pub(crate) fn resolve_ipc_timeout(model_name: &str) -> std::time::Duration {
    let suffix = model_env_key(model_name);
    let per_model_key = format!("EMBED_IPC_TIMEOUT_SECS_{suffix}");
    for key in [per_model_key.as_str(), "EMBED_IPC_TIMEOUT_SECS"] {
        // parsed-by: resolve_ipc_timeout
        // A blank value is unset, not a typo.
        let Some(raw) = crate::env_parse::raw(key).filter(|r| !r.trim().is_empty()) else {
            continue;
        };
        match raw.trim().parse::<u64>() {
            Ok(n) if n > 0 => {
                tracing::info!(model = %model_name, env = %key, secs = n, "IPC request deadline");
                return std::time::Duration::from_secs(n);
            }
            _ => crate::env_parse::note_fallback(
                key,
                raw.trim(),
                "a positive integer number of seconds",
                &"the next fallback",
            ),
        }
    }
    std::time::Duration::from_secs(DEFAULT_IPC_TIMEOUT_SECS)
}

/// Resolve the per-model `/v1/embeddings` concurrency cap — the number of
/// in-flight requests one model may hold before the handler answers
/// 429 + `Retry-After: 1`. Same TEI `try_acquire_permit` load-shed pattern
/// as the rerank semaphore (`main.rs`); a missing map entry means uncapped.
///
/// Lookup order:
///   1. `MAX_CONCURRENT_EMBED_REQUESTS_<MODEL_KEY>` — per-model override.
///   2. `MAX_CONCURRENT_EMBED_REQUESTS` — global.
///   3. Formula: `pool_size(model) × intra_threads × 8`, floor 16 — the
///      same headroom rule the rerank cap uses: the permit is held across
///      queue wait + inference, so capping at exactly `pool × intra` would
///      reject under bursts the worker gate already admits.
///
/// `0` at either level disables the cap for that model (legacy unlimited
/// behaviour). Unparseable warns and falls through.
///
/// `global_cap_raw` is the raw `MAX_CONCURRENT_EMBED_REQUESTS` value and
/// `global_pool_raw` the raw `EMBED_SESSION_POOL_SIZE` value — pass them in
/// once rather than re-reading env per model.
// `config` is compiled into both the lib and the binary; the only caller
// lives in the binary (main.rs), so the lib-target dead-code pass sees no
// caller — suppress rather than delete, same as
// `resolve_embed_pool_size_for_model` above.
#[allow(dead_code)]
pub(crate) fn resolve_embed_concurrency_for_model(
    model_name: &str,
    global_cap_raw: Option<&str>,
    global_pool_raw: Option<&str>,
    intra_threads: usize,
) -> Option<usize> {
    let per_model_key = format!(
        "MAX_CONCURRENT_EMBED_REQUESTS_{}",
        model_env_key(model_name)
    );

    // 1. Per-model override.
    // parsed-by: resolve_embed_concurrency_for_model
    // A blank value is unset, not a typo.
    if let Some(raw) = crate::env_parse::raw(&per_model_key).filter(|r| !r.trim().is_empty()) {
        match raw.trim().parse::<usize>() {
            Ok(0) => {
                tracing::info!(
                    model = %model_name,
                    env = %per_model_key,
                    "embed concurrency cap disabled"
                );
                return None;
            }
            Ok(n) => {
                tracing::info!(
                    model = %model_name,
                    env = %per_model_key,
                    cap = n,
                    "per-model embed concurrency cap"
                );
                return Some(n);
            }
            Err(_) => crate::env_parse::note_fallback(
                &per_model_key,
                raw.trim(),
                "a non-negative integer (0 disables the cap)",
                &"the next fallback",
            ),
        }
    }

    // 2. Global override.
    if let Some(raw) = global_cap_raw.filter(|r| !r.trim().is_empty()) {
        match raw.trim().parse::<usize>() {
            Ok(0) => {
                tracing::info!(
                    "embed concurrency cap disabled via MAX_CONCURRENT_EMBED_REQUESTS=0"
                );
                return None;
            }
            Ok(n) => {
                tracing::info!(
                    model = %model_name,
                    cap = n,
                    "global embed concurrency cap"
                );
                return Some(n);
            }
            Err(_) => crate::env_parse::note_fallback(
                "MAX_CONCURRENT_EMBED_REQUESTS",
                raw.trim(),
                "a non-negative integer (0 disables the cap)",
                &"the auto default (pool x intra x 8, floor 16)",
            ),
        }
    }

    // 3. Formula default — `pool × intra × 8`, floor 16.
    let (pool, _) = pool_size_for_model(model_name, global_pool_raw);
    let auto = pool.saturating_mul(intra_threads).saturating_mul(8).max(16);
    tracing::info!(
        model = %model_name,
        cap = auto,
        pool_size = pool,
        intra_threads,
        "embed concurrency cap (auto: pool × intra × 8, floor 16)"
    );
    Some(auto)
}

/// Parse `EMBED_MULTI_PROCESS` env value.
///
/// - Unset or `None` → `false` (default: single-process mode).
/// - `"1"` or `"true"` (case-insensitive) → `true`.
/// - Any other value → `false` (strict: unrecognized values don't activate
///   multi-process to avoid accidental double-memory by a typo).
///
/// Extracted as a pure function so it can be unit-tested without env mutation.
pub(crate) fn parse_multi_process_flag(raw: Option<&str>) -> bool {
    crate::env_parse::flag(
        "EMBED_MULTI_PROCESS",
        raw,
        false,
        &["1", "true"],
        &["0", "false"],
    )
}

/// Parse `EMBED_MEMORY_PATTERN_<MODEL_UPPER>` env value.
///
/// - Unset or empty → `true` (back-compat default).
/// - `"true"` / `"1"` → `true`.
/// - `"false"` / `"0"` → `false`.
/// - Any other value → `true` + warn (typo guard, same posture as other
///   bool knobs in this file).
///
/// Exposed for testing and for reranker/splade loaders; env lookup stays
/// in the caller.
pub(crate) fn parse_memory_pattern(var: &str, raw: Option<&str>) -> bool {
    crate::env_parse::flag(var, raw, true, &["true", "1"], &["false", "0"])
}

/// Parse `RERANKER_SESSION_POOL_SIZE` env value. Unset, empty, or
/// unparseable → 1 (single-session, mirrors the pre-pool behaviour
/// exactly). `0` is rejected with a warn rather than silently accepted —
/// it would `% 0` panic at request time, and follows the same
/// "0 is invalid, fall back" stance as `BATCH_MAX_TOKENS=0`. Exposed for
/// testing; env lookup stays in `from_env`.
fn parse_reranker_pool_size(raw: Option<&str>) -> usize {
    crate::env_parse::positive("RERANKER_SESSION_POOL_SIZE", raw, 1usize)
}

/// Parse comma-separated model definitions.
/// Each entry: `name:dir:dim:max_len:pad_id:has_tti`
///
/// `global_batch_max_seq` is the fallback used when no per-model
/// `BATCH_MAX_SEQ_<MODEL_UPPER>` env var is set.
/// `global_warmup_seq_len` is the global override from `EMBED_WARMUP_SEQ_LEN`;
/// when `None` (unset/default) each model defaults to `Some(max_len)`.
fn parse_models_with_globals(
    s: &str,
    global_batch_max_seq: usize,
    global_warmup_seq_len: Option<usize>,
) -> Result<Vec<ModelDef>, String> {
    s.split(',')
        .filter(|e| !e.trim().is_empty())
        .map(|e| parse_one_model(e, global_batch_max_seq, global_warmup_seq_len))
        .collect()
}

/// Parse comma-separated model definitions using global defaults.
///
/// This is the primary entry point called from `Config::from_env`. The
/// global batch_max_seq (from `BATCH_MAX_SEQ`) and warmup_seq_len (from
/// `EMBED_WARMUP_SEQ_LEN`) are resolved before calling this so per-model
/// env var lookups inside `parse_one_model` can fall back to them.
/// Parse models with default globals. Used in tests and as a
/// convenience wrapper.
#[allow(dead_code)]
pub(crate) fn parse_models(s: &str) -> Result<Vec<ModelDef>, String> {
    // Use defaults here; `from_env` calls `parse_models_with_globals`
    // directly with the resolved globals.
    parse_models_with_globals(s, parse_batch_max_seq(None), None)
}

fn parse_one_model(
    entry: &str,
    global_batch_max_seq: usize,
    global_warmup_seq_len: Option<usize>,
) -> Result<ModelDef, String> {
    let parts: Vec<&str> = entry.trim().split(':').collect();
    if parts.len() < 6 || parts.len() > 7 {
        return Err(format!(
            "model entry must have 6 or 7 colon-separated fields \
             (name:dir:dim:max_len:pad_id:has_tti[:onnx_filename]), got {}: '{entry}'",
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

    let name = parts[0].to_string();

    // Per-model env var key: uppercase name, '-' → '_'.
    let key = model_env_key(&name);

    // Per-model batch_max_seq: BATCH_MAX_SEQ_<KEY>. Falls back to global.
    let bms_var = format!("BATCH_MAX_SEQ_{key}");
    // Unusable (garbage or 0) → the global value, noted by `positive`.
    let batch_max_seq =
        crate::env_parse::raw(&bms_var) // parsed-by: positive
            .map(|v| crate::env_parse::positive(&bms_var, Some(&v), global_batch_max_seq))
            .unwrap_or(global_batch_max_seq);
    if batch_max_seq != global_batch_max_seq {
        tracing::info!(
            model = %name,
            per_model_batch_max_seq = batch_max_seq,
            "per-model BATCH_MAX_SEQ override"
        );
    }

    // Per-model warmup_seq_len: EMBED_WARMUP_SEQ_LEN_<KEY>.
    // Precedence: per-model env > global_warmup_seq_len > Some(max_len).
    let wsl_var = format!("EMBED_WARMUP_SEQ_LEN_{key}");
    // parsed-by: parse_one_model
    let warmup_seq_len = if let Some(raw) = crate::env_parse::raw(&wsl_var) {
        // Per-model override set.
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("max") {
            // "max" or empty → None: pad to max_len at warmup.
            tracing::info!(model = %name, "per-model EMBED_WARMUP_SEQ_LEN=max");
            None
        } else {
            match trimmed.parse::<usize>() {
                Ok(0) => {
                    let fallback = global_warmup_seq_len.unwrap_or(max_len);
                    crate::env_parse::note_fallback(
                        &wsl_var,
                        trimmed,
                        "a positive integer or \"max\"",
                        &fallback,
                    );
                    Some(fallback)
                }
                Ok(n) => {
                    tracing::info!(
                        model = %name,
                        per_model_warmup_seq_len = n,
                        "per-model EMBED_WARMUP_SEQ_LEN override"
                    );
                    Some(n)
                }
                Err(_) => {
                    let fallback = global_warmup_seq_len.unwrap_or(max_len);
                    crate::env_parse::note_fallback(
                        &wsl_var,
                        trimmed,
                        "a positive integer or \"max\"",
                        &fallback,
                    );
                    Some(fallback)
                }
            }
        }
    } else {
        // No per-model override. Use global if set, else default to Some(max_len).
        // `global_warmup_seq_len` is `None` when global is unset OR was "max".
        // We distinguish the two cases by whether from_env passes None (unset)
        // vs explicitly passes None for "max". For simplicity: if global is
        // None (unset/default path via parse_models called without globals),
        // fall back to Some(max_len) — which is the correct default for each
        // model regardless of global.
        //
        // When called via parse_models_with_globals from from_env, the global
        // is already resolved. If it's Some(n), use it. If it's None it means
        // the operator set EMBED_WARMUP_SEQ_LEN=max globally — honour that.
        // But we need to distinguish "global unset → use max_len" from "global
        // = max keyword → None". We solve this by encoding "unset" differently:
        // from_env passes `parse_embed_warmup_seq_len(raw)` which returns
        // Some(128) when unset and None when "max". However, we want the
        // default here to be Some(max_len), not Some(128).
        //
        // Solution: from_env passes a sentinel `Option<Option<usize>>` wrapped
        // one level deeper so we can tell apart "unset" from "max". But since
        // from_env calls parse_models_with_globals, we use a simpler approach:
        // pass the raw global as-is. When `global_warmup_seq_len` is None here
        // it means the operator wrote "max" globally — all models get None.
        // When the operator left EMBED_WARMUP_SEQ_LEN unset, from_env sees
        // Some(128) from parse_embed_warmup_seq_len — but we want per-model
        // to default to Some(max_len), ignoring the 128 default.
        //
        // To implement this cleanly, from_env will pass `None` for the global
        // when the env var is UNSET, and `Some(n)` when explicitly set.
        // The sentinel for "unset" = use each model's max_len.
        // See `parse_models_with_globals_raw` for the real from_env call.
        global_warmup_seq_len.or(Some(max_len))
    };

    // Per-model memory_pattern: EMBED_MEMORY_PATTERN_<KEY>.
    // Default true (back-compat). Set to false for models with variable seq
    // shapes (e.g. jina-code-v2) to avoid BFCArena monotonic fill.
    let mp_var = format!("EMBED_MEMORY_PATTERN_{key}");
    // parsed-by: parse_memory_pattern
    let memory_pattern = parse_memory_pattern(&mp_var, crate::env_parse::raw(&mp_var).as_deref());
    tracing::info!(model = %name, memory_pattern, "session config");
    let input_prefix = crate::input_prefix::InputPrefix::parse(
        // env-strict: an unparseable value aborts startup (map_err below)
        env::var(format!("EMBED_INPUT_PREFIX_{key}"))
            .ok()
            .as_deref(),
    )
    .map_err(|e| format!("model '{name}': EMBED_INPUT_PREFIX_{key}: {e}"))?;
    tracing::info!(model = %name, ?input_prefix, "input prefix");

    // Optional 7th segment: ONNX filename inside `dir`.
    // Absent → "model_quantized.onnx" (back-compat default).
    // Present → validate: non-empty, no path separators, no ".." component.
    let onnx_filename = if parts.len() == 7 {
        let raw = parts[6].trim();
        if raw.is_empty() {
            return Err(format!(
                "model '{name}': onnx_filename (7th segment) is empty; \
                 omit the segment to use the default 'model_quantized.onnx'"
            ));
        }
        if raw.contains('/') || raw.contains('\\') {
            return Err(format!(
                "model '{name}': onnx_filename '{raw}' must be a plain filename, \
                 not a path (no '/' or '\\')"
            ));
        }
        if raw == ".." || raw.starts_with("../") || raw.starts_with("..\\") {
            return Err(format!(
                "model '{name}': onnx_filename '{raw}' contains a path-traversal component"
            ));
        }
        raw.to_string()
    } else {
        "model_quantized.onnx".to_string()
    };

    Ok(ModelDef {
        name,
        dir: parts[1].to_string(),
        onnx_filename,
        dim,
        max_len,
        pad_id,
        has_token_type_ids: has_tti,
        batch_max_seq,
        warmup_seq_len,
        memory_pattern,
        input_prefix,
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
fn parse_warmup_batch_sizes(key: &str, raw: Option<&str>, defaults: &[usize]) -> Vec<usize> {
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
            crate::env_parse::note_fallback(
                key,
                s,
                "comma-separated positive integers",
                &format!("{defaults:?}"),
            );
        }
        return defaults.to_vec();
    }

    if !bad_tokens.is_empty() {
        tracing::warn!(
            bad_tokens = ?bad_tokens,
            "{key} had unparseable or zero entries; warming with the valid ones"
        );
        crate::env_parse::note_fallback(
            key,
            s,
            "comma-separated positive integers",
            &format!("the valid entries {out:?}"),
        );
    }

    out
}

/// Parse `SPLADE_SESSION_POOL_SIZE`. Same shape as
/// `parse_reranker_pool_size`: default 1, `0` → fall back with a warn,
/// garbage → fall back. Kept as a separate function so the warn
/// message names the right env var (operators grep for the literal).
fn parse_splade_pool_size(raw: Option<&str>) -> usize {
    crate::env_parse::positive("SPLADE_SESSION_POOL_SIZE", raw, 1usize)
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
    fn batch_max_seq_default_is_256_when_unset() {
        assert_eq!(parse_batch_max_seq(None), 256);
    }

    #[test]
    fn batch_max_seq_parses_valid_positive_integer() {
        assert_eq!(parse_batch_max_seq(Some("128")), 128);
        assert_eq!(parse_batch_max_seq(Some("512")), 512);
        assert_eq!(parse_batch_max_seq(Some("  384  ")), 384);
    }

    #[test]
    fn batch_max_seq_falls_back_on_zero_and_garbage() {
        assert_eq!(parse_batch_max_seq(Some("0")), 256);
        assert_eq!(parse_batch_max_seq(Some("nope")), 256);
        assert_eq!(parse_batch_max_seq(Some("-1")), 256);
        assert_eq!(parse_batch_max_seq(Some("")), 256);
    }

    #[test]
    fn embed_warmup_seq_len_default_is_128_when_unset() {
        assert_eq!(parse_embed_warmup_seq_len(None), Some(128));
    }

    #[test]
    fn embed_warmup_seq_len_max_keyword_returns_none() {
        // `"max"` is the documented opt-out: pad warmup to model max_len.
        assert_eq!(parse_embed_warmup_seq_len(Some("max")), None);
        assert_eq!(parse_embed_warmup_seq_len(Some("MAX")), None);
        assert_eq!(parse_embed_warmup_seq_len(Some("  Max  ")), None);
    }

    #[test]
    fn embed_warmup_seq_len_parses_positive_integers() {
        assert_eq!(parse_embed_warmup_seq_len(Some("64")), Some(64));
        assert_eq!(parse_embed_warmup_seq_len(Some("256")), Some(256));
        assert_eq!(parse_embed_warmup_seq_len(Some("  192  ")), Some(192));
    }

    #[test]
    fn embed_warmup_seq_len_falls_back_on_invalid() {
        // 0, negatives, garbage, empty string → default 128.
        assert_eq!(parse_embed_warmup_seq_len(Some("0")), Some(128));
        assert_eq!(parse_embed_warmup_seq_len(Some("-5")), Some(128));
        assert_eq!(parse_embed_warmup_seq_len(Some("nope")), Some(128));
        assert_eq!(parse_embed_warmup_seq_len(Some("")), Some(128));
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
    // EMBED_SESSION_POOL_SIZE parser. Same contract as
    // RERANKER_SESSION_POOL_SIZE — default 1, `0` rejected, garbage
    // falls back to default.
    // -----------------------------------------------------------------

    #[test]
    fn embed_pool_size_default_is_1_when_unset() {
        assert_eq!(parse_embed_pool_size(None, None), 1);
    }

    #[test]
    fn embed_pool_size_parses_valid_positive_integer() {
        assert_eq!(parse_embed_pool_size(Some("1"), None), 1);
        assert_eq!(parse_embed_pool_size(Some("2"), None), 2);
        assert_eq!(parse_embed_pool_size(Some("4"), None), 4);
        // Trim whitespace, same as reranker parser.
        assert_eq!(parse_embed_pool_size(Some("  3  "), None), 3);
    }

    #[test]
    fn embed_pool_size_rejects_zero() {
        // `0` would `% 0` panic in `embed_tokens`.
        assert_eq!(parse_embed_pool_size(Some("0"), None), 1);
        assert_eq!(parse_embed_pool_size(Some("  0  "), None), 1);
    }

    #[test]
    fn embed_pool_size_falls_back_on_garbage() {
        assert_eq!(parse_embed_pool_size(Some("nope"), None), 1);
        assert_eq!(parse_embed_pool_size(Some(""), None), 1);
        assert_eq!(parse_embed_pool_size(Some("-1"), None), 1);
    }

    #[test]
    fn parse_embed_pool_size_source_key_does_not_affect_return_value() {
        // source_key only affects the warn message; return value must be
        // identical regardless of what key is provided.
        assert_eq!(
            parse_embed_pool_size(Some("3"), Some("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2")),
            3
        );
        // Zero rejected → default=1, regardless of source_key.
        assert_eq!(
            parse_embed_pool_size(Some("0"), Some("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2")),
            1
        );
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
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", None, &[1, 5]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", None, &[1]),
            vec![1]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", None, &[1, 8]),
            vec![1, 8]
        );
    }

    #[test]
    fn warmup_batch_sizes_parses_csv_in_order() {
        // Order matters — operator picks "compile this first" by ordering
        // the list. We do not sort.
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("1,5"), &[1, 5]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("5,1"), &[1, 5]),
            vec![5, 1]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("1,2,5,10"), &[1]),
            vec![1, 2, 5, 10]
        );
        // Whitespace tolerance — env quoting in compose files is fragile.
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("  1 , 5 "), &[1]),
            vec![1, 5]
        );
    }

    #[test]
    fn warmup_batch_sizes_dedups_preserving_order() {
        // Duplicates collapse to first occurrence — keeps total warmup
        // time bounded when an operator pastes "1,5,1" by accident.
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("3,5,3"), &[1]),
            vec![3, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("1,1,1"), &[1]),
            vec![1]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("1,5,1,5,8"), &[1]),
            vec![1, 5, 8]
        );
    }

    #[test]
    fn warmup_batch_sizes_drops_zero_and_negatives() {
        // 0 would `% 0` panic in any per-batch dispatch loop. Negatives
        // can't parse as usize. Both get dropped without falling back —
        // the rest of the list is still useful.
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("0,1,5"), &[1]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("1,-3,5"), &[1]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("0"), &[1, 5]),
            vec![1, 5],
            "all-zero CSV must fall back to defaults, not return empty"
        );
    }

    #[test]
    fn warmup_batch_sizes_skips_unparseable_keeps_valid() {
        // Single typo in the middle should not kill warmup for the other
        // shapes — drop the bad token, keep the rest.
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("1,nope,5"), &[1]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("garbage"), &[1, 5]),
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
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some(""), &[1, 5]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some("   "), &[1, 5]),
            vec![1, 5]
        );
        assert_eq!(
            parse_warmup_batch_sizes("T_WARMUP_BATCH_SIZES", Some(",,"), &[1, 5]),
            vec![1, 5]
        );
    }

    // -----------------------------------------------------------------
    // Test helper: construct a ModelDef with resolved per-model knobs
    // without going through env var lookups. Mirrors the logic in
    // parse_one_model but takes explicit arguments for isolation.
    fn resolve_per_model_knobs(
        name: &str,
        max_len: usize,
        global_batch_max_seq: usize,
        per_model_batch_max_seq: Option<usize>,
        // None = "no per-model override" (use global+max_len default)
        // Some(None) = "max" keyword
        // Some(Some(n)) = explicit value
        per_model_warmup_seq_len: Option<Option<usize>>,
        per_model_memory_pattern: bool,
    ) -> ModelDef {
        let batch_max_seq = per_model_batch_max_seq.unwrap_or(global_batch_max_seq);
        let warmup_seq_len = match per_model_warmup_seq_len {
            None => Some(max_len), // no per-model override → default = max_len
            Some(v) => v,          // "max" → None, explicit → Some(n)
        };
        ModelDef {
            name: name.to_string(),
            dir: "/models".to_string(),
            onnx_filename: "model_quantized.onnx".to_string(),
            dim: 768,
            max_len,
            pad_id: 0,
            has_token_type_ids: false,
            batch_max_seq,
            warmup_seq_len,
            memory_pattern: per_model_memory_pattern,
            input_prefix: Default::default(),
        }
    }

    // -----------------------------------------------------------------
    // E2/E3: per-model batch_max_seq and warmup_seq_len fields on
    // ModelDef, resolved via env overrides.
    //
    // Convention:
    //   BATCH_MAX_SEQ_<MODEL_UPPER>=N   — per-model override for seq cap
    //   EMBED_WARMUP_SEQ_LEN_<MODEL_UPPER>=N — per-model warmup seq len
    //   <MODEL_UPPER> = uppercase(name), '-' → '_'
    // -----------------------------------------------------------------

    #[test]
    fn model_def_batch_max_seq_defaults_to_global_when_no_per_model_override() {
        // With no BATCH_MAX_SEQ_MULTILINGUAL_E5_LARGE set, batch_max_seq
        // must equal the global BATCH_MAX_SEQ (256 default).
        let global = 256usize;
        let def = resolve_per_model_knobs(
            "multilingual-e5-large",
            256, // max_len
            global,
            None, // no per-model override
            None, // no per-model warmup override
            true, // memory_pattern default
        );
        assert_eq!(def.batch_max_seq, global);
    }

    #[test]
    fn model_def_batch_max_seq_per_model_override_takes_precedence() {
        // BATCH_MAX_SEQ_JINA_CODE_V2=384 overrides global=256 for jina.
        let def = resolve_per_model_knobs(
            "jina-code-v2",
            512,       // max_len
            256,       // global batch_max_seq
            Some(384), // per-model override
            None,
            true, // memory_pattern default
        );
        assert_eq!(def.batch_max_seq, 384);
    }

    #[test]
    fn model_def_warmup_seq_len_defaults_to_max_len() {
        // When no per-model EMBED_WARMUP_SEQ_LEN_* is set, warmup_seq_len
        // must default to Some(max_len) so warmup pads to the model's
        // full context window — avoids memory_pattern replan on first
        // long prod request.
        let def = resolve_per_model_knobs(
            "multilingual-e5-large",
            256, // max_len
            256, // global batch_max_seq
            None,
            None, // no per-model warmup override
            true, // memory_pattern default
        );
        assert_eq!(def.warmup_seq_len, Some(256));
    }

    #[test]
    fn model_def_warmup_seq_len_per_model_override_takes_precedence() {
        // EMBED_WARMUP_SEQ_LEN_JINA_CODE_V2=256 while global would give
        // Some(512) (=max_len) — operator can lower it.
        let def = resolve_per_model_knobs(
            "jina-code-v2",
            512, // max_len
            256, // global batch_max_seq
            None,
            Some(Some(256)), // per-model warmup = Some(256)
            true,            // memory_pattern default
        );
        assert_eq!(def.warmup_seq_len, Some(256));
    }

    #[test]
    fn model_def_warmup_seq_len_per_model_max_keyword_gives_none() {
        // EMBED_WARMUP_SEQ_LEN_JINA_CODE_V2=max → None (pad to max_len at
        // warmup time). This re-enables the "legacy max_len warmup" for one
        // specific model while keeping others at explicit seq lens.
        let def = resolve_per_model_knobs(
            "jina-code-v2",
            512,
            256,
            None,
            Some(None), // per-model warmup = None (= "max" keyword)
            true,       // memory_pattern default
        );
        assert_eq!(def.warmup_seq_len, None);
    }

    #[test]
    fn parse_models_propagates_per_model_fields() {
        // A basic round-trip: parse_models should populate batch_max_seq
        // and warmup_seq_len on every ModelDef. The per-model env vars are
        // not set in this process (test isolation), so both values should
        // be their defaults: batch_max_seq=global_default(256),
        // warmup_seq_len=Some(max_len).
        let defs = parse_models("multilingual-e5-large:/models:1024:256:1:false").unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].batch_max_seq, 256); // global default
        assert_eq!(defs[0].warmup_seq_len, Some(256)); // = max_len
        assert!(defs[0].memory_pattern); // default true

        let defs2 = parse_models("jina-code-v2:/models-jina:768:512:0:false").unwrap();
        assert_eq!(defs2[0].batch_max_seq, 256); // still global default
        assert_eq!(defs2[0].warmup_seq_len, Some(512)); // = max_len=512
        assert!(defs2[0].memory_pattern); // default true (no env var set in tests)
    }

    // -----------------------------------------------------------------
    // E3b: onnx_filename — optional 7th segment in EMBED_MODELS spec.
    //
    // Contract:
    //   6-segment spec   → onnx_filename == "model_quantized.onnx" (default)
    //   7-segment spec   → onnx_filename == the given filename
    //   traversal        → parse error
    //   empty 7th seg    → parse error
    //   8-segment spec   → parse error (too many fields)
    // -----------------------------------------------------------------

    #[test]
    fn parse_models_six_segment_defaults_onnx_filename() {
        // 6-segment spec must populate onnx_filename with the legacy default
        // so existing prod configs are byte-identically unchanged.
        let defs = parse_models("multilingual-e5-large:/models:1024:256:1:false").unwrap();
        assert_eq!(defs[0].onnx_filename, "model_quantized.onnx");
    }

    #[test]
    fn parse_models_seven_segment_uses_given_filename() {
        // 7th segment overrides the default — CodeRankEmbed canonical spec.
        let defs = parse_models("code-rank-embed:/models-coderank:768:512:0:false:model_int8.onnx")
            .unwrap();
        assert_eq!(defs[0].onnx_filename, "model_int8.onnx");
        // Other fields parsed normally.
        assert_eq!(defs[0].name, "code-rank-embed");
        assert_eq!(defs[0].dir, "/models-coderank");
        assert_eq!(defs[0].dim, 768);
        assert_eq!(defs[0].max_len, 512);
        assert_eq!(defs[0].pad_id, 0);
        assert!(!defs[0].has_token_type_ids);
    }

    #[test]
    fn parse_models_seven_segment_whitespace_trimmed() {
        // Surrounding whitespace on the 7th segment must be stripped
        // (consistent with other field handling).
        let defs = parse_models("m:/d:768:512:0:false:  model_int8.onnx  ").unwrap();
        assert_eq!(defs[0].onnx_filename, "model_int8.onnx");
    }

    #[test]
    fn parse_models_rejects_path_separator_in_filename() {
        // Filenames with '/' are rejected — the value is joined onto dir;
        // allowing separators would enable path traversal.
        let result = parse_models("m:/d:768:512:0:false:sub/model.onnx");
        let err = result.err().expect("should have been Err");
        assert!(err.contains("plain filename"), "unexpected error: {err}");
    }

    #[test]
    fn parse_models_rejects_dotdot_traversal() {
        // ".." is a path-traversal component — must be rejected even without
        // a separator, because `dir.join("..")` would escape the model dir.
        let result = parse_models("m:/d:768:512:0:false:..");
        let err = result.err().expect("should have been Err");
        assert!(err.contains("path-traversal"), "unexpected error: {err}");
    }

    #[test]
    fn parse_models_rejects_empty_seventh_segment() {
        // An empty 7th segment is ambiguous — reject with a helpful message
        // so the operator knows to omit the trailing colon instead.
        let result = parse_models("m:/d:768:512:0:false:");
        let err = result.err().expect("should have been Err");
        assert!(err.contains("empty"), "unexpected error: {err}");
    }

    #[test]
    fn parse_models_rejects_eight_segments() {
        // 8 segments is always an error (format: name:dir:dim:max_len:pad_id:has_tti[:onnx_filename]).
        let result = parse_models("m:/d:768:512:0:false:model.onnx:extra");
        let err = result.err().expect("should have been Err");
        assert!(err.contains("6 or 7"), "unexpected error: {err}");
    }

    #[test]
    fn parse_models_five_segments_still_rejected() {
        // Fewer than 6 fields remain an error (unchanged pre-existing behaviour).
        let result = parse_models("m:/d:768:512:0");
        let err = result.err().expect("should have been Err");
        assert!(err.contains("6 or 7"), "unexpected error: {err}");
    }

    // -----------------------------------------------------------------
    // E4: per-model memory_pattern knob
    //
    // Convention:
    //   EMBED_MEMORY_PATTERN_<MODEL_UPPER>=true|false
    //   <MODEL_UPPER> = uppercase(name), '-' → '_'
    //
    // Default: true (back-compat — existing behaviour for all models).
    // False: disable memory_pattern for models with variable seq shapes
    // (jina-code-v2) where BFCArena fills monotonically under the plan.
    // -----------------------------------------------------------------

    // Prefixing changes every vector, so it must be opt-in per model.
    #[test]
    fn parse_models_input_prefix_defaults_off() {
        let defs = parse_models("e5-no-prefix-env:/m:1024:256:1:false").unwrap();
        assert_eq!(
            defs[0].input_prefix,
            crate::input_prefix::InputPrefix::Off,
            "a model without EMBED_INPUT_PREFIX_<MODEL> must not be prefixed"
        );
    }

    #[test]
    fn parse_per_model_memory_pattern_default_true() {
        // Unset → true (backwards-compatible default).
        assert!(parse_memory_pattern("EMBED_MEMORY_PATTERN_T", None));
        assert!(parse_memory_pattern("EMBED_MEMORY_PATTERN_T", Some("")));
    }

    #[test]
    fn parse_per_model_memory_pattern_env_false() {
        // EMBED_MEMORY_PATTERN_JINA_CODE_V2=false → false for jina.
        // e5-large has no override → stays true.
        assert!(!parse_memory_pattern(
            "EMBED_MEMORY_PATTERN_T",
            Some("false")
        ));
        assert!(!parse_memory_pattern("EMBED_MEMORY_PATTERN_T", Some("0")));
        // e5-large (no per-model env) defaults to true.
        assert!(parse_memory_pattern("EMBED_MEMORY_PATTERN_T", None));
    }

    #[test]
    fn parse_per_model_memory_pattern_invalid_value_defaults_true() {
        // Garbage → warn + default true (same fallback posture as other
        // bool knobs in this codebase).
        assert!(parse_memory_pattern(
            "EMBED_MEMORY_PATTERN_T",
            Some("garbage")
        ));
        assert!(parse_memory_pattern("EMBED_MEMORY_PATTERN_T", Some("nope")));
        assert!(parse_memory_pattern("EMBED_MEMORY_PATTERN_T", Some("2")));
    }

    #[test]
    fn model_def_memory_pattern_env_key_uses_screaming_snake_case() {
        // Key generation: uppercase + '-' → '_'. Tests that the correct
        // env-var name is derived from the model name so env var lookups
        // resolve to the right key (no dashes in env vars).
        //
        // We verify indirectly via resolve_per_model_knobs: passing
        // per_model_memory_pattern=false should land in model_def.
        let def = resolve_per_model_knobs(
            "jina-code-v2",
            512,
            256,
            None,
            None,
            false, // per_model_memory_pattern: false (override)
        );
        assert!(!def.memory_pattern);

        // e5-large with default true.
        let def2 = resolve_per_model_knobs(
            "multilingual-e5-large",
            256,
            256,
            None,
            None,
            true, // per_model_memory_pattern: true (default)
        );
        assert!(def2.memory_pattern);
    }

    // -----------------------------------------------------------------
    // EMBED_MULTI_PROCESS flag parser — pure function, no env mutation.
    // -----------------------------------------------------------------

    #[test]
    fn parse_multi_process_flag_cases() {
        use super::parse_multi_process_flag;
        // Default off when unset.
        assert!(!parse_multi_process_flag(None));
        // Explicit off values.
        assert!(!parse_multi_process_flag(Some("0")));
        assert!(!parse_multi_process_flag(Some("false")));
        assert!(!parse_multi_process_flag(Some("")));
        // On values.
        assert!(parse_multi_process_flag(Some("1")));
        assert!(parse_multi_process_flag(Some("true")));
        assert!(parse_multi_process_flag(Some("True")));
        assert!(parse_multi_process_flag(Some("TRUE")));
        // Unrecognized values must NOT activate multi-process (typo guard).
        assert!(!parse_multi_process_flag(Some("yes")));
        assert!(!parse_multi_process_flag(Some("on")));
    }

    // -----------------------------------------------------------------
    // model_env_key helper — shared by pool-size + arena override paths.
    // -----------------------------------------------------------------

    #[test]
    fn model_env_key_converts_dashes_to_underscores_and_uppercases() {
        assert_eq!(super::model_env_key("jina-code-v2"), "JINA_CODE_V2");
        assert_eq!(
            super::model_env_key("multilingual-e5-large"),
            "MULTILINGUAL_E5_LARGE"
        );
        assert_eq!(super::model_env_key("e5"), "E5");
    }

    /// Regression: all sites that previously used inline
    /// `to_uppercase().replace('-', "_")` must produce the same result as
    /// `model_env_key()` for an edge-case model name with multiple dashes.
    /// This guards against future callers accidentally reimplementing the
    /// transform with different semantics.
    #[test]
    fn model_env_key_consistent_for_edge_case_model_name() {
        let name = "gte-multi-rerank";
        let via_fn = super::model_env_key(name);
        // Verify it matches the transform that pool-size and arena env vars use.
        let pool_key = format!("EMBED_SESSION_POOL_SIZE_{via_fn}");
        let arena_key = format!("EMBED_MEMORY_PATTERN_{via_fn}");
        assert_eq!(via_fn, "GTE_MULTI_RERANK");
        assert_eq!(pool_key, "EMBED_SESSION_POOL_SIZE_GTE_MULTI_RERANK");
        assert_eq!(arena_key, "EMBED_MEMORY_PATTERN_GTE_MULTI_RERANK");
    }

    // -----------------------------------------------------------------
    // EMBED_SESSION_POOL_SIZE_<MODEL_KEY_UPPER> per-model override.
    //
    // resolve_embed_pool_size_for_model(model_key, global_raw) must:
    //   - prefer per-model env over global
    //   - fall back to global when per-model absent
    //   - fall back to default (1) when both absent
    //   - reject 0 with warn + default, identical to parse_embed_pool_size
    // -----------------------------------------------------------------

    #[test]
    fn resolve_embed_pool_size_for_model_default_when_nothing_set() {
        // Neither per-model nor global set → default 1.
        let size = super::resolve_embed_pool_size_for_model("jina-code-v2", None);
        assert_eq!(size, 1);
    }

    #[test]
    fn resolve_embed_pool_size_for_model_uses_global_when_no_per_model() {
        // Global set, no per-model → global wins.
        let size = super::resolve_embed_pool_size_for_model("jina-code-v2", Some("3"));
        assert_eq!(size, 3);
    }

    #[test]
    fn resolve_embed_pool_size_for_model_per_model_wins_over_global() {
        // EMBED_SESSION_POOL_SIZE_JINA_CODE_V2=1 must beat global=3.
        unsafe {
            std::env::set_var("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2", "1");
        }
        let size = super::resolve_embed_pool_size_for_model("jina-code-v2", Some("3"));
        unsafe {
            std::env::remove_var("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2");
        }
        assert_eq!(size, 1);
    }

    #[test]
    fn resolve_embed_pool_size_for_model_per_model_does_not_affect_other_models() {
        // Jina override must not bleed into e5-large.
        unsafe {
            std::env::set_var("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2", "1");
        }
        let size = super::resolve_embed_pool_size_for_model("multilingual-e5-large", Some("2"));
        unsafe {
            std::env::remove_var("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2");
        }
        assert_eq!(size, 2); // e5 gets global, not jina override
    }

    #[test]
    fn resolve_embed_pool_size_for_model_rejects_zero_same_as_global_parser() {
        unsafe {
            std::env::set_var("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2", "0");
        }
        let size = super::resolve_embed_pool_size_for_model("jina-code-v2", None);
        unsafe {
            std::env::remove_var("EMBED_SESSION_POOL_SIZE_JINA_CODE_V2");
        }
        assert_eq!(size, 1); // 0 rejected → default
    }

    // ── ready_probe_timeout_ms parsing ───────────────────────────────────────

    fn parse_ready_probe_timeout(raw: Option<&str>) -> u64 {
        match raw {
            Some(s) => match s.trim().parse::<u64>() {
                Ok(0) => 2000,
                Ok(v) => v,
                Err(_) => 2000,
            },
            None => 2000,
        }
    }

    #[test]
    fn ready_probe_timeout_default_when_unset() {
        assert_eq!(parse_ready_probe_timeout(None), 2000);
    }

    #[test]
    fn ready_probe_timeout_parses_valid_value() {
        assert_eq!(parse_ready_probe_timeout(Some("5000")), 5000);
        assert_eq!(parse_ready_probe_timeout(Some("  1000  ")), 1000);
    }

    #[test]
    fn ready_probe_timeout_zero_falls_back() {
        assert_eq!(parse_ready_probe_timeout(Some("0")), 2000);
    }

    #[test]
    fn ready_probe_timeout_garbage_falls_back() {
        assert_eq!(parse_ready_probe_timeout(Some("not-a-number")), 2000);
        assert_eq!(parse_ready_probe_timeout(Some("")), 2000);
    }

    // -----------------------------------------------------------------
    // IPC request deadline — resolve_ipc_timeout(model).
    //   - per-model EMBED_IPC_TIMEOUT_SECS_<KEY> beats global
    //   - global beats the 60 s default
    //   - 0 / garbage at any level falls through to the next
    // -----------------------------------------------------------------

    /// Snapshot + restore the two env vars the resolver reads. Every test
    /// below must run under this or a parallel `cargo test` run bleeds one
    /// test's override into another's "unset" expectation.
    fn with_ipc_timeout_env<R>(
        per_model: Option<(&str, &str)>,
        global: Option<&str>,
        f: impl FnOnce() -> R,
    ) -> R {
        let per_key =
            per_model.map(|(m, _)| format!("EMBED_IPC_TIMEOUT_SECS_{}", model_env_key(m)));
        let per_val = per_model.map(|(_, v)| v);
        let prev_per = per_key.as_ref().map(|k| (k.clone(), env::var(k).ok()));
        let prev_global = env::var("EMBED_IPC_TIMEOUT_SECS").ok();
        unsafe {
            if let Some(k) = &per_key {
                env::remove_var(k);
            }
            env::remove_var("EMBED_IPC_TIMEOUT_SECS");
            if let (Some(k), Some(v)) = (&per_key, per_val) {
                env::set_var(k, v);
            }
            if let Some(v) = global {
                env::set_var("EMBED_IPC_TIMEOUT_SECS", v);
            }
        }
        let r = f();
        unsafe {
            match prev_per {
                Some((k, Some(v))) => env::set_var(&k, v),
                Some((k, None)) => env::remove_var(&k),
                None => {}
            }
            match prev_global {
                Some(v) => env::set_var("EMBED_IPC_TIMEOUT_SECS", v),
                None => env::remove_var("EMBED_IPC_TIMEOUT_SECS"),
            }
        }
        r
    }

    #[test]
    #[serial_test::serial]
    fn ipc_timeout_default_is_60s() {
        let d = with_ipc_timeout_env(None, None, || resolve_ipc_timeout("any-model"));
        assert_eq!(d.as_secs(), DEFAULT_IPC_TIMEOUT_SECS);
    }

    #[test]
    #[serial_test::serial]
    fn ipc_timeout_global_override() {
        let d = with_ipc_timeout_env(None, Some("120"), || resolve_ipc_timeout("any-model"));
        assert_eq!(d.as_secs(), 120);
    }

    #[test]
    #[serial_test::serial]
    fn ipc_timeout_per_model_beats_global() {
        let d = with_ipc_timeout_env(Some(("jina-code-v2", "45")), Some("120"), || {
            resolve_ipc_timeout("jina-code-v2")
        });
        assert_eq!(d.as_secs(), 45);
    }

    #[test]
    #[serial_test::serial]
    fn ipc_timeout_per_model_does_not_leak_to_other_models() {
        let d = with_ipc_timeout_env(Some(("jina-code-v2", "45")), None, || {
            resolve_ipc_timeout("multilingual-e5-large")
        });
        assert_eq!(d.as_secs(), DEFAULT_IPC_TIMEOUT_SECS);
    }

    #[test]
    #[serial_test::serial]
    fn ipc_timeout_zero_falls_through_to_default() {
        // A 0 s deadline would fail every request instantly — invalid, not honoured.
        let d = with_ipc_timeout_env(Some(("jina-code-v2", "0")), Some("0"), || {
            resolve_ipc_timeout("jina-code-v2")
        });
        assert_eq!(d.as_secs(), DEFAULT_IPC_TIMEOUT_SECS);
    }

    #[test]
    #[serial_test::serial]
    fn ipc_timeout_garbage_falls_through_to_global() {
        let d = with_ipc_timeout_env(Some(("jina-code-v2", "soon")), Some("90"), || {
            resolve_ipc_timeout("jina-code-v2")
        });
        assert_eq!(d.as_secs(), 90);
    }

    // -----------------------------------------------------------------
    // Per-model embed concurrency cap — resolve_embed_concurrency_for_model.
    //   - per-model MAX_CONCURRENT_EMBED_REQUESTS_<KEY> beats global
    //   - global beats the pool × intra × 8 (floor 16) formula
    //   - 0 disables (None); garbage falls through
    // -----------------------------------------------------------------

    #[test]
    #[serial_test::serial]
    fn embed_concurrency_formula_default() {
        // pool 2 × intra 2 × 8 = 32 — same headroom rule as the rerank cap.
        unsafe { env::remove_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL") };
        let cap = resolve_embed_concurrency_for_model("test-mdl", None, Some("2"), 2);
        assert_eq!(cap, Some(32));
    }

    #[test]
    #[serial_test::serial]
    fn embed_concurrency_floor_is_16() {
        unsafe { env::remove_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL") };
        let cap = resolve_embed_concurrency_for_model("test-mdl", None, Some("1"), 1);
        assert_eq!(cap, Some(16));
    }

    #[test]
    #[serial_test::serial]
    fn embed_concurrency_global_override() {
        unsafe { env::remove_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL") };
        let cap = resolve_embed_concurrency_for_model("test-mdl", Some("64"), Some("2"), 2);
        assert_eq!(cap, Some(64));
    }

    #[test]
    #[serial_test::serial]
    fn embed_concurrency_global_zero_disables() {
        unsafe { env::remove_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL") };
        let cap = resolve_embed_concurrency_for_model("test-mdl", Some("0"), Some("2"), 2);
        assert_eq!(cap, None);
    }

    #[test]
    #[serial_test::serial]
    fn embed_concurrency_per_model_overrides_global() {
        let prev = env::var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL").ok();
        unsafe { env::set_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL", "7") };
        let cap = resolve_embed_concurrency_for_model("test-mdl", Some("64"), Some("2"), 2);
        unsafe {
            match prev {
                Some(v) => env::set_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL", v),
                None => env::remove_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL"),
            }
        }
        assert_eq!(cap, Some(7));
    }

    #[test]
    #[serial_test::serial]
    fn embed_concurrency_per_model_zero_disables_only_that_model() {
        let prev = env::var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL").ok();
        unsafe { env::set_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL", "0") };
        let cap = resolve_embed_concurrency_for_model("test-mdl", Some("64"), Some("2"), 2);
        let other = resolve_embed_concurrency_for_model("other-mdl", Some("64"), Some("2"), 2);
        unsafe {
            match prev {
                Some(v) => env::set_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL", v),
                None => env::remove_var("MAX_CONCURRENT_EMBED_REQUESTS_TEST_MDL"),
            }
        }
        assert_eq!(cap, None, "per-model 0 disables that model");
        assert_eq!(other, Some(64), "global cap still applies to other models");
    }
}

// ── config-reachability detector (issue #167) ───────────────────────────────
//
// Three times a `Config` field was parsed, range-validated, published to a
// gauge — and then discarded at the call site, so the knob did nothing in
// production. Each was found by accident, months apart:
//
//   1. BATCH_MAX / BATCH_MAX_TOKENS / BATCH_WAIT_MS / MAX_QUEUE_SIZE /
//      BATCH_LENGTH_RATIO_THRESHOLD / RERANKER_BATCH_MAX — dead because
//      `main.rs` sets `batcher: None` in the multi-process branch.
//   2. EMBED_IDLE_EVICT_SECS — `StandaloneEmbedder::load` passed a literal `0`.
//   3. AUTO_TRUNCATE — the same function passed a literal `false`.
//
// All three took `&cfg` as a parameter and ignored the field. Production runs
// `EMBED_MULTI_PROCESS=1`, so the worker path (`StandaloneEmbedder::load` →
// `EmbedModel::load`) is the one that matters.
//
// This module is a self-checking detector for that bug class. It derives the
// ground-truth field set from `struct Config` itself (parsed with `syn`), then
// enforces two invariants:
//
//   INV-1 (orphan knob) — every `pub` field of `Config` must be read as
//     `<receiver>.<field>` somewhere under `src/`. A field that only appears
//     in the `Config::from_env` struct literal (parsed but never consumed) is
//     an orphan. This is what catches a brand-new field that nobody wired up —
//     without a hand-maintained allowlist, which is the whole failure mode (a
//     list cannot detect a field missing from itself).
//
//   INV-2 (worker-path discard) — `EmbedModel::load` takes several scalar
//     parameters whose names match `Config` fields (`auto_truncate`,
//     `idle_evict_secs`, …). `StandaloneEmbedder::load` receives `&cfg`, so
//     every such parameter that is NOT already a parameter of
//     `StandaloneEmbedder::load` itself MUST be passed as `cfg.<field>` — not
//     a literal. Passing a literal is the exact regression that shipped for
//     `auto_truncate` and `idle_evict_secs`.
//
// What this detector FAILS to catch (documented limits):
//   - Fields read through an aliased receiver (`let c = &cfg; c.field`) — the
//     scan keys on the identifier `cfg`. This codebase uses `cfg.` uniformly,
//     so the limit is theoretical; a future alias would need a type-aware
//     analysis (clippy/flow) this AST walk is not.
//   - Fields consumed only through a getter method / trait — not present here.
//   - Rerank and SPLADE worker paths (`StandaloneReranker::load`,
//     `StandaloneSplade::load`) — follow-up, noted not built. INV-1 still
//     covers them for orphan detection; INV-2 is embed-only.
//   - A field whose value flows into `EmbedModel::load` correctly but is then
//     ignored INSIDE `EmbedModel::load` — that is a deeper invariant, out of
//     scope; this detector guards the call-site plumbing, not the callee body.
#[cfg(test)]
mod config_reachability {
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    use proc_macro2::{TokenStream, TokenTree};
    use syn::visit::{self, Visit};
    use syn::{
        Expr, ExprCall, ExprField, ExprLit, File, FnArg, ImplItem, ImplItemFn, Item, Macro, Member,
        Pat, Type, Visibility,
    };

    /// Crate `src/` directory, resolved from the compile-time manifest path so
    /// the test works regardless of the runtime working directory.
    fn src_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
    }

    fn parse_rel(rel: &str) -> File {
        let text = std::fs::read_to_string(src_dir().join(rel))
            .unwrap_or_else(|e| panic!("read src/{rel}: {e}"));
        syn::parse_file(&text).unwrap_or_else(|e| panic!("parse src/{rel}: {e}"))
    }

    /// True iff `self_ty` is a plain path type whose final segment is `name`
    /// (matches both `impl EmbedModel` and `impl Drop for EmbedModel`).
    fn type_is(self_ty: &Type, name: &str) -> bool {
        match self_ty {
            Type::Path(tp) => tp
                .path
                .segments
                .last()
                .map(|s| s.ident == name)
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Names of the typed (non-`self`) parameters of a method signature.
    fn sig_param_names(sig: &syn::Signature) -> Vec<String> {
        sig.inputs
            .iter()
            .filter_map(|a| match a {
                FnArg::Typed(t) => match &*t.pat {
                    Pat::Ident(pi) => Some(pi.ident.to_string()),
                    _ => None,
                },
                FnArg::Receiver(_) => None,
            })
            .collect()
    }

    /// The `fn` named `method` inside any `impl <type_name>` block.
    fn find_impl_fn<'a>(file: &'a File, type_name: &str, method: &str) -> &'a ImplItemFn {
        for item in &file.items {
            if let Item::Impl(i) = item
                && type_is(&i.self_ty, type_name)
            {
                for ii in &i.items {
                    if let ImplItem::Fn(f) = ii
                        && f.sig.ident == method
                    {
                        return f;
                    }
                }
            }
        }
        panic!("fn `{method}` not found in any `impl {type_name}`");
    }

    /// All `pub` named fields of `struct Config` in `src/config.rs`.
    /// This is the ground truth — derived from the struct, not hand-maintained.
    fn config_field_names() -> Vec<String> {
        let file = parse_rel("config.rs");
        for item in &file.items {
            if let Item::Struct(s) = item
                && s.ident == "Config"
            {
                return s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        let is_pub = matches!(f.vis, Visibility::Public(_));
                        // Skip tuple/unnamed fields — Config has none, but
                        // stay robust if one is ever added.
                        is_pub
                            .then_some(())
                            .and(f.ident.as_ref())
                            .map(|i| i.to_string())
                    })
                    .collect();
            }
        }
        panic!("struct Config not found in src/config.rs");
    }

    /// Visitor collecting every `<receiver>.<field>` member access where the
    /// receiver is the single identifier `cfg` (no leading `::`). This is the
    /// set of Config fields actually *read* by application code. The
    /// `Config::from_env` struct literal uses field-init syntax (not a member
    /// access), so construction does not count as a read — exactly what we
    /// want: a field parsed but never consumed is invisible here.
    ///
    /// Macro bodies (`format!`, `tracing::info!`, …) are opaque to the AST
    /// visitor, so `cfg.port` inside `format!("…", cfg.port)` would be missed.
    /// `visit_macro` therefore scans each macro's token stream at the token
    /// level for the `cfg . <ident>` pattern, recursing into nested groups.
    struct CfgReads {
        reads: HashSet<String>,
    }

    /// Scan a token stream for `cfg . <ident>` member accesses, recursing into
    /// nested groups (e.g. `format!("{}", inner!(cfg.x))`).
    fn scan_tokens(ts: &TokenStream, reads: &mut HashSet<String>) {
        let mut iter = ts.clone().into_iter().peekable();
        while let Some(tt) = iter.next() {
            match tt {
                TokenTree::Ident(ident) if ident == "cfg" => {
                    if matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '.') {
                        iter.next(); // consume '.'
                        if let Some(TokenTree::Ident(field)) = iter.peek() {
                            reads.insert(field.to_string());
                            let _ = iter.next(); // consume the field ident
                        }
                    }
                }
                TokenTree::Group(g) => scan_tokens(&g.stream(), reads),
                _ => {}
            }
        }
    }

    impl<'ast> Visit<'ast> for CfgReads {
        fn visit_expr_field(&mut self, i: &'ast ExprField) {
            if let Expr::Path(ep) = &*i.base
                && ep.path.leading_colon.is_none()
                && ep.path.segments.len() == 1
                && ep.path.segments[0].ident == "cfg"
                && let Member::Named(m) = &i.member
            {
                self.reads.insert(m.to_string());
            }
            visit::visit_expr_field(self, i);
        }

        fn visit_macro(&mut self, i: &'ast Macro) {
            scan_tokens(&i.tokens, &mut self.reads);
            visit::visit_macro(self, i);
        }
    }

    /// Walk every `.rs` file under `src/` and collect the union of `cfg.<field>`
    /// reads. Files that fail to parse (e.g. generated/proc-macro output) are
    /// skipped rather than fatal — a parse failure here must not mask a real
    /// orphan.
    fn all_cfg_reads() -> HashSet<String> {
        let mut reads = HashSet::new();
        walk_rs(&src_dir(), &mut reads);
        reads
    }

    fn walk_rs(dir: &Path, reads: &mut HashSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_rs(&path, reads);
            } else if path.extension().is_some_and(|e| e == "rs")
                && let Ok(text) = std::fs::read_to_string(&path)
                && let Ok(file) = syn::parse_file(&text)
            {
                let mut v = CfgReads {
                    reads: HashSet::new(),
                };
                v.visit_file(&file);
                reads.extend(v.reads);
            }
        }
    }

    /// Visitor collecting every `EmbedModel::load(...)` call's argument
    /// expressions, scoped to the function body it is run over.
    struct LoadCalls<'a> {
        calls: Vec<Vec<&'a Expr>>,
    }

    impl<'ast> Visit<'ast> for LoadCalls<'ast> {
        fn visit_expr_call(&mut self, i: &'ast ExprCall) {
            if let Expr::Path(ep) = &*i.func {
                let segs: Vec<String> = ep
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                if segs.as_slice() == ["EmbedModel", "load"] {
                    self.calls.push(i.args.iter().collect());
                }
            }
            visit::visit_expr_call(self, i);
        }
    }

    /// True iff `arg` is the member access `cfg.<field>`.
    fn is_cfg_field_access(arg: &Expr, field: &str) -> bool {
        if let Expr::Field(ef) = arg
            && let Expr::Path(ep) = &*ef.base
            && ep.path.leading_colon.is_none()
            && ep.path.segments.len() == 1
            && ep.path.segments[0].ident == "cfg"
            && let Member::Named(m) = &ef.member
        {
            return m == field;
        }
        false
    }

    /// Render an argument expression for the failure message: name it for what
    /// it is (literal / cfg-field / other) so the regression is obvious.
    fn describe_arg(arg: &Expr) -> String {
        match arg {
            Expr::Lit(ExprLit { lit, .. }) => format!("literal `{:?}`", lit),
            Expr::Field(ef) => {
                if let Expr::Path(ep) = &*ef.base
                    && ep.path.segments.len() == 1
                    && let Member::Named(m) = &ef.member
                {
                    return format!("{}.{m}", ep.path.segments[0].ident);
                }
                "field access".to_string()
            }
            other => format!("{:?}", other),
        }
    }

    /// The set of `EmbedModel::load` parameters that MUST be forwarded from
    /// `cfg` by `StandaloneEmbedder::load`: a parameter is in this set when its
    /// name matches a `Config` field AND it is not already a parameter of
    /// `StandaloneEmbedder::load` (the latter are sourced separately on the
    /// worker, e.g. `intra_threads` comes from `EMBED_WORKER_INTRA_THREADS`).
    fn must_flow_from_cfg(
        embed_params: &[String],
        standalone_params: &[String],
        config_fields: &HashSet<String>,
    ) -> Vec<String> {
        let standalone: HashSet<&str> = standalone_params.iter().map(|s| s.as_str()).collect();
        embed_params
            .iter()
            .filter(|p| config_fields.contains(p.as_str()) && !standalone.contains(p.as_str()))
            .cloned()
            .collect()
    }

    #[test]
    fn every_config_field_is_read_somewhere() {
        // INV-1: no orphan knobs. A field present in `struct Config` but never
        // read as `cfg.<field>` anywhere under `src/` was parsed and then
        // discarded — the bug class. This is what catches a brand-new field
        // that nobody told the detector about.
        let fields = config_field_names();
        let reads = all_cfg_reads();
        let orphan: Vec<&String> = fields
            .iter()
            .filter(|f| !reads.contains(f.as_str()))
            .collect();
        assert!(
            orphan.is_empty(),
            "Config fields with no `cfg.<field>` read anywhere under src/: {orphan:?}.\n\
             A field parsed in `Config::from_env` but never consumed is a dead knob —\n\
             the exact bug class of issue #167. Either wire the field into a consumer\n\
             or remove it from `Config`."
        );
    }

    #[test]
    fn standalone_embedder_load_forwards_cfg_fields_to_embed_model_load() {
        // INV-2: the worker-path discard regression. `StandaloneEmbedder::load`
        // takes `&cfg` and calls `EmbedModel::load`; any `EmbedModel::load`
        // parameter named like a `Config` field (and not already a parameter of
        // `StandaloneEmbedder::load`) must be forwarded as `cfg.<field>`, not a
        // literal. Passing `false`/`0` here is what shipped for `auto_truncate`
        // and `idle_evict_secs`.
        let model = parse_rel("model.rs");
        let embed_load = find_impl_fn(&model, "EmbedModel", "load");
        let standalone_load = find_impl_fn(&model, "StandaloneEmbedder", "load");

        let embed_params = sig_param_names(&embed_load.sig);
        let standalone_params = sig_param_names(&standalone_load.sig);
        let config_fields: HashSet<String> = config_field_names().into_iter().collect();
        let must_flow = must_flow_from_cfg(&embed_params, &standalone_params, &config_fields);

        // Collect the `EmbedModel::load(...)` call(s) inside the worker loader.
        let mut visitor = LoadCalls { calls: Vec::new() };
        visitor.visit_block(&standalone_load.block);
        assert_eq!(
            visitor.calls.len(),
            1,
            "expected exactly one `EmbedModel::load` call inside \
             `StandaloneEmbedder::load`, found {}",
            visitor.calls.len()
        );
        let call_args = &visitor.calls[0];

        for (i, param) in embed_params.iter().enumerate() {
            if !must_flow.contains(param) {
                continue;
            }
            let arg = call_args.get(i).unwrap_or_else(|| {
                panic!(
                    "`EmbedModel::load` call has {} args but parameter `{param}` is at \
                     position {i}",
                    call_args.len()
                );
            });
            assert!(
                is_cfg_field_access(arg, param),
                "`StandaloneEmbedder::load` passes {} for `EmbedModel::load` parameter \
                 `{param}` (arg position {i}); expected `cfg.{param}`.\n\
                 The `Config` field `{param}` is parsed but discarded on the worker path —\n\
                 the dead-knob bug class (issue #167). Production runs \
                 `EMBED_MULTI_PROCESS=1`, so the worker loader is the only call site that\n\
                 matters. Forward `cfg.{param}` instead of a literal.",
                describe_arg(arg)
            );
        }
    }
}

#[cfg(test)]
mod env_fallback_tests {
    use super::*;

    // ── Garbage is noted: one assertion per parse site ──────────────────────
    //
    // Each pure parser below used to fall back with at best a `warn!` and no
    // counter; `embed_config_fallback_total` read 0 while a typo ran on the
    // default (#151). The table pins, per parser, that garbage keeps the
    // default AND increments the counter under the variable's own name.

    fn counted(f: impl FnOnce()) -> String {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, f);
        handle.render()
    }

    #[test]
    fn every_parser_counts_garbage_and_keeps_its_default() {
        type Case = (&'static str, Box<dyn Fn() -> String>, String);
        let cases: Vec<Case> = vec![
            (
                "CACHE_MAX_ENTRIES",
                Box::new(|| parse_cache_max_entries(Some("lots")).to_string()),
                "10000".into(),
            ),
            (
                "TOKEN_CACHE_MAX_ENTRIES",
                Box::new(|| parse_token_cache_max_entries(Some("lots")).to_string()),
                "0".into(),
            ),
            (
                "BATCH_MAX_TOKENS",
                Box::new(|| parse_batch_max_tokens(Some("x")).to_string()),
                "16384".into(),
            ),
            (
                "BATCH_MAX_SEQ",
                Box::new(|| parse_batch_max_seq(Some("0")).to_string()),
                "256".into(),
            ),
            (
                "EMBED_WARMUP_SEQ_LEN",
                Box::new(|| format!("{:?}", parse_embed_warmup_seq_len(Some("x")))),
                "Some(128)".into(),
            ),
            (
                "EMBED_SESSION_POOL_SIZE",
                Box::new(|| parse_embed_pool_size(Some("x"), None).to_string()),
                "1".into(),
            ),
            (
                "RERANKER_SESSION_POOL_SIZE",
                Box::new(|| parse_reranker_pool_size(Some("x")).to_string()),
                "1".into(),
            ),
            (
                "SPLADE_SESSION_POOL_SIZE",
                Box::new(|| parse_splade_pool_size(Some("0")).to_string()),
                "1".into(),
            ),
            (
                "EMBED_WARMUP_BATCH_SIZES",
                Box::new(|| {
                    format!(
                        "{:?}",
                        parse_warmup_batch_sizes("EMBED_WARMUP_BATCH_SIZES", Some("x,y"), &[1, 8])
                    )
                }),
                "[1, 8]".into(),
            ),
            (
                "EMBED_MULTI_PROCESS",
                Box::new(|| parse_multi_process_flag(Some("yes")).to_string()),
                "false".into(),
            ),
            (
                "EMBED_MEMORY_PATTERN_X",
                Box::new(|| {
                    parse_memory_pattern("EMBED_MEMORY_PATTERN_X", Some("maybe")).to_string()
                }),
                "true".into(),
            ),
        ];
        for (key, run, want) in &cases {
            let mut got = String::new();
            let out = counted(|| got = run());
            assert_eq!(&got, want, "{key}: garbage must keep the default");
            assert!(
                out.contains(&format!(
                    "embed_config_fallback_total{{env_var=\"{key}\"}} 1"
                )),
                "{key}: a garbage value must be counted in embed_config_fallback_total:\n{out}"
            );
        }
    }

    /// The warmup-batch-size list keeps its valid neighbours of a bad token:
    /// that partial fallback is noted too, not only the all-bad one.
    #[test]
    fn warmup_batch_sizes_partial_garbage_is_counted() {
        let mut got = vec![];
        let out = counted(|| got = parse_warmup_batch_sizes("W", Some("1,x,5"), &[2]));
        assert_eq!(got, vec![1, 5]);
        assert!(
            out.contains("embed_config_fallback_total{env_var=\"W\"} 1"),
            "{out}"
        );
    }

    /// A blank value (empty line in an env file, `KEY=" "`) means unset: the
    /// hand-rolled resolvers must not count it as a typo, and a blank
    /// per-model value must fall through to the global exactly like unset.
    #[test]
    #[serial_test::serial]
    fn blank_values_are_unset_not_garbage() {
        const PM_IPC: &str = "EMBED_IPC_TIMEOUT_SECS_BLANK_MODEL";
        const PM_CAP: &str = "MAX_CONCURRENT_EMBED_REQUESTS_BLANK_MODEL";
        const PM_RATIO: &str = "EMBED_SUBBATCH_RATIO_BLANK_MODEL";
        for v in [PM_IPC, PM_CAP, PM_RATIO] {
            unsafe { std::env::set_var(v, "  ") };
        }
        unsafe { std::env::set_var("EMBED_IPC_TIMEOUT_SECS", " ") };
        let mut got = None;
        let out = counted(|| {
            got = Some((
                resolve_ipc_timeout("blank-model"),
                resolve_embed_concurrency_for_model("blank-model", Some("  "), None, 2),
                crate::model::parse_subbatch_ratio(Some("  "), "T"),
                crate::model::resolve_subbatch_ratio("blank-model"),
            ));
        });
        for v in [PM_IPC, PM_CAP, PM_RATIO] {
            unsafe { std::env::remove_var(v) };
        }
        unsafe { std::env::remove_var("EMBED_IPC_TIMEOUT_SECS") };
        let (ipc, cap, ratio, ratio_env) = got.unwrap();
        assert_eq!(
            ipc,
            std::time::Duration::from_secs(DEFAULT_IPC_TIMEOUT_SECS)
        );
        assert_eq!(cap, Some(16), "blank cap vars fall to the auto formula");
        assert_eq!(ratio, crate::model::DEFAULT_SUBBATCH_RATIO);
        assert_eq!(ratio_env, crate::model::DEFAULT_SUBBATCH_RATIO);
        assert!(
            !out.contains("embed_config_fallback_total"),
            "blank values must not be counted as garbage:\n{out}"
        );
    }

    /// A value of only separators (",,,", " , ,") is a quoting artefact
    /// like a blank — nothing was parseable, but nothing was typed either,
    /// so the fallback to defaults stays silent. The warn gate has two
    /// conditions and both decide observable behaviour:
    ///   - `&&` -> `||` counts ",,," as a typo;
    ///   - `c != ','` -> `c == ','` suppresses the warn for real input
    ///     without a comma (covered by the next test).
    #[test]
    fn warmup_batch_sizes_all_commas_falls_back_silently() {
        let mut got = vec![];
        let out = counted(|| got = parse_warmup_batch_sizes("W", Some(",,,"), &[2]));
        assert_eq!(got, vec![2]);
        assert!(
            !out.contains("embed_config_fallback_total"),
            "a value of only commas is a quoting artefact, not a typo:\n{out}"
        );
    }

    /// The other half of that gate: a non-blank value with no valid
    /// entries must be counted. `"0"` carries no comma, so the
    /// `c != ','` -> `c == ','` mutation mutes the warn entirely.
    #[test]
    fn warmup_batch_sizes_all_invalid_counts_the_fallback() {
        let mut got = vec![];
        let out = counted(|| got = parse_warmup_batch_sizes("W", Some("0"), &[2]));
        assert_eq!(got, vec![2]);
        assert!(
            out.contains("embed_config_fallback_total{env_var=\"W\"} 1"),
            "a real value that cannot be used must be counted:\n{out}"
        );
    }

    /// `EMBED_INTRA_THREADS=0` is unusable — 0 would mean ORT auto-select,
    /// but the session pool expects an explicit count — so it falls back
    /// to 2, while a valid value passes through. Asserted on the `Config`
    /// field itself so the whole `from_env` line is covered.
    ///
    /// RED-proven: `raw == 0` -> `raw != 0` turns "0" into 0 and "5" into 2.
    #[test]
    #[serial_test::serial]
    fn from_env_intra_threads_zero_falls_back_to_two() {
        // Vars whose bad values abort startup; cleared so an ambient CI
        // env cannot fail the parse for an unrelated reason.
        const HARD_VARS: &[&str] = &[
            "EMBED_PORT",
            "EMBED_MODELS",
            "EMBED_DEFAULT_MODEL",
            "EMBED_INTRA_THREADS",
            "RERANKER_MODELS",
            "SPLADE_MODELS",
            "EMBED_INPUT_PREFIX_PROBE_MDL",
            "BATCH_MAX_SEQ_PROBE_MDL",
            "EMBED_WARMUP_SEQ_LEN_PROBE_MDL",
            "EMBED_MEMORY_PATTERN_PROBE_MDL",
        ];
        let saved: Vec<(&str, Option<String>)> = HARD_VARS
            .iter()
            .map(|k| {
                let prev = env::var(k).ok();
                unsafe { env::remove_var(k) };
                (*k, prev)
            })
            .collect();
        unsafe { env::set_var("EMBED_MODELS", "probe-mdl:/models:8:64:0:false") };

        unsafe { env::set_var("EMBED_INTRA_THREADS", "0") };
        let zero = Config::from_env().map(|c| c.intra_threads);
        unsafe { env::set_var("EMBED_INTRA_THREADS", "5") };
        let five = Config::from_env().map(|c| c.intra_threads);

        for (k, prev) in saved {
            unsafe {
                match prev {
                    Some(v) => env::set_var(k, v),
                    None => env::remove_var(k),
                }
            }
        }

        assert_eq!(
            zero.expect("from_env with EMBED_INTRA_THREADS=0"),
            2,
            "0 is unusable: the pool needs an explicit count"
        );
        assert_eq!(
            five.expect("from_env with EMBED_INTRA_THREADS=5"),
            5,
            "a valid override must pass through, not fall back to 2"
        );
    }

    /// The per-model `BATCH_MAX_SEQ_<KEY>` override announces itself with
    /// exactly one info line — only when the resolved value differs from
    /// the global. `!=` -> `==` at that gate would log an override that is
    /// not there and stay silent for one that is.
    ///
    /// RED-proven: the mutant fails both assertions below.
    #[test]
    #[serial_test::serial]
    fn per_model_batch_max_seq_logs_only_on_a_real_override() {
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct LogBuf(Arc<Mutex<Vec<u8>>>);
        impl Write for LogBuf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().expect("log buf").write(b)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.0.lock().expect("log buf").flush()
            }
        }

        const VAR: &str = "BATCH_MAX_SEQ_LOG_PROBE";
        let prev = env::var(VAR).ok();
        let entry = "log-probe:/models:8:64:0:false";

        // Parse under a subscriber capturing INFO lines, with VAR set to
        // `raw` (None = unset, so the resolved value equals the global).
        let run = |raw: Option<&str>| -> String {
            match raw {
                Some(v) => unsafe { env::set_var(VAR, v) },
                None => unsafe { env::remove_var(VAR) },
            }
            let buf = LogBuf(Arc::new(Mutex::new(Vec::new())));
            let writer = buf.clone();
            let sub = tracing_subscriber::fmt()
                .with_ansi(false)
                .without_time()
                .with_max_level(tracing::Level::INFO)
                .with_writer(move || writer.clone())
                .finish();
            tracing::subscriber::with_default(sub, || {
                parse_one_model(entry, 256, None).expect("valid model entry")
            });
            String::from_utf8(buf.0.lock().expect("log buf").clone()).expect("utf8 logs")
        };

        let differing = run(Some("128"));
        let equal = run(None);
        match prev {
            Some(v) => unsafe { env::set_var(VAR, v) },
            None => unsafe { env::remove_var(VAR) },
        }

        assert!(
            differing.contains("per-model BATCH_MAX_SEQ override"),
            "an override differing from the global must be logged:\n{differing}"
        );
        assert!(
            !equal.contains("per-model BATCH_MAX_SEQ override"),
            "no override must not log one:\n{equal}"
        );
    }
}
