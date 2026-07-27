use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ort::ep;
use ort::session::RunOptions;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};

use crate::config::ModelDef;
use crate::evictable_pool::{AcquireError, EvictablePool};
use crate::mlock::{MlockedSession, read_and_mlock};
use crate::onnx_cache::{self, CacheDir, LoadPlan};
use crate::pool;

/// Configure tokenizer truncation.
///
/// When `auto_truncate` is true, the tokenizer will silently truncate any
/// input longer than `max_len` tokens (TEI-compat default — matches Hugging
/// Face's `text-embeddings-inference` behaviour).
///
/// When `auto_truncate` is false, truncation is cleared; overlong inputs
/// encode to more than `max_len` tokens and downstream code decides how to
/// handle them (currently `pool::build_tensors` still clips to `max_len`,
/// but this may change — keeping the strict switch lets callers detect
/// overlong inputs if we ever wire that up).
pub fn configure_truncation(
    tokenizer: &mut Tokenizer,
    auto_truncate: bool,
    max_len: usize,
) -> Result<(), String> {
    // Truncation knobs we care about:
    //
    // `direction: Right` — drop trailing tokens, preserve the leading `[CLS]`
    //   / BOS and query content. Matters for sentence-pair inputs (Phase E
    //   reranker) where `[CLS] query [SEP] document [SEP]` must keep the
    //   query intact and truncate the document tail.
    // `strategy: LongestFirst` — when the input is a pair, truncate the
    //   longer side first so a short query isn't clipped just because the
    //   document is long. For single-input embedding it's effectively a
    //   no-op (there's only one side), but setting it consistently keeps
    //   Phase E behaviour aligned with Phase A.
    let params = if auto_truncate {
        Some(TruncationParams {
            direction: TruncationDirection::Right,
            max_length: max_len,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        })
    } else {
        None
    };
    tokenizer
        .with_truncation(params)
        .map(|_| ())
        .map_err(|e| format!("with_truncation: {e}"))
}

/// Parse the `ORT_OPT_LEVEL` env var (0..=3) into an ort
/// `GraphOptimizationLevel`. Defaults to `Level3` (all optimizations) when
/// the variable is unset or unparseable.
fn parse_opt_level() -> GraphOptimizationLevel {
    let raw = std::env::var("ORT_OPT_LEVEL")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(3);
    match raw {
        0 => GraphOptimizationLevel::Disable,
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    }
}

/// Wraps an evictable pool of ONNX sessions + a single tokenizer for one
/// embedding model.
///
/// `sessions` is an [`EvictablePool`] that replaces the former
/// `Vec<Mutex<Session>>` + round-robin `AtomicUsize`. Key differences:
///
/// - When `idle_evict_secs == 0` (default) the pool never evicts and
///   behaves exactly like the old code (pre-allocated, mutex-per-slot).
/// - When `idle_evict_secs > 0` (opt-in via `EMBED_IDLE_EVICT_SECS`),
///   sessions idle longer than the threshold are freed and lazily rebuilt
///   on the next acquire — saving ~250-400 MiB per slot during idle periods.
/// - Acquire semantics: first non-busy slot is returned. Under low
///   concurrency (pool_size=1 or pool_size=2 with non-overlapping requests)
///   this is equivalent to the old round-robin; under high concurrency the
///   first-available policy maximises throughput without coordination.
///
/// IMPORTANT: ort 2.0-rc has no shared-weights mode, so each pool member
/// holds its own ~400 MiB weight buffer. The pool size is intentionally
/// off by default (operator opt-in via `EMBED_SESSION_POOL_SIZE`).
pub struct EmbedModel {
    /// Model name for metric labels (e.g. "jina-code-v2").
    name: String,
    sessions: Arc<EvictablePool<MlockedSession>>,
    /// Number of slots for warmup iteration (acquired sequentially).
    pool_size: usize,
    tokenizer: Tokenizer,
    pub dim: usize,
    max_len: usize,
    pad_id: u32,
    has_token_type_ids: bool,
    /// Number of attention heads — used to estimate self-attention scratch
    /// bytes: `B × num_heads × S² × 4`.  Hardcoded per model family since
    /// ort 2.0-rc does not expose metadata from the ONNX graph ergonomically.
    ///
    /// jina-code-v2:        12 heads (RoBERTa-base backbone)
    /// multilingual-e5-large: 16 heads (XLM-RoBERTa-large backbone)
    /// Unknown models:      12 (conservative fallback)
    num_heads: usize,
    /// Whether per-run BFCArena shrinkage is enabled for this model.
    ///
    /// When `true`, `embed_tokens` and `warmup_at_shape` call
    /// `session.run_with_options` with a [`RunOptions`] that carries
    /// `"memory.enable_memory_arena_shrinkage" = "cpu:0"`. This triggers
    /// `BFCArena::ShrinkRegion()` after each inference, returning fully-free
    /// AllocationRegions to the OS and preventing `total_allocated_bytes`
    /// from growing monotonically.
    ///
    /// Only set for `memory_pattern=false` models (jina-code-v2) by default.
    /// For `memory_pattern=true` models (e5-large, reranker, splade), ORT
    /// plans the entire forward pass as one block — shrinkage would tear it
    /// down on each run, forcing a cold re-plan. Operator escape hatch:
    /// `EMBED_ARENA_SHRINK_<MODEL>=true|false|auto` (see `arena_shrink_enabled_for_model`).
    ///
    /// IMPORTANT: `run_options.is_some()` is the authoritative runtime gate.
    /// This field is kept for ops tooling (`arena_shrink_enabled()` accessor)
    /// and test assertions.
    #[allow(dead_code)]
    arena_shrink_enabled: bool,
    /// Pre-constructed run options with the arena-shrink config entry, reused
    /// across every inference call. `None` when shrinkage is disabled for
    /// this model.
    ///
    /// `RunOptions<NoSelectedOutputs>` is `Send + Sync` — safe to share
    /// across threads behind the pool mutex.
    run_options: Option<RunOptions>,
    /// Background eviction-loop task handle. `Some` when
    /// `idle_evict_secs > 0`. Aborted on `Drop` to avoid leaking the
    /// task across hot reloads / test teardown.
    eviction_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for EmbedModel {
    fn drop(&mut self) {
        if let Some(h) = self.eviction_handle.take() {
            h.abort();
        }
    }
}

impl EmbedModel {
    /// Load model from a directory containing model_quantized.onnx
    /// and tokenizer.json.
    ///
    /// `auto_truncate`: if true (TEI-compat default), the tokenizer silently
    /// truncates inputs longer than `def.max_len`. If false, truncation is
    /// left disabled on the tokenizer.
    ///
    /// `pool_size` controls how many independent ONNX sessions are created.
    /// `1` (the historical default) preserves the legacy single-session path
    /// exactly. Values >1 enable concurrent inference at N× the per-session
    /// weight memory cost — see the struct doc comment.
    ///
    /// `idle_evict_secs == 0` disables eviction (default). Positive value
    /// enables idle eviction — sessions unused for that many seconds are freed
    /// and lazily rebuilt on next acquire (cold start ~5-10s).
    pub fn load(
        def: &ModelDef,
        intra_threads: usize,
        auto_truncate: bool,
        pool_size: usize,
        idle_evict_secs: u64,
    ) -> Result<Self, String> {
        // Defensive clamp — caller contract says >=1, but a stray 0 from
        // misconfigured plumbing would leave an empty pool.
        let pool_size = pool_size.max(1);
        // Infer num_heads from the model name. Known families:
        //   multilingual-e5-large → XLM-RoBERTa-large → 16 heads
        //   jina-code-v2          → RoBERTa-base       → 12 heads
        //   (all others)          → 12 heads (conservative default)
        let num_heads = if def.name.contains("e5-large") {
            16
        } else {
            12
        };

        let dir = Path::new(&def.dir);

        let onnx_path = dir.join(&def.onnx_filename);
        if !onnx_path.exists() {
            return Err(format!("ONNX file not found: {}", onnx_path.display()));
        }

        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(format!("tokenizer.json not found: {}", tok_path.display()));
        }

        let opt_level = parse_opt_level();
        tracing::info!(
            path = %onnx_path.display(),
            ?opt_level,
            pool_size,
            intra_threads,
            idle_evict_secs,
            "creating ONNX session(s)"
        );

        let sessions = build_evictable_pool(
            &def.name,
            &onnx_path,
            opt_level,
            intra_threads,
            pool_size,
            def.memory_pattern,
            idle_evict_secs,
        )?;

        tracing::info!(path = %tok_path.display(), "loading tokenizer");
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| format!("load tokenizer: {e}"))?;
        configure_truncation(&mut tokenizer, auto_truncate, def.max_len)?;
        tracing::info!(auto_truncate, "tokenizer loaded");

        tracing::info!(
            model = %def.name,
            dim = def.dim,
            max_len = def.max_len,
            pad_id = def.pad_id,
            has_tti = def.has_token_type_ids,
            auto_truncate,
            pool_size,
            idle_evict_secs,
            "loaded model"
        );

        let sessions = Arc::new(sessions);

        // Spawn the idle-eviction loop when opt-in. Tick interval is
        // `max(idle_evict_secs/4, 5s)` — short enough to evict near the
        // threshold, never busy enough to hammer the pool. Mirrors the
        // ox-whisper v0.7.0 default formula.
        let eviction_handle = if idle_evict_secs > 0 {
            let tick = std::time::Duration::from_secs((idle_evict_secs / 4).max(5));
            tracing::info!(
                model = %def.name,
                idle_evict_secs,
                tick_secs = tick.as_secs(),
                "spawning ONNX session eviction loop"
            );
            Some(sessions.spawn_eviction_loop(tick))
        } else {
            None
        };

        // ── per-run arena shrinkage (Phase A of BUG-004 fragmentation fix) ──
        // Gate: enabled by default for memory_pattern=false models (jina-code-v2).
        // Operator escape hatch via EMBED_ARENA_SHRINK_<MODEL_UPPER>.
        let arena_shrink_enabled = arena_shrink_enabled_for_model(&def.name, def.memory_pattern);

        // Build the RunOptions once at load time; reuse across every inference.
        // Creating RunOptions requires the ORT library to be loaded (it calls
        // OrtApi::CreateRunOptions). We do this at startup where an error is
        // fatal — consistent with how we treat session-builder failures.
        let run_options = if arena_shrink_enabled {
            let mut opts = RunOptions::new().map_err(|e| {
                format!(
                    "model {}: failed to create RunOptions for arena shrinkage: {e}",
                    def.name
                )
            })?;
            opts.add_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
                .map_err(|e| {
                    format!(
                        "model {}: failed to add arena shrinkage config entry: {e}",
                        def.name
                    )
                })?;
            tracing::info!(
                model = %def.name,
                "arena shrinkage enabled: run_with_options(memory.enable_memory_arena_shrinkage=cpu:0)"
            );
            Some(opts)
        } else {
            tracing::info!(
                model = %def.name,
                memory_pattern = def.memory_pattern,
                "arena shrinkage disabled (memory_pattern=true or env override)"
            );
            None
        };

        // Publish to /metrics so operators can verify config without reading logs.
        crate::metrics::set_arena_shrink_enabled(&def.name, arena_shrink_enabled);

        Ok(Self {
            name: def.name.clone(),
            sessions,
            pool_size,
            tokenizer,
            dim: def.dim,
            max_len: def.max_len,
            pad_id: def.pad_id,
            has_token_type_ids: def.has_token_type_ids,
            num_heads,
            arena_shrink_enabled,
            run_options,
            eviction_handle,
        })
    }

    /// Configured pool size (number of ONNX session slots).
    ///
    /// Returns the number of slots configured at load time, which may exceed
    /// the count of live sessions after idle eviction. Use for capacity
    /// assertions and ops tooling, not for live session availability checks.
    #[allow(dead_code)]
    pub fn session_count(&self) -> usize {
        self.pool_size
    }

    /// Tokenize a batch of texts into their `input_ids`. Truncation is
    /// applied according to the tokenizer's configuration (see
    /// `configure_truncation`), then defensively capped at `self.max_len`
    /// per sequence. Runs the tokenizer only — no ONNX forward pass —
    /// so callers can cheaply compute token counts before dispatching
    /// a batch (enables token-budget accounting in the batcher).
    pub fn tokenize(&self, texts: &[String]) -> Result<Vec<Vec<u32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        // Per-seq truncation: the tokenizer is already configured to truncate
        // when auto_truncate=true, but we defensively cap here too so callers
        // can rely on the bound regardless of tokenizer state.
        Ok(encodings
            .iter()
            .map(|e| {
                let ids = e.get_ids();
                let len = ids.len().min(self.max_len);
                ids[..len].to_vec()
            })
            .collect())
    }

    /// Embed a batch of pre-tokenized `input_ids`, returning one vector
    /// per sequence. Skips the tokenizer entirely — callers are
    /// responsible for having already run `tokenize()`.
    pub fn embed_tokens(&self, token_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, String> {
        if token_ids.is_empty() {
            return Ok(vec![]);
        }

        // Pad to the longest sequence in the batch, capped at model max,
        // then round UP to the next power of two (also capped at
        // `self.max_len`). NEON INT8 GEMM tile sizes on ARM Neoverse-N1
        // are 4×4 / 8×8 — power-of-two seq dims hit cleaner tiling and
        // typically save a partial-tile epilogue on the last block.
        // Correctness is preserved: `pool::build_tensors_from_ids` zero-
        // fills the mask over padded positions, and
        // `pool::mean_pool_normalize` averages only positions where
        // mask > 0, so the extra padded tokens contribute neither to the
        // mean nor to the L2 norm. Kept on its own commit so it can be
        // reverted in isolation if a future model produces drift.
        let real_max_seq = token_ids
            .iter()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .min(self.max_len);
        let max_seq = round_up_seq_len(real_max_seq, self.max_len);

        let batch = token_ids.len();

        // ── forensic metrics: pre-inference observation ───────────────────
        // Record (B, S) distribution and token-budget before the forward
        // pass so they are available even when inference fails.
        crate::metrics::record_batch_dimensions(&self.name, batch, max_seq);
        crate::metrics::record_batch_token_budget(&self.name, batch, max_seq);
        crate::metrics::record_attention_scratch(&self.name, batch, self.num_heads, max_seq);
        // Snapshot RSS before inference for peak-bytes delta.
        let rss_before = read_rss_bytes();

        let (ids, mask_i64, tti_opt) = pool::build_tensors_from_ids(
            token_ids,
            batch,
            max_seq,
            self.pad_id,
            self.has_token_type_ids,
        );

        let ids_arr =
            Array2::from_shape_vec([batch, max_seq], ids).map_err(|e| format!("ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
            .map_err(|e| format!("mask shape: {e}"))?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(|e| format!("ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(|e| format!("mask tensor: {e}"))?;

        // Build tti tensor only when the model uses it (Some branch).
        // XLM-RoBERTa / RoBERTa / DistilBERT don't have a token_type_ids input,
        // so tti_opt is None for those models and we skip the tensor entirely.
        let tti_tensor_opt = match tti_opt {
            Some(tti_vec) => {
                let tti_arr = Array2::from_shape_vec([batch, max_seq], tti_vec)
                    .map_err(|e| format!("tti shape: {e}"))?;
                Some(Tensor::from_array(tti_arr).map_err(|e| format!("tti tensor: {e}"))?)
            }
            None => None,
        };

        // Acquire a free session from the pool. With pool_size==1 this always
        // picks the single slot — identical to the legacy single-Mutex<Session>
        // path. With pool_size>1, concurrent callers land on different slots
        // and run inference in parallel under independent slot mutexes.
        // AllBusy = all sessions held concurrently; caller retries or queues.
        let mut session = self.sessions.acquire().map_err(|e| match e {
            AcquireError::AllBusy => "embed session pool: all slots busy".to_string(),
            AcquireError::ReinitFailed(s) => format!("embed session pool: reinit failed: {s}"),
        })?;

        // Choose inference path: run_with_options (arena shrinkage) or plain run.
        // The shrink RunOptions are pre-built at load time; we just pass a reference.
        let run_result = match &self.run_options {
            Some(opts) => {
                crate::metrics::record_arena_shrink_call(&self.name);
                if let Some(tti_tensor) = tti_tensor_opt {
                    session.run_with_options(
                        ort::inputs! {
                            "input_ids" => ids_tensor,
                            "attention_mask" => mask_tensor,
                            "token_type_ids" => tti_tensor,
                        },
                        opts,
                    )
                } else {
                    session.run_with_options(
                        ort::inputs! {
                            "input_ids" => ids_tensor,
                            "attention_mask" => mask_tensor,
                        },
                        opts,
                    )
                }
            }
            None => {
                if let Some(tti_tensor) = tti_tensor_opt {
                    session.run(ort::inputs! {
                        "input_ids" => ids_tensor,
                        "attention_mask" => mask_tensor,
                        "token_type_ids" => tti_tensor,
                    })
                } else {
                    session.run(ort::inputs! {
                        "input_ids" => ids_tensor,
                        "attention_mask" => mask_tensor,
                    })
                }
            }
        };

        // ── forensic metrics: post-inference observation ──────────────────
        let rss_after = read_rss_bytes();
        let peak_delta = rss_after.saturating_sub(rss_before);
        crate::metrics::record_inference_peak_bytes(&self.name, peak_delta);

        let outputs = run_result.map_err(|e| {
            // Classify the failure for the failures counter.
            let msg = e.to_string();
            let (reason, bin_num) = classify_ort_error(&msg);
            crate::metrics::record_inference_failure(&self.name, reason, bin_num);
            format!("inference: {e}")
        })?;

        // Output shape: [batch, seq_len, dim]
        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let mask_arr_f = Array2::from_shape_vec([batch, max_seq], pool::mask_i64_to_f32(&mask_i64))
            .map_err(|e| format!("mask_f shape: {e}"))?;

        pool::mean_pool_normalize(&raw, &mask_arr_f, batch, max_seq, self.dim)
    }

    /// Return the model name (for test assertions and metric label reuse).
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run a dummy inference at each requested batch shape to force ORT
    /// kernel binding + arena allocation BEFORE the first production
    /// request. Same motivation as `RerankerModel::warmup`: the cold
    /// path on `[B, max_seq]` is meaningfully slower than steady-state,
    /// and prod sees several distinct B values (1 for trivial callers,
    /// 8 for memdb's `texts_per_req` default — and operators may set
    /// `EMBED_WARMUP_BATCH_SIZES` for other deployments).
    ///
    /// Each shape's pass uses the SAME bytes through the SAME tensor
    /// builders that production traffic uses (`pool::build_tensors_from_ids`),
    /// so ORT's per-shape memory pattern records what real inference
    /// will need rather than a pathological alternate code path.
    ///
    /// Best-effort: per-shape failure logs a warn and we continue with
    /// the next shape. The server still serves correctly without
    /// warmup; this is purely a tail-latency optimisation.
    pub fn warmup(
        &self,
        name: &str,
        shapes: &[usize],
        warmup_seq_len: Option<usize>,
    ) -> Result<(), String> {
        if shapes.is_empty() {
            tracing::warn!(
                model = %name,
                "embed warmup called with empty shapes — skipping"
            );
            return Ok(());
        }
        for &batch in shapes {
            if let Err(e) = self.warmup_at_shape(name, batch, warmup_seq_len) {
                tracing::warn!(
                    model = %name,
                    batch,
                    error = %e,
                    "embed shape warmup failed (continuing with remaining shapes)"
                );
            }
        }
        Ok(())
    }

    /// One pass at exactly `batch` items. Synthesises `batch` short
    /// dummy texts (content irrelevant — only the resulting tensor
    /// shape `[batch, max_seq]` matters for ORT pre-binding).
    ///
    /// `warmup_seq_len` controls the second tensor dim:
    /// - `None` → pad to `self.max_len` (worst-case prod shape; legacy).
    /// - `Some(n)` → pad to `n.min(self.max_len)`. With memory_pattern
    ///   enabled, ORT re-plans on the first prod request that needs a
    ///   longer shape — we just bind kernels here without committing
    ///   worst-case scratch.
    fn warmup_at_shape(
        &self,
        name: &str,
        batch: usize,
        warmup_seq_len: Option<usize>,
    ) -> Result<(), String> {
        // `batch` copies of a tiny placeholder. We deliberately keep
        // the text very short so tokenization is cheap; the ONNX
        // forward pass dominates wall time anyway.
        let texts: Vec<String> = (0..batch).map(|_| "warmup".to_string()).collect();
        let token_ids = self.tokenize(&texts)?;
        if token_ids.iter().all(|v| v.is_empty()) {
            return Err("warmup tokens produced empty sequence".to_string());
        }
        // Choose the warmup seq_len. `None` preserves legacy behaviour
        // (pad to `self.max_len`, exercising the worst-case prod shape).
        // `Some(n)` clamps to `min(n, max_len)` — with memory_pattern=true
        // ORT re-plans on the first long prod request, so binding kernels
        // at a shorter shape avoids committing worst-case scratch slabs at
        // startup. `build_tensors_from_ids` still zero-pads beyond the
        // real token count.
        let max_seq = match warmup_seq_len {
            None => self.max_len,
            Some(n) => n.min(self.max_len).max(1),
        };
        let (ids, mask_i64, tti_opt) = pool::build_tensors_from_ids(
            &token_ids,
            batch,
            max_seq,
            self.pad_id,
            self.has_token_type_ids,
        );

        // Warm EVERY session in the pool — without this, only the first
        // session served would be hot; the rest pay the cold-start cost
        // on their first concurrent request.
        //
        // Collect ALL guards first so acquire() skips already-busy slots.
        // If guards were acquired and dropped inside the same loop iteration,
        // each acquire would find slot 0 free and never reach slot 1+.
        let mut guards = Vec::with_capacity(self.pool_size);
        for i in 0..self.pool_size {
            match self.sessions.acquire() {
                Ok(g) => guards.push(g),
                Err(e) => {
                    tracing::warn!(
                        model = %name,
                        slot = i,
                        batch,
                        error = %e,
                        "embed warmup: could not acquire session slot (skipping)"
                    );
                }
            }
        }

        // Run the warmup forward pass on each acquired slot.
        for (i, session) in guards.iter_mut().enumerate() {
            let ids_arr = Array2::from_shape_vec([batch, max_seq], ids.clone())
                .map_err(|e| format!("warmup ids shape (batch={batch}): {e}"))?;
            let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
                .map_err(|e| format!("warmup mask shape (batch={batch}): {e}"))?;
            let ids_tensor =
                Tensor::from_array(ids_arr).map_err(|e| format!("warmup ids tensor: {e}"))?;
            let mask_tensor =
                Tensor::from_array(mask_arr).map_err(|e| format!("warmup mask tensor: {e}"))?;

            // Build tti tensor once per loop iteration when the model uses it.
            let tti_tensor_opt = match &tti_opt {
                Some(tti_vec) => {
                    let tti_arr = Array2::from_shape_vec([batch, max_seq], tti_vec.clone())
                        .map_err(|e| format!("warmup tti shape (batch={batch}): {e}"))?;
                    Some(
                        Tensor::from_array(tti_arr)
                            .map_err(|e| format!("warmup tti tensor: {e}"))?,
                    )
                }
                None => None,
            };

            let start = Instant::now();

            // Mirror the production path: use run_with_options when shrinkage
            // is enabled so warmup itself also exercises the shrink path and
            // verifies the RunOptions are valid before the first real request.
            let run_result = match &self.run_options {
                Some(opts) => {
                    if let Some(tti_tensor) = tti_tensor_opt {
                        session.run_with_options(
                            ort::inputs! {
                                "input_ids" => ids_tensor,
                                "attention_mask" => mask_tensor,
                                "token_type_ids" => tti_tensor,
                            },
                            opts,
                        )
                    } else {
                        session.run_with_options(
                            ort::inputs! {
                                "input_ids" => ids_tensor,
                                "attention_mask" => mask_tensor,
                            },
                            opts,
                        )
                    }
                }
                None => {
                    if let Some(tti_tensor) = tti_tensor_opt {
                        session.run(ort::inputs! {
                            "input_ids" => ids_tensor,
                            "attention_mask" => mask_tensor,
                            "token_type_ids" => tti_tensor,
                        })
                    } else {
                        session.run(ort::inputs! {
                            "input_ids" => ids_tensor,
                            "attention_mask" => mask_tensor,
                        })
                    }
                }
            };

            match run_result {
                Ok(_) => tracing::info!(
                    model = %name,
                    session = i,
                    batch,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "embed session warmed"
                ),
                Err(e) => tracing::error!(
                    model = %name,
                    session = i,
                    batch,
                    error = %e,
                    "embed session warmup failed (continuing)"
                ),
            }
        }
        Ok(())
    }
}

/// Round `n` up to the next power of two, capped at `cap` (and at least
/// 1). Used by `embed_tokens` to align the per-batch `max_seq` to NEON
/// INT8 GEMM tile sizes (4×4 / 8×8 on ARM Neoverse-N1) while never
/// exceeding the model's static `max_len`. `n == 0` → 1, `n == cap` →
/// `cap` (no rounding above the cap), `n` already a power of two →
/// itself.
fn round_up_seq_len(n: usize, cap: usize) -> usize {
    if n <= 1 {
        // Empty/singleton input → 1-token tensor. `max_len` is always
        // >= 1 for any real model (config parser would reject 0), so
        // we don't bother re-clamping at the cap here.
        return 1;
    }
    // `next_power_of_two` is a no-op when `n` is already a power of two.
    // Saturating math protects against the (theoretical) overflow on
    // very large `cap` — in practice `cap == self.max_len ≤ 512`.
    n.next_power_of_two().min(cap)
}

/// Build an [`EvictablePool`] of `pool_size` ONNX sessions for one model.
///
/// The factory closure captures the ONNX path + build parameters and is
/// called once per slot at startup (and again lazily after eviction).
/// `idle_evict_secs == 0` disables eviction — the pool never calls the
/// factory after startup.
///
/// Per-model ONNX cache override (`ONNX_OPT_CACHE_DIR_<MODEL_NAME>`) is
/// resolved inside `build_one_session`, which the factory invokes — so
/// every cold start (initial fill + post-eviction) re-reads the env.
fn build_evictable_pool(
    model_name: &str,
    onnx_path: &Path,
    opt_level: GraphOptimizationLevel,
    intra_threads: usize,
    pool_size: usize,
    memory_pattern: bool,
    idle_evict_secs: u64,
) -> Result<EvictablePool<MlockedSession>, String> {
    // Capture everything the factory needs to build one session. The
    // factory re-reads `ONNX_OPT_CACHE_DIR_<MODEL>` (or global fallback)
    // on every call so post-eviction cold starts pick up cache changes.
    let onnx_path_buf = onnx_path.to_path_buf();
    let model_name_owned = model_name.to_string();
    let factory: Arc<dyn Fn() -> Result<MlockedSession, String> + Send + Sync> =
        Arc::new(move || {
            build_one_session(
                &model_name_owned,
                &onnx_path_buf,
                opt_level,
                intra_threads,
                memory_pattern,
            )
        });

    // Build the initial pool_size sessions up front (startup). Any failure
    // here is fatal (wrong ONNX path / bad config) — panic is intentional.
    let mut items = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let session = (factory)().map_err(|e| format!("embed session #{i} failed: {e}"))?;
        items.push(session);
    }
    tracing::info!(
        model = %model_name,
        count = pool_size,
        idle_evict_secs,
        "embed ONNX session pool created"
    );
    Ok(EvictablePool::from_items(
        items,
        idle_evict_secs,
        model_name,
        factory,
    ))
}

/// Return `true` when a sibling `<onnx_path>.data` file exists next to the
/// **original** model file.
///
/// Must be called with the original `onnx_path`, never with a cache-redirected
/// path: the `.data` sidecar lives next to the source model, not in any cache
/// directory. When external data is present we must also disable caching
/// (`LoadPlan::NoCache`) because an ORT-optimized graph written into the cache
/// directory would lose its relative reference to the sidecar — there is no
/// coherent way to cache an external-data model.
fn should_use_external_data(onnx_path: &Path) -> bool {
    let mut p = onnx_path.as_os_str().to_owned();
    p.push(".data");
    std::path::Path::new(&p).exists()
}

/// Build a single ONNX session with the configured parameters.
/// Called by the factory closure in [`build_evictable_pool`] — both at
/// startup (for initial pool fill) and lazily after eviction.
fn build_one_session(
    model_name: &str,
    onnx_path: &std::path::Path,
    opt_level: GraphOptimizationLevel,
    intra_threads: usize,
    memory_pattern: bool,
) -> Result<MlockedSession, String> {
    // Defence-in-depth: assert the shared CPU arena is registered BEFORE
    // any Session::builder() call. The reranker (model_reranker/load.rs)
    // and splade (model_splade.rs) already do this; the embed path
    // previously relied only on the worker's startup check
    // (register_arena_for_worker in src/bin/worker.rs). If a future
    // refactor bypasses that check, this assert catches it with a clear
    // panic instead of silently falling back to per-session BFCArena
    // (unbounded memory growth). See arena.rs:68.
    crate::arena::assert_arena_registered_before_session();
    // External-data detection MUST happen against the original onnx_path before
    // any cache redirection. If a `<onnx_path>.data` sidecar exists, caching is
    // incoherent (an optimized graph in the cache dir has no `.data` next to it)
    // so we force NoCache regardless of what the env says. This also guarantees
    // that both sessions in a pool_size=2 scenario take identical, deterministic
    // paths: session #0 (which would be a Miss) and session #1 (which would see
    // the Miss-written file as a Hit) both become NoCache → commit_from_file on
    // onnx_path, with no order-dependent divergence.
    let has_external_data = should_use_external_data(onnx_path);

    // Per-model override: `ONNX_OPT_CACHE_DIR_<MODEL_NAME_UPPER>` takes
    // precedence over the global `ONNX_OPT_CACHE_DIR`.
    let cache = CacheDir::from_env_for_model(model_name);
    let plan = if has_external_data {
        // External-data models cannot be cached — force NoCache so every pool
        // session loads from the original path and can locate the sidecar.
        LoadPlan::NoCache
    } else {
        LoadPlan::decide(cache.as_ref(), onnx_path)
    };
    let load_path = plan.load_source(onnx_path).to_path_buf();
    let t_commit = std::time::Instant::now();
    // memory_pattern=true: ORT plans scratch reuse within the shared
    // env-level arena (registered in `arena.rs`). Combined with
    // `DisableCpuMemArena` (PR #34), the session has only the shared
    // arena to draw from — pattern planning amortizes per-shape
    // scratch allocations across requests.
    let builder = Session::builder().map_err(|e| format!("session builder: {e}"))?;
    let builder = onnx_cache::apply_plan(builder, &plan, opt_level)
        .map_err(|e| format!("apply cache plan: {e}"))?;
    let allow_spinning = crate::arena::parse_intra_op_spinning();
    let mut builder = builder
        .with_intra_threads(intra_threads)
        .map_err(|e| format!("set threads: {e}"))?
        // Gate ORT's intra-op spin via env (ORT_INTRA_OP_SPINNING, default
        // false). `OMP_WAIT_POLICY=PASSIVE` only governs OpenMP, NOT ORT's
        // own intra pool — explicit `with_intra_op_spinning(false)` is the
        // only way to stop the spin on a shared multi-tenant CPU. See
        // arena::parse_intra_op_spinning for the full rationale.
        .with_intra_op_spinning(allow_spinning)
        .map_err(|e| format!("set intra spinning: {e}"))?
        // Inter-op parallelism: BERT-family encoders are largely sequential
        // (attention -> FFN -> norm), so inter-op threads yield no benefit
        // and over-subscribe the 4-core CPU. ORT defaults inter_op_num_threads
        // to the intra-op count; pinning to 1 eliminates the over-subscription.
        .with_inter_threads(1)
        .map_err(|e| format!("set inter threads: {e}"))?
        .with_memory_pattern(memory_pattern)
        .map_err(|e| format!("enable memory pattern: {e}"))?
        // Use the shared env-level arena registered in arena.rs.
        .with_env_allocators()
        .map_err(|e| format!("enable env allocators: {e}"))?
        // Belt-and-braces: disable the per-session CPU mem arena.
        // Without this, ORT's CPU EP defaults to EnableCpuMemArena=1
        // and may still spawn a session-local BFCArena alongside our
        // shared one, doubling allocator state.
        .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
        .map_err(|e| format!("disable per-session cpu mem arena: {e}"))?;

    // When external data is present, load_path == onnx_path (NoCache above),
    // so ORT can resolve the `.data` sidecar relative to the model directory.
    // commit_from_memory has no directory context and cannot locate the sidecar.
    // The mlock optimisation is skipped for such models — external-data models
    // keep their weights in ORT's own heap.
    if has_external_data {
        let data_path = {
            let mut p = onnx_path.as_os_str().to_owned();
            p.push(".data");
            std::path::PathBuf::from(p)
        };
        tracing::info!(
            path = %onnx_path.display(),
            data = %data_path.display(),
            "ONNX has external data — loading via commit_from_file (mlock skipped, cache disabled)"
        );
        let session = builder
            .commit_from_file(onnx_path)
            .map_err(|e| format!("load ONNX {}: {e}", onnx_path.display()))?;
        onnx_cache::observe_post_commit(&plan, t_commit.elapsed().as_millis());
        Ok(MlockedSession::new_without_mlock(session))
    } else {
        // Read the ONNX file (original or cached) into a mlocked buffer so the
        // kernel cannot swap those pages out under host memory pressure. The
        // buffer is passed to ORT via commit_from_memory rather than
        // commit_from_file so we own the source bytes and can mlock them. If
        // mlock fails (RLIMIT too low, no privilege) read_and_mlock logs a
        // warning and still returns the bytes — load proceeds normally without
        // pinning.
        let mlocked = read_and_mlock(&load_path)
            .map_err(|e| format!("read ONNX bytes {}: {e}", load_path.display()))?;
        let session = builder
            .commit_from_memory(mlocked.as_slice())
            .map_err(|e| format!("load ONNX {}: {e}", load_path.display()))?;
        onnx_cache::observe_post_commit(&plan, t_commit.elapsed().as_millis());
        Ok(MlockedSession::new(session, mlocked))
    }
}

#[cfg(test)]
mod seq_pad_tests {
    use super::round_up_seq_len;

    #[test]
    fn zero_rounds_to_one() {
        // Empty input — degenerate, but an empty batch should not
        // produce a 0-dim tensor. Cap=256 is a normal e5-large limit.
        assert_eq!(round_up_seq_len(0, 256), 1);
    }

    #[test]
    fn one_stays_one() {
        // Power of two already; no rounding work.
        assert_eq!(round_up_seq_len(1, 256), 1);
    }

    #[test]
    fn rounds_up_to_next_power_of_two() {
        // Sub-power values lift to the next tile boundary.
        assert_eq!(round_up_seq_len(3, 256), 4);
        assert_eq!(round_up_seq_len(5, 256), 8);
        assert_eq!(round_up_seq_len(9, 256), 16);
        assert_eq!(round_up_seq_len(17, 256), 32);
        assert_eq!(round_up_seq_len(65, 256), 128);
    }

    #[test]
    fn already_power_of_two_passes_through() {
        for n in [2usize, 4, 8, 16, 32, 64, 128, 256] {
            assert_eq!(
                round_up_seq_len(n, 256),
                n,
                "{n} is already a power of two, must not round up"
            );
        }
    }

    #[test]
    fn capped_at_model_max_len() {
        // 200 → 256 if cap allows, else clamp at cap.
        assert_eq!(round_up_seq_len(200, 256), 256);
        // The static-shape cap is 256 — never exceed the model's
        // declared max_len even if the next power of two would.
        assert_eq!(round_up_seq_len(200, 200), 200);
        assert_eq!(round_up_seq_len(257, 256), 256);
        // Non-power-of-two cap (e.g. jina-code-v2 max_len=512 is fine,
        // but cap=300 would clamp).
        assert_eq!(round_up_seq_len(150, 300), 256);
        // 250.next_power_of_two() == 256, well under cap=300.
        assert_eq!(round_up_seq_len(250, 300), 256);
        // 257.next_power_of_two() == 512, capped to 300.
        assert_eq!(round_up_seq_len(257, 300), 300);
    }
}

/// Tests for the external-data / cache-policy decision logic.
///
/// These are pure filesystem-state tests — no ORT sessions involved.
/// The key invariant: `should_use_external_data` must read the ORIGINAL
/// onnx path, never a cache-redirected path, so a cache HIT cannot flip
/// the decision and cause a startup failure on session #1 of a pool.
#[cfg(test)]
mod session_load_decision_tests {
    use super::should_use_external_data;

    /// No `.data` sidecar → mlock path should be taken (external data = false).
    #[test]
    fn no_sidecar_returns_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let onnx = dir.path().join("model.onnx");
        std::fs::write(&onnx, b"fake").expect("write onnx");
        // No model.onnx.data created.
        assert!(
            !should_use_external_data(&onnx),
            "no sidecar → should_use_external_data must be false"
        );
    }

    /// A sibling `<model>.onnx.data` exists → external-data path must be taken.
    #[test]
    fn with_sidecar_returns_true() {
        let dir = tempfile::tempdir().expect("tempdir");
        let onnx = dir.path().join("model_int8.onnx");
        let data = dir.path().join("model_int8.onnx.data");
        std::fs::write(&onnx, b"fake onnx").expect("write onnx");
        std::fs::write(&data, b"fake weights").expect("write data");
        assert!(
            should_use_external_data(&onnx),
            "sidecar present → should_use_external_data must be true"
        );
    }

    /// Cache-HIT regression guard: a redirected load_path in a *different*
    /// directory (simulating what `LoadPlan::Hit` would produce) must NOT
    /// cause a false-positive even if a file with the cache name happens to
    /// exist. The decision is anchored to the original onnx_path.
    ///
    /// Scenario: session #0 writes an optimized graph to cache_dir; session #1
    /// sees a Hit and previously probed `cache_dir/<basename>.data` (doesn't
    /// exist) → returned false → commit_from_memory → ORT fails to find the
    /// sidecar. This test verifies that the corrected code anchors the probe to
    /// onnx_path, so both sessions agree regardless of cache state.
    #[test]
    fn cache_hit_path_does_not_affect_decision() {
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let cache_dir = tempfile::tempdir().expect("cache tempdir");

        let onnx_path = model_dir.path().join("model_int8.onnx");
        let onnx_data = model_dir.path().join("model_int8.onnx.data");
        std::fs::write(&onnx_path, b"fake onnx").expect("write onnx");
        std::fs::write(&onnx_data, b"fake weights").expect("write data");

        // Simulate what a cache Hit would redirect to — in a different dir,
        // with NO corresponding .data file there.
        let cached_path = cache_dir.path().join("model_int8.onnx.1234.optimized.onnx");
        std::fs::write(&cached_path, b"optimized graph").expect("write cached");
        // Deliberately do NOT create cached_path + ".data".

        // The decision must be anchored to onnx_path (has sidecar → true),
        // not to cached_path (no sidecar → would return false if broken).
        assert!(
            should_use_external_data(&onnx_path),
            "must probe original onnx_path, not a hypothetical cache-redirected path"
        );
        // Confirm the cached path itself would return false if probed — proving
        // the test is actually exercising the discrimination.
        assert!(
            !should_use_external_data(&cached_path),
            "cache-redirected path has no sidecar → false (this is the broken behaviour the fix prevents)"
        );
    }
}

/// Determine whether per-run BFCArena shrinkage should be enabled for a model.
///
/// Logic (in priority order):
/// 1. `EMBED_ARENA_SHRINK_<MODEL_UPPER>` env var:
///    - `"true"` / `"1"` → force on.
///    - `"false"` / `"0"` → force off.
///    - Unset / `"auto"` / any other value → fall through to step 2.
/// 2. Auto gate: enabled iff `memory_pattern == false`.
///    - `memory_pattern=false` models (jina-code-v2): ORT allocates per-op,
///      `total_allocated_bytes` grows monotonically over hours → shrinkage is
///      the targeted fix.
///    - `memory_pattern=true` models (e5-large, reranker, splade): ORT plans
///      the entire forward pass as one block; shrinkage tears it down on each
///      run, forcing a cold re-plan → devastating perf regression, must NOT
///      be enabled by default.
///
/// `model_name` is the raw model name (e.g. `"jina-code-v2"`); the env var
/// key is derived by uppercasing and replacing `-` with `_` — mirroring the
/// `EMBED_MEMORY_PATTERN_<MODEL>` convention.
pub(crate) fn arena_shrink_enabled_for_model(model_name: &str, memory_pattern: bool) -> bool {
    let key = crate::config::model_env_key(model_name);
    let env_val = std::env::var(format!("EMBED_ARENA_SHRINK_{key}")).ok();
    match env_val.as_deref() {
        Some("true") | Some("1") => {
            tracing::info!(
                model = %model_name,
                "EMBED_ARENA_SHRINK_{key}=true: arena shrinkage forced on"
            );
            true
        }
        Some("false") | Some("0") => {
            tracing::info!(
                model = %model_name,
                "EMBED_ARENA_SHRINK_{key}=false: arena shrinkage forced off"
            );
            false
        }
        // `auto` is the documented sentinel for "follow memory_pattern" — silent.
        None | Some("auto") | Some("") => !memory_pattern,
        Some(other) => {
            // Operator typo (e.g. "yes" / "on" / "enabled"). Mirror the
            // warn-on-failure pattern from `parse_memory_pattern` (config.rs)
            // so misconfiguration is visible rather than silently auto-gated.
            tracing::warn!(
                model = %model_name,
                value = %other,
                "EMBED_ARENA_SHRINK_{key} unrecognised value (expected true/false/1/0/auto/empty); falling back to auto-gate (memory_pattern={memory_pattern})"
            );
            !memory_pattern
        }
    }
}

/// Read process RSS (resident set size) in bytes from `/proc/self/statm`.
///
/// Returns `0` on any error (file absent on non-Linux, parse failure, etc.)
/// so callers never panic.  Used only for a best-effort peak-bytes delta
/// around `session.run()`.
///
/// `/proc/self/statm` format (space-separated integers, all in pages):
///   `size  resident  shared  text  lib  data  dt`
/// We want field 1 (resident).  `page_size()` converts to bytes.
fn read_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let Ok(content) = std::fs::read_to_string("/proc/self/statm") else {
            return 0;
        };
        let mut parts = content.split_ascii_whitespace();
        let _size = parts.next();
        let Some(rss_pages_str) = parts.next() else {
            return 0;
        };
        let Ok(rss_pages) = rss_pages_str.parse::<u64>() else {
            return 0;
        };
        // SAFETY: `sysconf(_SC_PAGESIZE)` is thread-safe (POSIX), always
        // returns a power-of-two ≥ 4096 on Linux. libc is a Linux-only dep.
        let page_size = unsafe { ::libc::sysconf(::libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return rss_pages * 4096;
        }
        rss_pages * page_size as u64
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Classify an ORT error message into a `(reason, bin_num)` pair for
/// `embed_inference_failures_total`.
///
/// Recognised pattern (from BFCArena C++ source):
///   "Available memory of X bytes is smaller than requested bytes of Y"
/// Returns `("arena_oom", 0)` for that pattern.
///
/// Returns `("other", 0)` for everything else.
fn classify_ort_error(msg: &str) -> (&'static str, u32) {
    if msg.contains("Available memory of") && msg.contains("smaller than requested") {
        ("arena_oom", 0)
    } else {
        ("other", 0)
    }
}

// ── unit tests for new forensic helpers ──────────────────────────────────────

#[cfg(test)]
mod forensic_tests {
    use super::*;

    // ── classify_ort_error ────────────────────────────────────────────────

    #[test]
    fn classify_arena_oom_message() {
        let msg =
            "Available memory of 1073741824 bytes is smaller than requested bytes of 1258291200";
        let (reason, bin_num) = classify_ort_error(msg);
        assert_eq!(reason, "arena_oom");
        assert_eq!(bin_num, 0);
    }

    #[test]
    fn classify_other_error_message() {
        let msg = "ONNX graph is invalid: input tensor not found";
        let (reason, bin_num) = classify_ort_error(msg);
        assert_eq!(reason, "other");
        assert_eq!(bin_num, 0);
    }

    #[test]
    fn classify_partial_match_is_other() {
        // Contains "Available memory" but not "smaller than requested"
        let msg = "Available memory of 1 GiB";
        let (reason, _) = classify_ort_error(msg);
        assert_eq!(reason, "other");
    }

    // ── read_rss_bytes ────────────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "linux")]
    fn read_rss_returns_nonzero_on_linux() {
        let rss = read_rss_bytes();
        // The process always has resident pages; anything ≥ 4096 is sane.
        assert!(rss >= 4096, "rss={rss} expected > 0");
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn read_rss_returns_zero_on_non_linux() {
        assert_eq!(read_rss_bytes(), 0);
    }
}

#[cfg(test)]
mod truncation_tests {
    use super::*;
    use tokenizers::Tokenizer;

    /// Load a real e5-compatible tokenizer.json for truncation tests.
    ///
    /// Path resolution order:
    /// 1. `E5_TOKENIZER_PATH` env var (lets CI / other dev boxes point at
    ///    wherever they've staged the model bundle).
    /// 2. Default on-box path
    ///    `/models/multilingual-e5-large/tokenizer.json`.
    ///
    /// When neither exists, the test returns `None` and the caller
    /// early-returns after printing a visible skip notice. The skip line
    /// is loud on purpose so the test doesn't silently vanish from CI
    /// output the way `#[ignore]` would.
    fn load_tokenizer_or_skip() -> Option<Tokenizer> {
        const DEFAULT_PATH: &str = "/models/multilingual-e5-large/tokenizer.json";
        let p = std::env::var("E5_TOKENIZER_PATH").unwrap_or_else(|_| DEFAULT_PATH.to_string());
        if !std::path::Path::new(&p).exists() {
            eprintln!(
                "SKIP truncation test: tokenizer.json not found at {p} \
                 (set E5_TOKENIZER_PATH to override)"
            );
            return None;
        }
        Some(Tokenizer::from_file(&p).expect("load tokenizer"))
    }

    #[test]
    fn configure_truncation_enables_when_auto_true() {
        let Some(mut tok) = load_tokenizer_or_skip() else {
            return;
        };
        // Precondition: the on-disk tokenizer.json has truncation: null.
        assert!(
            tok.get_truncation().is_none(),
            "precondition: shipped tokenizer.json should have no truncation"
        );

        configure_truncation(&mut tok, true, 512).expect("configure_truncation");

        let params = tok
            .get_truncation()
            .expect("truncation should be enabled when auto_truncate=true");
        assert_eq!(params.max_length, 512);
    }

    #[test]
    fn configure_truncation_disabled_when_auto_false() {
        let Some(mut tok) = load_tokenizer_or_skip() else {
            return;
        };
        // Pre-seed truncation so we can assert it gets cleared.
        configure_truncation(&mut tok, true, 512).expect("seed");
        assert!(tok.get_truncation().is_some());

        configure_truncation(&mut tok, false, 512).expect("configure_truncation");

        assert!(
            tok.get_truncation().is_none(),
            "truncation should be disabled when auto_truncate=false"
        );
    }

    #[test]
    fn overlong_input_encodes_within_max_len_when_auto_truncate_on() {
        let Some(mut tok) = load_tokenizer_or_skip() else {
            return;
        };
        configure_truncation(&mut tok, true, 512).expect("configure_truncation");

        // Make an input that tokenises to well over 512 tokens.
        let long = "word ".repeat(5000);
        let enc = tok.encode(long, true).expect("encode");
        assert!(
            enc.get_ids().len() <= 512,
            "expected <= 512 ids, got {}",
            enc.get_ids().len()
        );
    }

    #[test]
    fn overlong_input_exceeds_max_len_when_auto_truncate_off() {
        let Some(mut tok) = load_tokenizer_or_skip() else {
            return;
        };
        // Explicitly disabled: we keep the current (pre-A3) strict-ish behaviour
        // where the encoder emits full-length output and downstream code decides.
        configure_truncation(&mut tok, false, 512).expect("configure_truncation");

        let long = "word ".repeat(5000);
        let enc = tok.encode(long, true).expect("encode");
        assert!(
            enc.get_ids().len() > 512,
            "expected overlong ids when truncation off, got {}",
            enc.get_ids().len()
        );
    }
}

#[cfg(test)]
mod opt_level_tests {
    use super::*;

    /// Sets/unsets an env var around a closure and restores the previous value.
    fn with_env<F: FnOnce()>(key: &str, val: Option<&str>, f: F) {
        let prev = std::env::var(key).ok();
        match val {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        f();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn parse_opt_level_env_mapping() {
        // Run cases sequentially in one thread to avoid env-var races with
        // other tests (Rust test runner is multi-threaded by default).
        with_env("ORT_OPT_LEVEL", None, || {
            assert_eq!(parse_opt_level(), GraphOptimizationLevel::Level3);
        });
        for (val, want) in [
            ("0", GraphOptimizationLevel::Disable),
            ("1", GraphOptimizationLevel::Level1),
            ("2", GraphOptimizationLevel::Level2),
            ("3", GraphOptimizationLevel::Level3),
        ] {
            with_env("ORT_OPT_LEVEL", Some(val), || {
                assert_eq!(
                    parse_opt_level(),
                    want,
                    "value {:?} should map to {:?}",
                    val,
                    want
                );
            });
        }
        // Garbage / out-of-range → Level3 fallback.
        for garbage in ["not-a-number", "99", ""] {
            with_env("ORT_OPT_LEVEL", Some(garbage), || {
                assert_eq!(parse_opt_level(), GraphOptimizationLevel::Level3);
            });
        }
    }
}

// ── arena shrink gate tests ───────────────────────────────────────────────────
//
// These tests exercise the pure `arena_shrink_enabled_for_model` function
// (defined below in the implementation) and the metrics counter. No real ORT
// session is needed — all assertions are on return values and Prometheus
// counters.

#[cfg(test)]
mod arena_shrink_tests {
    use super::*;
    use serial_test::serial;

    /// Sets/unsets an env var around a closure, restoring the previous value.
    fn with_env<F: FnOnce()>(key: &str, val: Option<&str>, f: F) {
        let prev = std::env::var(key).ok();
        match val {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        f();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    // ── Test 1: memory_pattern=false → shrinkage enabled (auto gate) ─────────

    #[test]
    #[serial]
    fn embed_model_shrink_enabled_when_memory_pattern_false() {
        // No env override — gate follows memory_pattern.
        with_env("EMBED_ARENA_SHRINK_JINA_CODE_V2", None, || {
            let enabled = arena_shrink_enabled_for_model("jina-code-v2", false);
            assert!(
                enabled,
                "arena shrinkage should be enabled for memory_pattern=false (jina)"
            );
        });
    }

    // ── Test 2: memory_pattern=true → shrinkage disabled (auto gate) ─────────

    #[test]
    #[serial]
    fn embed_model_no_shrink_when_memory_pattern_true() {
        // No env override — gate follows memory_pattern.
        with_env("EMBED_ARENA_SHRINK_MULTILINGUAL_E5_LARGE", None, || {
            let enabled = arena_shrink_enabled_for_model("multilingual-e5-large", true);
            assert!(
                !enabled,
                "arena shrinkage must NOT be enabled for memory_pattern=true (e5-large)"
            );
        });
    }

    // ── Test 3: env force-on overrides memory_pattern=true ───────────────────

    #[test]
    #[serial]
    fn embed_arena_shrink_env_force_on_overrides_memory_pattern_true() {
        with_env(
            "EMBED_ARENA_SHRINK_MULTILINGUAL_E5_LARGE",
            Some("true"),
            || {
                // memory_pattern=true but env forces shrinkage on.
                let enabled = arena_shrink_enabled_for_model("multilingual-e5-large", true);
                assert!(
                    enabled,
                    "EMBED_ARENA_SHRINK_MULTILINGUAL_E5_LARGE=true must force shrinkage on \
                 regardless of memory_pattern"
                );
            },
        );
    }

    // ── Test 4: env force-off overrides memory_pattern=false ─────────────────

    #[test]
    #[serial]
    fn embed_arena_shrink_env_force_off_overrides_memory_pattern_false() {
        with_env("EMBED_ARENA_SHRINK_JINA_CODE_V2", Some("false"), || {
            // memory_pattern=false but env forces shrinkage off.
            let enabled = arena_shrink_enabled_for_model("jina-code-v2", false);
            assert!(
                !enabled,
                "EMBED_ARENA_SHRINK_JINA_CODE_V2=false must disable shrinkage \
                 regardless of memory_pattern"
            );
        });
    }

    // ── Test 4b: explicit `auto` sentinel = same as unset = follow memory_pattern.

    #[test]
    #[serial]
    fn embed_arena_shrink_env_auto_follows_memory_pattern() {
        with_env("EMBED_ARENA_SHRINK_JINA_CODE_V2", Some("auto"), || {
            assert!(
                arena_shrink_enabled_for_model("jina-code-v2", false),
                "auto + memory_pattern=false → shrinkage on"
            );
        });
        with_env(
            "EMBED_ARENA_SHRINK_MULTILINGUAL_E5_LARGE",
            Some("auto"),
            || {
                assert!(
                    !arena_shrink_enabled_for_model("multilingual-e5-large", true),
                    "auto + memory_pattern=true → shrinkage off"
                );
            },
        );
    }

    // ── Test 4c: invalid env value warns + falls back to auto-gate.

    #[test]
    #[serial]
    fn embed_arena_shrink_env_invalid_falls_back_to_auto() {
        // Operator typo "yes" — falls back to auto-gate behaviour
        // (warn emitted at runtime, not asserted here).
        with_env("EMBED_ARENA_SHRINK_JINA_CODE_V2", Some("yes"), || {
            assert!(
                arena_shrink_enabled_for_model("jina-code-v2", false),
                "invalid value 'yes' + memory_pattern=false → auto-gate enables shrinkage"
            );
        });
        with_env(
            "EMBED_ARENA_SHRINK_MULTILINGUAL_E5_LARGE",
            Some("nope"),
            || {
                assert!(
                    !arena_shrink_enabled_for_model("multilingual-e5-large", true),
                    "invalid value 'nope' + memory_pattern=true → auto-gate disables shrinkage"
                );
            },
        );
    }

    // ── Test 5: shrink call counter increments when enabled ───────────────────
    //
    // We cannot call `embed_tokens` without a real ONNX session, so we test
    // the metric helper directly: `record_arena_shrink_call` must increment
    // `embed_arena_shrink_calls_total{model}` by 1.

    #[test]
    fn embed_arena_shrink_calls_counter_increments_per_run() {
        // Use the shared test recorder (OnceLock — safe under parallel tests).
        let handle = crate::metrics::test_prometheus_handle();

        // Use a unique model label so this test is not sensitive to other
        // tests also calling record_arena_shrink_call with the same label.
        let model = "jina-code-v2-shrink-counter-test";

        // Read counter before and after the increment.
        let before = read_shrink_counter(handle, model);
        crate::metrics::record_arena_shrink_call(model);
        let after = read_shrink_counter(handle, model);

        assert_eq!(
            after,
            before + 1.0,
            "embed_arena_shrink_calls_total should increment by exactly 1 per call; \
             before={before} after={after}"
        );
    }

    /// Read `embed_arena_shrink_calls_total{model=...}` from the rendered
    /// Prometheus output. Returns 0.0 if the counter has not been emitted yet.
    fn read_shrink_counter(
        handle: &metrics_exporter_prometheus::PrometheusHandle,
        model: &str,
    ) -> f64 {
        let rendered = handle.render();
        let target = format!("embed_arena_shrink_calls_total{{model=\"{model}\"}}");
        rendered
            .lines()
            .find(|l| l.starts_with(&target))
            .and_then(|l| l.split_whitespace().last())
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
    }
}

// ── StandaloneEmbedder ────────────────────────────────────────────────────────

/// Standalone embedder for worker process — owns model state, no batcher/queue.
///
/// Worker creates one on startup and calls `infer` per IPC request.
/// Concurrency limit is enforced by the caller (worker main loop) via
/// `tokio::Semaphore` matching `EMBED_WORKER_POOL_SIZE`.
// Used by the `embed-worker` binary; the `embed-server` binary does not
// construct it. The allow below is intentional: the lib crate is compiled
// for both binaries; from `embed-server`'s perspective these items are unused.
#[allow(dead_code)]
pub struct StandaloneEmbedder {
    inner: EmbedModel,
    dims: u32,
}

impl StandaloneEmbedder {
    /// Load a single model by name. Looks up `ModelDef` in `cfg.models` and
    /// calls `EmbedModel::load`.
    ///
    /// `pool_size` maps to the ONNX session pool; for the worker process a
    /// value of 1 is typical (concurrency limited externally by the semaphore).
    // Called only from embed-worker binary, not embed-server binary.
    #[allow(dead_code)]
    pub fn load(
        model_name: &str,
        cfg: &crate::config::Config,
        intra_threads: usize,
        pool_size: usize,
    ) -> Result<Self, String> {
        let def = cfg
            .models
            .iter()
            .find(|m| m.name == model_name)
            .ok_or_else(|| format!("model {model_name} not found in EMBED_MODELS"))?;
        let dims = def.dim as u32;
        let inner = EmbedModel::load(
            def,
            intra_threads,
            false, // auto_truncate — worker does not silently truncate
            pool_size,
            0, // idle_evict_secs — disabled; worker is short-lived
        )?;
        Ok(Self { inner, dims })
    }

    /// Tokenize + embed in one call. Returns `(vectors, dims)`.
    ///
    /// `_max_seq_len` is currently unused — sequence capping is handled by the
    /// tokenizer config and `EmbedModel` internals. Phase 5 will wire
    /// per-request seq-len overrides once the IPC protocol is extended.
    // Called only from embed-worker binary, not embed-server binary.
    #[allow(dead_code)]
    #[allow(unused_variables)] // TODO(phase-5): wire _max_seq_len into tokenizer truncation
    pub fn infer(
        &self,
        texts: Vec<String>,
        _max_seq_len: u32,
    ) -> Result<(Vec<Vec<f32>>, u32), String> {
        let ids = self.inner.tokenize(&texts)?;
        let vecs = self.inner.embed_tokens(&ids)?;
        Ok((vecs, self.dims))
    }
}

// ── StandaloneReranker ────────────────────────────────────────────────────────

/// Standalone reranker for worker process — owns model state, no batcher/queue.
///
/// Worker creates one on startup and calls `score` per IPC request.
/// Concurrency limit is enforced by the caller via `tokio::Semaphore`.
#[allow(dead_code)]
pub struct StandaloneReranker {
    inner: crate::model_reranker::RerankerModel,
}

impl StandaloneReranker {
    /// Load a single reranker by name from `cfg.rerankers`.
    #[allow(dead_code)]
    pub fn load(
        model_name: &str,
        cfg: &crate::config::Config,
        intra_threads: usize,
        pool_size: usize,
    ) -> Result<Self, String> {
        let def = cfg
            .rerankers
            .iter()
            .find(|r| r.name == model_name)
            .ok_or_else(|| format!("reranker {model_name} not found in RERANKER_MODELS"))?;
        let inner = crate::model_reranker::RerankerModel::load(
            &def.name,
            &def.dir,
            def.max_len,
            def.padded_model,
            intra_threads,
            pool_size,
        )?;
        tracing::info!(
            model = %model_name,
            tokenizer_max_len = def.max_len,
            "reranker loaded; max_seq_len from IPC is advisory (tokenizer truncates at load-time max_len)"
        );
        Ok(Self { inner })
    }

    /// Tokenize query+docs, run cross-encoder, return one score per document.
    ///
    /// `_max_seq_len` is passed through for protocol parity but not yet used
    /// — truncation is governed by the tokenizer config at load time.
    #[allow(dead_code)]
    #[allow(unused_variables)]
    pub fn score(
        &self,
        query: String,
        documents: Vec<String>,
        _max_seq_len: u32,
    ) -> Result<Vec<f32>, String> {
        let token_ids = self.inner.tokenize_pairs(&query, &documents)?;
        self.inner.score_pairs(&token_ids)
    }
}

// ── StandaloneSplade ──────────────────────────────────────────────────────────

/// Standalone SPLADE encoder for worker process — owns model state, no batcher.
///
/// Worker creates one on startup and calls `encode` per IPC request.
/// Concurrency limit is enforced by the caller via `tokio::Semaphore`.
#[allow(dead_code)]
pub struct StandaloneSplade {
    inner: crate::model_splade::SpladeModel,
}

impl StandaloneSplade {
    /// Load a single SPLADE model by name from `cfg.splades`.
    #[allow(dead_code)]
    pub fn load(
        model_name: &str,
        cfg: &crate::config::Config,
        intra_threads: usize,
        pool_size: usize,
    ) -> Result<Self, String> {
        let def = cfg
            .splades
            .iter()
            .find(|s| s.name == model_name)
            .ok_or_else(|| format!("splade {model_name} not found in SPLADE_MODELS"))?;
        let inner = crate::model_splade::SpladeModel::load(
            &def.name,
            &def.dir,
            def.max_len,
            intra_threads,
            pool_size,
        )?;
        tracing::info!(
            model = %model_name,
            tokenizer_max_len = def.max_len,
            "splade loaded; max_seq_len from IPC is advisory (tokenizer truncates at load-time max_len)"
        );
        Ok(Self { inner })
    }

    /// Tokenize and encode a batch of texts into sparse vectors.
    ///
    /// Each text is encoded independently (SPLADE's `encode_sparse` is
    /// single-text-per-call by design). Returns one `Vec<(token_id, weight)>`
    /// per input text.
    ///
    /// `_max_seq_len` is passed for protocol parity but not yet wired —
    /// truncation is governed by the tokenizer config at load time.
    ///
    /// `top_k`: maximum sparse entries per output. 0 means unlimited (passes
    /// `usize::MAX` to `encode_sparse`, which applies no top-k truncation).
    ///
    /// `min_weight`: drop entries with weight <= this threshold. 0.0 disables
    /// filtering (only exact zeros from the post-ReLU output are dropped).
    #[allow(dead_code)]
    #[allow(unused_variables)]
    pub fn encode(
        &self,
        texts: Vec<String>,
        _max_seq_len: u32,
        top_k: u32,
        min_weight: f32,
    ) -> Result<Vec<Vec<(u32, f32)>>, String> {
        let effective_top_k: usize = if top_k == 0 {
            usize::MAX
        } else {
            top_k as usize
        };
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = self.inner.tokenize(&text)?;
            let sparse = self.inner.encode_sparse(ids, effective_top_k, min_weight)?;
            results.push(sparse);
        }
        Ok(results)
    }
}
