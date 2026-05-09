use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::Array2;
use ort::ep;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};

use crate::config::ModelDef;
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

/// Wraps a pool of ONNX sessions + a single tokenizer for one embedding
/// model.
///
/// `sessions` mirrors the reranker pool (`model_reranker/load.rs`):
/// `pool_size==1` is byte-for-byte equivalent to the legacy
/// `Mutex<Session>` path (single-element vector, `next` always %1 == 0).
/// `pool_size>1` lets concurrent `embed_tokens` calls run inference in
/// parallel under independent Mutexes — round-robin via `next` keeps the
/// load even without coordination.
///
/// IMPORTANT: ort 2.0-rc has no shared-weights mode, so each pool member
/// holds its own ~400 MiB weight buffer. The pool size is intentionally
/// off by default (operator opt-in via `EMBED_SESSION_POOL_SIZE`).
pub struct EmbedModel {
    /// Model name for metric labels (e.g. "jina-code-v2").
    name: String,
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
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
}

impl EmbedModel {
    /// Load model from a directory containing model_quantized.onnx
    /// and tokenizer.json.
    ///
    /// `auto_truncate`: if true (TEI-compat default), the tokenizer silently
    /// truncates inputs longer than `def.max_len`. If false, truncation is
    /// left disabled on the tokenizer.
    ///
    /// `pool_size` controls how many independent ONNX sessions are
    /// created. `1` (the historical default) preserves the legacy
    /// single-session path exactly. Values >1 enable concurrent
    /// inference, at N× the per-session weight memory cost — see the
    /// struct doc comment.
    pub fn load(
        def: &ModelDef,
        intra_threads: usize,
        auto_truncate: bool,
        pool_size: usize,
    ) -> Result<Self, String> {
        // Defensive clamp — caller contract says >=1, but a stray 0 from
        // misconfigured plumbing would `% 0` panic in `embed_tokens`.
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

        let onnx_path = dir.join("model_quantized.onnx");
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
            "creating ONNX session(s)"
        );

        let sessions = build_session_pool(
            &def.name,
            &onnx_path,
            opt_level,
            intra_threads,
            pool_size,
            def.memory_pattern,
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
            "loaded model"
        );

        Ok(Self {
            name: def.name.clone(),
            sessions,
            next: AtomicUsize::new(0),
            tokenizer,
            dim: def.dim,
            max_len: def.max_len,
            pad_id: def.pad_id,
            has_token_type_ids: def.has_token_type_ids,
            num_heads,
        })
    }

    /// Number of sessions in the inference pool. Used by tests to assert
    /// `pool_size` plumbing and by future ops tooling. Held even when the
    /// production hot path doesn't read it — mirrors the reranker
    /// `session_count` accessor for symmetry.
    #[allow(dead_code)]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
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

        let (ids, mask_i64, tti) =
            pool::build_tensors_from_ids(token_ids, batch, max_seq, self.pad_id);

        let ids_arr =
            Array2::from_shape_vec([batch, max_seq], ids).map_err(|e| format!("ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
            .map_err(|e| format!("mask shape: {e}"))?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(|e| format!("ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(|e| format!("mask tensor: {e}"))?;

        // Round-robin pick from the pool. With pool_size==1 this always
        // resolves to index 0 — identical lock pattern to the legacy
        // single-Mutex<Session> code. With pool_size>1, concurrent callers
        // (in the steady state) land on different sessions and run
        // inference in parallel under separate locks.
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[idx]
            .lock()
            .map_err(|e| format!("lock session #{idx}: {e}"))?;

        let run_result = if self.has_token_type_ids {
            let tti_arr = Array2::from_shape_vec([batch, max_seq], tti)
                .map_err(|e| format!("tti shape: {e}"))?;
            let tti_tensor = Tensor::from_array(tti_arr).map_err(|e| format!("tti tensor: {e}"))?;
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
        let (ids, mask_i64, tti) =
            pool::build_tensors_from_ids(&token_ids, batch, max_seq, self.pad_id);

        // Warm EVERY session in the pool — without this, only the first
        // session served by round-robin would be hot; the second would
        // pay the cold-start cost on its first concurrent request.
        for (i, sess_mu) in self.sessions.iter().enumerate() {
            let ids_arr = Array2::from_shape_vec([batch, max_seq], ids.clone())
                .map_err(|e| format!("warmup ids shape (batch={batch}): {e}"))?;
            let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
                .map_err(|e| format!("warmup mask shape (batch={batch}): {e}"))?;
            let ids_tensor =
                Tensor::from_array(ids_arr).map_err(|e| format!("warmup ids tensor: {e}"))?;
            let mask_tensor =
                Tensor::from_array(mask_arr).map_err(|e| format!("warmup mask tensor: {e}"))?;

            let start = Instant::now();
            let mut session = sess_mu
                .lock()
                .map_err(|e| format!("warmup lock session #{i} (batch={batch}): {e}"))?;

            let run_result = if self.has_token_type_ids {
                let tti_arr = Array2::from_shape_vec([batch, max_seq], tti.clone())
                    .map_err(|e| format!("warmup tti shape (batch={batch}): {e}"))?;
                let tti_tensor =
                    Tensor::from_array(tti_arr).map_err(|e| format!("warmup tti tensor: {e}"))?;
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

/// Build N independent ONNX sessions over the same model file. Mirrors
/// the reranker `build_session_pool` in `model_reranker/load.rs` —
/// there is no shared-weights mode in ort 2.0-rc, so each session pays
/// its own ~400 MiB (multilingual-e5-large) / ~250 MiB (jina-code-v2)
/// weight buffer cost in exchange for true parallelism under
/// independent Mutexes.
///
/// memory_pattern + env-allocator + per-session-CPU-arena-disabled
/// configuration is identical to the pre-pool single-session path —
/// see the in-place comments at the call site below for rationale.
fn build_session_pool(
    model_key: &str,
    onnx_path: &Path,
    opt_level: GraphOptimizationLevel,
    intra_threads: usize,
    pool_size: usize,
    memory_pattern: bool,
) -> Result<Vec<Mutex<Session>>, String> {
    crate::arena::assert_arena_registered_before_session();
    // Resolve the cache dir once per pool. The decision (hit / miss) is
    // re-evaluated *per session* inside the loop: session 0 sees a miss
    // and writes the optimized graph; sessions 1..N see a hit on their
    // own re-check and skip the Level3 pass entirely.
    // Per-model override: ONNX_OPT_CACHE_DIR_<MODEL_KEY_UPPER> takes
    // precedence over the global ONNX_OPT_CACHE_DIR.
    let cache = CacheDir::from_env_for_model(model_key);
    let mut sessions: Vec<Mutex<Session>> = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let plan = LoadPlan::decide(cache.as_ref(), onnx_path);
        let load_path = plan.load_source(onnx_path).to_path_buf();
        let t_commit = std::time::Instant::now();
        // memory_pattern=true: ORT plans scratch reuse within the shared
        // env-level arena (registered in `arena.rs`). Combined with
        // `DisableCpuMemArena` (PR #34), the session has only the shared
        // arena to draw from — pattern planning amortizes per-shape
        // scratch allocations across requests.
        let builder = Session::builder().map_err(|e| format!("session builder #{i}: {e}"))?;
        let builder = onnx_cache::apply_plan(builder, &plan, opt_level)
            .map_err(|e| format!("apply cache plan #{i}: {e}"))?;
        let session = builder
            .with_intra_threads(intra_threads)
            .map_err(|e| format!("set threads #{i}: {e}"))?
            .with_memory_pattern(memory_pattern)
            .map_err(|e| format!("enable memory pattern #{i}: {e}"))?
            // Use the shared env-level arena registered in arena.rs.
            .with_env_allocators()
            .map_err(|e| format!("enable env allocators #{i}: {e}"))?
            // Belt-and-braces: disable the per-session CPU mem arena.
            // Without this, ORT's CPU EP defaults to EnableCpuMemArena=1
            // and may still spawn a session-local BFCArena alongside our
            // shared one, doubling allocator state.
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(|e| format!("disable per-session cpu mem arena #{i}: {e}"))?
            .commit_from_file(&load_path)
            .map_err(|e| format!("load ONNX #{i} {}: {e}", load_path.display()))?;
        onnx_cache::observe_post_commit(&plan, t_commit.elapsed().as_millis());
        sessions.push(Mutex::new(session));
    }
    tracing::info!(count = sessions.len(), "embed ONNX session(s) created");
    Ok(sessions)
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
    ///    `/home/krolik/deploy/krolik-server/models/multilingual-e5-large/tokenizer.json`.
    ///
    /// When neither exists, the test returns `None` and the caller
    /// early-returns after printing a visible skip notice. The skip line
    /// is loud on purpose so the test doesn't silently vanish from CI
    /// output the way `#[ignore]` would.
    fn load_tokenizer_or_skip() -> Option<Tokenizer> {
        const DEFAULT_PATH: &str =
            "/home/krolik/deploy/krolik-server/models/multilingual-e5-large/tokenizer.json";
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
