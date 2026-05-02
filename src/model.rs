use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ndarray::Array2;
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

/// Wraps an ONNX session + tokenizer for a single embedding model.
pub struct EmbedModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub dim: usize,
    max_len: usize,
    pad_id: u32,
    has_token_type_ids: bool,
}

impl EmbedModel {
    /// Load model from a directory containing model_quantized.onnx
    /// and tokenizer.json.
    ///
    /// `auto_truncate`: if true (TEI-compat default), the tokenizer silently
    /// truncates inputs longer than `def.max_len`. If false, truncation is
    /// left disabled on the tokenizer.
    pub fn load(def: &ModelDef, intra_threads: usize, auto_truncate: bool) -> Result<Self, String> {
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
        let cache = CacheDir::from_env();
        let plan = LoadPlan::decide(cache.as_ref(), &onnx_path);
        tracing::info!(
            path = %onnx_path.display(),
            ?opt_level,
            cache_state = ?std::mem::discriminant(&plan),
            "creating ONNX session"
        );
        // Disable memory pattern: ORT pre-allocates per static input shape.
        // Our DynamicBatcher produces variable batch sizes (1..BATCH_MAX) and
        // variable seq_len (truncated per token budget). With pattern enabled,
        // each new (batch, seq_len) shape causes a fresh BFCArena extension
        // that's never released — a 31-min run grew from 3GB to 8GB and
        // cancelled queues across all models. Disabled, allocations are
        // sized per-request; couple-ms latency hit per batch is dwarfed by
        // the queue stalls we get under memory pressure.
        let builder = Session::builder().map_err(|e| format!("session builder: {e}"))?;
        let builder = onnx_cache::apply_plan(builder, &plan, opt_level)?;
        let load_path = plan.load_source(&onnx_path).to_path_buf();
        let t_commit = std::time::Instant::now();
        let session = builder
            .with_intra_threads(intra_threads)
            .map_err(|e| format!("set threads: {e}"))?
            // Phase H.17 (2026-05-01) — flipped false → true. Earlier comment
            // claimed memory_pattern caused n=10 batch regression; that was
            // before BATCH_MAX_TOKENS was capped (Phase H.17 compose change).
            // With shapes now bounded (max 32 items × max_seq=256 for e5-large),
            // memory_pattern lets ORT pre-plan the tensor layout per shape
            // and reuse buffers across same-shape calls — eliminating the
            // arena extend cycle that was producing 1.25 GiB slabs.
            .with_memory_pattern(true)
            .map_err(|e| format!("enable memory pattern: {e}"))?
            // Use the shared env-level arena registered in arena.rs (kSameAsRequested + bounded max_mem).
            // Avoids per-session BFCArena duplication and unbounded extension growth.
            .with_env_allocators()
            .map_err(|e| format!("enable env allocators: {e}"))?
            .commit_from_file(&load_path)
            .map_err(|e| format!("load ONNX {}: {e}", load_path.display()))?;
        onnx_cache::observe_post_commit(&plan, t_commit.elapsed().as_millis());

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
            "loaded model"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: def.dim,
            max_len: def.max_len,
            pad_id: def.pad_id,
            has_token_type_ids: def.has_token_type_ids,
        })
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

        // Pad to the longest sequence in the batch, capped at model max.
        let max_seq = token_ids
            .iter()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .min(self.max_len);

        let batch = token_ids.len();
        let (ids, mask_i64, tti) =
            pool::build_tensors_from_ids(token_ids, batch, max_seq, self.pad_id);

        let ids_arr =
            Array2::from_shape_vec([batch, max_seq], ids).map_err(|e| format!("ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
            .map_err(|e| format!("mask shape: {e}"))?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(|e| format!("ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(|e| format!("mask tensor: {e}"))?;

        let mut session = self.session.lock().map_err(|e| format!("lock: {e}"))?;

        let outputs = if self.has_token_type_ids {
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
        }
        .map_err(|e| format!("inference: {e}"))?;

        // Output shape: [batch, seq_len, dim]
        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let mask_arr_f = Array2::from_shape_vec([batch, max_seq], pool::mask_i64_to_f32(&mask_i64))
            .map_err(|e| format!("mask_f shape: {e}"))?;

        pool::mean_pool_normalize(&raw, &mask_arr_f, batch, max_seq, self.dim)
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
    pub fn warmup(&self, name: &str, shapes: &[usize]) -> Result<(), String> {
        if shapes.is_empty() {
            tracing::warn!(
                model = %name,
                "embed warmup called with empty shapes — skipping"
            );
            return Ok(());
        }
        for &batch in shapes {
            if let Err(e) = self.warmup_at_shape(name, batch) {
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
    fn warmup_at_shape(&self, name: &str, batch: usize) -> Result<(), String> {
        // `batch` copies of a tiny placeholder. We deliberately keep
        // the text very short so tokenization is cheap; the ONNX
        // forward pass dominates wall time anyway.
        let texts: Vec<String> = (0..batch).map(|_| "warmup".to_string()).collect();
        let token_ids = self.tokenize(&texts)?;
        let max_seq = token_ids
            .iter()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .min(self.max_len);
        if max_seq == 0 {
            return Err("warmup tokens produced empty sequence".to_string());
        }
        let (ids, mask_i64, tti) =
            pool::build_tensors_from_ids(&token_ids, batch, max_seq, self.pad_id);
        let ids_arr = Array2::from_shape_vec([batch, max_seq], ids)
            .map_err(|e| format!("warmup ids shape (batch={batch}): {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64)
            .map_err(|e| format!("warmup mask shape (batch={batch}): {e}"))?;
        let ids_tensor =
            Tensor::from_array(ids_arr).map_err(|e| format!("warmup ids tensor: {e}"))?;
        let mask_tensor =
            Tensor::from_array(mask_arr).map_err(|e| format!("warmup mask tensor: {e}"))?;

        let start = Instant::now();
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("warmup lock (batch={batch}): {e}"))?;

        let run_result = if self.has_token_type_ids {
            let tti_arr = Array2::from_shape_vec([batch, max_seq], tti)
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
                // EmbedModel currently holds a single Mutex<Session>
                // (no pool support yet — see struct comment). Stamp
                // session=0 so the log shape is identical across the
                // three model kinds; future EmbedModel pooling can
                // expand this to a `for (i, s) in pool.iter()` loop
                // without changing the log schema.
                session = 0,
                batch,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "embed session warmed"
            ),
            Err(e) => tracing::error!(
                model = %name,
                session = 0,
                batch,
                error = %e,
                "embed session warmup failed (continuing)"
            ),
        }
        Ok(())
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
