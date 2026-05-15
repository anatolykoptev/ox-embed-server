//! SPLADE-v3 sparse retrieval model: a learned sparse encoder that
//! turns a single text into a sparse vector over the BERT vocabulary.
//!
//! SPLADE = **S**parse **L**exical **A**n**D** **E**xpansion.
//!
//! # Output transform
//!
//! The ONNX graph emits dense logits of shape `[batch, seq, vocab]`.
//! SPLADE post-processes them as:
//!
//! ```text
//! relu     = max(logits, 0)
//! weights  = log(1 + relu)
//! masked   = weights * attention_mask.unsqueeze(-1)   # zero out pads
//! sparse   = max(masked, dim=seq)                     # max-pool over seq
//! ```
//!
//! The `[batch, vocab]` result is sparse: most entries are zero, and
//! the top-k weighted indices are the "expanded" terms a downstream
//! sparse retrieval index (e.g. Qdrant) consumes.
//!
//! # Differences from the other model kinds in this server
//!
//! * vs `EmbedModel` (bi-encoder dense): same single-text input, but
//!   output is sparse `(token_id, weight)` pairs, not a fixed-dim
//!   dense vector. No mean pooling or L2 normalisation — SPLADE's
//!   own log/relu/max-pool produces the final weights.
//! * vs `RerankerModel` (cross-encoder): different I/O shape entirely
//!   — single text, not a `(query, doc)` pair, and the output is a
//!   per-vocab vector, not a scalar logit.
//!
//! # Concurrency
//!
//! Loaded as a pool of N independent `Session` instances behind
//! `Vec<Mutex<Session>>`, with an `AtomicUsize` round-robin selector.
//! Mirrors `RerankerModel` exactly — see that module for the full
//! rationale (TL;DR: pool_size>1 lets concurrent requests run inference
//! in parallel, at N× the per-session memory cost).
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// Thread-local accumulator for the [vocab] sparse pool buffer
// (~122 KiB at vocab=30522). Reused across `encode_sparse` calls on
// the same tokio worker thread — saves a per-request heap alloc +
// zero-fill. `clear() + resize()` guarantees a fresh-zeroed buffer
// while preserving capacity. Per-thread isolation is safe: each
// `encode_sparse` call holds the `RefCell` mutably for the duration
// of its sweep + top-k extraction; no nested calls possible (sync
// code, no `.await` inside the `with` closure).
thread_local! {
    static SPARSE_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

use ndarray::Array2;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::model::configure_truncation;
use crate::onnx_cache::{self, CacheDir, LoadPlan};

/// Parse `ORT_OPT_LEVEL` the same way `EmbedModel` and `RerankerModel`
/// do — single shared knob for the whole process.
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

/// Wraps an ONNX session pool + tokenizer for one SPLADE model.
///
/// The ONNX graph has two inputs (`input_ids`, `attention_mask`) and one
/// output (`logits` of shape `[batch, seq, vocab_size]`). DistilBERT-
/// based SPLADE has no `token_type_ids` input, so we never feed one.
pub struct SpladeModel {
    name: String,
    /// Pool of ONNX sessions — round-robined per inference call. With
    /// pool_size==1 this behaves exactly like a single-Mutex<Session>
    /// model. See `RerankerModel` for the design rationale.
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
    tokenizer: Tokenizer,
    max_len: usize,
    /// Vocabulary size — output dimension. Captured from the model's
    /// own ONNX graph at load time so we don't hard-code 30522 anywhere
    /// (future SPLADE variants may use larger or rotated vocabularies).
    vocab_size: usize,
}

impl SpladeModel {
    /// Load the SPLADE model from `dir`. Expects:
    ///   - `<dir>/model.onnx` (NOTE: not `model_quantized.onnx` like the
    ///     reranker — splade-v3-distilbert ships as a single fp32 graph)
    ///   - `<dir>/tokenizer.json`
    ///
    /// `intra_threads` plumbs to ORT's `with_intra_threads` (per-session).
    /// `pool_size` is the number of independent ONNX sessions loaded.
    /// Same caveat as the reranker: each session uses the FULL
    /// `intra_threads` value, so the operator should ensure
    /// `pool_size * intra_threads <= total_cores`.
    pub fn load(
        name: &str,
        dir: &str,
        max_len: usize,
        intra_threads: usize,
        pool_size: usize,
    ) -> Result<Self, String> {
        crate::arena::assert_arena_registered_before_session();
        // Defensive clamp — `0` would `% 0` panic in `encode_sparse`.
        let pool_size = pool_size.max(1);

        let dir_p = Path::new(dir);

        let onnx_path = dir_p.join("model.onnx");
        if !onnx_path.exists() {
            return Err(format!("ONNX file not found: {}", onnx_path.display()));
        }

        let tok_path = dir_p.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(format!("tokenizer.json not found: {}", tok_path.display()));
        }

        let opt_level = parse_opt_level();
        // Per-model override: ONNX_OPT_CACHE_DIR_<MODEL_KEY_UPPER> takes
        // precedence over the global ONNX_OPT_CACHE_DIR.
        let cache = CacheDir::from_env_for_model(name);
        let key = crate::config::model_env_key(&name);
        let memory_pattern = crate::config::parse_memory_pattern(
            std::env::var(format!("EMBED_MEMORY_PATTERN_{key}"))
                .ok()
                .as_deref(),
        );
        tracing::info!(
            path = %onnx_path.display(),
            ?opt_level,
            pool_size,
            intra_threads,
            memory_pattern,
            "creating splade ONNX session(s)"
        );
        let mut sessions: Vec<Mutex<Session>> = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            // Re-check cache state per iteration: session 0 misses and
            // writes, sessions 1..N hit on the file written by session 0.
            let plan = LoadPlan::decide(cache.as_ref(), &onnx_path);
            let load_path = plan.load_source(&onnx_path).to_path_buf();
            let t_commit = std::time::Instant::now();
            // memory_pattern: per-model knob (EMBED_MEMORY_PATTERN_<MODEL_UPPER>).
            // Default true (back-compat). See config::ModelDef::memory_pattern for rationale.
            let builder = Session::builder().map_err(|e| format!("session builder #{i}: {e}"))?;
            let builder = onnx_cache::apply_plan(builder, &plan, opt_level)
                .map_err(|e| format!("apply cache plan #{i}: {e}"))?;
            let session = builder
                .with_intra_threads(intra_threads)
                .map_err(|e| format!("set threads #{i}: {e}"))?
                // memory_pattern: per-model knob parsed from env at load time.
                .with_memory_pattern(memory_pattern)
                .map_err(|e| format!("enable memory pattern #{i}: {e}"))?
                .with_env_allocators()
                .map_err(|e| format!("enable env allocators #{i}: {e}"))?
                // Disable per-session CPU mem arena (see model.rs for detail).
                .with_execution_providers([ort::ep::CPU::default()
                    .with_arena_allocator(false)
                    .build()])
                .map_err(|e| format!("disable per-session cpu mem arena #{i}: {e}"))?
                .commit_from_file(&load_path)
                .map_err(|e| format!("load ONNX #{i} {}: {e}", load_path.display()))?;
            onnx_cache::observe_post_commit(&plan, t_commit.elapsed().as_millis());
            sessions.push(Mutex::new(session));
        }
        tracing::info!(count = sessions.len(), "splade ONNX session(s) created");

        // Discover the output vocab dim from the ONNX graph rather than
        // hard-coding 30522. Stays robust if a future SPLADE variant
        // ships with a different head size, and matches the principled
        // load-time-introspection stance the reranker takes for pad_id.
        // We probe the first session's `outputs()` metadata; the per-
        // session lock here is held only long enough to read the static
        // shape info, never during inference.
        let vocab_size: usize = {
            let s = sessions[0].lock().map_err(|e| format!("lock #0: {e}"))?;
            let outputs = s.outputs();
            if outputs.is_empty() {
                return Err("splade ONNX graph has no outputs".to_string());
            }
            // Output 0 is `logits` with shape [batch, seq, vocab].
            // The first two dims are symbolic (-1 in ort's `Shape`),
            // the last is the concrete vocab size.
            let shape = outputs[0]
                .dtype()
                .tensor_shape()
                .ok_or_else(|| "splade output 0 is not a tensor".to_string())?;
            // `Shape` derefs to `[i64]`.
            let dims: &[i64] = shape;
            let last = dims.last().copied().unwrap_or(-1);
            if last <= 0 {
                return Err(format!(
                    "splade output last dim must be a concrete vocab size, got shape {:?}",
                    dims
                ));
            }
            last as usize
        };
        tracing::info!(vocab_size, "splade vocab discovered");

        tracing::info!(path = %tok_path.display(), "loading splade tokenizer");
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| format!("load tokenizer: {e}"))?;
        // Always auto-truncate: SPLADE has no pair semantics, but a
        // single document still routinely overruns 512 tokens — the
        // standard `LongestFirst` + right truncation drops the tail and
        // keeps the [CLS] / leading content intact.
        configure_truncation(&mut tokenizer, /*auto_truncate*/ true, max_len)?;

        tracing::info!(
            model = %name,
            max_len,
            vocab_size,
            "loaded splade model"
        );

        Ok(Self {
            name: name.to_string(),
            sessions,
            next: AtomicUsize::new(0),
            tokenizer,
            max_len,
            vocab_size,
        })
    }

    /// Display name (used in API responses).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Output vocab size — the dense dimensionality the sparse vector
    /// is drawn from. Exposed for handler-side sanity checks / metrics.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Convert a vocab token id back to its surface form. Used by the
    /// smoke test for visibility ("what does the model think 'cat'
    /// expands to?") and available to handlers if they ever want to
    /// stream decoded terms instead of raw indices.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Tokenize a single text into `input_ids`. The tokenizer adds the
    /// standard `[CLS] ... [SEP]` BERT special tokens. Truncation is
    /// applied at load time; we defensively cap at `self.max_len` here
    /// too so callers get a hard guarantee on the bound.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self
            .tokenizer
            .encode(text, /*add_special_tokens*/ true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids = enc.get_ids();
        let len = ids.len().min(self.max_len);
        Ok(ids[..len].to_vec())
    }

    /// Run a single-text SPLADE forward pass and return a sparse vector
    /// as a list of `(token_id, weight)` pairs sorted by weight descending.
    ///
    /// Steps:
    ///   1. Build `[1, seq]` input_ids tensor and an all-ones attention mask
    ///      (single-text input has no pad positions inside the sequence —
    ///      the tokenizer already truncated to max_len).
    ///   2. Run inference, extract logits `[1, seq, vocab]`.
    ///   3. Compute `log(1 + ReLU(logits))` per (seq, vocab) cell.
    ///   4. Multiply by the attention mask broadcast across the vocab axis
    ///      (zero out pad positions). Even though the mask is all-ones in
    ///      single-text mode, we keep the multiply so the code shape matches
    ///      the formula and a future batched version doesn't need to change
    ///      anything but the input prep.
    ///   5. Max-pool across the seq axis → `[vocab]`.
    ///   6. Filter `weight > min_weight`, sort desc, take `top_k`.
    pub fn encode_sparse(
        &self,
        token_ids: Vec<u32>,
        top_k: usize,
        min_weight: f32,
    ) -> Result<Vec<(u32, f32)>, String> {
        if token_ids.is_empty() {
            return Ok(vec![]);
        }
        if top_k == 0 {
            return Ok(vec![]);
        }

        let seq_len = token_ids.len().min(self.max_len);
        let batch = 1usize;

        // Build `[1, seq]` ids tensor directly — no padding needed for
        // a single-text request, so we skip `pool::build_tensors_from_ids`
        // (which is engineered around batched padding + the optional
        // token_type_ids slot SPLADE doesn't have).
        let ids_i64: Vec<i64> = token_ids[..seq_len].iter().map(|&id| id as i64).collect();
        let mask_i64: Vec<i64> = vec![1i64; seq_len];

        let ids_arr = Array2::from_shape_vec([batch, seq_len], ids_i64)
            .map_err(|e| format!("ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, seq_len], mask_i64.clone())
            .map_err(|e| format!("mask shape: {e}"))?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(|e| format!("ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(|e| format!("mask tensor: {e}"))?;

        // Round-robin pick from the pool — identical pattern to the
        // reranker. `Relaxed` ordering is sufficient: the per-session
        // Mutex provides the actual memory ordering for inference.
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[idx]
            .lock()
            .map_err(|e| format!("lock session #{idx}: {e}"))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            })
            .map_err(|e| format!("splade inference: {e}"))?;

        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract logits: {e}"))?;
        let shape = raw.shape();
        if shape.len() != 3
            || shape[0] != batch
            || shape[1] != seq_len
            || shape[2] != self.vocab_size
        {
            return Err(format!(
                "unexpected splade output shape: {:?}, expected [{batch}, {seq_len}, {}]",
                shape, self.vocab_size
            ));
        }

        // We can't `drop(session)` here — `outputs` (and thus `raw`)
        // borrows from the SessionOutputs lifetime, which itself is
        // tied to the live `&mut session`. The pooling sweep below is
        // pure CPU on the borrowed view; we accept holding the per-
        // session lock for that extra ~1ms rather than copying the full
        // [1, seq, vocab] f32 buffer (~60MB at seq=512) just to release
        // a lock that's already round-robined across the pool. A
        // pool_size>1 deployment still gets concurrency: another caller
        // hits the next session in the rotation.
        // Compute log(1 + ReLU(logits)) and max-pool over seq simultaneously.
        // Allocating `[seq, vocab]` intermediate would waste memory
        // (~512 * 30522 * 4 bytes = 62 MB per request). Instead we keep
        // a running `[vocab]` max as we sweep seq positions.
        //
        // The `[vocab]` accumulator buffer (~122 KiB at vocab=30522) is
        // reused via `thread_local!` across requests on the same tokio
        // worker thread — avoids per-request heap alloc + zero-fill.
        let vocab = self.vocab_size;
        SPARSE_BUF.with(|cell| -> Result<Vec<(u32, f32)>, String> {
            let mut sparse = cell.borrow_mut();
            sparse.clear();
            sparse.resize(vocab, 0.0);

            for j in 0..seq_len {
                // Mask is all-ones in single-text mode (built above), but we
                // honour it for parity with the formula. A 0 mask would zero
                // the contribution from this seq position.
                let m = mask_i64[j] as f32;
                if m == 0.0 {
                    continue;
                }
                for k in 0..vocab {
                    let lo = raw[[0, j, k]];
                    if lo > 0.0 {
                        // log1p(x) = ln(1 + x); SPLADE uses natural log.
                        let w = (1.0 + lo).ln() * m;
                        if w > sparse[k] {
                            sparse[k] = w;
                        }
                    }
                }
            }

            // Collect indices/weights above min_weight, then take top_k via
            // `select_nth_unstable_by` (O(n) partial-sort) instead of full
            // O(n log n) sort. For n~800 active terms and top_k=256, that's
            // 800 + 256·log(256) = ~2850 ops vs 7700 for full sort — ~2.7× win.
            let mut entries: Vec<(u32, f32)> = sparse
                .iter()
                .enumerate()
                .filter_map(|(idx, &w)| {
                    if w > min_weight {
                        Some((idx as u32, w))
                    } else {
                        None
                    }
                })
                .collect();
            if entries.len() > top_k {
                // Partition: first `top_k` entries are the top-k by weight desc
                // (unordered among themselves).
                entries.select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
                entries.truncate(top_k);
            }
            // Final sort of the (now small) top-k slice.
            entries.sort_by(|a, b| b.1.total_cmp(&a.1));
            Ok(entries)
        })
    }

    /// Pre-warm every session in the pool by running one dummy
    /// inference per requested shape. SPLADE's `encode_sparse` is
    /// hard-coded to `batch=1` (single text in, sparse vector out), so
    /// each entry in `shapes` triggers a batch=1 inference — the list
    /// length controls how many warmup passes run, not the input shape
    /// itself. Default `[1]` runs one warmup per session, the legacy
    /// (pre-shape-warmup) coverage.
    ///
    /// Best-effort per shape: a failure logs a warn and we move on,
    /// matching `RerankerModel::warmup` and `EmbedModel::warmup`.
    pub fn warmup(&self, shapes: &[usize]) -> Result<(), String> {
        if shapes.is_empty() {
            tracing::warn!(
                model = %self.name,
                "splade warmup called with empty shapes — skipping"
            );
            return Ok(());
        }
        for &batch in shapes {
            if let Err(e) = self.warmup_at_shape(batch) {
                tracing::warn!(
                    model = %self.name,
                    batch,
                    error = %e,
                    "splade shape warmup failed (continuing with remaining shapes)"
                );
            }
        }
        Ok(())
    }

    /// One warmup pass. SPLADE always runs batch=1 inference at the
    /// graph level (the wrapping API takes a single `Vec<u32>`), so
    /// `batch` here is informational — we still feed a `[1, seq]`
    /// tensor. Logging stamps the requested batch so the operator can
    /// confirm their `SPLADE_WARMUP_BATCH_SIZES` was honoured.
    fn warmup_at_shape(&self, batch: usize) -> Result<(), String> {
        // Mirror `encode_sparse` shape-construction: tokenize a tiny
        // placeholder and build [1, seq_len] tensors directly. The
        // tokenizer is configured with truncation at load time so
        // we're bounded above by `self.max_len`.
        let token_ids = self
            .tokenize("warmup splade")
            .map_err(|e| format!("warmup tokenize (batch={batch}): {e}"))?;
        if token_ids.is_empty() {
            return Err("warmup tokens produced empty sequence".to_string());
        }
        let seq_len = token_ids.len().min(self.max_len);
        let ids_i64: Vec<i64> = token_ids[..seq_len].iter().map(|&id| id as i64).collect();
        let mask_i64: Vec<i64> = vec![1i64; seq_len];

        // Warm EVERY session in the pool — same rationale as the
        // reranker. Round-robin would otherwise leave session #1+ cold
        // until prod traffic happens to land there.
        for (i, sess_mu) in self.sessions.iter().enumerate() {
            let ids_arr = Array2::from_shape_vec([1, seq_len], ids_i64.clone())
                .map_err(|e| format!("warmup ids shape (batch={batch}): {e}"))?;
            let mask_arr = Array2::from_shape_vec([1, seq_len], mask_i64.clone())
                .map_err(|e| format!("warmup mask shape (batch={batch}): {e}"))?;
            let ids_tensor = Tensor::from_array(ids_arr)
                .map_err(|e| format!("warmup ids tensor (batch={batch}): {e}"))?;
            let mask_tensor = Tensor::from_array(mask_arr)
                .map_err(|e| format!("warmup mask tensor (batch={batch}): {e}"))?;

            let start = Instant::now();
            let mut session = sess_mu
                .lock()
                .map_err(|e| format!("warmup lock session #{i} (batch={batch}): {e}"))?;
            match session.run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            }) {
                Ok(_) => tracing::info!(
                    model = %self.name,
                    session = i,
                    batch,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "splade session warmed"
                ),
                Err(e) => tracing::error!(
                    model = %self.name,
                    session = i,
                    batch,
                    error = %e,
                    "splade session warmup failed (continuing)"
                ),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load a real SPLADE model from disk if available, else SKIP.
    /// Mirrors `model_reranker::tests::load_reranker_or_skip`.
    fn load_splade_or_skip() -> Option<SpladeModel> {
        load_splade_or_skip_with_pool(1)
    }

    fn load_splade_or_skip_with_pool(pool_size: usize) -> Option<SpladeModel> {
        const DEFAULT_DIR: &str = "/home/krolik/deploy/krolik-server/models/splade-v3-distilbert";
        let dir = std::env::var("SPLADE_TEST_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string());
        if !Path::new(&dir).join("tokenizer.json").exists()
            || !Path::new(&dir).join("model.onnx").exists()
        {
            eprintln!(
                "SKIP splade test: model files not found at {dir} \
                 (set SPLADE_TEST_DIR to override)"
            );
            return None;
        }
        // Simulate the init sequence: tests call load() directly without going
        // through main.rs, so we set the arena flag manually to satisfy the
        // assert in SpladeModel::load.
        crate::arena::ARENA_REGISTERED.store(true, std::sync::atomic::Ordering::Release);
        Some(
            SpladeModel::load("splade-v3-distilbert", &dir, 512, 1, pool_size)
                .expect("load splade"),
        )
    }

    #[test]
    fn tokenize_basic_query() {
        let Some(m) = load_splade_or_skip() else {
            return;
        };
        let ids = m.tokenize("what is a cat").expect("tokenize");
        assert!(
            !ids.is_empty(),
            "tokenize must return at least [CLS] ... [SEP]"
        );
        assert!(
            ids.len() <= 512,
            "tokenize must stay within max_len, got {}",
            ids.len()
        );
        // BERT-style: first token is [CLS] (id 101), last is [SEP] (id 102).
        // Quick sanity that we asked for special tokens.
        assert_eq!(ids[0], 101, "expected [CLS] at position 0, got {}", ids[0]);
        assert_eq!(
            ids[ids.len() - 1],
            102,
            "expected [SEP] at last position, got {}",
            ids[ids.len() - 1]
        );
    }

    #[test]
    fn encode_sparse_returns_terms() {
        let Some(m) = load_splade_or_skip() else {
            return;
        };
        let ids = m.tokenize("what is a cat").expect("tokenize");
        let entries = m.encode_sparse(ids, 256, 0.0).expect("encode_sparse");

        assert!(
            entries.len() >= 5,
            "splade should produce at least 5 expansion terms, got {}",
            entries.len()
        );
        // ReLU + log(1+x) — every weight strictly positive.
        for (idx, w) in &entries {
            assert!(
                *w > 0.0,
                "weight for token {idx} must be > 0 (post-ReLU), got {w}"
            );
        }
        // Sorted descending — clients depend on this order.
        for window in entries.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "entries must be sorted by weight desc: {:?} then {:?}",
                window[0],
                window[1]
            );
        }

        // Visibility: print the top-5 decoded tokens so the operator can
        // eyeball the semantic expansion (e.g. cat → feline, kitten...).
        // Only printed under `--nocapture`. Not asserted — vocabulary
        // strings are checkpoint-dependent.
        let top5: Vec<(String, f32)> = entries
            .iter()
            .take(5)
            .map(|(id, w)| {
                let tok = m.id_to_token(*id).unwrap_or_else(|| format!("<id:{id}>"));
                (tok, *w)
            })
            .collect();
        println!("SPLADE expansion top-5 for 'what is a cat': {top5:?}");
    }

    #[test]
    fn encode_sparse_pool_concurrent() {
        let Some(m) = load_splade_or_skip_with_pool(2) else {
            return;
        };
        // Sanity: the model really did load 2 sessions.
        assert_eq!(
            m.sessions.len(),
            2,
            "load_splade_or_skip_with_pool(2) should produce 2 sessions"
        );

        let ids_a = m.tokenize("what is a cat").expect("tokenize a");
        let ids_b = m.tokenize("how do dogs bark").expect("tokenize b");

        // `std::thread::scope` lets both threads borrow `&m` directly:
        // `SpladeModel` is `Sync` because every field is (`Mutex<Session>`,
        // `AtomicUsize`, `Tokenizer`, `String`, `usize`).
        std::thread::scope(|s| {
            let h_a = s.spawn(|| m.encode_sparse(ids_a.clone(), 64, 0.0));
            let h_b = s.spawn(|| m.encode_sparse(ids_b.clone(), 64, 0.0));
            let entries_a = h_a.join().expect("thread A panicked").expect("encode A");
            let entries_b = h_b.join().expect("thread B panicked").expect("encode B");

            assert!(!entries_a.is_empty(), "thread A produced no entries");
            assert!(!entries_b.is_empty(), "thread B produced no entries");
            assert!(
                entries_a.iter().all(|(_, w)| w.is_finite() && *w > 0.0),
                "thread A weights must all be finite positive: {entries_a:?}"
            );
            assert!(
                entries_b.iter().all(|(_, w)| w.is_finite() && *w > 0.0),
                "thread B weights must all be finite positive: {entries_b:?}"
            );
        });
    }
}
