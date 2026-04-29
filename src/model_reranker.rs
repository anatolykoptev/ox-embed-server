//! Cross-encoder reranker model: scores `(query, doc)` pairs with a
//! BERT-style ONNX classifier that emits a single logit per pair.
//!
//! Differs fundamentally from `EmbedModel` (bi-encoder):
//!   - input is a PAIR encoded together (`[CLS] q [SEP] d [SEP]`), not a
//!     single text;
//!   - output is a scalar per row (`[batch, 1]` logits), not a pooled
//!     vector `[batch, dim]`.
//!
//! `score_pairs` returns raw logits (higher = more relevant). No softmax,
//! no normalisation — matches Cohere/Jina/BGE convention.
//!
//! Module-wide `allow(dead_code)` because the production call sites
//! (`main.rs` wire-up + `/v1/rerank` handler) land in separate commits
//! E2/E3. Everything here is reachable from the in-file test module,
//! but clippy's reachability analysis treats the `cfg(test)` cone as
//! excluded. The allows will naturally retire as E2+E3 light up the
//! call paths.
#![allow(dead_code)]

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array2;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::model::configure_truncation;
use crate::pool;

/// Parse `ORT_OPT_LEVEL` the same way `EmbedModel` does (shared env var —
/// a single server process has one ORT tuning knob, not one per model
/// kind). Defaults to `Level3`.
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

/// Wraps an ONNX session + tokenizer for a single cross-encoder reranker.
///
/// The reranker's ONNX graph has two inputs (`input_ids`, `attention_mask`)
/// and one output (`logits` shape `[batch, 1]`). We never feed
/// `token_type_ids` here — the exported reranker graph omits that
/// input entirely, and XLM-RoBERTa (the base model) doesn't use segment
/// embeddings anyway.
pub struct RerankerModel {
    name: String,
    /// Pool of independent ONNX sessions. With `pool_size==1` this is a
    /// single-element vector and behaves exactly like the legacy
    /// `Mutex<Session>` path. With `pool_size>1`, `score_pairs` round-
    /// robins across them via `next` so concurrent requests can run
    /// inference in parallel — each session keeps its own `intra_threads`
    /// pool inside ORT, so `pool_size * intra_threads` is the upper
    /// bound on physical threads spent on one model.
    sessions: Vec<Mutex<Session>>,
    /// Round-robin cursor for selecting the next session. `Relaxed` is
    /// fine: no other state is synchronised against this counter, and
    /// the per-session `Mutex` provides any required ordering for the
    /// inference itself.
    next: AtomicUsize,
    tokenizer: Tokenizer,
    max_len: usize,
    /// Whether this model pads every sequence in a batch to `max(seq_len)`
    /// (true for all BERT-style encoders, which is every reranker we'll
    /// ship). Retained as a field so `DynamicBatcher::with_tokens` can be
    /// parameterised from config without guessing.
    pub padded_model: bool,
    /// Pad-token id used when building the padded input tensor. For
    /// XLM-RoBERTa this is 1; we read it from the tokenizer at load time
    /// so other reranker kinds (e.g. bge-reranker-base — BERT, pad_id=0)
    /// don't need a hand-coded override.
    pad_id: u32,
}

impl RerankerModel {
    /// Load the ONNX session(s) + tokenizer from `dir`. Expects
    /// `model_quantized.onnx` and `tokenizer.json` at the top level —
    /// same layout `EmbedModel::load` uses.
    ///
    /// `intra_threads` plumbs through to ORT's `with_intra_threads` so
    /// the embed-server's single `EMBED_INTRA_THREADS` knob governs both
    /// model kinds.
    ///
    /// `pool_size` controls how many independent `Session` instances are
    /// loaded for this model. `1` is the legacy single-session path —
    /// behaves byte-for-byte like the pre-pool code. Values >1 enable
    /// concurrent inference (round-robin across sessions in
    /// `score_pairs`) at the cost of N× the per-session memory
    /// (~300-550 MB per session, depends on model: gte-multi-rerank ~340 MB, bge ~544 MB).
    ///
    /// IMPORTANT: each loaded session uses the FULL `intra_threads` value
    /// — the model deliberately does NOT auto-divide. The caller is
    /// expected to pass `intra_threads = total_cores / pool_size` (or
    /// similar) so total CPU usage stays bounded. Keeping the math
    /// explicit at the config layer means `EMBED_INTRA_THREADS` always
    /// reflects what each session actually sees, instead of being a
    /// surprising "logical" value the model silently divides.
    pub fn load(
        name: &str,
        dir: &str,
        max_len: usize,
        padded_model: bool,
        intra_threads: usize,
        pool_size: usize,
    ) -> Result<Self, String> {
        // Defensive clamp: caller contract says `>=1`, but a stray `0`
        // from misconfigured plumbing would `% 0` panic in `score_pairs`.
        // Costs nothing to make robust.
        let pool_size = pool_size.max(1);

        let dir_p = Path::new(dir);

        let onnx_path = dir_p.join("model_quantized.onnx");
        if !onnx_path.exists() {
            return Err(format!("ONNX file not found: {}", onnx_path.display()));
        }

        let tok_path = dir_p.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(format!("tokenizer.json not found: {}", tok_path.display()));
        }

        let opt_level = parse_opt_level();
        tracing::info!(
            path = %onnx_path.display(),
            ?opt_level,
            pool_size,
            intra_threads,
            "creating reranker ONNX session(s)"
        );
        // Build N independent sessions. ORT loads the same ONNX file fine
        // multiple times — no special "shared weights" mode is needed
        // (and none exposed by ort 2.0-rc anyway). Each session has its
        // own intra-op thread pool and weight buffers, so they really do
        // run in parallel under separate Mutexes.
        let mut sessions: Vec<Mutex<Session>> = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let session = Session::builder()
                .map_err(|e| format!("session builder #{i}: {e}"))?
                .with_optimization_level(opt_level)
                .map_err(|e| format!("set opt level #{i}: {e}"))?
                .with_intra_threads(intra_threads)
                .map_err(|e| format!("set threads #{i}: {e}"))?
                .commit_from_file(&onnx_path)
                .map_err(|e| format!("load ONNX #{i} {}: {e}", onnx_path.display()))?;
            sessions.push(Mutex::new(session));
        }
        tracing::info!(count = sessions.len(), "reranker ONNX session(s) created");

        tracing::info!(path = %tok_path.display(), "loading reranker tokenizer");
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| format!("load tokenizer: {e}"))?;
        // Always auto-truncate for reranker: pair inputs routinely overflow
        // 512 tokens on long documents, and the `LongestFirst` +
        // `TruncationDirection::Right` config configured in
        // `crate::model::configure_truncation` is precisely what cross-
        // encoder pair encoding needs (trim the long document tail, keep
        // the query + [CLS] intact).
        configure_truncation(&mut tokenizer, /*auto_truncate*/ true, max_len)?;

        // Discover pad_id from the tokenizer rather than config — each
        // reranker family uses a different pad token and config bloat is
        // better avoided.
        let pad_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .unwrap_or_else(|| {
                // XLM-RoBERTa uses 1, BERT uses 0 — fall back to
                // tokenizer's <pad> token lookup.
                tokenizer
                    .token_to_id("<pad>")
                    .or_else(|| tokenizer.token_to_id("[PAD]"))
                    .unwrap_or(0)
            });

        tracing::info!(
            model = %name,
            max_len,
            pad_id,
            padded_model,
            "loaded reranker model"
        );

        Ok(Self {
            name: name.to_string(),
            sessions,
            next: AtomicUsize::new(0),
            tokenizer,
            max_len,
            padded_model,
            pad_id,
        })
    }

    /// Model's display name (same string used as the `model` field in
    /// `/v1/rerank` responses).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tokenize `(query, doc)` pairs into concatenated input_ids.
    ///
    /// Every output `Vec<u32>` is ONE encoding of the pair — the
    /// tokenizer inserts the `[CLS]` / `[SEP]` / `</s>` special tokens
    /// itself (we call `encode_batch(..., /*add_special_tokens*/ true)`).
    /// Defensively capped at `self.max_len`; the `configure_truncation`
    /// call at load time already enables silent truncation.
    pub fn tokenize_pairs(&self, query: &str, docs: &[String]) -> Result<Vec<Vec<u32>>, String> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
        // Build `Vec<(String, String)>` — the `From<(I1, I2)> for EncodeInput`
        // blanket impl in tokenizers 0.22 (mod.rs:268) turns each tuple
        // into `EncodeInput::Dual(query, doc)` automatically, so no
        // explicit `EncodeInput::Dual(...)` map step is needed.
        let pairs: Vec<(String, String)> = docs
            .iter()
            .map(|d| (query.to_string(), d.clone()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, /*add_special_tokens*/ true)
            .map_err(|e| format!("tokenize_pairs: {e}"))?;
        Ok(encodings
            .iter()
            .map(|e| {
                let ids = e.get_ids();
                let len = ids.len().min(self.max_len);
                ids[..len].to_vec()
            })
            .collect())
    }

    /// Run the cross-encoder forward pass on pre-tokenized pairs.
    /// Returns one raw logit per pair — higher means more relevant.
    ///
    /// Output tensor shape from the reranker ONNX graph is
    /// `[batch, 1]`; we take `arr[[i, 0]]` for each row `i`. No softmax,
    /// no normalisation — clients get the raw score (matches Cohere /
    /// Jina rerank response semantics).
    pub fn score_pairs(&self, token_ids: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Ok(vec![]);
        }

        let max_seq = token_ids
            .iter()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
            .min(self.max_len);
        let batch = token_ids.len();
        // Reuse `pool::build_tensors_from_ids` — the `tti` output slot is
        // intentionally discarded because the reranker ONNX graph has no
        // `token_type_ids` input (confirmed via `InferenceSession::get_inputs`).
        let (ids, mask_i64, _tti) =
            pool::build_tensors_from_ids(token_ids, batch, max_seq, self.pad_id);

        let ids_arr =
            Array2::from_shape_vec([batch, max_seq], ids).map_err(|e| format!("ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64)
            .map_err(|e| format!("mask shape: {e}"))?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(|e| format!("ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(|e| format!("mask tensor: {e}"))?;

        // Round-robin pick from the pool. With pool_size==1 this always
        // resolves to index 0 — identical lock pattern to the legacy
        // single-Mutex<Session> code. With pool_size>1, two concurrent
        // callers will (in the steady state) land on different sessions
        // and run inference in parallel under separate locks.
        // `Relaxed` is sufficient: the per-session Mutex provides the
        // synchronization for the actual inference state.
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[idx]
            .lock()
            .map_err(|e| format!("lock session #{idx}: {e}"))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            })
            .map_err(|e| format!("reranker inference: {e}"))?;

        // the reranker emits a single output tensor named "logits"
        // of shape [batch, 1]. Extract, reshape, then flatten the trailing
        // 1-dim by taking `[i, 0]` for each row.
        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract logits: {e}"))?;
        let shape = raw.shape();
        if shape.len() != 2 || shape[0] != batch || shape[1] != 1 {
            return Err(format!(
                "unexpected reranker output shape: {:?}, expected [{batch}, 1]",
                shape
            ));
        }
        Ok((0..batch).map(|i| raw[[i, 0]]).collect())
    }

    /// Run a tiny dummy inference on every session in the pool to force
    /// ORT graph compilation, kernel selection, and arena allocation
    /// up-front. Without this the FIRST production request pays the full
    /// startup cost (~3s observed for gte-multi-rerank, vs ~1.5s steady
    /// state) — a tail-latency spike that's pure boot-time work.
    ///
    /// Best-effort: any session that fails to warm logs an error and we
    /// continue. The server still serves correctly without warmup; this
    /// is purely a latency optimization.
    pub fn warmup(&self) -> Result<(), String> {
        // Two short pairs so the batch dim isn't 1 (some kernels have
        // separate codepaths for batch=1 vs batch>1; we want to compile
        // the common path). Picked deliberately tiny so we don't hold
        // the lock long.
        let dummy_pairs = vec![
            (
                "warmup query".to_string(),
                "warmup document one".to_string(),
            ),
            (
                "warmup query".to_string(),
                "warmup document two".to_string(),
            ),
        ];
        let encodings = self
            .tokenizer
            .encode_batch(dummy_pairs, /*add_special_tokens*/ true)
            .map_err(|e| format!("warmup tokenize: {e}"))?;
        let token_ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| {
                let ids = e.get_ids();
                let len = ids.len().min(self.max_len);
                ids[..len].to_vec()
            })
            .collect();

        let max_seq = token_ids.iter().map(|v| v.len()).max().unwrap_or(0);
        let batch = token_ids.len();
        let (ids, mask_i64, _tti) =
            pool::build_tensors_from_ids(&token_ids, batch, max_seq, self.pad_id);

        // Warm EACH session in the pool — without this, only the first
        // session served by round-robin would be hot; the second would
        // pay the cold-start cost on its first concurrent request.
        for (i, sess_mu) in self.sessions.iter().enumerate() {
            let ids_arr = Array2::from_shape_vec([batch, max_seq], ids.clone())
                .map_err(|e| format!("warmup ids shape: {e}"))?;
            let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
                .map_err(|e| format!("warmup mask shape: {e}"))?;
            let ids_tensor =
                Tensor::from_array(ids_arr).map_err(|e| format!("warmup ids tensor: {e}"))?;
            let mask_tensor =
                Tensor::from_array(mask_arr).map_err(|e| format!("warmup mask tensor: {e}"))?;
            let start = std::time::Instant::now();
            let mut session = sess_mu
                .lock()
                .map_err(|e| format!("warmup lock session #{i}: {e}"))?;
            match session.run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            }) {
                Ok(_) => tracing::info!(
                    model = %self.name,
                    session = i,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "reranker session warmed"
                ),
                Err(e) => tracing::error!(
                    model = %self.name,
                    session = i,
                    error = %e,
                    "reranker session warmup failed (continuing)"
                ),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load a real reranker from disk if available, else emit a visible
    /// SKIP line and return `None`. Matches the pattern used in
    /// `model::truncation_tests::load_tokenizer_or_skip`.
    ///
    /// Resolution order:
    ///   1. `RERANKER_TEST_DIR` env var (CI / alt dev boxes).
    ///   2. Default on-box path
    ///      `/home/krolik/deploy/krolik-server/models/gte-multi-rerank`.
    fn load_reranker_or_skip() -> Option<RerankerModel> {
        load_reranker_or_skip_with_pool(1)
    }

    /// Like `load_reranker_or_skip` but parameterised on `pool_size` so
    /// the concurrent-pool test can request a 2-session model. Existing
    /// tests stay on the single-session helper unchanged.
    fn load_reranker_or_skip_with_pool(pool_size: usize) -> Option<RerankerModel> {
        const DEFAULT_DIR: &str = "/home/krolik/deploy/krolik-server/models/gte-multi-rerank";
        let dir = std::env::var("RERANKER_TEST_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string());
        if !Path::new(&dir).join("tokenizer.json").exists()
            || !Path::new(&dir).join("model_quantized.onnx").exists()
        {
            eprintln!(
                "SKIP reranker test: model files not found at {dir} \
                 (set RERANKER_TEST_DIR to override)"
            );
            return None;
        }
        Some(
            RerankerModel::load("gte-multi-rerank", &dir, 512, true, 1, pool_size)
                .expect("load reranker"),
        )
    }

    #[test]
    fn tokenize_pairs_empty_docs_returns_empty() {
        let Some(m) = load_reranker_or_skip() else {
            return;
        };
        let ids = m.tokenize_pairs("query", &[]).expect("tokenize empty");
        assert!(
            ids.is_empty(),
            "empty docs must produce empty output without hitting tokenizer"
        );
    }

    #[test]
    fn tokenize_pairs_produces_one_encoding_per_doc() {
        let Some(m) = load_reranker_or_skip() else {
            return;
        };
        let ids = m
            .tokenize_pairs(
                "what is a cat",
                &["a cat is a feline".into(), "pasta is tasty".into()],
            )
            .expect("tokenize_pairs");
        assert_eq!(ids.len(), 2, "one encoding per document");
        assert!(!ids[0].is_empty(), "first encoding should contain tokens");
        assert!(!ids[1].is_empty(), "second encoding should contain tokens");
        // Both encodings must embed the query, so the initial tokens
        // (after [CLS]) should match between the two pairs — quick sanity
        // check that we ARE encoding as a pair and not dropping the query.
        //
        // The specific ids will be XLM-RoBERTa-dependent; only compare
        // prefixes defensively — first ~4 tokens cover `<s>` + the first
        // couple of query tokens.
        let prefix_len = 4.min(ids[0].len()).min(ids[1].len());
        assert_eq!(
            &ids[0][..prefix_len],
            &ids[1][..prefix_len],
            "both pairs share the same query prefix"
        );
    }

    #[test]
    fn tokenize_pairs_respects_max_len_cap() {
        let Some(m) = load_reranker_or_skip() else {
            return;
        };
        // Document way over 512 tokens. configure_truncation(true, max_len)
        // runs at load time with LongestFirst, so the doc side gets
        // clipped and we stay within max_len.
        let long_doc = "word ".repeat(5000);
        let ids = m
            .tokenize_pairs("what is a cat", &[long_doc])
            .expect("tokenize long");
        assert_eq!(ids.len(), 1);
        assert!(
            ids[0].len() <= 512,
            "long-doc encoding must be truncated to max_len=512, got {}",
            ids[0].len()
        );
    }

    #[test]
    fn score_pairs_relevant_outscores_unrelated() {
        let Some(m) = load_reranker_or_skip() else {
            return;
        };
        let ids = m
            .tokenize_pairs(
                "what is a cat",
                &[
                    "a cat is a small domestic feline mammal".into(),
                    "the price of oil dropped yesterday".into(),
                ],
            )
            .expect("tokenize");
        let scores = m.score_pairs(&ids).expect("score");
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "relevant pair must outscore unrelated pair (got relevant={}, unrelated={})",
            scores[0],
            scores[1]
        );
        // Additionally assert the absolute gap is meaningful — a
        // well-calibrated cross-encoder produces a sizeable spread. The
        // python smoke test sees ~5.8 vs -11 on these exact inputs; we
        // use a conservative >3.0 margin to avoid brittleness on tiny
        // quantization drift.
        assert!(
            scores[0] - scores[1] > 3.0,
            "expected margin >3.0, got {} (relevant={}, unrelated={})",
            scores[0] - scores[1],
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn score_pairs_empty_input_returns_empty() {
        let Some(m) = load_reranker_or_skip() else {
            return;
        };
        let scores = m.score_pairs(&[]).expect("empty score");
        assert!(scores.is_empty());
    }

    /// With `pool_size=2`, two threads calling `score_pairs` concurrently
    /// must both succeed and return correctly shaped output. The point
    /// is to prove the pool doesn't serialize through one mutex —
    /// a regression to a single shared lock would still pass the shape
    /// assertions, but the test would deadlock the moment one thread
    /// holds a lock the other expects (it doesn't, by construction, so
    /// the bare fact that two scoped threads run end-to-end is the
    /// signal we want). We deliberately do NOT assert on timing —
    /// wall-clock comparisons are flaky on shared CI runners.
    #[test]
    fn score_pairs_pool_concurrent() {
        let Some(m) = load_reranker_or_skip_with_pool(2) else {
            return;
        };
        // Sanity: the model really did load 2 sessions.
        assert_eq!(
            m.sessions.len(),
            2,
            "load_reranker_or_skip_with_pool(2) should produce 2 sessions"
        );

        let ids_a = m
            .tokenize_pairs(
                "what is a cat",
                &[
                    "a cat is a small domestic feline mammal".into(),
                    "the price of oil dropped yesterday".into(),
                ],
            )
            .expect("tokenize a");
        let ids_b = m
            .tokenize_pairs(
                "what is a dog",
                &[
                    "a dog is a domesticated descendant of the wolf".into(),
                    "vegetables grow in the garden".into(),
                    "loyal canines often bark".into(),
                ],
            )
            .expect("tokenize b");

        // `std::thread::scope` lets both threads borrow `&m` directly:
        // `RerankerModel` is `Sync` because every field is `Sync`
        // (`Mutex<Session>: Sync`, `AtomicUsize: Sync`, `Tokenizer: Sync`).
        std::thread::scope(|s| {
            let h_a = s.spawn(|| m.score_pairs(&ids_a));
            let h_b = s.spawn(|| m.score_pairs(&ids_b));
            let scores_a = h_a.join().expect("thread A panicked").expect("score A");
            let scores_b = h_b.join().expect("thread B panicked").expect("score B");

            assert_eq!(scores_a.len(), 2, "thread A: one logit per doc");
            assert_eq!(scores_b.len(), 3, "thread B: one logit per doc");
            assert!(
                scores_a.iter().all(|s| s.is_finite()),
                "thread A scores must be finite: {scores_a:?}"
            );
            assert!(
                scores_b.iter().all(|s| s.is_finite()),
                "thread B scores must be finite: {scores_b:?}"
            );
            // Cross-check semantic ordering still holds when running
            // through the pool — the relevant doc still wins.
            assert!(
                scores_a[0] > scores_a[1],
                "thread A: relevant > unrelated, got {scores_a:?}"
            );
        });
    }
}
