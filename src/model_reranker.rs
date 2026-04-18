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
/// `token_type_ids` here — the exported BGE-reranker graph omits that
/// input entirely, and XLM-RoBERTa (the base model) doesn't use segment
/// embeddings anyway.
pub struct RerankerModel {
    name: String,
    session: Mutex<Session>,
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
    /// Load the ONNX session + tokenizer from `dir`. Expects
    /// `model_quantized.onnx` and `tokenizer.json` at the top level —
    /// same layout `EmbedModel::load` uses.
    ///
    /// `intra_threads` plumbs through to ORT's `with_intra_threads` so
    /// the embed-server's single `EMBED_INTRA_THREADS` knob governs both
    /// model kinds.
    pub fn load(
        name: &str,
        dir: &str,
        max_len: usize,
        padded_model: bool,
        intra_threads: usize,
    ) -> Result<Self, String> {
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
            "creating reranker ONNX session"
        );
        let session = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .with_optimization_level(opt_level)
            .map_err(|e| format!("set opt level: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| format!("set threads: {e}"))?
            .commit_from_file(&onnx_path)
            .map_err(|e| format!("load ONNX {}: {e}", onnx_path.display()))?;
        tracing::info!("reranker ONNX session created");

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
            session: Mutex::new(session),
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
    /// Output tensor shape from the bge-reranker-v2-m3 ONNX graph is
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

        let mut session = self.session.lock().map_err(|e| format!("lock: {e}"))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            })
            .map_err(|e| format!("reranker inference: {e}"))?;

        // bge-reranker-v2-m3 emits a single output tensor named "logits"
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
    ///      `/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3`.
    fn load_reranker_or_skip() -> Option<RerankerModel> {
        const DEFAULT_DIR: &str = "/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3";
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
        Some(RerankerModel::load("bge-reranker-v2-m3", &dir, 512, true, 1).expect("load reranker"))
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
}
