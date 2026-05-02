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
//! ## Module layout
//!
//! Split across files at concern boundaries (per CLAUDE.md):
//!
//! | File          | Concern                                              |
//! |---------------|------------------------------------------------------|
//! | `mod.rs`      | struct + runtime hot path (`tokenize_pairs`, `score_pairs`, `warmup`) |
//! | `load.rs`     | `RerankerModel::load` + ONNX-graph introspection (Phase 1B)           |
//! | `tests.rs`    | integration tests (skipped when no model on disk)    |
//!
//! Module-wide `allow(dead_code)` because the production call sites
//! (`main.rs` wire-up + `/v1/rerank` handler) land in separate commits
//! E2/E3. Everything here is reachable from the in-file test module,
//! but clippy's reachability analysis treats the `cfg(test)` cone as
//! excluded. The allows will naturally retire as E2+E3 light up the
//! call paths.
#![allow(dead_code)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::pool;

mod load;
mod tests;

/// Wraps an ONNX session + tokenizer for a single cross-encoder reranker.
///
/// The reranker's ONNX graph has two inputs (`input_ids`, `attention_mask`)
/// and one output (`logits` shape `[batch, 1]`). We never feed
/// `token_type_ids` here — the exported reranker graph omits that input
/// entirely, and XLM-RoBERTa / OLMo / ModernBERT base models don't use
/// segment embeddings anyway. Phase 1B (`load::introspect_graph_inputs`)
/// logs a warning at startup if a future model export changes this
/// assumption.
pub struct RerankerModel {
    pub(super) name: String,
    /// Pool of independent ONNX sessions. With `pool_size==1` this is a
    /// single-element vector and behaves exactly like the legacy
    /// `Mutex<Session>` path. With `pool_size>1`, `score_pairs` round-
    /// robins across them via `next` so concurrent requests can run
    /// inference in parallel — each session keeps its own `intra_threads`
    /// pool inside ORT, so `pool_size * intra_threads` is the upper
    /// bound on physical threads spent on one model.
    pub(super) sessions: Vec<Mutex<Session>>,
    /// Round-robin cursor for selecting the next session. `Relaxed` is
    /// fine: no other state is synchronised against this counter, and
    /// the per-session `Mutex` provides any required ordering for the
    /// inference itself.
    pub(super) next: AtomicUsize,
    pub(super) tokenizer: Tokenizer,
    pub(super) max_len: usize,
    /// Whether this model pads every sequence in a batch to `max(seq_len)`
    /// (true for all BERT-style encoders, which is every reranker we'll
    /// ship). Retained as a field so `DynamicBatcher::with_tokens` can be
    /// parameterised from config without guessing.
    pub padded_model: bool,
    /// Pad-token id used when building the padded input tensor. For
    /// XLM-RoBERTa this is 1; we read it from the tokenizer at load time
    /// so other reranker kinds (e.g. bge-reranker-base — BERT, pad_id=0)
    /// don't need a hand-coded override.
    pub(super) pad_id: u32,
}

impl RerankerModel {
    /// Model's display name (same string used as the `model` field in
    /// `/v1/rerank` responses).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of sessions in the inference pool. Used by tests to assert
    /// that `pool_size` plumbed through, and by future ops tooling to
    /// expose pool capacity in `/health` if needed. Cheap (`Vec::len`).
    pub fn session_count(&self) -> usize {
        self.sessions.len()
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
        // 2026-05-01 — switched from `Vec<(String, String)>` to
        // `Vec<(&str, &str)>`. Old code allocated 2N Strings per call:
        // `query.to_string()` for each doc + `d.clone()` for each doc.
        // For batch=32, that was 64 heap allocations on the hot path.
        // tokenizers 0.22 `encode_batch` accepts any `I: Into<EncodeInput>`,
        // and the blanket `From<(I1, I2)> for EncodeInput` covers
        // `(&str, &str)` via `InputSequence::from(&str)`. Zero allocs.
        let pairs: Vec<(&str, &str)> = docs.iter().map(|d| (query, d.as_str())).collect();
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
    /// Output tensor shape from the reranker ONNX graph is `[batch, 1]`;
    /// we take `arr[[i, 0]]` for each row `i`. No softmax, no
    /// normalisation — clients get the raw score (matches Cohere / Jina
    /// rerank response semantics).
    ///
    /// Emits Phase 1A metrics:
    ///   - `embed_rerank_pool_acquire_duration_seconds{model}` — mutex wait
    ///   - `embed_rerank_inference_duration_seconds{model}` — `session.run` only
    ///   - `embed_rerank_batch_size{model}` — number of pairs in this pass
    ///   - `embed_rerank_padding_waste_ratio{model}` — `(padded - real) / padded`
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
        // Sum of clipped real lengths — used for padding-waste ratio. Each
        // sequence contributes `min(len, max_len)` tokens of actual compute;
        // the rest of the `[batch, max_seq]` tensor is padding.
        let real_tokens: usize = token_ids.iter().map(|v| v.len().min(self.max_len)).sum();
        // Reuse `pool::build_tensors_from_ids` — the `tti` output slot is
        // intentionally discarded because the reranker ONNX graph has no
        // `token_type_ids` input (confirmed via `Session::inputs()` at load).
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
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let acquire_start = Instant::now();
        let mut session = self.sessions[idx]
            .lock()
            .map_err(|e| format!("lock session #{idx}: {e}"))?;
        crate::metrics::record_rerank_pool_acquire(&self.name, acquire_start.elapsed());

        let inference_start = Instant::now();
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            })
            .map_err(|e| format!("reranker inference: {e}"))?;
        crate::metrics::record_rerank_inference(&self.name, inference_start.elapsed(), batch);
        crate::metrics::record_rerank_padding_waste(
            &self.name,
            batch.saturating_mul(max_seq),
            real_tokens,
        );

        // The reranker emits a single output tensor named "logits" of
        // shape [batch, 1]. Extract, reshape, then flatten the trailing
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
            let start = Instant::now();
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
