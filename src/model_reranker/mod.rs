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
    /// Phase H.20 — optional fast-path session pool with FIXED static shape
    /// `[1, max_len]`. Loaded only if `<dir>/model_quantized_static.onnx`
    /// exists alongside the dynamic file. ORT pre-folds 700+ runtime
    /// shape-computation nodes into constants, giving 1.74x speedup
    /// per-call vs the dynamic graph (bench: 1359ms → 781ms on
    /// gte-reranker-modernbert-base, batch=1, seq=256, ARM Neoverse-N1).
    ///
    /// Routing: `score_pairs` checks the actual batch size — a single-pair
    /// call (batch=1, the dominant prod path for memdb-go's per-pair
    /// rerank loop) goes through the static pool; multi-pair calls
    /// (batch>1) keep using the dynamic pool because the static graph's
    /// input shape is literally `[1, ?]` and ORT will reject `[N, ?]`.
    pub(super) static_sessions: Option<Vec<Mutex<Session>>>,
    /// Round-robin cursor for the static pool. Separate from `next` so
    /// the two pools cycle independently and a busy dynamic pool can't
    /// stall the static fast path.
    pub(super) static_next: AtomicUsize,
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

        // Phase H.20 fast path: route batch=1 calls to the static-shape
        // session if loaded. The static graph is `[1, max_len]` fixed —
        // ORT pre-folded ~700 runtime shape ops into constants giving
        // 1.74× speedup vs the dynamic graph (1359ms → 781ms standalone
        // bench, gte-reranker-modernbert-base, ARM Neoverse-N1).
        // batch>1 falls through to the dynamic path below because the
        // static graph rejects shapes other than `[1, max_len]`.
        if token_ids.len() == 1
            && let Some(static_sessions) = &self.static_sessions
        {
            return self.score_pairs_static(token_ids, static_sessions);
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

    /// Phase H.20 fast-path inference for batch=1 calls. Uses the
    /// static-shape session pool (`[1, max_len]` fixed graph) which
    /// avoids the ~700 runtime shape-computation nodes of the dynamic
    /// graph. Always pads to `self.max_len` (no per-call max_seq calc)
    /// because that's literally the only shape the static graph accepts.
    ///
    /// Trade-off: short single-pair calls pad to full max_len even when
    /// the actual content is shorter (lose padding-waste advantage of
    /// the dynamic path). The 1.74× per-call speedup more than offsets
    /// this for typical seq>=64; for seq<32 the dynamic path may be
    /// faster but the difference is sub-millisecond and not worth
    /// switching back per call.
    fn score_pairs_static(
        &self,
        token_ids: &[Vec<u32>],
        static_sessions: &[Mutex<Session>],
    ) -> Result<Vec<f32>, String> {
        debug_assert_eq!(token_ids.len(), 1);
        let static_seq = self.max_len;
        let real_tokens = token_ids[0].len().min(static_seq);

        let (ids, mask_i64, _tti) =
            pool::build_tensors_from_ids(token_ids, 1, static_seq, self.pad_id);

        let ids_arr = Array2::from_shape_vec([1, static_seq], ids)
            .map_err(|e| format!("static ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([1, static_seq], mask_i64)
            .map_err(|e| format!("static mask shape: {e}"))?;

        let ids_tensor =
            Tensor::from_array(ids_arr).map_err(|e| format!("static ids tensor: {e}"))?;
        let mask_tensor =
            Tensor::from_array(mask_arr).map_err(|e| format!("static mask tensor: {e}"))?;

        let idx = self.static_next.fetch_add(1, Ordering::Relaxed) % static_sessions.len();
        let acquire_start = Instant::now();
        let mut session = static_sessions[idx]
            .lock()
            .map_err(|e| format!("lock static session #{idx}: {e}"))?;
        crate::metrics::record_rerank_pool_acquire(&self.name, acquire_start.elapsed());

        let inference_start = Instant::now();
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            })
            .map_err(|e| format!("static reranker inference: {e}"))?;
        crate::metrics::record_rerank_inference(&self.name, inference_start.elapsed(), 1);
        crate::metrics::record_rerank_padding_waste(&self.name, static_seq, real_tokens);

        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("static extract logits: {e}"))?;
        let shape = raw.shape();
        if shape.len() != 2 || shape[0] != 1 || shape[1] != 1 {
            return Err(format!(
                "unexpected static reranker output shape: {:?}, expected [1, 1]",
                shape
            ));
        }
        Ok(vec![raw[[0, 0]]])
    }

    /// Run a tiny dummy inference on every session in the pool, once per
    /// requested batch shape, to force ORT graph compilation, kernel
    /// selection, and arena allocation up-front. Without this the FIRST
    /// production request at a given batch shape pays the full startup
    /// cost (~3s observed for gte-multi-rerank cold, vs ~1.5s steady) —
    /// a tail-latency spike that's pure boot-time work and that ALSO
    /// re-fires when production traffic shifts to a new batch size
    /// (e.g. memdb-go's batch=5 fanout vs batch=1 single-pair calls).
    ///
    /// `shapes` is a list of batch sizes (operator-controlled via
    /// `RERANK_WARMUP_BATCH_SIZES`). For each shape `B` we synthesise
    /// `B` dummy `(query, doc)` pairs so ORT compiles its kernels for
    /// that exact `[B, max_seq]` graph shape. Order in `shapes` is
    /// preserved — the first listed shape is compiled first and is the
    /// "best path" if the static-shape fast-path (Phase H.20) is
    /// loaded; the rest force the dynamic graph to bind kernels for the
    /// other shapes prod will see.
    ///
    /// Best-effort per shape: a single shape failing (e.g. operator put
    /// `batch=99` and the static graph rejects it) only logs a warn and
    /// the next shape proceeds. Total startup cost is bounded by
    /// `shapes.len() * sessions.len()` — `parse_warmup_batch_sizes`
    /// dedupes the input so duplicate shapes don't double-count.
    pub fn warmup(&self, shapes: &[usize]) -> Result<(), String> {
        if shapes.is_empty() {
            // Defensive — `parse_warmup_batch_sizes` falls back to
            // defaults on empty input, so this branch never trips in
            // production. Still guard against direct callers (tests,
            // future code paths) so `shapes[0]` indexing in the loop
            // can't panic.
            tracing::warn!(
                model = %self.name,
                "reranker warmup called with empty shapes — skipping"
            );
            return Ok(());
        }
        for &batch in shapes {
            if let Err(e) = self.warmup_at_shape(batch) {
                // Per-shape failure is non-fatal: log and try the next
                // shape. The static-shape fast-path will loudly reject
                // batch>1 with `[N, ?]` — that's expected for any shape
                // we throw at it that isn't 1, so we just continue.
                tracing::warn!(
                    model = %self.name,
                    batch,
                    error = %e,
                    "reranker shape warmup failed (continuing with remaining shapes)"
                );
            }
        }
        Ok(())
    }

    /// Run one pre-warm pass at exactly `batch` items across every
    /// session in the dynamic pool. Helper extracted from `warmup` so
    /// per-shape error handling stays clean (one fallible scope per
    /// shape, no manual cleanup of partial state).
    fn warmup_at_shape(&self, batch: usize) -> Result<(), String> {
        // `batch` of the same query-doc pair — content doesn't matter,
        // only the resulting `[batch, max_seq]` tensor shape does.
        // Tiny strings keep tokenization cheap; we cap to `max_len`
        // below so the mask shape matches what production traffic
        // produces at this batch dim.
        let pairs: Vec<(&str, &str)> =
            (0..batch).map(|_| ("warmup query", "warmup document")).collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, /*add_special_tokens*/ true)
            .map_err(|e| format!("warmup tokenize batch={batch}: {e}"))?;
        let token_ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| {
                let ids = e.get_ids();
                let len = ids.len().min(self.max_len);
                ids[..len].to_vec()
            })
            .collect();
        // All rows tokenize identically (same query + doc), so max =
        // their common length. Compute defensively in case a future
        // tokenizer change breaks that assumption.
        let max_seq = token_ids.iter().map(|v| v.len()).max().unwrap_or(0);
        let (ids, mask_i64, _tti) =
            pool::build_tensors_from_ids(&token_ids, batch, max_seq, self.pad_id);

        // Warm EACH session in the pool — without this, only the first
        // session served by round-robin would be hot; the second would
        // pay the cold-start cost on its first concurrent request.
        for (i, sess_mu) in self.sessions.iter().enumerate() {
            let ids_arr = Array2::from_shape_vec([batch, max_seq], ids.clone())
                .map_err(|e| format!("warmup ids shape (batch={batch}): {e}"))?;
            let mask_arr = Array2::from_shape_vec([batch, max_seq], mask_i64.clone())
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
                    "reranker session warmed"
                ),
                Err(e) => tracing::error!(
                    model = %self.name,
                    session = i,
                    batch,
                    error = %e,
                    "reranker session warmup failed (continuing)"
                ),
            }
        }
        Ok(())
    }
}
