//! `RerankerModel::load` — startup-time construction. Splitting it out
//! of the runtime hot path (mod.rs) keeps each file under the
//! maintainability budget AND mirrors the natural concern boundary:
//! everything in this file runs ONCE at process boot, behind no metric.
//!
//! Includes:
//!   - ONNX session pool creation
//!   - tokenizer load + truncation config
//!   - graph-input introspection (Phase 1B — startup self-documenting
//!     guard against unexpected ONNX graph shapes)
//!   - pad_id discovery
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use tokenizers::Tokenizer;

use super::RerankerModel;
use crate::model::configure_truncation;

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

/// Parse `ORT_INTRA_OP_SPINNING` — controls whether ORT's intra-op thread
/// pool busy-waits before parking. Default `false` because embed-server
/// shares a 4-core ARM Neoverse-N1 host with memdb-go, postgres, and
/// other services; busy-waiting wastes CPU that other services need
/// during their own spikes. Flip to `1` only on dedicated-CPU hardware
/// where the ~5-15% latency reduction justifies a 100%-of-allocated-cores
/// idle CPU floor while inference is in flight. Per ORT docs: spinning
/// faster on tight inference loops, worse on bursty multi-tenant boxes.
fn parse_intra_op_spinning() -> bool {
    matches!(
        std::env::var("ORT_INTRA_OP_SPINNING").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
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
        let allow_spinning = parse_intra_op_spinning();
        tracing::info!(
            path = %onnx_path.display(),
            ?opt_level,
            pool_size,
            intra_threads,
            allow_spinning,
            "creating reranker ONNX session(s)"
        );

        let sessions = build_session_pool(
            name,
            &onnx_path,
            opt_level,
            intra_threads,
            pool_size,
            allow_spinning,
        )?;
        introspect_graph_inputs(name, &sessions)?;

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

        let pad_id = discover_pad_id(&tokenizer);

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
}

/// Build N independent ONNX sessions over the same model file. ORT loads
/// the same path multiple times fine — there is no special "shared
/// weights" mode in ort 2.0-rc — so each session pays its own ~340 MB to
/// ~550 MB weight buffer cost, in exchange for true parallelism under
/// independent Mutexes.
fn build_session_pool(
    _name: &str,
    onnx_path: &Path,
    opt_level: GraphOptimizationLevel,
    intra_threads: usize,
    pool_size: usize,
    allow_spinning: bool,
) -> Result<Vec<Mutex<Session>>, String> {
    let mut sessions: Vec<Mutex<Session>> = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        // See model.rs for the rationale: memory pattern + dynamic batches
        // = unbounded BFCArena growth. Reranker pool members are even more
        // sensitive because they also see variable doc counts.
        let session = Session::builder()
            .map_err(|e| format!("session builder #{i}: {e}"))?
            .with_optimization_level(opt_level)
            .map_err(|e| format!("set opt level #{i}: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| format!("set threads #{i}: {e}"))?
            // Phase 3B (Plan 2026-05-01) — gate ORT's intra-op spin via env.
            // `OMP_WAIT_POLICY=PASSIVE` only governs OpenMP, NOT ORT's own
            // intra pool — explicit `with_intra_op_spinning(false)` is the
            // only way to stop the spin on a shared multi-tenant CPU.
            .with_intra_op_spinning(allow_spinning)
            .map_err(|e| format!("set intra spinning #{i}: {e}"))?
            // Phase H.17 (2026-05-01) — flipped false → true. Earlier
            // experiment (n=10 batch, 3.55s → 6.87s) was contaminated by
            // unbounded BATCH_MAX_TOKENS=32768 and a leaky shared arena;
            // memory_pattern re-planned constantly under huge variable
            // shapes, dominated by re-plan cost. Now with H.17 cap of
            // BATCH_MAX_TOKENS=8192 + RERANKER_BATCH_MAX=8 the rerank
            // batch shape varies only across small bounded set
            // ({1,2,...,8} items × actual_max_seq), so ORT pre-plans for
            // each shape and reuses across same-shape calls — exactly
            // what eliminates the arena extend cycle.
            .with_memory_pattern(true)
            .map_err(|e| format!("enable memory pattern #{i}: {e}"))?
            .with_env_allocators()
            .map_err(|e| format!("enable env allocators #{i}: {e}"))?
            .commit_from_file(onnx_path)
            .map_err(|e| format!("load ONNX #{i} {}: {e}", onnx_path.display()))?;
        sessions.push(Mutex::new(session));
    }
    tracing::info!(count = sessions.len(), "reranker ONNX session(s) created");
    Ok(sessions)
}

/// Phase 1B — log the actual graph input names + dtypes the loaded ONNX
/// expects, and warn loudly if they differ from `{input_ids,
/// attention_mask}`. The runtime call site
/// (`super::inference::score_pairs`) statically passes only those two
/// inputs, so a model export that adds an unexpected input
/// (`token_type_ids`, `position_ids`) would otherwise surface only as an
/// inference-time cryptic error. We warn and let the server come up —
/// ORT will fail loudly itself if the graph genuinely cannot run.
fn introspect_graph_inputs(name: &str, sessions: &[Mutex<Session>]) -> Result<(), String> {
    let session = sessions[0]
        .lock()
        .map_err(|e| format!("introspect session #0: {e}"))?;
    let inputs = session.inputs();
    let names: Vec<&str> = inputs.iter().map(|o| o.name()).collect();
    tracing::info!(
        model = %name,
        inputs = ?names,
        "reranker ONNX graph inputs"
    );
    let expected: std::collections::HashSet<&str> =
        ["input_ids", "attention_mask"].into_iter().collect();
    let actual: std::collections::HashSet<&str> = names.iter().copied().collect();
    if actual != expected {
        tracing::warn!(
            model = %name,
            expected = ?expected,
            actual = ?actual,
            "reranker ONNX graph inputs differ from {{input_ids, attention_mask}} — \
             the inference call may fail or silently ignore inputs"
        );
    }
    for outlet in inputs.iter() {
        tracing::debug!(
            model = %name,
            input = outlet.name(),
            dtype = ?outlet.dtype(),
            "reranker input dtype"
        );
    }
    Ok(())
}

/// Discover the pad token id from the tokenizer rather than config —
/// every reranker family ships a different one (XLM-RoBERTa = 1, BERT
/// = 0, OLMo / ModernBERT = 1) and config bloat is better avoided.
fn discover_pad_id(tokenizer: &Tokenizer) -> u32 {
    tokenizer
        .get_padding()
        .map(|p| p.pad_id)
        .unwrap_or_else(|| {
            tokenizer
                .token_to_id("<pad>")
                .or_else(|| tokenizer.token_to_id("[PAD]"))
                .unwrap_or(0)
        })
}
