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
//!   - static-shape ONNX discovery (Phase H.20 + multi-shape extension,
//!     2026-05-02): scans `<dir>/model_quantized_static_b<N>.onnx` files
//!     and the legacy unsuffixed `model_quantized_static.onnx` (treated
//!     as `b=1` for backwards compatibility with PR #27).
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use tokenizers::Tokenizer;

use super::RerankerModel;
use crate::model::configure_truncation;
use crate::onnx_cache::{self, CacheDir, LoadPlan};

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

        // Phase H.20 + 2026-05-02 multi-shape extension —
        // opportunistically load static-shape sibling session pools from
        // `<dir>/model_quantized_static_b<N>.onnx` files. Each batch-size
        // axis (`N`) gets its own session pool keyed in a `BTreeMap`, so
        // routing in `score_pairs` is a single `.get(&len)` lookup.
        // Backwards-compat: a legacy unsuffixed `model_quantized_static.onnx`
        // (the PR #27 convention) is treated as `b=1`.
        // Convention-based, no env config — operators drop the file in
        // place and the matching shape activates automatically. Per-shape
        // pool size hard-coded to 2 (mirrors the dynamic default and
        // matches the typical 4-core / 4-inflight config).
        let static_session_pools = load_static_session_pools(
            name,
            dir_p,
            opt_level,
            intra_threads,
            allow_spinning,
        );
        let static_session_cursors = static_session_pools
            .keys()
            .map(|k| (*k, AtomicUsize::new(0)))
            .collect();

        tracing::info!(
            model = %name,
            max_len,
            pad_id,
            padded_model,
            static_pool_shapes = ?static_session_pools.keys().copied().collect::<Vec<_>>(),
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
            static_session_pools,
            static_session_cursors,
        })
    }
}

/// Per-shape session pool size. PR #27 hard-coded the legacy single-shape
/// pool to 2. Matches the typical 4-core / 4-inflight config and keeps
/// the multi-shape memory budget bounded (see
/// `docs/plans/2026-05-02-multi-shape-static-export.md` § Memory budget).
/// Hoisted to a module const so the discovery + load split share a
/// single source of truth.
const STATIC_POOL_SIZE_PER_SHAPE: usize = 2;

/// Discovery — scan `dir` for static-shape ONNX files and produce a
/// `(batch_size, path)` map. Pure: no I/O beyond `read_dir`, no session
/// build, deterministic ordering (`BTreeMap` sorts by key).
///
/// Recognises:
///   - `model_quantized_static_b<N>.onnx` — explicit batch axis (N>=1).
///   - `model_quantized_static.onnx` — legacy PR #27 convention,
///     treated as `b=1`.
///
/// If both an explicit `_b1` file AND the legacy unsuffixed file exist,
/// the explicit file wins (explicit > implicit) and the duplicate is
/// logged at `warn`. Files matching the prefix but failing the
/// `_b<digits>` regex (e.g. `_btest`, `_b1.bak`) are ignored with a
/// `debug` log.
///
/// Returns an empty map when no static files are present — that's the
/// expected default for any reranker without an export, not an error.
fn discover_static_shape_files(name: &str, dir: &Path) -> BTreeMap<usize, PathBuf> {
    const PREFIX_EXPLICIT: &str = "model_quantized_static_b";
    const LEGACY_FILENAME: &str = "model_quantized_static.onnx";

    let mut out: BTreeMap<usize, PathBuf> = BTreeMap::new();

    // Phase 1 — scan the directory for `model_quantized_static_b<N>.onnx`.
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(
                model = %name,
                dir = %dir.display(),
                error = %e,
                "static-shape discovery: read_dir failed (treating as empty)"
            );
            return out;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !file_name.starts_with(PREFIX_EXPLICIT) || !file_name.ends_with(".onnx") {
            continue;
        }
        // Carve out the `<N>` between the prefix and `.onnx`.
        let mid = &file_name[PREFIX_EXPLICIT.len()..file_name.len() - ".onnx".len()];
        match mid.parse::<usize>() {
            Ok(0) => {
                tracing::warn!(
                    model = %name,
                    path = %path.display(),
                    "static-shape file has batch=0 — ignoring (no zero-batch graph is meaningful)"
                );
            }
            Ok(n) => {
                if let Some(prev) = out.insert(n, path.clone()) {
                    tracing::warn!(
                        model = %name,
                        batch = n,
                        kept = %path.display(),
                        dropped = %prev.display(),
                        "duplicate static-shape file for same batch size — keeping last-seen"
                    );
                }
            }
            Err(_) => {
                tracing::debug!(
                    model = %name,
                    path = %path.display(),
                    suffix = %mid,
                    "static-shape filename matches prefix but suffix is not a number — ignoring"
                );
            }
        }
    }

    // Phase 2 — legacy PR #27 convention: unsuffixed file = batch=1.
    let legacy = dir.join(LEGACY_FILENAME);
    if legacy.exists() {
        match out.insert(1, legacy.clone()) {
            None => {
                // Most common path: only the legacy file is present, no
                // explicit `_b1`. Use it as `b=1` silently.
            }
            Some(replaced) => {
                // Conflict: explicit `_b1.onnx` was already in the map.
                // Restore the explicit file (explicit > implicit) and
                // warn about the duplicate so ops can clean up.
                tracing::warn!(
                    model = %name,
                    explicit = %replaced.display(),
                    legacy = %legacy.display(),
                    "both model_quantized_static_b1.onnx and model_quantized_static.onnx present — \
                     keeping the explicit _b1 file; remove the legacy file to silence this warning"
                );
                out.insert(1, replaced);
            }
        }
    }

    out
}

/// Phase H.20 + 2026-05-02 multi-shape extension — load every
/// `model_quantized_static_b<N>.onnx` discovered under `dir` as a
/// per-shape session pool. Returns an empty `BTreeMap` (with `debug`
/// log) when no static files are present — that's the normal case for
/// rerankers that don't have a static export, and not an error.
///
/// Logs a `warn` (and skips that shape) when a discovered file fails
/// session creation (e.g. corrupted ONNX, unsupported opset) so the
/// fallback to the dynamic-only path for that batch size is visible in
/// ops, not silent. The model still serves correctly via the dynamic
/// pool for the failed shape; other successfully loaded shapes are
/// unaffected.
fn load_static_session_pools(
    name: &str,
    dir_p: &Path,
    opt_level: GraphOptimizationLevel,
    intra_threads: usize,
    allow_spinning: bool,
) -> BTreeMap<usize, Vec<Mutex<Session>>> {
    let discovered = discover_static_shape_files(name, dir_p);
    if discovered.is_empty() {
        tracing::debug!(
            model = %name,
            "no static-shape ONNX siblings found — dynamic-only inference path"
        );
        return BTreeMap::new();
    }

    let mut pools: BTreeMap<usize, Vec<Mutex<Session>>> = BTreeMap::new();
    for (batch, path) in discovered {
        tracing::info!(
            model = %name,
            batch,
            path = %path.display(),
            pool_size = STATIC_POOL_SIZE_PER_SHAPE,
            "loading static-shape ONNX fast-path session pool"
        );
        // Pool size hard-coded to STATIC_POOL_SIZE_PER_SHAPE — same
        // default as the dynamic pool. Memory cost on ARM Neoverse-N1 /
        // 8 GiB container: ~510 MiB per shape for ModernBERT
        // (255 MiB × 2). Fits under the 3 GiB shared arena cap (Phase H.16).
        // See docs/plans/2026-05-02-multi-shape-static-export.md
        // § Memory budget for the {b=1, b=5} math.
        match build_session_pool(
            name,
            &path,
            opt_level,
            intra_threads,
            STATIC_POOL_SIZE_PER_SHAPE,
            allow_spinning,
        ) {
            Ok(p) => {
                tracing::info!(
                    model = %name,
                    batch,
                    count = p.len(),
                    "loaded static session pool: model={} batch={} count={}",
                    name,
                    batch,
                    p.len()
                );
                pools.insert(batch, p);
            }
            Err(e) => {
                tracing::warn!(
                    model = %name,
                    batch,
                    error = %e,
                    "static-shape ONNX session creation failed — \
                     falling back to dynamic for this batch size"
                );
            }
        }
    }
    pools
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
    // Resolve the cache dir once per pool. The decision (hit / miss) is
    // re-evaluated *per session* inside the loop: session 0 sees a miss
    // and writes the optimized graph; sessions 1..N see a hit on their
    // own re-check and skip the Level3 pass entirely.
    let cache = CacheDir::from_env();
    let mut sessions: Vec<Mutex<Session>> = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        // Re-check inside the loop — see comment above. The first
        // iteration almost always misses; subsequent iterations hit.
        let plan = LoadPlan::decide(cache.as_ref(), onnx_path);
        let load_path = plan.load_source(onnx_path).to_path_buf();
        let t_commit = std::time::Instant::now();
        // See model.rs for the rationale: memory pattern + dynamic batches
        // = unbounded BFCArena growth. Reranker pool members are even more
        // sensitive because they also see variable doc counts.
        let builder = Session::builder().map_err(|e| format!("session builder #{i}: {e}"))?;
        let builder = onnx_cache::apply_plan(builder, &plan, opt_level)
            .map_err(|e| format!("apply cache plan #{i}: {e}"))?;
        let session = builder
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
            .commit_from_file(&load_path)
            .map_err(|e| format!("load ONNX #{i} {}: {e}", load_path.display()))?;
        onnx_cache::observe_post_commit(&plan, t_commit.elapsed().as_millis());
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

#[cfg(test)]
mod discovery_tests {
    //! Unit tests for `discover_static_shape_files`. The discovery
    //! function is pure (only `read_dir` + filename parsing — no ORT
    //! session build), so synthetic temp directories with empty
    //! placeholder files are sufficient to drive the parser through
    //! every branch. No real ONNX bytes required.
    //!
    //! Tests cover:
    //!   - empty dir → empty map
    //!   - single explicit file `_b1.onnx` → `{1: ...}`
    //!   - multiple explicit files `_b1.onnx + _b5.onnx` → `{1: ..., 5: ...}`
    //!   - legacy unsuffixed file → treated as `b=1`
    //!   - legacy + explicit `_b1` both present → explicit wins
    //!   - filename matches prefix but suffix is non-numeric → ignored
    //!   - explicit `_b0.onnx` → ignored (not meaningful)
    //!   - dir does not exist → empty map (no panic)
    //!
    //! Integration tests for the full session-pool load + score path
    //! live in `tests.rs`, gated on a real ModernBERT ONNX file being
    //! present on disk.
    use std::fs::File;

    use super::*;

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        File::create(&p).expect("create test file");
        p
    }

    #[test]
    fn discovery_empty_dir_returns_empty_map() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let map = discover_static_shape_files("test-model", tmp.path());
        assert!(
            map.is_empty(),
            "no files in dir → no static pools, got {map:?}"
        );
    }

    #[test]
    fn discovery_finds_explicit_b1_only() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p1 = touch(tmp.path(), "model_quantized_static_b1.onnx");
        let map = discover_static_shape_files("test-model", tmp.path());
        assert_eq!(map.len(), 1, "exactly one shape discovered");
        assert_eq!(map.get(&1), Some(&p1), "b=1 must point at _b1 file");
    }

    #[test]
    fn discovery_finds_b1_and_b5() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p1 = touch(tmp.path(), "model_quantized_static_b1.onnx");
        let p5 = touch(tmp.path(), "model_quantized_static_b5.onnx");
        // Realistic dir also has the dynamic + tokenizer files; they
        // must not appear in the discovery output.
        touch(tmp.path(), "model_quantized.onnx");
        touch(tmp.path(), "tokenizer.json");

        let map = discover_static_shape_files("test-model", tmp.path());
        assert_eq!(map.len(), 2, "two static shapes discovered, got {map:?}");
        assert_eq!(map.get(&1), Some(&p1));
        assert_eq!(map.get(&5), Some(&p5));
        // BTreeMap iteration is sorted — tests that depend on ordering
        // (logging, deterministic load sequence) get it for free.
        let keys: Vec<usize> = map.keys().copied().collect();
        assert_eq!(keys, vec![1, 5], "keys must be sorted ascending");
    }

    #[test]
    fn discovery_legacy_unsuffixed_treated_as_b1() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // PR #27 convention — unsuffixed file. Backwards-compat
        // contract: existing prod deployments keep working with no
        // operator action on this branch.
        let legacy = touch(tmp.path(), "model_quantized_static.onnx");
        let map = discover_static_shape_files("test-model", tmp.path());
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&1),
            Some(&legacy),
            "legacy unsuffixed file is treated as b=1 for PR #27 backwards compat"
        );
    }

    #[test]
    fn discovery_explicit_b1_wins_over_legacy_when_both_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _legacy = touch(tmp.path(), "model_quantized_static.onnx");
        let explicit = touch(tmp.path(), "model_quantized_static_b1.onnx");
        let map = discover_static_shape_files("test-model", tmp.path());
        assert_eq!(map.len(), 1, "still one b=1 entry, not two");
        assert_eq!(
            map.get(&1),
            Some(&explicit),
            "explicit _b1 file takes precedence over the unsuffixed legacy file"
        );
    }

    #[test]
    fn discovery_ignores_non_numeric_suffix() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // Stray files matching the prefix but not the `_b<digits>.onnx`
        // shape — must not panic, must not appear.
        touch(tmp.path(), "model_quantized_static_btest.onnx");
        touch(tmp.path(), "model_quantized_static_b1.onnx.bak");
        // A real `_b1` file SHOULD still be found.
        let p1 = touch(tmp.path(), "model_quantized_static_b1.onnx");
        let map = discover_static_shape_files("test-model", tmp.path());
        assert_eq!(map.len(), 1, "non-numeric / .bak suffixes ignored");
        assert_eq!(map.get(&1), Some(&p1));
    }

    #[test]
    fn discovery_ignores_b0() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // `b=0` is not a meaningful graph shape; reject explicitly so a
        // typo like `_b0` doesn't load a corrupted graph and crash at
        // first inference.
        touch(tmp.path(), "model_quantized_static_b0.onnx");
        let map = discover_static_shape_files("test-model", tmp.path());
        assert!(
            map.is_empty(),
            "b=0 must not produce a discovered shape, got {map:?}"
        );
    }

    #[test]
    fn discovery_missing_dir_returns_empty_map_no_panic() {
        // Non-existent dir → debug log, empty map. Loader stays
        // dynamic-only without aborting model boot.
        let map = discover_static_shape_files(
            "test-model",
            Path::new("/definitely/not/a/real/path/embed-server-tests"),
        );
        assert!(map.is_empty());
    }
}
