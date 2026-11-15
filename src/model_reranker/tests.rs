//! Integration tests for `RerankerModel`. Live as a sibling module so
//! `mod.rs` stays under the per-file maintainability budget. All tests
//! load a real ONNX file from disk and SKIP gracefully when not
//! available — same pattern as `model::truncation_tests`.
#![cfg(test)]

use std::path::Path;

use super::RerankerModel;

/// Load a real reranker from disk if available, else emit a visible
/// SKIP line and return `None`.
///
/// Resolution order:
///   1. `RERANKER_TEST_DIR` env var (CI / alt dev boxes).
///   2. Default on-box path
///      the directory passed via `RERANKER_MODELS` env.
fn load_reranker_or_skip() -> Option<RerankerModel> {
    load_reranker_or_skip_with_pool(1)
}

/// Like `load_reranker_or_skip` but parameterised on `pool_size` so the
/// concurrent-pool test can request a 2-session model. Existing tests
/// stay on the single-session helper unchanged.
fn load_reranker_or_skip_with_pool(pool_size: usize) -> Option<RerankerModel> {
    let default_dir = std::env::var("TEST_RERANKER_DIR")
        .unwrap_or_else(|_| "/models/gte-multi-rerank".to_string());
    let dir = std::env::var("RERANKER_TEST_DIR").unwrap_or(default_dir);
    if !Path::new(&dir).join("tokenizer.json").exists()
        || !Path::new(&dir).join("model_quantized.onnx").exists()
    {
        eprintln!(
            "SKIP reranker test: model files not found at {dir} \
             (set RERANKER_TEST_DIR to override)"
        );
        return None;
    }
    // Simulate the init sequence: tests call load() directly without going
    // through main.rs, so we set the arena flag manually to satisfy the
    // assert in build_session_pool.
    crate::arena::ARENA_REGISTERED.store(true, std::sync::atomic::Ordering::Release);
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
    // runs at load time with LongestFirst, so the doc side gets clipped
    // and we stay within max_len.
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
    // python smoke test sees ~5.8 vs -11 on these exact inputs; we use
    // a conservative >3.0 margin to avoid brittleness on tiny
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

/// `warmup(&[1])` and `warmup(&[1, 5])` must both return Ok and not
/// touch state visible to subsequent inference calls. SKIPs without
/// model files — the integration coverage we get is "the new
/// shape-list signature actually executes the inference path the same
/// way score_pairs does." Parser-level coverage of the env-var
/// parsing lives in `config::tests::warmup_batch_sizes_*`.
#[test]
fn warmup_runs_for_all_requested_shapes() {
    let Some(m) = load_reranker_or_skip() else {
        return;
    };
    // Single-shape call — parity with the legacy single-warmup path
    // (which used to be `warmup()` with no args, hard-coded batch=2).
    m.warmup(&[1], None).expect("warmup at batch=1");
    // Multi-shape call — the load-bearing new behaviour. Both shapes
    // should compile their kernels; the function returns Ok even if
    // one shape internally fails (best-effort logging contract).
    m.warmup(&[1, 5], None).expect("warmup at batches [1, 5]");
    // Empty shape list is a no-op (logged as a warning) — must not
    // error or panic. Defensive coverage: `parse_warmup_batch_sizes`
    // already falls back to defaults on empty input, so production
    // can't reach here, but direct callers (future code paths,
    // tests) shouldn't have to know that.
    m.warmup(&[], None).expect("warmup with empty shapes");
    // Bounded seq_len path — `Some(64)` clamps the warmup tensor's
    // second dim regardless of model max_len. Must still produce a
    // working session; assert no panic / error.
    m.warmup(&[1], Some(64))
        .expect("warmup at batch=1 with bounded seq_len");
    // Post-warmup, inference still works at both shapes — assert the
    // warmup didn't somehow pollute session state. Uses the same
    // semantic spread `score_pairs_relevant_outscores_unrelated`
    // checks (relevant doc beats unrelated by >3.0 logits).
    let ids = m
        .tokenize_pairs(
            "what is a cat",
            &[
                "a cat is a small domestic feline mammal".into(),
                "the price of oil dropped yesterday".into(),
            ],
        )
        .expect("post-warmup tokenize");
    let scores = m.score_pairs(&ids).expect("post-warmup score");
    assert_eq!(scores.len(), 2);
    assert!(
        scores[0] > scores[1],
        "post-warmup: relevant > unrelated (got {scores:?})"
    );
}

/// Backwards-compat integration test for PR #27 → multi-shape
/// transition: a model dir containing the legacy unsuffixed
/// `model_quantized_static.onnx` (no `_b<N>` suffix) must load that
/// file as the `b=1` static pool. Production gte-reranker-modernbert-
/// base ships exactly this layout (see PR #27 commit `c5ac856`).
///
/// Skips when the ModernBERT static file is not on disk — same SKIP
/// pattern as the rest of this file.
#[test]
fn legacy_unsuffixed_static_loads_as_b1() {
    let default_dir = std::env::var("TEST_MODERNBERT_DIR")
        .unwrap_or_else(|_| "/models/gte-reranker-modernbert-base".to_string());
    let dir = std::env::var("MODERNBERT_TEST_DIR").unwrap_or(default_dir);
    let static_legacy = Path::new(&dir).join("model_quantized_static.onnx");
    let dynamic = Path::new(&dir).join("model_quantized.onnx");
    let tok = Path::new(&dir).join("tokenizer.json");
    if !dynamic.exists() || !tok.exists() || !static_legacy.exists() {
        eprintln!("SKIP legacy_unsuffixed_static_loads_as_b1: required files not present at {dir}");
        return;
    }
    let m = RerankerModel::load("gte-modernbert", &dir, 256, true, 1, 1)
        .expect("load gte-modernbert with legacy static file");
    let shapes: Vec<usize> = m.static_pool_shapes();
    assert!(
        shapes.contains(&1),
        "legacy unsuffixed static file must register as b=1, got shapes={shapes:?}"
    );
}

/// With `pool_size=2`, two threads calling `score_pairs` concurrently
/// must both succeed and return correctly shaped output. The point is
/// to prove the pool doesn't serialize through one mutex — a regression
/// to a single shared lock would still pass the shape assertions, but
/// the test would deadlock the moment one thread holds a lock the
/// other expects (it doesn't, by construction, so the bare fact that
/// two scoped threads run end-to-end is the signal we want). We
/// deliberately do NOT assert on timing — wall-clock comparisons are
/// flaky on shared CI runners.
#[test]
fn score_pairs_pool_concurrent() {
    let Some(m) = load_reranker_or_skip_with_pool(2) else {
        return;
    };
    // Sanity: the model really did load 2 sessions.
    assert_eq!(
        m.session_count(),
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
        // Cross-check semantic ordering still holds when running through
        // the pool — the relevant doc still wins.
        assert!(
            scores_a[0] > scores_a[1],
            "thread A: relevant > unrelated, got {scores_a:?}"
        );
    });
}

/// Load under a caller-chosen name so tests that read the process-wide
/// Prometheus registry get a label no other test touches.
fn load_named_or_skip(name: &str) -> Option<RerankerModel> {
    let default_dir = std::env::var("TEST_RERANKER_DIR")
        .unwrap_or_else(|_| "/models/gte-multi-rerank".to_string());
    let dir = std::env::var("RERANKER_TEST_DIR").unwrap_or(default_dir);
    if !Path::new(&dir).join("model_quantized.onnx").exists() {
        eprintln!("SKIP reranker test: model files not found at {dir}");
        return None;
    }
    crate::arena::ARENA_REGISTERED.store(true, std::sync::atomic::Ordering::Release);
    Some(RerankerModel::load(name, &dir, 256, true, 1, 1).expect("load reranker"))
}

/// Forward passes run for `model`, read off `embed_rerank_batch_size_count`
/// (one observation per pass). Counts passes actually executed, not the plan:
/// a plan that is computed and then ignored would leave this unchanged.
fn passes_recorded(model: &str) -> f64 {
    let needle = format!("embed_rerank_batch_size_count{{model=\"{model}\"}}");
    crate::metrics::test_prometheus_handle()
        .render()
        .lines()
        .find_map(|l| l.strip_prefix(needle.as_str()))
        .and_then(|rest| rest.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Pairs whose lengths differ by far more than the default 2.0 ratio: three
/// short snippets interleaved with three long page chunks.
fn mixed_length_docs() -> Vec<String> {
    let long =
        "the quick brown fox jumps over the lazy dog while the cat sleeps by the fire. ".repeat(12);
    vec![
        "a cat is a small feline".into(),
        format!("cats are domestic animals. {long}"),
        "oil prices fell".into(),
        format!("a cat sat on the mat. {long}"),
        "felines purr when content".into(),
        format!("stock markets and bonds. {long}"),
    ]
}

/// Sub-batching must (a) really split a heterogeneous request, (b) return
/// scores at the INPUT positions, and (c) agree with a single padded pass.
///
/// Numeric agreement is not bit-exact and is not expected to be: the model's
/// int8 dynamic-quantised MatMuls derive their activation scale from the
/// whole tensor, so a pair's logit shifts a little with its pad width and
/// batch-mates (measured max drift on this fixture: 0.085 logits, against a
/// score spread of ~3.5). `LOGIT_TOL` is therefore measured, not derived, and
/// the ranking is compared on pairs separated by more than twice that
/// tolerance.
#[test]
fn subbatched_scores_match_single_pass_and_keep_input_order() {
    const LOGIT_TOL: f32 = 0.15;
    let name = "rerank-subbatch-eq";
    let Some(mut m) = load_named_or_skip(name) else {
        return;
    };
    let ids = m
        .tokenize_pairs("what do cats do", &mixed_length_docs())
        .expect("tokenize");

    m.subbatch_ratio = 1.0;
    let before = passes_recorded(name);
    let single = m.score_pairs(&ids).expect("single pass");
    assert_eq!(
        passes_recorded(name) - before,
        1.0,
        "ratio 1.0 must run exactly one forward pass"
    );

    m.subbatch_ratio = 2.0;
    let before = passes_recorded(name);
    let split = m.score_pairs(&ids).expect("sub-batched");
    assert!(
        passes_recorded(name) - before >= 2.0,
        "short and long pairs must run in separate forward passes at ratio 2.0"
    );

    eprintln!("single: {single:?}\nsplit:  {split:?}");
    assert_eq!(split.len(), single.len(), "one score per pair");
    for (i, (s, b)) in split.iter().zip(&single).enumerate() {
        assert!(
            (s - b).abs() <= LOGIT_TOL,
            "pair {i}: sub-batched {s} vs single-pass {b} (tol {LOGIT_TOL})"
        );
    }
    for i in 0..single.len() {
        for j in 0..single.len() {
            if single[i] - single[j] > 2.0 * LOGIT_TOL {
                assert!(
                    split[i] > split[j],
                    "ranking flipped for pair {i} vs {j}: {split:?} vs {single:?}"
                );
            }
        }
    }

    // Reversing the input must reverse the output: a scatter that writes a
    // group's rows to the wrong positions survives the comparison above only
    // by luck, and cannot survive this one.
    let mut reversed_ids = ids.clone();
    reversed_ids.reverse();
    let mut from_reversed = m.score_pairs(&reversed_ids).expect("reversed");
    from_reversed.reverse();
    for (i, (r, s)) in from_reversed.iter().zip(&split).enumerate() {
        assert!(
            (r - s).abs() <= LOGIT_TOL,
            "pair {i}: reversed-input score {r} vs {s} — scatter order"
        );
    }
}
