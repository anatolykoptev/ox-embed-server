//! Disk-backed cache for Level3-optimized ONNX graphs.
//!
//! ## Why
//!
//! `Session::with_optimization_level(Level3)` runs constant folding,
//! redundant-node elimination, layout transforms, etc. on every model load.
//! For a ~340-550 MB reranker / e5-large model on ARM Neoverse-N1 this is
//! ~5-10 sec of CPU work *per session*. With a 2-session pool × 2 rerankers
//! + 1 embedder + 1 splade, container restart pays ~30-60 sec of repeat
//! optimization work. ORT exposes `with_optimized_model_path(p)` which
//! serializes the post-optimization graph as a side effect of the first
//! load; subsequent loads can read that file directly with
//! `GraphOptimizationLevel::Disable` and skip the optimization passes.
//!
//! ## Activation
//!
//! Opt-in via `ONNX_OPT_CACHE_DIR=/some/writable/path`. Unset → behaviour
//! is byte-for-byte identical to pre-cache code (no probing, no logging,
//! no perf risk). This is deliberate: the deployed compose file marks the
//! container `read_only: true` with only `/tmp` writable, and we don't
//! want to assume any particular mount layout.
//!
//! ## Cache invalidation
//!
//! Cache filename includes the *source* model file's mtime in nanoseconds
//! (`<basename>.<mtime_ns>.optimized.onnx`). Swapping the source file
//! transparently invalidates the cache — the new mtime produces a new
//! cache path, the old file is left orphaned (operator's job to GC). This
//! avoids hashing 500 MB of weights on every startup; mtime is sufficient
//! because models are read-only host bind mounts that change only when
//! the operator deploys a new version.
//!
//! ## Failure mode
//!
//! Any I/O error (cache dir missing, read-only, full disk, permission
//! denied) downgrades to a `warn` log and the model loads with the
//! original Level3-on-the-fly path. Startup never fails because of the
//! cache — the cache is a perf optimization, not a correctness dependency.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

/// Resolution of the `ONNX_OPT_CACHE_DIR` env var into a usable cache
/// directory, or `None` when the feature is disabled / unusable.
///
/// `None` means "don't change anything" — callers fall through to the
/// original Level3-on-the-fly load path. `Some(dir)` means we have a
/// writable directory we can stash optimized-graph files into.
#[derive(Debug, Clone)]
pub struct CacheDir(PathBuf);

impl CacheDir {
    /// Read `ONNX_OPT_CACHE_DIR` and verify the directory is writable
    /// (touch-and-delete a sentinel file). Returns `None` when:
    ///   - env var unset (feature disabled — default)
    ///   - env var set to empty string
    ///   - directory does not exist and cannot be created
    ///   - directory is not writable (read-only FS, permission denied)
    ///
    /// All non-default outcomes log a warning so operators can tell the
    /// difference between "feature off" and "feature broken".
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("ONNX_OPT_CACHE_DIR").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        Self::from_path(Path::new(trimmed))
    }

    /// Same as `from_env` but with an explicit path (used by tests).
    pub fn from_path(dir: &Path) -> Option<Self> {
        // Auto-create the dir if it doesn't exist. Operators typically
        // mount a tmpfs / volume here, but a stray `mkdir -p` is harmless
        // and avoids a setup gotcha.
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "ONNX_OPT_CACHE_DIR not usable (mkdir failed) — caching disabled"
            );
            return None;
        }
        if !dir_is_writable(dir) {
            tracing::warn!(
                dir = %dir.display(),
                "ONNX_OPT_CACHE_DIR not writable — caching disabled"
            );
            return None;
        }
        tracing::info!(
            dir = %dir.display(),
            "ONNX optimized-graph cache enabled"
        );
        Some(Self(dir.to_path_buf()))
    }

    /// Compute the cache-file path for a given source ONNX file.
    ///
    /// Format: `<cache_dir>/<source_basename>.<mtime_ns>.optimized.onnx`.
    ///
    /// Returns `None` when the source file's metadata can't be read —
    /// without an mtime we can't safely key the cache, so we fall back to
    /// the unmodified load path rather than risk serving a stale cache.
    pub fn cache_path(&self, source: &Path) -> Option<PathBuf> {
        let basename = source.file_name()?.to_str()?;
        let mtime_ns = mtime_ns(source)?;
        Some(self.0.join(format!(
            "{basename}.{mtime_ns}.optimized.onnx"
        )))
    }
}

/// Touch-and-delete probe: create a randomly-named sentinel file in
/// `dir`, then immediately remove it. Cheaper than a full file write and
/// catches the common "mounted read-only" / "wrong owner" cases at
/// startup before we try to use the cache.
fn dir_is_writable(dir: &Path) -> bool {
    // A nanosecond-keyed name avoids collisions if multiple processes
    // probe simultaneously (compose `up -d` of >1 service in parallel).
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let probe = dir.join(format!(".onnx_cache_probe.{nanos}"));
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Source file's modification time in nanoseconds since UNIX epoch.
/// Used as the cache-key disambiguator so model swaps invalidate the
/// cache without operator action.
fn mtime_ns(p: &Path) -> Option<u128> {
    let meta = std::fs::metadata(p).ok()?;
    let mtime = meta.modified().ok()?;
    mtime
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
}

/// Outcome of the cache decision for a single session load. Surfaces
/// the right `(opt_level, write_path)` pair to the call site so the
/// session builder chain stays linear and readable.
#[derive(Debug, Clone)]
pub enum LoadPlan {
    /// No cache directory configured / writable. Use the caller's
    /// requested optimization level and don't write a cache file.
    NoCache,
    /// Cache hit: load the pre-optimized graph from `cached`. Force
    /// `GraphOptimizationLevel::Disable` because the graph already has
    /// every transformation baked in — re-running Level3 on top would be
    /// wasted work (and the whole point of this feature).
    Hit { cached: PathBuf },
    /// Cache miss: load the original `source`, keep the requested opt
    /// level, AND ask ORT to serialize the optimized graph at `target`
    /// as a side effect of `commit_from_file`.
    Miss { target: PathBuf },
}

impl LoadPlan {
    /// Decide cache hit vs miss for a single session. Re-evaluated per
    /// session in a pool: the first miss writes the cache, the second
    /// session sees a hit on its own re-check.
    pub fn decide(cache: Option<&CacheDir>, source: &Path) -> Self {
        let Some(cache) = cache else {
            return Self::NoCache;
        };
        let Some(target) = cache.cache_path(source) else {
            tracing::warn!(
                source = %source.display(),
                "could not compute cache path (mtime unreadable) — caching disabled for this load"
            );
            return Self::NoCache;
        };
        if target.exists() {
            Self::Hit { cached: target }
        } else {
            Self::Miss { target }
        }
    }

    /// The path actually passed to `commit_from_file`. For a hit this is
    /// the cached optimized graph; otherwise the original source.
    pub fn load_source<'a>(&'a self, source: &'a Path) -> &'a Path {
        match self {
            Self::Hit { cached } => cached.as_path(),
            Self::NoCache | Self::Miss { .. } => source,
        }
    }
}

/// Apply the cache plan to a `SessionBuilder` chain. On a hit, force
/// `GraphOptimizationLevel::Disable` so we don't pay Level3 again on top
/// of an already-optimized graph. On a miss, keep the caller's
/// `requested` opt level and configure ORT to serialize the optimized
/// graph at the target path. On `NoCache`, pass through the requested
/// level unchanged.
///
/// Returns the (possibly modified) builder so the chain stays fluent.
pub fn apply_plan(
    builder: SessionBuilder,
    plan: &LoadPlan,
    requested: GraphOptimizationLevel,
) -> Result<SessionBuilder, String> {
    match plan {
        LoadPlan::NoCache => builder
            .with_optimization_level(requested)
            .map_err(|e| format!("set opt level: {e}")),
        LoadPlan::Hit { .. } => builder
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| format!("set opt level (cache hit): {e}")),
        LoadPlan::Miss { target } => builder
            .with_optimization_level(requested)
            .map_err(|e| format!("set opt level (cache miss): {e}"))?
            .with_optimized_model_path(target)
            .map_err(|e| format!("set optimized model path: {e}")),
    }
}

/// Log + verify the post-load cache state. Called immediately after
/// `commit_from_file` returns. On a miss, ORT should have written the
/// optimized graph at `target`; if it didn't (rare — usually means
/// quantized model + a transformation that ORT skips serializing), log a
/// warn so ops can tell the cache is degenerate. On a hit, just log the
/// kernel-init time so we have evidence the warm path is faster.
pub fn observe_post_commit(plan: &LoadPlan, elapsed_ms: u128) {
    match plan {
        LoadPlan::NoCache => {
            tracing::info!(elapsed_ms, "ONNX session committed (cache disabled)");
        }
        LoadPlan::Hit { cached } => {
            tracing::info!(
                cached = %cached.display(),
                elapsed_ms,
                "ONNX session committed from optimized-graph cache (warm)"
            );
        }
        LoadPlan::Miss { target } => {
            let size_mib = std::fs::metadata(target).ok().map(|m| m.len() / (1024 * 1024));
            match size_mib {
                Some(mib) => tracing::info!(
                    target = %target.display(),
                    size_mib = mib,
                    elapsed_ms,
                    "ONNX session committed (cold) — wrote optimized graph"
                ),
                None => tracing::warn!(
                    target = %target.display(),
                    elapsed_ms,
                    "ONNX session committed (cold) but optimized graph not written — \
                     ORT may have skipped serialization for this model; subsequent \
                     loads will re-optimize"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Sets/unsets an env var around a closure and restores it.
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

    fn tempdir(suffix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("onnx_cache_test_{nanos}_{suffix}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn from_env_unset_returns_none() {
        with_env("ONNX_OPT_CACHE_DIR", None, || {
            assert!(CacheDir::from_env().is_none());
        });
    }

    #[test]
    fn from_env_empty_returns_none() {
        with_env("ONNX_OPT_CACHE_DIR", Some(""), || {
            assert!(CacheDir::from_env().is_none());
        });
        with_env("ONNX_OPT_CACHE_DIR", Some("   "), || {
            assert!(CacheDir::from_env().is_none());
        });
    }

    #[test]
    fn from_env_writable_returns_some() {
        let dir = tempdir("writable");
        with_env("ONNX_OPT_CACHE_DIR", Some(dir.to_str().unwrap()), || {
            let cache = CacheDir::from_env().expect("writable dir should be Some");
            assert_eq!(cache.0, dir);
        });
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_path_creates_missing_dir() {
        let parent = tempdir("create");
        let nested = parent.join("does/not/exist/yet");
        let cache = CacheDir::from_path(&nested);
        assert!(cache.is_some(), "from_path should mkdir -p");
        assert!(nested.is_dir(), "directory should now exist");
        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn from_path_readonly_dir_returns_none() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir("readonly");
        // 0o555 = r-x r-x r-x — readable + traversable, NOT writable.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let cache = CacheDir::from_path(&dir);
        // Restore perms before assert so cleanup works.
        let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o755));
        let _ = fs::remove_dir_all(&dir);
        assert!(cache.is_none(), "read-only dir should disable caching");
    }

    #[test]
    fn cache_path_includes_basename_and_mtime() {
        let dir = tempdir("cachepath");
        let cache = CacheDir::from_path(&dir).unwrap();

        let src = dir.join("model_quantized.onnx");
        fs::write(&src, b"fake onnx bytes").unwrap();

        let p = cache.cache_path(&src).expect("metadata readable");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("model_quantized.onnx."),
            "should contain source basename, got {name}"
        );
        assert!(
            name.ends_with(".optimized.onnx"),
            "should end with .optimized.onnx, got {name}"
        );
        // The middle segment must be a parseable u128 (mtime_ns).
        let middle = name
            .strip_prefix("model_quantized.onnx.")
            .unwrap()
            .strip_suffix(".optimized.onnx")
            .unwrap();
        middle
            .parse::<u128>()
            .expect("middle segment should be u128 mtime_ns");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_path_changes_when_source_mtime_changes() {
        use std::time::Duration;
        let dir = tempdir("mtime");
        let cache = CacheDir::from_path(&dir).unwrap();
        let src = dir.join("m.onnx");
        fs::write(&src, b"v1").unwrap();
        let p1 = cache.cache_path(&src).unwrap();

        // Sleep just long enough for filesystem mtime resolution.
        // ext4/zfs/apfs all give us at least ms; 20ms is comfortable.
        std::thread::sleep(Duration::from_millis(20));
        fs::write(&src, b"v2-different-bytes").unwrap();
        let p2 = cache.cache_path(&src).unwrap();

        assert_ne!(p1, p2, "different mtime should yield different cache path");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_path_distinct_for_different_basenames() {
        // Dynamic vs static reranker variants must NOT collide.
        let dir = tempdir("basename");
        let cache = CacheDir::from_path(&dir).unwrap();
        let dynamic = dir.join("model_quantized.onnx");
        let static_ = dir.join("model_quantized_static.onnx");
        fs::write(&dynamic, b"a").unwrap();
        fs::write(&static_, b"b").unwrap();

        let pa = cache.cache_path(&dynamic).unwrap();
        let pb = cache.cache_path(&static_).unwrap();
        assert_ne!(pa, pb, "dynamic and static cache paths must differ");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn decide_no_cache_when_dir_unset() {
        let src = std::env::temp_dir().join("nonexistent.onnx");
        let plan = LoadPlan::decide(None, &src);
        assert!(matches!(plan, LoadPlan::NoCache));
    }

    #[test]
    fn decide_miss_when_target_absent() {
        let dir = tempdir("miss");
        let cache = CacheDir::from_path(&dir).unwrap();
        let src = dir.join("m.onnx");
        fs::write(&src, b"x").unwrap();

        let plan = LoadPlan::decide(Some(&cache), &src);
        assert!(matches!(plan, LoadPlan::Miss { .. }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn decide_hit_when_target_exists() {
        let dir = tempdir("hit");
        let cache = CacheDir::from_path(&dir).unwrap();
        let src = dir.join("m.onnx");
        fs::write(&src, b"x").unwrap();
        let target = cache.cache_path(&src).unwrap();
        // Pre-create the cache file to simulate a prior cold load.
        fs::write(&target, b"optimized").unwrap();

        let plan = LoadPlan::decide(Some(&cache), &src);
        assert!(matches!(plan, LoadPlan::Hit { .. }));
        if let LoadPlan::Hit { cached } = plan {
            assert_eq!(cached, target);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_source_returns_cached_on_hit_and_source_otherwise() {
        let src = Path::new("/tmp/source.onnx");
        let cached = PathBuf::from("/tmp/cached.onnx");

        let no_cache = LoadPlan::NoCache;
        assert_eq!(no_cache.load_source(src), src);

        let miss = LoadPlan::Miss {
            target: cached.clone(),
        };
        assert_eq!(miss.load_source(src), src);

        let hit = LoadPlan::Hit {
            cached: cached.clone(),
        };
        assert_eq!(hit.load_source(src), cached.as_path());
    }

    #[test]
    fn dir_is_writable_true_for_writable_dir() {
        let dir = tempdir("writable_probe");
        assert!(dir_is_writable(&dir));
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn dir_is_writable_false_for_readonly_dir() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir("readonly_probe");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let writable = dir_is_writable(&dir);
        let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o755));
        let _ = fs::remove_dir_all(&dir);
        assert!(!writable);
    }

    /// Empirical proof that the warm path is materially faster than cold.
    ///
    /// Marked `#[ignore]` because it requires a real ONNX model on disk
    /// — point it at one with `ONNX_CACHE_BENCH_PATH=/path/to/model.onnx`
    /// and run with `cargo test --package embed-server -- --ignored
    /// bench_warm_vs_cold --nocapture`.
    ///
    /// The test:
    ///   1. cold load (no cache) — Level3 from scratch
    ///   2. cold load WITH cache write — Level3 + serialize
    ///   3. warm load (cache hit) — Disable + read pre-optimized graph
    ///
    /// Asserts step 3 is at least 30% faster than step 1. If this fails
    /// the feature isn't delivering and should not ship — see the task's
    /// STOP-and-report clause.
    #[test]
    #[ignore]
    fn bench_warm_vs_cold() {
        use ort::session::Session;
        use std::time::Instant;

        let onnx_path = match std::env::var("ONNX_CACHE_BENCH_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => {
                eprintln!(
                    "SKIP bench_warm_vs_cold: set ONNX_CACHE_BENCH_PATH to a real .onnx file"
                );
                return;
            }
        };
        assert!(
            onnx_path.exists(),
            "ONNX_CACHE_BENCH_PATH does not exist: {}",
            onnx_path.display()
        );

        let dir = tempdir("bench");
        let cache = CacheDir::from_path(&dir).expect("cache dir writable");

        // Sanity: pre-clean any stale cache file.
        let target = cache.cache_path(&onnx_path).unwrap();
        let _ = fs::remove_file(&target);

        // Step 1 — cold, no cache.
        let plan_cold = LoadPlan::NoCache;
        let t0 = Instant::now();
        let builder = Session::builder().unwrap();
        let builder =
            apply_plan(builder, &plan_cold, GraphOptimizationLevel::Level3).unwrap();
        let _s = builder
            .with_intra_threads(1)
            .unwrap()
            .commit_from_file(&onnx_path)
            .expect("cold load");
        let cold_ms = t0.elapsed().as_millis();
        drop(_s);

        // Step 2 — cold + write cache.
        let plan_miss = LoadPlan::decide(Some(&cache), &onnx_path);
        assert!(matches!(plan_miss, LoadPlan::Miss { .. }));
        let t1 = Instant::now();
        let builder = Session::builder().unwrap();
        let builder = apply_plan(builder, &plan_miss, GraphOptimizationLevel::Level3).unwrap();
        let _s = builder
            .with_intra_threads(1)
            .unwrap()
            .commit_from_file(plan_miss.load_source(&onnx_path))
            .expect("cold+write load");
        let cold_write_ms = t1.elapsed().as_millis();
        drop(_s);
        assert!(
            target.exists(),
            "cache file should exist after a Miss commit"
        );

        // Step 3 — warm.
        let plan_hit = LoadPlan::decide(Some(&cache), &onnx_path);
        assert!(matches!(plan_hit, LoadPlan::Hit { .. }));
        let t2 = Instant::now();
        let builder = Session::builder().unwrap();
        let builder = apply_plan(builder, &plan_hit, GraphOptimizationLevel::Level3).unwrap();
        let _s = builder
            .with_intra_threads(1)
            .unwrap()
            .commit_from_file(plan_hit.load_source(&onnx_path))
            .expect("warm load");
        let warm_ms = t2.elapsed().as_millis();

        eprintln!(
            "bench: cold={}ms cold+write={}ms warm={}ms ratio={:.2}",
            cold_ms,
            cold_write_ms,
            warm_ms,
            warm_ms as f64 / cold_ms as f64
        );

        let _ = fs::remove_dir_all(&dir);

        // The warm path must be measurably faster. We pick 30% as a
        // conservative floor — typical Level3 cost on ARM is 5-10s on a
        // 340-550 MB model, so warm should be ~1-2s. If warm >= 0.7×cold
        // ORT is silently re-running the optimization passes despite our
        // `Disable` flag — ship-stopper, see task spec.
        assert!(
            (warm_ms as f64) < 0.7 * (cold_ms as f64),
            "warm load not materially faster than cold — feature not delivering"
        );
    }
}
