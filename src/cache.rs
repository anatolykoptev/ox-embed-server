//! Process-local concurrent cache for embeddings keyed by (model, sha256(text)).
//!
//! Embeddings are deterministic: for a given (model, text) pair the output
//! vector is always identical. MemDB re-queries the same search strings
//! often, so turning a ~200ms forward pass into a ~1µs cache lookup is a
//! large win.
//!
//! Scope (intentional):
//! - Process-local (not distributed).
//! - Not persisted across restarts.
//! - Size-bounded with **LRU** eviction via `moka`. No TTL (embeddings
//!   are deterministic; size is the only bound).
//!
//! Threading:
//! - `moka::sync::Cache` is internally sharded and lock-free on the fast
//!   path. No single global `Mutex` around the map, so concurrent probes
//!   under load don't contend the way a `Mutex<LruCache>` would.
//!
//! Eviction policy — LRU (was TinyLFU):
//! - moka's default TinyLFU admission filter rejects one-shot inserts
//!   without a frequency signal. The `jina-code-v2` workload is
//!   ~unique-per-request code chunks, so admission rejected effectively
//!   every insert: live `cache_size` stuck at the cold-start residents (4)
//!   and hit rate stayed at 0% indefinitely while `multilingual-e5-large`
//!   re-queried search strings hit ~92.9%.
//! - Switching to LRU accepts every insert: jina entries now live until
//!   normal LRU eviction, so re-queried code chunks (which do exist —
//!   e.g. shared imports across files) actually hit. The cost is a slight
//!   theoretical regression for e5 under adversarial scan-style traffic,
//!   which we don't see in production.
//!
//! Disable semantic:
//! - `EmbeddingCache::new(0)` constructs a cache with no backing store;
//!   `get` always returns `None` and `insert` is a no-op. This lets
//!   callers keep `Arc<EmbeddingCache>` in shared state regardless of
//!   whether caching is enabled at runtime.

use moka::policy::EvictionPolicy;
use moka::sync::Cache;
use sha2::{Digest, Sha256};

/// Cache key: (model name, sha256 of the input text).
///
/// We hash the text rather than store the full string because inputs can
/// be large; 32 bytes per key is uniform and tiny.
pub type CacheKey = (String, [u8; 32]);

/// Hash an input text for use as the second component of a CacheKey.
pub fn hash_text(text: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    h.finalize().into()
}

#[derive(Debug)]
pub struct EmbeddingCache {
    /// `None` when the cache is disabled (constructed with capacity 0).
    inner: Option<Cache<CacheKey, Vec<f32>>>,
}

impl EmbeddingCache {
    /// Create a cache with the given maximum entry count.
    ///
    /// `max_entries == 0` constructs a disabled cache (get/insert are no-ops).
    pub fn new(max_entries: usize) -> Self {
        let inner = if max_entries == 0 {
            None
        } else {
            // LRU rather than the moka default TinyLFU — TinyLFU's admission
            // filter starves caches whose workload is unique-per-request
            // (jina-code-v2 saw 0% hit rate in prod). See module docstring.
            Some(
                Cache::builder()
                    .max_capacity(max_entries as u64)
                    .eviction_policy(EvictionPolicy::lru())
                    .build(),
            )
        };
        Self { inner }
    }

    /// Returns whether the cache is enabled (capacity > 0).
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Look up an embedding. Returns a clone of the stored vector on hit,
    /// None on miss or when the cache is disabled.
    ///
    /// moka's `get` returns `Option<V>` (value is cloned out of the shard
    /// internally) rather than `Option<&V>`, so no explicit `.cloned()`.
    /// It also updates the LRU recency tracker.
    pub fn get(&self, model: &str, text: &str) -> Option<Vec<f32>> {
        let cache = self.inner.as_ref()?;
        let key = (model.to_string(), hash_text(text));
        cache.get(&key)
    }

    /// Insert an embedding. No-op when the cache is disabled.
    ///
    /// Eviction (when over capacity) is performed asynchronously by a
    /// background worker — the insert itself does not block on eviction.
    /// For deterministic size assertions in tests, call
    /// `run_pending_tasks_for_test` first.
    pub fn insert(&self, model: &str, text: &str, vec: Vec<f32>) {
        let Some(cache) = self.inner.as_ref() else {
            return;
        };
        let key = (model.to_string(), hash_text(text));
        cache.insert(key, vec);
    }

    /// Current entry count (0 when disabled).
    ///
    /// Note: moka reports `entry_count()` eventually-consistent — freshly
    /// inserted entries may be counted slightly after they're visible to
    /// `get`. For production telemetry this is the right signal (it
    /// reflects after-eviction steady-state). Tests that need an exact
    /// post-insert count should `run_pending_tasks_for_test()` first.
    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map(|c| c.entry_count() as usize)
            .unwrap_or(0)
    }

    /// Drain the background maintenance queue so eviction and size
    /// bookkeeping reflect writes-so-far. Test-only helper; production
    /// code should never need this (moka drains continuously under load).
    #[cfg(test)]
    fn run_pending_tasks_for_test(&self) {
        if let Some(c) = self.inner.as_ref() {
            c.run_pending_tasks();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_after_insert() {
        let c = EmbeddingCache::new(10);
        assert!(c.get("e5", "hello").is_none());
        c.insert("e5", "hello", vec![1.0, 2.0, 3.0]);
        assert_eq!(c.get("e5", "hello"), Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn miss_on_different_model() {
        let c = EmbeddingCache::new(10);
        c.insert("e5", "hello", vec![1.0]);
        assert_eq!(c.get("jina", "hello"), None, "per-model keyspace");
    }

    #[test]
    fn miss_on_different_text() {
        let c = EmbeddingCache::new(10);
        c.insert("e5", "hello", vec![1.0]);
        assert_eq!(c.get("e5", "world"), None);
    }

    #[test]
    fn bounded_capacity_evicts_under_pressure() {
        // We use LRU eviction (see module docstring): inserts are always
        // admitted, eviction follows recency. This test only asserts the
        // size invariant — it does not pin which key survives so the test
        // stays robust against future eviction-policy tuning.
        let c = EmbeddingCache::new(2);
        c.insert("m", "a", vec![1.0]);
        c.insert("m", "b", vec![2.0]);
        c.insert("m", "c", vec![3.0]);
        c.run_pending_tasks_for_test();

        let present = ["a", "b", "c"]
            .iter()
            .filter(|t| c.get("m", t).is_some())
            .count();
        assert!(
            present <= 2,
            "cache must respect max_capacity=2, got {present} entries present"
        );
        assert!(
            present >= 1,
            "at least one of the inserted entries should be retained"
        );
        assert!(
            c.len() <= 2,
            "entry_count must not exceed capacity; got {}",
            c.len()
        );
    }

    #[test]
    fn hash_text_is_stable_and_distinct() {
        assert_eq!(hash_text("foo"), hash_text("foo"));
        assert_ne!(hash_text("foo"), hash_text("bar"));
        // Empty string hashes consistently.
        assert_eq!(hash_text(""), hash_text(""));
        assert_ne!(hash_text(""), hash_text("a"));
    }

    #[test]
    fn disabled_cache_is_noop() {
        let c = EmbeddingCache::new(0);
        assert!(!c.is_enabled());
        c.insert("e5", "hello", vec![1.0, 2.0, 3.0]);
        assert_eq!(c.get("e5", "hello"), None);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn enabled_cache_reports_enabled() {
        let c = EmbeddingCache::new(4);
        assert!(c.is_enabled());
    }

    #[test]
    fn repeated_get_returns_same_value() {
        // moka updates frequency estimators on read; make sure repeated
        // reads still return Some with the same value (no internal
        // invalidation on touch).
        let c = EmbeddingCache::new(4);
        c.insert("m", "a", vec![1.0, 2.0]);
        assert_eq!(c.get("m", "a"), Some(vec![1.0, 2.0]));
        assert_eq!(c.get("m", "a"), Some(vec![1.0, 2.0]));
        assert_eq!(c.get("m", "a"), Some(vec![1.0, 2.0]));
    }
}
