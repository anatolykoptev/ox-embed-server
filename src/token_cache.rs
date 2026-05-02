//! Per-pair tokenizer cache for the reranker hot path.
//!
//! ## Why per-pair, not per-doc
//!
//! The reranker is a *cross-encoder*: it jointly encodes `[CLS] q [SEP] d
//! [SEP]` as a single input. The tokenizer produces one encoding that
//! depends on BOTH the query and the document — you cannot cache the doc
//! half independently and glue it together later. Therefore the cache key
//! is `(model, query, doc)` expressed as `(model_name, sha256(query ∥ "\0" ∥ doc))`.
//!
//! The D7 sub-query rewrite in memdb-go re-issues the same top-N docs
//! against multiple rephrased queries. A per-pair cache still wins here
//! when memdb-go re-scores the same `(query_variant, doc)` pair — which
//! happens when the query rewrites are near-duplicates or when the same
//! documents appear across independent search sessions.
//!
//! ## Key construction
//!
//! `sha256(query.as_bytes() + b"\x00" + doc.as_bytes())` — the null byte
//! separator guarantees that `("ab", "c")` ≠ `("a", "bc")`. We hash
//! rather than store the full strings because pair inputs can be thousands
//! of characters; 32 bytes per key is uniform.
//!
//! ## Disabled mode
//!
//! `TokenCache::new(0)` constructs a disabled (no-op) cache — `get` always
//! returns `None` and `insert` is a no-op. Setting `TOKEN_CACHE_MAX_ENTRIES=0`
//! in the environment produces identical byte output to the pre-cache code
//! path; it is the documented runtime kill-switch.
//!
//! ## Threading
//!
//! `moka::sync::Cache` is internally sharded and lock-free on the fast
//! path. No single global `Mutex` around the map, so concurrent probes
//! under load do not contend.
//!
//! ## Value type
//!
//! `Arc<Vec<u32>>` avoids cloning the token ID vector on every `get` call.
//! moka's `get` internally clones the stored value out of its shard;
//! with a bare `Vec<u32>` that would copy potentially hundreds of u32
//! entries. Wrapping in `Arc` reduces the clone to a single atomic
//! reference-count increment.

use std::sync::Arc;

use moka::sync::Cache;
use sha2::{Digest, Sha256};

/// Cache key: (model name, sha256 of the pair bytes).
pub type TokenCacheKey = (String, [u8; 32]);

/// Hash a `(query, doc)` pair for use as the second component of a
/// `TokenCacheKey`. The null-byte separator prevents the key from
/// colliding when the query suffix and doc prefix share characters.
fn hash_pair(query: &str, doc: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(query.as_bytes());
    h.update(b"\x00");
    h.update(doc.as_bytes());
    h.finalize().into()
}

/// Process-local concurrent cache for cross-encoder token IDs.
///
/// Keys are `(model_name, sha256(query ∥ NUL ∥ doc))` — per-pair because
/// cross-encoder tokenization is pair-dependent (see module doc). Values
/// are `Arc<Vec<u32>>` to avoid copying the ID vector on every `get`.
#[derive(Debug)]
pub struct TokenCache {
    /// `None` when the cache is disabled (constructed with capacity 0).
    inner: Option<Cache<TokenCacheKey, Arc<Vec<u32>>>>,
}

impl TokenCache {
    /// Create a cache with the given maximum entry count.
    ///
    /// `max_entries == 0` constructs a disabled cache (get/insert are no-ops).
    /// This lets callers keep `Arc<TokenCache>` in shared state regardless of
    /// whether caching is enabled at runtime, without branch logic at every
    /// call site.
    pub fn new(max_entries: usize) -> Self {
        let inner = if max_entries == 0 {
            None
        } else {
            Some(Cache::builder().max_capacity(max_entries as u64).build())
        };
        Self { inner }
    }

    /// Returns whether the cache is enabled (capacity > 0).
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Look up a tokenized pair.
    ///
    /// Returns a cloned `Arc` (cheap: one atomic increment) on hit.
    /// Returns `None` on miss or when the cache is disabled.
    ///
    /// moka's `get` updates TinyLFU frequency estimators + recency
    /// tracker on each hit — the admission policy will retain hot pairs
    /// across eviction pressure from cold one-shot inputs.
    pub fn get(&self, model: &str, query: &str, doc: &str) -> Option<Arc<Vec<u32>>> {
        let cache = self.inner.as_ref()?;
        let key = (model.to_string(), hash_pair(query, doc));
        cache.get(&key)
    }

    /// Insert a tokenized pair into the cache.
    ///
    /// No-op when the cache is disabled. Eviction (when over capacity) is
    /// performed asynchronously by moka's background worker — `insert` does
    /// not block on eviction. Tests that need a deterministic post-insert
    /// count should call `run_pending_tasks_for_test()` first.
    pub fn insert(&self, model: &str, query: &str, doc: &str, ids: Arc<Vec<u32>>) {
        let Some(cache) = self.inner.as_ref() else {
            return;
        };
        let key = (model.to_string(), hash_pair(query, doc));
        cache.insert(key, ids);
    }

    /// Current entry count (0 when disabled).
    ///
    /// `moka::entry_count()` is eventually consistent — freshly inserted
    /// entries may be counted slightly after they are visible to `get`.
    /// For production telemetry this is the right signal; tests should
    /// drain via `run_pending_tasks_for_test()` for an exact count.
    ///
    /// Production code uses Prometheus counters for hit/miss accounting
    /// rather than polling `len()` directly; the method is retained for
    /// test assertions and future telemetry hooks.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map(|c| c.entry_count() as usize)
            .unwrap_or(0)
    }

    /// Drain moka's background maintenance queue so eviction and size
    /// bookkeeping reflect writes-so-far.
    ///
    /// Test-only helper — production code should never need this (moka
    /// drains continuously under load).
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

    // -----------------------------------------------------------------------
    // Basic hit / miss / disabled behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn hit_after_insert() {
        let c = TokenCache::new(10);
        assert!(c.get("m", "q", "d").is_none());
        let ids = Arc::new(vec![1u32, 2, 3]);
        c.insert("m", "q", "d", ids.clone());
        let got = c.get("m", "q", "d").expect("should hit after insert");
        assert_eq!(*got, vec![1u32, 2, 3]);
    }

    #[test]
    fn disabled_cache_always_misses() {
        let c = TokenCache::new(0);
        assert!(!c.is_enabled());
        c.insert("m", "q", "d", Arc::new(vec![1u32]));
        assert!(c.get("m", "q", "d").is_none());
        assert_eq!(c.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Per-model namespacing
    //
    // Different reranker tokenizers produce different IDs for the same text.
    // gte-multi-rerank (XLM-RoBERTa vocab) and gte-modernbert (BertTokenizer)
    // must NOT share cache entries even for identical (query, doc) strings.
    // -----------------------------------------------------------------------

    #[test]
    fn different_models_produce_separate_keys() {
        let c = TokenCache::new(10);
        let a = Arc::new(vec![10u32, 20]);
        let b = Arc::new(vec![30u32, 40]);
        c.insert("gte-multi", "hello", "world", a.clone());
        c.insert("gte-modernbert", "hello", "world", b.clone());

        assert_eq!(
            *c.get("gte-multi", "hello", "world").unwrap(),
            vec![10u32, 20]
        );
        assert_eq!(
            *c.get("gte-modernbert", "hello", "world").unwrap(),
            vec![30u32, 40]
        );
    }

    // -----------------------------------------------------------------------
    // Per-pair key correctness
    //
    // Different queries with the same doc must NOT share a cache entry.
    // This verifies the pair-cache rationale: the cross-encoder tokenization
    // output depends on both halves.
    // -----------------------------------------------------------------------

    #[test]
    fn different_queries_same_doc_are_separate_keys() {
        let c = TokenCache::new(10);
        let a = Arc::new(vec![1u32, 2]);
        let b = Arc::new(vec![3u32, 4]);
        c.insert("m", "query A", "same doc", a.clone());
        c.insert("m", "query B", "same doc", b.clone());

        assert_eq!(*c.get("m", "query A", "same doc").unwrap(), vec![1u32, 2]);
        assert_eq!(*c.get("m", "query B", "same doc").unwrap(), vec![3u32, 4]);
    }

    // -----------------------------------------------------------------------
    // Separator collision prevention
    //
    // ("ab", "c") must not collide with ("a", "bc") because both concatenate
    // to "abc". The NUL separator in `hash_pair` prevents this.
    // -----------------------------------------------------------------------

    #[test]
    fn no_key_collision_across_pair_split_boundary() {
        let c = TokenCache::new(10);
        c.insert("m", "ab", "c", Arc::new(vec![1u32]));
        c.insert("m", "a", "bc", Arc::new(vec![2u32]));

        assert_eq!(*c.get("m", "ab", "c").unwrap(), vec![1u32]);
        assert_eq!(*c.get("m", "a", "bc").unwrap(), vec![2u32]);
    }

    // -----------------------------------------------------------------------
    // Capacity / eviction
    //
    // moka uses TinyLFU admission — eviction order is not pure LRU, so we
    // assert bounds rather than exact survivors.
    // -----------------------------------------------------------------------

    #[test]
    fn lru_eviction_at_capacity() {
        let c = TokenCache::new(2);
        c.insert("m", "q", "a", Arc::new(vec![1u32]));
        c.insert("m", "q", "b", Arc::new(vec![2u32]));
        c.insert("m", "q", "c", Arc::new(vec![3u32]));
        c.run_pending_tasks_for_test();

        let present = ["a", "b", "c"]
            .iter()
            .filter(|d| c.get("m", "q", d).is_some())
            .count();
        assert!(
            present <= 2,
            "cache must respect max_capacity=2, got {present} entries"
        );
        assert!(present >= 1, "at least one inserted entry should survive");
        assert!(c.len() <= 2, "entry_count must not exceed capacity");
    }

    // -----------------------------------------------------------------------
    // Arc semantics: get returns Arc (cheap clone), not a deep copy
    // -----------------------------------------------------------------------

    #[test]
    fn get_returns_shared_arc() {
        let c = TokenCache::new(4);
        let original = Arc::new(vec![42u32, 43, 44]);
        c.insert("m", "q", "d", original.clone());
        let got = c.get("m", "q", "d").expect("hit");
        assert_eq!(*got, *original);
    }

    // -----------------------------------------------------------------------
    // hash_pair stability
    // -----------------------------------------------------------------------

    #[test]
    fn hash_pair_is_stable_and_distinct() {
        assert_eq!(hash_pair("q", "d"), hash_pair("q", "d"));
        assert_ne!(hash_pair("q", "d"), hash_pair("q", "x"));
        assert_ne!(hash_pair("q", "d"), hash_pair("x", "d"));
        // Separator check: left-heavy vs right-heavy split must differ.
        assert_ne!(hash_pair("ab", "c"), hash_pair("a", "bc"));
    }
}
