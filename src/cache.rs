//! Process-local LRU cache for embeddings keyed by (model, sha256(text)).
//!
//! Embeddings are deterministic: for a given (model, text) pair the output
//! vector is always identical. MemDB re-queries the same search strings
//! often, so turning a ~200ms forward pass into a ~1µs cache lookup is a
//! large win.
//!
//! Scope (intentional):
//! - Process-local (not distributed).
//! - Not persisted across restarts.
//! - Pure LRU — no TTL.
//!
//! Threading:
//! - Single `Mutex` around the LRU map is intentional. Probe and insert are
//!   µs-scale; inference is performed OUTSIDE the lock by the caller.
//!
//! Disable semantic:
//! - `EmbeddingCache::new(0)` constructs a cache with no backing store;
//!   `get` always returns `None` and `insert` is a no-op. This lets
//!   callers keep `Arc<EmbeddingCache>` in shared state regardless of
//!   whether caching is enabled at runtime.

use lru::LruCache;
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::Mutex;

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
    inner: Mutex<Option<LruCache<CacheKey, Vec<f32>>>>,
}

impl EmbeddingCache {
    /// Create a cache with the given maximum entry count.
    ///
    /// `max_entries == 0` constructs a disabled cache (get/insert are no-ops).
    pub fn new(max_entries: usize) -> Self {
        let inner = if max_entries == 0 {
            None
        } else {
            // SAFETY: guarded by the `== 0` check above.
            let cap = NonZeroUsize::new(max_entries).unwrap();
            Some(LruCache::new(cap))
        };
        Self {
            inner: Mutex::new(inner),
        }
    }

    /// Returns whether the cache is enabled (capacity > 0).
    pub fn is_enabled(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    /// Look up an embedding. Returns a clone of the stored vector on hit,
    /// None on miss or when the cache is disabled. Marks the entry as MRU.
    pub fn get(&self, model: &str, text: &str) -> Option<Vec<f32>> {
        let key = (model.to_string(), hash_text(text));
        let mut guard = self.inner.lock().unwrap();
        guard.as_mut()?.get(&key).cloned()
    }

    /// Insert an embedding. No-op when the cache is disabled.
    pub fn insert(&self, model: &str, text: &str, vec: Vec<f32>) {
        let key = (model.to_string(), hash_text(text));
        let mut guard = self.inner.lock().unwrap();
        if let Some(lru) = guard.as_mut() {
            lru.put(key, vec);
        }
    }

    /// Current entry count (0 when disabled).
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap()
            .as_ref()
            .map(|l| l.len())
            .unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    fn lru_evicts_oldest() {
        let c = EmbeddingCache::new(2);
        c.insert("m", "a", vec![1.0]);
        c.insert("m", "b", vec![2.0]);
        assert_eq!(c.len(), 2);
        // Third insert evicts the LRU entry ("a").
        c.insert("m", "c", vec![3.0]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.get("m", "a"), None, "a should have been evicted");
        assert!(c.get("m", "b").is_some());
        assert!(c.get("m", "c").is_some());
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
    fn get_marks_mru() {
        // With capacity 2: insert a, b; get a; insert c → b should be evicted (not a).
        let c = EmbeddingCache::new(2);
        c.insert("m", "a", vec![1.0]);
        c.insert("m", "b", vec![2.0]);
        assert!(c.get("m", "a").is_some(), "touch makes a MRU");
        c.insert("m", "c", vec![3.0]);
        assert!(c.get("m", "a").is_some(), "a survives as MRU");
        assert_eq!(c.get("m", "b"), None, "b evicted as LRU");
        assert!(c.get("m", "c").is_some());
    }
}
