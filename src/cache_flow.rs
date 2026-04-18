//! Per-request hit/miss partitioning for the embedding cache.
//!
//! Extracted as a free function so `api::embeddings` stays focused on
//! the HTTP orchestration and the cache bookkeeping is independently
//! testable without an `AppState` harness.
//!
//! The partition pass serves two goals simultaneously:
//!
//! 1. **Cache hits** short-circuit inference entirely.
//! 2. **In-request dedup**: when the same text appears at multiple
//!    positions in one request (e.g. MemDB re-querying identical
//!    search strings across columns), we tokenize + embed it ONCE
//!    and scatter the resulting vector to every original position.
//!
//! Ownership note: the returned `HashMap<String, Vec<usize>>` uses
//! owned `String` keys rather than `&str` because the caller moves
//! `texts` into a `spawn_blocking` closure for tokenization, so any
//! borrow would outlive its source.

use std::collections::HashMap;

use crate::cache::EmbeddingCache;

/// Slot for each original input position: `Some(vec)` when served from
/// cache, `None` when the position awaits scatter from a freshly-computed
/// miss vector.
pub type CachedSlots = Vec<Option<Vec<f32>>>;

/// Unique miss texts mapped to the original input positions that await
/// them. `keys()` is the minimal dedup'd set to tokenize + embed.
pub type PendingMisses = HashMap<String, Vec<usize>>;

/// Probe the cache for each input text; return a `(cached, pending)` pair.
///
/// - `cached[i]` is `Some(vec)` when a hit was found for original position `i`,
///   else `None` (position will be filled by the subsequent inference pass).
/// - `pending` maps each unique miss text to the list of original positions
///   that need its vector. Iterating `pending.keys()` yields the minimal set
///   of texts to tokenize + embed.
///
/// Invariants:
/// - `cached.len() == texts.len()`.
/// - Every index in `0..texts.len()` appears either as `Some` in `cached`
///   OR in exactly one `pending` value's position list — never both, never
///   neither.
pub fn partition_hits_and_misses(
    cache: &EmbeddingCache,
    model: &str,
    texts: &[String],
) -> (CachedSlots, PendingMisses) {
    let mut cached: CachedSlots = vec![None; texts.len()];
    let mut pending: PendingMisses = HashMap::new();

    for (i, text) in texts.iter().enumerate() {
        if let Some(v) = cache.get(model, text) {
            cached[i] = Some(v);
        } else {
            pending.entry(text.clone()).or_default().push(i);
        }
    }

    (cached, pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn all_miss_when_cache_empty() {
        let cache = EmbeddingCache::new(10);
        let texts = vec![s("a"), s("b"), s("c")];
        let (cached, pending) = partition_hits_and_misses(&cache, "m", &texts);

        assert_eq!(cached.len(), 3);
        assert!(cached.iter().all(|c| c.is_none()), "all positions miss");
        assert_eq!(pending.len(), 3, "three unique miss texts");
        assert_eq!(pending.get("a"), Some(&vec![0]));
        assert_eq!(pending.get("b"), Some(&vec![1]));
        assert_eq!(pending.get("c"), Some(&vec![2]));
    }

    #[test]
    fn all_hit_when_cache_warm() {
        let cache = EmbeddingCache::new(10);
        cache.insert("m", "a", vec![1.0]);
        cache.insert("m", "b", vec![2.0]);

        let texts = vec![s("a"), s("b"), s("a")];
        let (cached, pending) = partition_hits_and_misses(&cache, "m", &texts);

        assert_eq!(cached[0], Some(vec![1.0]));
        assert_eq!(cached[1], Some(vec![2.0]));
        assert_eq!(cached[2], Some(vec![1.0]));
        assert!(pending.is_empty(), "no misses to embed");
    }

    #[test]
    fn in_request_dedup_groups_positions() {
        // Same text "x" at positions 0, 2, 3 — should appear as one pending
        // entry with all three positions, tokenized/embedded once.
        let cache = EmbeddingCache::new(10);
        let texts = vec![s("x"), s("y"), s("x"), s("x")];
        let (cached, pending) = partition_hits_and_misses(&cache, "m", &texts);

        assert!(cached.iter().all(|c| c.is_none()));
        assert_eq!(pending.len(), 2, "two unique miss texts (x, y)");

        let x_positions = pending.get("x").expect("x present");
        assert_eq!(x_positions.len(), 3, "x dedup'd across 3 positions");
        assert!(x_positions.contains(&0));
        assert!(x_positions.contains(&2));
        assert!(x_positions.contains(&3));

        assert_eq!(pending.get("y"), Some(&vec![1]));
    }

    #[test]
    fn mixed_hit_and_miss_separate_correctly() {
        let cache = EmbeddingCache::new(10);
        cache.insert("m", "hit1", vec![10.0]);
        cache.insert("m", "hit2", vec![20.0]);

        let texts = vec![
            s("hit1"),
            s("miss_a"),
            s("hit2"),
            s("miss_b"),
            s("miss_a"), // dup with position 1
        ];
        let (cached, pending) = partition_hits_and_misses(&cache, "m", &texts);

        // Hits populate cached[] directly.
        assert_eq!(cached[0], Some(vec![10.0]));
        assert_eq!(cached[2], Some(vec![20.0]));
        // Misses leave cached[] as None.
        assert!(cached[1].is_none());
        assert!(cached[3].is_none());
        assert!(cached[4].is_none());

        // Pending dedup: miss_a appears once with two positions.
        assert_eq!(pending.len(), 2);
        let a_pos = pending.get("miss_a").expect("miss_a present");
        assert_eq!(a_pos.len(), 2);
        assert!(a_pos.contains(&1));
        assert!(a_pos.contains(&4));
        assert_eq!(pending.get("miss_b"), Some(&vec![3]));
    }

    #[test]
    fn per_model_keyspace_respected() {
        let cache = EmbeddingCache::new(10);
        cache.insert("model_a", "hello", vec![1.0]);

        // Same text, different model → miss.
        let texts = vec![s("hello")];
        let (cached, pending) = partition_hits_and_misses(&cache, "model_b", &texts);

        assert!(cached[0].is_none(), "different model must miss");
        assert_eq!(pending.get("hello"), Some(&vec![0]));
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let cache = EmbeddingCache::new(10);
        let (cached, pending) = partition_hits_and_misses(&cache, "m", &[]);
        assert!(cached.is_empty());
        assert!(pending.is_empty());
    }

    #[test]
    fn disabled_cache_always_misses() {
        let cache = EmbeddingCache::new(0);
        cache.insert("m", "would_hit", vec![99.0]); // no-op on disabled

        let texts = vec![s("would_hit")];
        let (cached, pending) = partition_hits_and_misses(&cache, "m", &texts);

        assert!(cached[0].is_none());
        assert_eq!(pending.get("would_hit"), Some(&vec![0]));
    }
}
