//! Generic evictable session pool for ONNX inference sessions.
//!
//! Port of `ox-whisper/src/pool.rs` (v0.7.0) adapted for embed-server:
//!
//! - Metrics carry a `model` label (embed-server convention vs ox-whisper
//!   which uses per-process counter names without labels).
//! - `factory` returns `Result<T, String>` matching the rest of embed-server's
//!   error-handling convention (no `anyhow` dependency here).
//! - `idle_secs == 0` disables eviction entirely (opt-in via `EMBED_IDLE_EVICT_SECS`).
//!
//! # Pool lifecycle
//!
//! 1. `EvictablePool::from_items` wraps already-constructed items (startup path).
//! 2. `acquire()` returns an `EvictableGuard<T>` which returns the item on `Drop`.
//! 3. Evicted slots (`None`) are lazily re-created by the next `acquire()` call.
//! 4. `spawn_eviction_loop` runs a background tokio task; abort the handle to stop it.
//!
//! # Concurrency guarantees
//!
//! - B1: `busy` flag is set *inside* the `MutexGuard` scope to eliminate the
//!   TOCTOU window between mutex unlock and flag store.
//! - M4: factory is called *outside* the mutex so a slow reinit does not block
//!   other slots from being acquired.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ── time helper ───────────────────────────────────────────────────────────────

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── AcquireError ─────────────────────────────────────────────────────────────

/// Errors returned by [`EvictablePool::acquire`].
#[derive(Debug)]
pub enum AcquireError {
    /// All slots are currently in use.
    AllBusy,
    /// Factory returned an error when re-initialising an evicted slot.
    ReinitFailed(String),
}

impl std::fmt::Display for AcquireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquireError::AllBusy => write!(f, "all pool slots are busy"),
            AcquireError::ReinitFailed(e) => write!(f, "pool slot reinit failed: {e}"),
        }
    }
}

impl std::error::Error for AcquireError {}

// ── EvictableSlot ─────────────────────────────────────────────────────────────

/// Internal slot state machine:
/// - `item = Some(T)` + `busy = false` → idle, ready to acquire.
/// - `item = None`   + `busy = false` → evicted, reinit on next acquire.
/// - `item = None`   + `busy = true`  → held by a guard (in use).
struct EvictableSlot<T> {
    item: Mutex<Option<T>>,
    busy: AtomicBool,
    last_used: AtomicU64,
}

// ── EvictablePool ─────────────────────────────────────────────────────────────

/// Pool with opt-in idle eviction. Items are lazily re-created via `factory`
/// when a slot was evicted and then acquired again.
///
/// When `idle_secs == 0` eviction is **disabled** and the pool behaves like
/// a simple pre-allocated pool (legacy behaviour preserved for existing deployments).
pub struct EvictablePool<T> {
    slots: Vec<Arc<EvictableSlot<T>>>,
    factory: Arc<dyn Fn() -> Result<T, String> + Send + Sync>,
    pub(crate) idle_secs: u64,
    /// Model name for Prometheus labels (`embed_pool_*{model="..."}` series).
    model: String,
}

impl<T: Send + 'static> EvictablePool<T> {
    /// Create a pool pre-filled with already-constructed items.
    ///
    /// `factory` is used only for lazy re-init after eviction.
    /// Useful in tests or when items are loaded via external code (e.g. warmup).
    pub fn from_items(
        items: Vec<T>,
        idle_secs: u64,
        model: impl Into<String>,
        factory: Arc<dyn Fn() -> Result<T, String> + Send + Sync>,
    ) -> Self {
        let now = unix_now_secs();
        let slots = items
            .into_iter()
            .map(|item| {
                Arc::new(EvictableSlot {
                    item: Mutex::new(Some(item)),
                    busy: AtomicBool::new(false),
                    last_used: AtomicU64::new(now),
                })
            })
            .collect();
        Self {
            slots,
            factory,
            idle_secs,
            model: model.into(),
        }
    }

    /// Acquire an idle slot. Re-initializes evicted slots via factory (cold start).
    ///
    /// Returns `Err(AcquireError::AllBusy)` if all slots are in use.
    /// Returns `Err(AcquireError::ReinitFailed)` if an evicted slot's factory call fails;
    /// in this case the slot stays as `None` and is retryable on next acquire.
    pub fn acquire(&self) -> Result<EvictableGuard<T>, AcquireError> {
        let now = unix_now_secs();
        for slot in &self.slots {
            // Skip slots that are already in use.
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }

            // ── M4: factory called OUTSIDE the mutex ─────────────────────────
            // Step 1: take lock, check state, determine if reinit is needed.
            // B1: busy is set INSIDE the MutexGuard scope to close the TOCTOU
            // window between lock release and busy.store — without this, two
            // threads could both see needs_reinit=true and both call factory.
            let needs_reinit = {
                let guard = match slot.item.lock() {
                    Ok(g) => g,
                    Err(poisoned) => {
                        tracing::warn!("pool mutex poisoned — recovering inner value");
                        metrics::counter!(
                            "embed_pool_mutex_poisoned_total",
                            "model" => self.model.clone()
                        )
                        .increment(1);
                        poisoned.into_inner()
                    }
                };
                // Re-check busy inside lock to avoid TOCTOU.
                if slot.busy.load(Ordering::Acquire) {
                    continue;
                }
                if guard.is_none() {
                    // Claim the slot NOW, while still holding the lock.
                    // This prevents a second thread from also deciding to reinit.
                    slot.busy.store(true, Ordering::Release);
                    true
                } else {
                    false
                }
            }; // lock released HERE — busy=true already set, slot is claimed

            if needs_reinit {
                // Step 2: slot is already marked busy (done inside the lock above).

                // Step 3: call factory WITHOUT holding the mutex.
                metrics::counter!(
                    "embed_pool_cold_starts_total",
                    "model" => self.model.clone()
                )
                .increment(1);
                tracing::info!(model = %self.model, "pool cold start: reinitializing evicted slot");
                let new_item = match (self.factory)() {
                    Ok(item) => item,
                    Err(e) => {
                        tracing::error!(model = %self.model, "pool slot reinit failed: {e}");
                        metrics::counter!(
                            "embed_pool_reinit_failures_total",
                            "model" => self.model.clone()
                        )
                        .increment(1);
                        // Leave slot as None; clear busy so next acquire can retry.
                        slot.busy.store(false, Ordering::Release);
                        return Err(AcquireError::ReinitFailed(e));
                    }
                };

                // Step 4: take lock again, store item, take it out for the guard.
                let mut guard = match slot.item.lock() {
                    Ok(g) => g,
                    Err(poisoned) => {
                        tracing::warn!("pool mutex poisoned after reinit — recovering");
                        metrics::counter!(
                            "embed_pool_mutex_poisoned_total",
                            "model" => self.model.clone()
                        )
                        .increment(1);
                        poisoned.into_inner()
                    }
                };
                *guard = Some(new_item);
                slot.last_used.store(now, Ordering::Relaxed);
                let item = guard
                    .take()
                    .expect("BUG: slot item missing after reinit completed");
                return Ok(EvictableGuard {
                    slot: Arc::clone(slot),
                    item: Some(item),
                });
            }

            // Slot has an item — claim it under lock.
            let mut guard = match slot.item.lock() {
                Ok(g) => g,
                Err(poisoned) => {
                    tracing::warn!("pool mutex poisoned — recovering inner value");
                    metrics::counter!(
                        "embed_pool_mutex_poisoned_total",
                        "model" => self.model.clone()
                    )
                    .increment(1);
                    poisoned.into_inner()
                }
            };
            // ── B1: set busy BEFORE releasing MutexGuard ─────────────────────
            // This prevents a race window between mutex unlock and busy.store.
            if slot.busy.load(Ordering::Acquire) {
                // Another thread grabbed it between our check and the lock.
                continue;
            }
            slot.busy.store(true, Ordering::Release);
            slot.last_used.store(now, Ordering::Relaxed);
            let item = guard
                .take()
                .expect("BUG: slot item missing after lock acquired");
            // guard drops here (MutexGuard released) — busy is already true.
            return Ok(EvictableGuard {
                slot: Arc::clone(slot),
                item: Some(item),
            });
        }
        Err(AcquireError::AllBusy)
    }

    /// Evict slots idle longer than `threshold_secs`. Returns count evicted.
    /// When `self.idle_secs == 0`, does nothing and returns 0.
    pub fn evict_idle(&self, threshold_secs: u64) -> usize {
        if self.idle_secs == 0 {
            return 0;
        }
        let now = unix_now_secs();
        let mut count = 0;
        for slot in &self.slots {
            // Fast-path skip for busy slots without taking the mutex.
            // This is a performance optimisation only — the inner re-check
            // (!slot.busy.load inside the let-chain below) is the actual
            // race protection that prevents evicting a slot being acquired.
            if slot.busy.load(Ordering::Acquire) {
                continue;
            }
            let age = now.saturating_sub(slot.last_used.load(Ordering::Relaxed));
            if age >= threshold_secs
                && let Ok(mut guard) = slot.item.lock()
                && guard.is_some()
                && !slot.busy.load(Ordering::Acquire)
            {
                *guard = None;
                count += 1;
                metrics::counter!(
                    "embed_pool_evictions_total",
                    "model" => self.model.clone()
                )
                .increment(1);
                tracing::info!(model = %self.model, age, "pool evicted idle slot");
            }
        }
        count
    }

    /// Spawn a background tokio task that calls `evict_idle` every `tick` interval.
    ///
    /// The returned `JoinHandle` can be aborted to stop the loop.
    /// Callers should gate on `idle_secs > 0` before calling this.
    pub fn spawn_eviction_loop(
        self: &Arc<Self>,
        tick: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let pool = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            // tokio::time::interval fires the first tick immediately on creation.
            // Skip it so eviction doesn't run before the first idle window elapses.
            interval.tick().await; // skip the immediate first tick
            loop {
                interval.tick().await;
                let threshold = pool.idle_secs;
                // Catch panics in evict_idle to avoid silently killing the loop.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pool.evict_idle(threshold);
                }));
                if let Err(e) = result {
                    let msg = e
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| e.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("unknown panic");
                    tracing::error!(model = %pool.model, "eviction loop panicked: {msg}");
                    metrics::counter!(
                        "embed_pool_eviction_loop_panics_total",
                        "model" => pool.model.clone()
                    )
                    .increment(1);
                }
            }
        })
    }

    /// Test helper: push all slots' `last_used` `secs` seconds into the past.
    #[cfg(test)]
    pub fn force_last_used_ago(&self, secs: u64) {
        let past = unix_now_secs().saturating_sub(secs);
        for slot in &self.slots {
            slot.last_used.store(past, Ordering::Relaxed);
        }
    }

    /// Test helper: check if slot at index is evicted (None).
    #[cfg(test)]
    pub fn slot_is_evicted(&self, idx: usize) -> bool {
        self.slots[idx].item.lock().unwrap().is_none()
    }
}

impl<T> EvictableGuard<T> {
    /// Test helper: returns the slot index this guard holds within its pool.
    #[cfg(test)]
    pub fn slot_index(&self, pool: &EvictablePool<T>) -> usize {
        pool.slots
            .iter()
            .position(|s| Arc::ptr_eq(s, &self.slot))
            .expect("guard's slot not found in pool (should never happen)")
    }
}

// ── EvictableGuard ────────────────────────────────────────────────────────────

/// RAII guard that returns the item to its slot on drop and refreshes `last_used`.
pub struct EvictableGuard<T> {
    slot: Arc<EvictableSlot<T>>,
    item: Option<T>,
}

impl<T> std::ops::Deref for EvictableGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for EvictableGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.item.as_mut().unwrap()
    }
}

impl<T> Drop for EvictableGuard<T> {
    fn drop(&mut self) {
        let Some(item) = self.item.take() else {
            return;
        };
        let now = unix_now_secs();

        // ── ONE lock() call, one arm per outcome ─────────────────────────────
        // The previous shape was `if let Ok(..) = lock() { } else if let
        // Err(..) = lock() { }` — two `lock()` calls on the same mutex inside
        // `Drop`. It was deadlock-free only because edition 2024 drops the
        // `if let` scrutinee temporary before the `else` arm runs, and that
        // temporary is a `PoisonError<MutexGuard>` which still OWNS the guard.
        // On edition 2021 the guard is still alive when the second `lock()`
        // fires, so the same source self-deadlocks inside a destructor — with
        // no compile error. `Cargo.toml` says `edition = "2024"` today; a
        // future edition change must not be able to reintroduce that.
        //
        // A `match` also removes the second defect of the old shape: `if/else
        // if` had no final `else`, so a state matching neither arm dropped the
        // item silently — the exact loss that #137 added this recovery path to
        // prevent. `LockResult` has exactly two variants, so every outcome is
        // now handled by construction rather than by inspection.
        let mut guard = match self.slot.item.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Recover the inner value (matching the `acquire()` recovery
                // pattern above) so the item returns to its slot instead of
                // being lost. `into_inner()` yields the `Option<T>` regardless
                // of poison state; the item we hold is still valid.
                tracing::warn!("pool mutex poisoned on guard drop — item recovered");
                poisoned.into_inner()
            }
        };

        self.slot.last_used.store(now, Ordering::Relaxed);
        *guard = Some(item);
        // ── B1: busy.store INSIDE the MutexGuard scope ───────────────────────
        // Publishing busy=false while still holding the lock eliminates the
        // race window that existed when busy.store fired after unlock.
        self.slot.busy.store(false, Ordering::Release);
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AOrdering};

    fn make_pool(size: usize, idle_secs: u64) -> EvictablePool<u32> {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        EvictablePool::from_items(
            vec![42u32; size],
            idle_secs,
            "test-model",
            Arc::new(move || {
                c.fetch_add(1, AOrdering::SeqCst);
                Ok(42u32)
            }),
        )
    }

    // ── 1. eviction_disabled_by_default ─────────────────────────────────────
    /// When idle_secs=0, evict_idle returns 0 regardless of age.
    #[test]
    fn eviction_disabled_by_default() {
        let pool = make_pool(2, 0);
        {
            let _guard = pool.acquire().expect("should acquire");
        }
        pool.force_last_used_ago(9999);
        assert_eq!(pool.evict_idle(1), 0, "eviction disabled => no evictions");
    }

    // ── 2. eviction_after_idle_threshold ────────────────────────────────────
    /// With idle_secs=1 and slots idle for 2s, both slots are evicted.
    #[test]
    fn eviction_after_idle_threshold() {
        let pool = make_pool(2, 1);
        {
            let _guard = pool.acquire().expect("should acquire");
        }
        pool.force_last_used_ago(2);

        let evicted = pool.evict_idle(1);
        assert_eq!(
            evicted, 2,
            "both idle slots should be evicted after threshold"
        );
    }

    // ── 3. lazy_reinit_after_eviction ───────────────────────────────────────
    /// After eviction, next acquire calls factory and returns the new item.
    #[test]
    fn lazy_reinit_after_eviction() {
        let pool = make_pool(1, 1);
        {
            let _guard = pool.acquire().expect("acquire 1");
        }
        pool.force_last_used_ago(5);
        let evicted = pool.evict_idle(1);
        assert_eq!(evicted, 1, "one slot evicted");

        let guard = pool.acquire().expect("reinit acquire");
        assert_eq!(*guard, 42u32, "reinit should produce factory value");
    }

    // ── 4. factory_error_returns_err_and_slot_stays_alive ──────────────────
    /// On factory failure: Err returned, slot stays None, busy cleared,
    /// next acquire retries and succeeds when factory recovers.
    #[test]
    fn factory_error_returns_err_and_slot_stays_alive() {
        use std::sync::atomic::AtomicUsize;
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        // Use from_items so pool starts pre-filled (factory not called at init).
        let pool = Arc::new(EvictablePool::from_items(
            vec![0u32],
            1,
            "test-model",
            Arc::new(move || {
                let n = cc.fetch_add(1, AOrdering::SeqCst);
                if n == 0 {
                    Err("factory fail".to_string())
                } else {
                    Ok(99u32)
                }
            }),
        ));

        // Force evict.
        pool.force_last_used_ago(10);
        pool.evict_idle(1);
        assert!(pool.slot_is_evicted(0), "slot must be evicted");

        // First acquire: factory fails → Err.
        let result = pool.acquire();
        assert!(
            result.is_err(),
            "acquire must return Err when factory fails"
        );

        // Slot stays None, busy cleared.
        assert!(
            pool.slot_is_evicted(0),
            "slot stays None after reinit failure"
        );
        assert!(
            !pool.slots[0].busy.load(Ordering::Acquire),
            "busy must be cleared after reinit failure"
        );

        // Second acquire: factory succeeds.
        let guard = pool
            .acquire()
            .expect("second acquire must succeed after factory recovers");
        assert_eq!(*guard, 99u32);
    }

    // ── 5. tick_interval_quarter_of_idle_secs ───────────────────────────────
    /// tick = max(idle_secs / 4, 5s). Pure formula test.
    #[test]
    fn tick_interval_quarter_of_idle_secs() {
        use std::time::Duration;
        let compute_tick = |idle_secs: u64| -> Duration {
            let quarter = Duration::from_secs(idle_secs / 4);
            quarter.max(Duration::from_secs(5))
        };

        assert_eq!(compute_tick(120), Duration::from_secs(30));
        assert_eq!(compute_tick(40), Duration::from_secs(10));
        assert_eq!(
            compute_tick(16),
            Duration::from_secs(5),
            "minimum 5s applies"
        );
        assert_eq!(
            compute_tick(4),
            Duration::from_secs(5),
            "aggressive threshold: 4s idle → 5s tick minimum"
        );
    }

    // ── 6. eviction_loop_runs_and_evicts ────────────────────────────────────
    /// `spawn_eviction_loop` calls evict_idle periodically and evicts slots.
    #[tokio::test]
    async fn eviction_loop_runs_and_evicts() {
        let pool = Arc::new(make_pool(1, 1));

        {
            let _guard = pool.acquire().expect("acquire for loop test");
        }
        pool.force_last_used_ago(5);

        let handle = pool.spawn_eviction_loop(std::time::Duration::from_millis(100));
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        handle.abort();

        assert!(
            pool.slot_is_evicted(0),
            "slot should be evicted after loop ran"
        );
    }

    // ── 7. no_eviction_when_idle_secs_zero ──────────────────────────────────
    /// When idle_secs=0, loop must not evict even with stale last_used.
    #[tokio::test]
    async fn no_eviction_when_idle_secs_zero() {
        let pool = Arc::new(make_pool(1, 0));
        {
            let _guard = pool.acquire().expect("acquire");
        }
        pool.force_last_used_ago(9999);

        let handle = pool.spawn_eviction_loop(std::time::Duration::from_millis(100));
        tokio::time::sleep(std::time::Duration::from_millis(350)).await;
        handle.abort();

        assert!(
            !pool.slot_is_evicted(0),
            "idle_secs=0 → no eviction even with stale last_used"
        );
    }

    // ── drop ordering: concurrent stress ────────────────────────────────────
    #[test]
    fn drop_ordering_no_race() {
        let pool = Arc::new(EvictablePool::from_items(
            vec![42u32; 2],
            0,
            "stress-model",
            Arc::new(|| Ok(42u32)),
        ));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let p = pool.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    if let Ok(guard) = p.acquire() {
                        drop(guard);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("thread must not panic");
        }

        for slot in &pool.slots {
            assert!(slot.item.lock().unwrap().is_some(), "item must be returned");
            assert!(!slot.busy.load(AOrdering::Acquire), "busy must be false");
        }
    }

    // ── #94: poisoned mutex on guard drop must not lose the item ───────────
    /// Acquire a guard, then poison the slot's mutex from another thread by
    /// panicking while holding the lock. Drop the guard — the item must be
    /// recovered into the slot (via `poisoned.into_inner()`) rather than
    /// lost. The slot must be reusable afterwards.
    #[test]
    fn poisoned_mutex_drop_recovers_item() {
        let pool = Arc::new(EvictablePool::from_items(
            vec![100u32],
            0,
            "poison-test",
            Arc::new(|| Ok(100u32)),
        ));

        // Acquire the guard (takes the item out of the slot).
        let guard = pool.acquire().expect("slot must be acquirable");
        assert_eq!(*guard, 100u32);

        // Poison the mutex from a separate thread: hold the lock, then panic.
        // The panic propagates out of the thread (caught by join), leaving the
        // mutex poisoned.
        let slot = pool.slots[0].clone();
        let h = std::thread::spawn(move || {
            let _lock = slot.item.lock().unwrap();
            panic!("intentional poison");
        });
        assert!(h.join().is_err(), "poison thread must panic");

        // Drop the guard — the Drop impl must recover the item via
        // poisoned.into_inner() instead of losing it.
        drop(guard);

        // The slot must contain the item again and busy must be cleared.
        // The mutex is still poisoned (we recovered via into_inner but didn't
        // clear the poison flag), so use into_inner() to read.
        let slot0 = &pool.slots[0];
        let guard = slot0.item.lock().unwrap_or_else(|p| p.into_inner());
        assert!(
            guard.is_some(),
            "item must be recovered after poisoned drop"
        );
        drop(guard);
        assert!(
            !slot0.busy.load(AOrdering::Acquire),
            "busy must be cleared after poisoned drop"
        );

        // The slot must be reusable: a fresh acquire must succeed (acquire()
        // already handles poisoned mutex via into_inner() at lines 139-146).
        let guard2 = pool
            .acquire()
            .expect("slot must be reusable after recovery");
        assert_eq!(*guard2, 100u32);
    }

    // ── #152: a poisoned drop must COMPLETE, not merely recover ─────────────
    /// `poisoned_mutex_drop_recovers_item` above asserts the item comes back,
    /// but asserts nothing about the destructor terminating — and termination
    /// is the property #152 is about. The shape this replaced was
    /// `if let Ok(..) = lock() { } else if let Err(..) = lock() { }`: two
    /// `lock()` calls on one mutex, safe only because edition 2024 drops the
    /// `if let` scrutinee temporary — a `PoisonError<MutexGuard>` that still
    /// owns the guard — before the `else` arm runs. On an edition that does
    /// not, the second `lock()` blocks forever inside `Drop`.
    ///
    /// A deadlocked test HANGS rather than fails, and a hanging job reads as a
    /// slow one in CI, so the completion is asserted explicitly and with a
    /// bound. Kills: reverting the `match` to the double-`lock()` chain under
    /// an edition where the temporary outlives the arm.
    #[test]
    fn poisoned_drop_completes_and_does_not_deadlock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let pool = Arc::new(EvictablePool::from_items(
            vec![7u32],
            0,
            "deadlock-test",
            Arc::new(|| Ok(7u32)),
        ));

        let guard = pool.acquire().expect("slot must be acquirable");

        // Poison the mutex: hold the lock in another thread, then panic.
        let slot = pool.slots[0].clone();
        let h = std::thread::spawn(move || {
            let _lock = slot.item.lock().unwrap();
            panic!("intentional poison");
        });
        assert!(h.join().is_err(), "poison thread must panic");

        // Run the drop on its own thread so a deadlock cannot hang the suite.
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(guard);
            let _ = tx.send(());
        });

        rx.recv_timeout(Duration::from_secs(5))
            .expect("Drop must complete on a poisoned mutex — it deadlocked");

        // And the recovery itself still holds.
        let slot0 = &pool.slots[0];
        let inner = slot0.item.lock().unwrap_or_else(|p| p.into_inner());
        assert!(inner.is_some(), "item must be back in the slot");
        drop(inner);
        assert!(
            !slot0.busy.load(AOrdering::Acquire),
            "busy must be cleared after a poisoned drop"
        );
    }

    // ── factory called outside lock (MAJOR 3 rewrite) ───────────────────────
    /// M4 proof: with pool_size=1, a slow factory (100ms) must NOT block a
    /// concurrent acquire on the mutex. Thread 2 should return AllBusy quickly
    /// (within 10ms) rather than blocking for the full 100ms factory duration.
    #[test]
    fn factory_called_outside_lock() {
        use std::time::{Duration, Instant};

        // pool_size=1: only one slot, so both threads compete for the same slot.
        let pool = Arc::new(EvictablePool::from_items(
            vec![0u32],
            1,
            "slow-factory-m4",
            Arc::new(|| {
                // Simulate slow ONNX cold start — 100ms.
                std::thread::sleep(Duration::from_millis(100));
                Ok(77u32)
            }),
        ));

        // Evict the single slot so t1 will trigger a reinit (and call factory).
        pool.force_last_used_ago(10);
        pool.evict_idle(1);
        assert!(pool.slot_is_evicted(0), "slot must be evicted before test");

        let p1 = pool.clone();
        let p2 = pool.clone();

        // t1 starts first and enters the slow factory path.
        let t1 = std::thread::spawn(move || p1.acquire());

        // Give t1 time to enter the reinit branch and start the factory call,
        // but NOT enough time to finish it (factory takes 100ms).
        std::thread::sleep(Duration::from_millis(10));

        // t2 should return AllBusy quickly WITHOUT blocking on the mutex.
        let t2_start = Instant::now();
        let t2 = std::thread::spawn(move || p2.acquire());
        let r2 = t2.join().expect("t2 must not panic");
        let t2_elapsed = t2_start.elapsed();

        let _r1 = t1.join().expect("t1 must not panic");

        // If factory was called inside the mutex, t2 would block ~90ms.
        // With factory outside the mutex, t2 returns AllBusy within ~5ms.
        assert!(
            t2_elapsed < Duration::from_millis(20),
            "t2 blocked {t2_elapsed:?} — factory must be called OUTSIDE the mutex (M4)"
        );
        assert!(
            matches!(r2, Err(AcquireError::AllBusy)),
            "t2 must get AllBusy (slot is being reinitialised)"
        );
    }

    // ── BLOCKER 1: dual-reinit race in acquire reinit path ───────────────────
    /// Two threads racing on an evicted slot must call factory exactly once.
    /// The winner gets the guard; the loser gets AllBusy.
    ///
    /// Factory sleeps 50ms to widen the race window.
    #[test]
    fn dual_reinit_race_yields_single_factory_call() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;

        let factory_calls = Arc::new(AtomicUsize::new(0));
        let fc = factory_calls.clone();

        let pool = Arc::new(EvictablePool::from_items(
            vec![0u32],
            1,
            "race-test",
            Arc::new(move || {
                fc.fetch_add(1, AOrdering::SeqCst);
                // Sleep to widen the race window between needs_reinit detection
                // and busy.store — without the fix, both threads pass the race.
                std::thread::sleep(Duration::from_millis(50));
                Ok(99u32)
            }),
        ));

        // Evict the only slot.
        pool.force_last_used_ago(10);
        pool.evict_idle(1);
        assert!(pool.slot_is_evicted(0), "slot must be evicted before race");

        let p1 = pool.clone();
        let p2 = pool.clone();

        // Launch both threads simultaneously — no sleep gap between them.
        let t1 = std::thread::spawn(move || p1.acquire());
        let t2 = std::thread::spawn(move || p2.acquire());

        let r1 = t1.join().expect("t1 must not panic");
        let r2 = t2.join().expect("t2 must not panic");

        // Exactly one factory call — if both reinit, this fails.
        let calls = factory_calls.load(AOrdering::SeqCst);
        assert_eq!(
            calls, 1,
            "factory must be called exactly once; got {calls} calls (dual-reinit race triggered)"
        );

        // Exactly one success, one AllBusy.
        let ok_count = r1.is_ok() as usize + r2.is_ok() as usize;
        let busy_count = matches!(r1, Err(AcquireError::AllBusy)) as usize
            + matches!(r2, Err(AcquireError::AllBusy)) as usize;
        assert_eq!(ok_count, 1, "exactly one acquire must succeed");
        assert_eq!(busy_count, 1, "exactly one acquire must get AllBusy");
    }

    // ── BLOCKER 2: acquire skips busy slots in order ─────────────────────────
    /// With pool_size=3, three sequential acquires (each held) must return
    /// guards for distinct slot indices (0, 1, 2 in some order).
    #[test]
    fn acquire_skips_busy_slots_in_order() {
        let pool = EvictablePool::from_items(
            vec![10u32, 20u32, 30u32],
            0,
            "order-test",
            Arc::new(|| Ok(0u32)),
        );

        let g0 = pool.acquire().expect("first acquire");
        let g1 = pool.acquire().expect("second acquire");
        let g2 = pool.acquire().expect("third acquire");

        let i0 = g0.slot_index(&pool);
        let i1 = g1.slot_index(&pool);
        let i2 = g2.slot_index(&pool);

        assert_ne!(i0, i1, "first and second guards must be on different slots");
        assert_ne!(i1, i2, "second and third guards must be on different slots");
        assert_ne!(i0, i2, "first and third guards must be on different slots");

        // All three slot indices together must cover {0, 1, 2}.
        let mut indices = [i0, i1, i2];
        indices.sort_unstable();
        assert_eq!(
            indices,
            [0, 1, 2],
            "guards must cover all three slots exactly"
        );
    }
}
