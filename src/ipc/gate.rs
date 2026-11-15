//! Priority-aware admission gate for the supervisor -> worker connection pool.
//!
//! Replaces the plain `tokio::sync::Semaphore` that capped in-flight worker
//! connections at `pool_size`. Same capacity semantics; the difference is the
//! order waiters are admitted in:
//!
//! * [`Priority::Interactive`] (the default) is admitted before
//!   [`Priority::Bulk`].
//! * Bulk cannot starve: a bulk waiter that has waited at least the aging
//!   bound is treated as interactive and competes by arrival time.
//!
//! # Why aging and not a ratio
//!
//! A ratio ("1 bulk per N interactive") bounds throughput share but not
//! latency: how long a bulk request waits then depends on the interactive
//! arrival rate, so no number can be quoted to a bulk caller. An aging bound
//! gives one knob in seconds, `EMBED_BULK_AGING_MS`, and a worst-case
//! wait of roughly `aging + the work already queued ahead of it`. It also
//! costs nothing when idle: with no interactive waiter a bulk request is
//! admitted immediately.
//!
//! # Cancel safety
//!
//! A permit is handed to a waiter through a oneshot channel. If the waiting
//! future is dropped (axum timeout, client disconnect) the [`WaitGuard`]
//! removes it from the queue, and if a permit was already granted but never
//! picked up, returns it. A grant to a receiver that is already gone is
//! detected at send time and offered to the next waiter. No path leaks a
//! permit.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::Instant;

/// Env var: how long a bulk waiter waits before it is served as interactive.
pub const BULK_AGING_ENV: &str = "EMBED_BULK_AGING_MS";
/// Default aging bound (2 s).
pub const DEFAULT_BULK_AGING: Duration = Duration::from_millis(2000);

/// Request class on the supervisor wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Priority {
    #[default]
    Interactive,
    Bulk,
    /// Internal liveness probes (heartbeat, `/ready`). Queued as interactive;
    /// a separate value only so they do not pollute the wait metric that
    /// callers' latency is read from. Never produced from a header.
    Probe,
}

impl Priority {
    /// Bounded metric label value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Bulk => "bulk",
            Self::Probe => "probe",
        }
    }

    /// Parse the `X-Embed-Priority` header value. Anything that is not
    /// `bulk` (case-insensitive) is interactive: an unknown value must never
    /// demote a caller, and omitting the header must not change behaviour.
    pub fn from_header(value: &str) -> Self {
        if value.trim().eq_ignore_ascii_case("bulk") {
            Self::Bulk
        } else {
            Self::Interactive
        }
    }
}

tokio::task_local! {
    /// Priority of the request being served, set by the HTTP middleware
    /// ([`scope_from_headers`]). Read where the supervisor waits for a permit.
    static REQUEST_PRIORITY: Priority;
}

/// Priority of the current request; [`Priority::Interactive`] outside a scope
/// (health probes, heartbeat, tests).
pub fn current_priority() -> Priority {
    REQUEST_PRIORITY.try_with(|p| *p).unwrap_or_default()
}

/// Run `fut` with `priority` as the current request priority.
pub async fn with_priority<F: std::future::Future>(priority: Priority, fut: F) -> F::Output {
    REQUEST_PRIORITY.scope(priority, fut).await
}

/// Axum middleware: read `X-Embed-Priority` and scope the request with it.
pub async fn scope_from_headers(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let priority = req
        .headers()
        .get("x-embed-priority")
        .and_then(|v| v.to_str().ok())
        .map(Priority::from_header)
        .unwrap_or_default();
    with_priority(priority, next.run(req)).await
}

/// Install the request-priority layer on a router. The single place both the
/// server and its test build the layering, so the test covers the real wiring.
pub fn with_priority_layer<S>(router: axum::Router<S>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(axum::middleware::from_fn(scope_from_headers))
}

/// Resolve the aging bound from `EMBED_BULK_AGING_MS`. Zero or garbage falls
/// back to the default and bumps the config-fallback counter.
pub fn bulk_aging_from_env() -> Duration {
    parse_aging(std::env::var(BULK_AGING_ENV).ok().as_deref())
}

fn parse_aging(raw: Option<&str>) -> Duration {
    let Some(raw) = raw else {
        return DEFAULT_BULK_AGING;
    };
    match raw.trim().parse::<u64>() {
        Ok(n) if n > 0 => Duration::from_millis(n),
        _ => {
            tracing::warn!(
                env_key = BULK_AGING_ENV,
                value = %raw,
                "invalid bulk aging; using default"
            );
            crate::metrics::record_config_fallback(BULK_AGING_ENV);
            DEFAULT_BULK_AGING
        }
    }
}

struct Waiter {
    id: u64,
    enqueued: Instant,
    tx: oneshot::Sender<()>,
}

struct State {
    /// Free permits. Invariant: `permits > 0` implies both queues are empty
    /// (a release hands the permit straight to a waiter when one exists).
    permits: usize,
    next_id: u64,
    interactive: VecDeque<Waiter>,
    bulk: VecDeque<Waiter>,
}

pub struct PriorityGate {
    state: Mutex<State>,
    aging: Duration,
}

impl PriorityGate {
    pub fn new(permits: usize, aging: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                permits,
                next_id: 0,
                interactive: VecDeque::new(),
                bulk: VecDeque::new(),
            }),
            aging,
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // The critical sections do not panic; a poisoned lock still holds
        // consistent data, so keep serving instead of cascading the panic.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Free permits right now (tests / diagnostics).
    #[cfg(test)]
    fn available(&self) -> usize {
        self.lock().permits
    }

    /// Queued waiters (interactive, bulk) right now (tests).
    #[cfg(test)]
    fn queued(&self) -> (usize, usize) {
        let st = self.lock();
        (st.interactive.len(), st.bulk.len())
    }

    /// Wait for a permit. Cancel-safe: dropping the returned future at any
    /// point leaves the gate's permit count unchanged.
    pub async fn acquire(self: &Arc<Self>, priority: Priority) -> Permit {
        let (id, rx) = {
            let mut st = self.lock();
            if st.permits > 0 {
                st.permits -= 1;
                return Permit {
                    gate: Arc::clone(self),
                };
            }
            let (tx, rx) = oneshot::channel();
            let id = st.next_id;
            st.next_id += 1;
            let waiter = Waiter {
                id,
                enqueued: Instant::now(),
                tx,
            };
            match priority {
                Priority::Interactive | Priority::Probe => st.interactive.push_back(waiter),
                Priority::Bulk => st.bulk.push_back(waiter),
            }
            (id, rx)
        };
        let mut guard = WaitGuard {
            gate: self,
            id,
            rx,
            armed: true,
        };
        let granted = (&mut guard.rx).await;
        // Resolved: the permit (or the teardown error) is ours; disarm.
        guard.armed = false;
        // The sender lives in the gate's own queue, which `self` keeps alive,
        // so a dropped sender without a grant cannot happen.
        debug_assert!(granted.is_ok(), "waiter sender dropped without a grant");
        Permit {
            gate: Arc::clone(self),
        }
    }

    /// Pick the next waiter: interactive first, except an aged bulk waiter
    /// that arrived before the interactive head.
    fn pop_next(&self, st: &mut State) -> Option<Waiter> {
        let now = Instant::now();
        let aged_bulk_first = match (st.interactive.front(), st.bulk.front()) {
            (None, Some(_)) => true,
            (Some(i), Some(b)) => {
                now.duration_since(b.enqueued) >= self.aging && b.enqueued <= i.enqueued
            }
            _ => false,
        };
        if aged_bulk_first {
            st.bulk.pop_front()
        } else {
            st.interactive.pop_front()
        }
    }

    /// Return one permit: hand it to the next live waiter, else bank it.
    fn release_locked(&self, st: &mut State) {
        while let Some(w) = self.pop_next(st) {
            if w.tx.send(()).is_ok() {
                return;
            }
            // Receiver gone (cancelled, guard not yet run): try the next one.
        }
        st.permits += 1;
    }

    fn release(&self) {
        let mut st = self.lock();
        self.release_locked(&mut st);
    }
}

/// An admitted request. Dropping it frees the slot.
#[must_use = "dropping the permit immediately releases the slot"]
pub struct Permit {
    gate: Arc<PriorityGate>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// Armed while a waiter is queued; cleans up if the future is dropped.
struct WaitGuard<'a> {
    gate: &'a Arc<PriorityGate>,
    id: u64,
    rx: oneshot::Receiver<()>,
    armed: bool,
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return; // permit was received; Permit now owns the release
        }
        let mut st = self.gate.lock();
        st.interactive.retain(|w| w.id != self.id);
        st.bulk.retain(|w| w.id != self.id);
        self.rx.close();
        // A grant that landed before we got the lock was never picked up.
        if self.rx.try_recv().is_ok() {
            self.gate.release_locked(&mut st);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{advance, sleep};

    const AGING: Duration = Duration::from_secs(2);

    /// Spawn a waiter that records its admission order, then holds the permit
    /// until `hold` elapses.
    fn spawn_waiter(
        gate: &Arc<PriorityGate>,
        prio: Priority,
        label: &'static str,
        order: &Arc<Mutex<Vec<&'static str>>>,
        hold: Duration,
    ) -> tokio::task::JoinHandle<()> {
        let gate = Arc::clone(gate);
        let order = Arc::clone(order);
        tokio::spawn(async move {
            let _p = gate.acquire(prio).await;
            order.lock().unwrap().push(label);
            sleep(hold).await;
        })
    }

    /// Every paused-time test body runs under this. A lost wakeup (a waiter
    /// that is never granted) leaves nothing runnable, so paused time would
    /// stall CI forever; the timeout turns that into a failure.
    async fn bounded<F: std::future::Future>(fut: F) -> F::Output {
        tokio::time::timeout(Duration::from_secs(120), fut)
            .await
            .expect("lost wakeup: test did not finish in 120 s of virtual time")
    }

    async fn settle() {
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
    }

    #[test]
    fn header_parsing_defaults_to_interactive() {
        assert_eq!(Priority::from_header("bulk"), Priority::Bulk);
        assert_eq!(Priority::from_header(" BULK "), Priority::Bulk);
        assert_eq!(Priority::from_header("interactive"), Priority::Interactive);
        assert_eq!(Priority::from_header("urgent"), Priority::Interactive);
        assert_eq!(Priority::from_header(""), Priority::Interactive);
        assert_eq!(current_priority(), Priority::Interactive);
    }

    #[tokio::test]
    async fn scope_sets_current_priority() {
        let seen = with_priority(Priority::Bulk, async { current_priority() }).await;
        assert_eq!(seen, Priority::Bulk);
    }

    #[tokio::test(start_paused = true)]
    async fn interactive_is_admitted_before_earlier_bulk() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let order = Arc::new(Mutex::new(Vec::new()));
            let holder = gate.acquire(Priority::Interactive).await;

            // Bulk arrives FIRST, interactive second: priority must flip them.
            let b = spawn_waiter(&gate, Priority::Bulk, "bulk", &order, Duration::ZERO);
            settle().await;
            let i = spawn_waiter(
                &gate,
                Priority::Interactive,
                "inter",
                &order,
                Duration::ZERO,
            );
            settle().await;

            drop(holder);
            let _ = tokio::join!(b, i);
            assert_eq!(*order.lock().unwrap(), vec!["inter", "bulk"]);
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn same_class_is_fifo() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let order = Arc::new(Mutex::new(Vec::new()));
            let holder = gate.acquire(Priority::Interactive).await;
            let a = spawn_waiter(&gate, Priority::Interactive, "a", &order, Duration::ZERO);
            settle().await;
            let b = spawn_waiter(&gate, Priority::Interactive, "b", &order, Duration::ZERO);
            settle().await;
            drop(holder);
            let _ = tokio::join!(a, b);
            assert_eq!(*order.lock().unwrap(), vec!["a", "b"]);
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn idle_bulk_is_admitted_immediately() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let _p = gate.acquire(Priority::Bulk).await;
            assert_eq!(gate.available(), 0);
        })
        .await;
    }

    /// Sustained interactive load: every permit release finds a fresh
    /// interactive waiter queued behind it. Without aging the bulk request
    /// never runs; with aging it is served once it is older than the bound.
    #[tokio::test(start_paused = true)]
    async fn bulk_is_not_starved_by_sustained_interactive_load() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let order = Arc::new(Mutex::new(Vec::new()));
            let bulk_done = Arc::new(AtomicUsize::new(0));

            let holder = gate.acquire(Priority::Interactive).await;
            let bulk = {
                let gate = Arc::clone(&gate);
                let done = Arc::clone(&bulk_done);
                tokio::spawn(async move {
                    let _p = gate.acquire(Priority::Bulk).await;
                    done.fetch_add(1, Ordering::SeqCst);
                })
            };
            settle().await;

            // Interactive chain: 20 waiters, each holds 500 ms, always one queued.
            let mut chain = Vec::new();
            for _ in 0..20 {
                chain.push(spawn_waiter(
                    &gate,
                    Priority::Interactive,
                    "i",
                    &order,
                    Duration::from_millis(500),
                ));
                settle().await;
            }
            drop(holder);
            // 8 s of virtual time >> 2 s aging bound, with interactive still queued.
            for _ in 0..80 {
                advance(Duration::from_millis(100)).await;
                settle().await;
            }
            assert_eq!(
                bulk_done.load(Ordering::SeqCst),
                1,
                "bulk waiter starved despite aging bound"
            );
            // And it did not jump the whole chain: interactive work was served too.
            assert!(order.lock().unwrap().len() >= 3);
            bulk.await.unwrap();
            for c in chain {
                c.abort();
            }
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn bulk_is_held_back_until_it_ages() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let order = Arc::new(Mutex::new(Vec::new()));
            let holder = gate.acquire(Priority::Interactive).await;
            let b = spawn_waiter(&gate, Priority::Bulk, "bulk", &order, Duration::ZERO);
            settle().await;
            advance(Duration::from_millis(500)).await; // bulk is 0.5 s old: young
            let i = spawn_waiter(
                &gate,
                Priority::Interactive,
                "inter",
                &order,
                Duration::ZERO,
            );
            settle().await;
            drop(holder);
            let _ = tokio::join!(b, i);
            assert_eq!(*order.lock().unwrap(), vec!["inter", "bulk"]);
        })
        .await;
    }

    /// Cancelling a queued waiter must not lose a permit.
    #[tokio::test(start_paused = true)]
    async fn cancelled_waiter_leaks_no_permit() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let holder = gate.acquire(Priority::Interactive).await;

            let victim = {
                let gate = Arc::clone(&gate);
                tokio::spawn(async move {
                    let _p = gate.acquire(Priority::Interactive).await;
                })
            };
            settle().await;
            assert_eq!(gate.queued(), (1, 0), "victim should be queued");
            victim.abort(); // client disconnect / axum timeout
            let _ = victim.await;
            settle().await;
            assert_eq!(gate.queued(), (0, 0), "cancelled waiter left in queue");

            drop(holder);
            assert_eq!(gate.available(), 1, "permit leaked after waiter cancel");
            // And the gate still admits.
            let _p = gate.acquire(Priority::Bulk).await;
        })
        .await;
    }

    /// The nasty race: the permit was already granted to a waiter whose
    /// future is dropped before it ever polls the grant.
    #[tokio::test(start_paused = true)]
    async fn grant_then_cancel_before_pickup_returns_permit() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let holder = gate.acquire(Priority::Interactive).await;

            let victim = {
                let gate = Arc::clone(&gate);
                tokio::spawn(async move {
                    let _p = gate.acquire(Priority::Interactive).await;
                    sleep(Duration::from_secs(60)).await;
                })
            };
            settle().await;
            // Grant (releases to the queued victim), then cancel it before it runs.
            drop(holder);
            victim.abort();
            let _ = victim.await;
            settle().await;
            assert_eq!(gate.available(), 1, "granted-but-cancelled permit leaked");
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_around_acquire_leaks_nothing() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let _holder = gate.acquire(Priority::Interactive).await;
            for _ in 0..10 {
                let r =
                    tokio::time::timeout(Duration::from_millis(50), gate.acquire(Priority::Bulk))
                        .await;
                assert!(r.is_err());
            }
            drop(_holder);
            assert_eq!(gate.available(), 1);
        })
        .await;
    }

    /// Interactive arrived FIRST, bulk later; both are past the aging bound
    /// when the permit frees. Aging only promotes bulk to "interactive
    /// class", where arrival order rules, so interactive still goes first.
    #[tokio::test(start_paused = true)]
    async fn aged_bulk_does_not_jump_an_older_interactive() {
        bounded(async {
            let gate = PriorityGate::new(1, AGING);
            let order = Arc::new(Mutex::new(Vec::new()));
            let holder = gate.acquire(Priority::Interactive).await;
            let i = spawn_waiter(
                &gate,
                Priority::Interactive,
                "inter",
                &order,
                Duration::ZERO,
            );
            settle().await;
            advance(Duration::from_millis(100)).await;
            let b = spawn_waiter(&gate, Priority::Bulk, "bulk", &order, Duration::ZERO);
            settle().await;
            advance(AGING * 2).await; // both now far older than the bound
            drop(holder);
            let _ = tokio::join!(i, b);
            assert_eq!(*order.lock().unwrap(), vec!["inter", "bulk"]);
        })
        .await;
    }

    /// A bulk waiter exactly `aging` old is promoted (`>=`); one tick younger
    /// is not.
    #[tokio::test(start_paused = true)]
    async fn aging_boundary_is_inclusive() {
        for (age, expect) in [
            (AGING, vec!["bulk", "inter"]),
            (AGING - Duration::from_millis(1), vec!["inter", "bulk"]),
        ] {
            bounded(async {
                let gate = PriorityGate::new(1, AGING);
                let order = Arc::new(Mutex::new(Vec::new()));
                let holder = gate.acquire(Priority::Interactive).await;
                let b = spawn_waiter(&gate, Priority::Bulk, "bulk", &order, Duration::ZERO);
                settle().await;
                advance(age).await;
                let i = spawn_waiter(
                    &gate,
                    Priority::Interactive,
                    "inter",
                    &order,
                    Duration::ZERO,
                );
                settle().await;
                drop(holder); // no time passes between the advance and the grant
                let _ = tokio::join!(b, i);
                assert_eq!(*order.lock().unwrap(), expect, "bulk age {age:?}");
            })
            .await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn probe_queues_as_interactive_and_has_its_own_label() {
        bounded(async {
            assert_eq!(Priority::Probe.as_str(), "probe");
            assert_eq!(Priority::from_header("probe"), Priority::Interactive);
            let gate = PriorityGate::new(1, AGING);
            let order = Arc::new(Mutex::new(Vec::new()));
            let holder = gate.acquire(Priority::Interactive).await;
            let b = spawn_waiter(&gate, Priority::Bulk, "bulk", &order, Duration::ZERO);
            settle().await;
            let p = spawn_waiter(&gate, Priority::Probe, "probe", &order, Duration::ZERO);
            settle().await;
            drop(holder);
            let _ = tokio::join!(b, p);
            assert_eq!(*order.lock().unwrap(), vec!["probe", "bulk"]);
        })
        .await;
    }

    /// The header reaches the gate only through `with_priority_layer`, the
    /// function `main` builds its router with. Dropping the layer there makes
    /// the bulk case read Interactive.
    #[tokio::test]
    async fn router_layer_carries_header_to_current_priority() {
        use tower::ServiceExt;
        let app = with_priority_layer(axum::Router::new().route(
            "/p",
            axum::routing::post(|| async { current_priority().as_str() }),
        ));
        for (header, expect) in [
            (Some("bulk"), "bulk"),
            (Some("BULK"), "bulk"),
            (Some("nonsense"), "interactive"),
            (None, "interactive"),
        ] {
            let mut req = axum::http::Request::post("/p");
            if let Some(h) = header {
                req = req.header("X-Embed-Priority", h);
            }
            let resp = app
                .clone()
                .oneshot(req.body(axum::body::Body::empty()).unwrap())
                .await
                .unwrap();
            let body = axum::body::to_bytes(resp.into_body(), 64).await.unwrap();
            assert_eq!(&body[..], expect.as_bytes(), "header {header:?}");
        }
    }

    #[test]
    fn aging_parse_default_override_and_garbage() {
        assert_eq!(parse_aging(None), DEFAULT_BULK_AGING);
        assert_eq!(parse_aging(Some("750")), Duration::from_millis(750));
        assert_eq!(parse_aging(Some("0")), DEFAULT_BULK_AGING);
        assert_eq!(parse_aging(Some("soon")), DEFAULT_BULK_AGING);
    }
}
