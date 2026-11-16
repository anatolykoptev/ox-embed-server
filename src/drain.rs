//! In-flight request tracking for graceful shutdown.
//!
//! `shutdown_signal` used to sleep the full `drain_timeout` on every
//! SIGTERM whether or not anything was actually running, so each restart
//! stayed unavailable for the whole timeout. Instead we count live HTTP
//! requests and wait until the count reaches zero — or the timeout
//! elapses, whichever comes first. Idle shutdown returns immediately;
//! a busy shutdown unblocks the moment the last handler completes.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;

struct Inner {
    count: AtomicUsize,
    /// Woken when `count` transitions to 0. Used with the `enable()` +
    /// re-check pattern in `wait_idle` so a decrement that races with
    /// registration can never be missed.
    zero: tokio::sync::Notify,
}

/// Shared in-flight request counter. Clone to hand a handle to the
/// middleware layer and another to the shutdown path.
#[derive(Clone)]
pub(crate) struct InFlight {
    inner: Arc<Inner>,
}

/// RAII handle for one in-flight request: decrements the counter (and
/// wakes drain waiters on reaching 0) on drop, so handler success, error
/// and cancellation all release it exactly once.
pub(crate) struct InFlightGuard {
    inner: Arc<Inner>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if self.inner.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.inner.zero.notify_waiters();
        }
    }
}

impl InFlight {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                count: AtomicUsize::new(0),
                zero: tokio::sync::Notify::new(),
            }),
        }
    }

    /// Increment the in-flight count; the returned guard decrements on drop.
    pub(crate) fn enter(&self) -> InFlightGuard {
        self.inner.count.fetch_add(1, Ordering::SeqCst);
        InFlightGuard {
            inner: Arc::clone(&self.inner),
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.inner.count.load(Ordering::SeqCst)
    }

    /// Wait until the in-flight count reaches zero or `timeout` elapses,
    /// whichever comes first. Returns `true` when idle was reached,
    /// `false` on timeout (requests still running).
    pub(crate) async fn wait_idle(&self, timeout: Duration) -> bool {
        let idle = async {
            loop {
                // Register interest BEFORE checking the count: a decrement
                // that lands between enable() and the count check still
                // wakes this future, and a count already at 0 short-
                // circuits without ever waiting.
                let notified = self.inner.zero.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.count() == 0 {
                    return;
                }
                notified.await;
            }
        };
        tokio::time::timeout(timeout, idle).await.is_ok()
    }
}

/// Axum `from_fn_with_state` middleware: count the request as in-flight
/// for the lifetime of its handler. The guard drops — decrementing the
/// counter and waking drain waiters — when `next.run` resolves, so
/// success, error and handler cancellation are all accounted exactly once.
pub(crate) async fn track(State(in_flight): State<InFlight>, req: Request, next: Next) -> Response {
    let _guard = in_flight.enter();
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::routing::get;
    use axum::{Router, http::StatusCode};
    use tower::ServiceExt as _;

    /// A request must be counted while its handler runs and released as
    /// soon as the handler completes — this is the counter the drain
    /// path reads.
    #[tokio::test]
    async fn middleware_counts_request_for_handler_lifetime() {
        let in_flight = InFlight::new();
        let probe = in_flight.clone();
        let app = Router::new()
            .route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    "ok"
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                in_flight.clone(),
                track,
            ));

        let call = tokio::spawn(async move {
            app.oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap())
                .await
                .unwrap()
        });

        // Let the request reach the handler before sampling the counter.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            probe.count(),
            1,
            "request must be counted while its handler runs"
        );
        let resp = call.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            probe.count(),
            0,
            "counter must return to zero when the handler completes"
        );
    }

    /// A counter already at zero must not wait at all — the enable() +
    /// re-check pattern relies on this short-circuit.
    #[tokio::test]
    async fn wait_idle_returns_immediately_when_already_idle() {
        let in_flight = InFlight::new();
        let start = std::time::Instant::now();
        assert!(
            in_flight.wait_idle(Duration::from_secs(30)).await,
            "idle tracker must report idle"
        );
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "wait_idle on an idle tracker must not wait: {:?}",
            start.elapsed()
        );
    }

    /// A decrement racing with waiter registration must still unblock the
    /// wait — the whole reason for `enable()` + re-check.
    #[tokio::test]
    async fn wait_idle_unblocks_on_racing_release() {
        let in_flight = InFlight::new();
        let guard = in_flight.enter();
        let waiter = tokio::spawn({
            let tracker = in_flight.clone();
            async move { tracker.wait_idle(Duration::from_secs(30)).await }
        });
        // Release at a random-ish moment inside the wait.
        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(guard);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), waiter)
                .await
                .expect("waiter hung — missed the racing release")
                .unwrap(),
            "waiter must report idle after release"
        );
    }

    /// A held guard that never releases keeps wait_idle waiting until the
    /// bound — the timeout path.
    #[tokio::test]
    async fn wait_idle_times_out_when_request_never_finishes() {
        let in_flight = InFlight::new();
        let _guard = in_flight.enter();
        let start = std::time::Instant::now();
        assert!(
            !in_flight.wait_idle(Duration::from_millis(150)).await,
            "a live request must hold the wait until the timeout"
        );
        assert!(
            start.elapsed() >= Duration::from_millis(150),
            "timeout bound must hold: {:?}",
            start.elapsed()
        );
    }
}
