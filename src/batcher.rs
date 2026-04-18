//! Dynamic batching: coalesces concurrent embed calls, dispatches one embed_fn per batch.
//!
//! Phase B: token-budget accounting with padded-model formula
//! (port of HuggingFace text-embeddings-inference `core/src/queue.rs`).
//! The batcher accepts pre-tokenized input_ids and caps batches by the
//! padded total tokens `max(seq_len_in_batch) * n_items`, not by item count.
#![allow(dead_code)]
use std::cmp::max;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

#[derive(Debug)]
pub enum BatchError {
    QueueFull(QueueFullError),
    Inference(String),
    Shutdown,
}
impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchError::QueueFull(e) => write!(f, "queue full: {e}"),
            BatchError::Inference(msg) => write!(f, "inference error: {msg}"),
            BatchError::Shutdown => write!(f, "batcher shut down"),
        }
    }
}
impl std::error::Error for BatchError {}

#[derive(Debug)]
pub struct QueueFullError {
    pub batcher_name: String,
}
impl fmt::Display for QueueFullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "batcher '{}' queue is full", self.batcher_name)
    }
}
impl std::error::Error for QueueFullError {}

#[derive(Debug)]
struct Item {
    /// One entry per text in the request. Each entry is the tokenizer's
    /// input_ids (already truncated to the model's `max_len`).
    token_ids: Vec<Vec<u32>>,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
}

impl Item {
    /// Number of texts (≥ 1) this Item represents in a batch.
    fn n_texts(&self) -> usize {
        self.token_ids.len()
    }

    /// Longest token sequence inside this Item.
    fn max_seq_len(&self) -> usize {
        self.token_ids.iter().map(|v| v.len()).max().unwrap_or(0)
    }

    /// Sum of all token lengths across this Item's texts (non-padded).
    fn total_tokens(&self) -> usize {
        self.token_ids.iter().map(|v| v.len()).sum()
    }
}

#[derive(Debug)]
pub struct DynamicBatcher {
    name: Arc<String>,
    sender: mpsc::Sender<Item>,
    worker: JoinHandle<()>,
}

/// Configuration for the token-budget batcher. Mirrors TEI `core/src/queue.rs`.
#[derive(Debug, Clone, Copy)]
struct BatcherConfig {
    /// Hard cap on padded batch tokens: `max(seq_len) * n_items` must be
    /// strictly less than this value (a newly arriving item whose inclusion
    /// would make the product ≥ this limit is deferred to the next batch).
    max_batch_tokens: usize,
    /// Soft cap on items (texts) per batch. Kept for fairness — otherwise
    /// a single giant multi-text request could monopolise a dispatch.
    max_batch_items: usize,
    /// True for BERT-style encoders where every sequence in a batch is
    /// padded to `max(seq_len)`: total compute scales with the padded
    /// product, so the budget must account for that. False for models
    /// that can handle ragged batches (rare on ONNX CPU — mostly a
    /// future-proofing knob / testability hook).
    padded_model: bool,
    /// How long to wait after the first item before dispatching, giving
    /// concurrent requests a chance to coalesce into the same batch.
    wait: Duration,
}

impl DynamicBatcher {
    /// Create a token-budget batcher and start its worker. `embed_fn` runs
    /// in `spawn_blocking` and receives the batch's pre-tokenized input_ids
    /// (one `Vec<u32>` per text, flattened across all Items in the batch).
    ///
    /// Token accounting formula (padded model, ported from TEI):
    /// ```text
    /// new_max_len = max(current_max_len, entry.max_seq_len())
    /// padded_total = new_max_len * (current_items + entry.n_texts())
    /// // Gate: padded_total < max_batch_tokens.  If ≥, defer the entry
    /// // into the next batch instead of joining this one.
    /// ```
    /// The first item of a fresh batch is ALWAYS admitted, even if on its
    /// own it would exceed the budget (otherwise a single long request
    /// could never make progress).
    pub fn with_tokens<F>(
        name: &str,
        embed_fn: F,
        max_batch_tokens: usize,
        max_batch_items: usize,
        padded_model: bool,
        wait_ms: u64,
        max_queue: usize,
    ) -> Self
    where
        F: Fn(Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, String> + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel::<Item>(max_queue);
        let arc_name = Arc::new(name.to_string());
        let cfg = BatcherConfig {
            max_batch_tokens,
            max_batch_items,
            padded_model,
            wait: Duration::from_millis(wait_ms),
        };
        let handle = tokio::spawn(run_worker(rx, Arc::new(embed_fn), arc_name.clone(), cfg));
        DynamicBatcher {
            name: arc_name,
            sender: tx,
            worker: handle,
        }
    }

    /// Submit a request carrying pre-tokenized input_ids (one `Vec<u32>` per text).
    pub async fn embed_tokens(
        &self,
        token_ids: Vec<Vec<u32>>,
    ) -> Result<Vec<Vec<f32>>, BatchError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .sender
            .try_send(Item {
                token_ids,
                reply: reply_tx,
            })
            .is_err()
        {
            crate::metrics::record_queue_rejected(&self.name);
            return Err(BatchError::QueueFull(QueueFullError {
                batcher_name: self.name.as_ref().clone(),
            }));
        }
        match reply_rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(m)) => Err(BatchError::Inference(m)),
            Err(_) => Err(BatchError::Shutdown),
        }
    }

    pub async fn shutdown(self, timeout: Duration) {
        let DynamicBatcher { sender, worker, .. } = self;
        drop(sender);
        let _ = tokio::time::timeout(timeout, worker).await;
    }

    /// Test-only: enqueue an item with a caller-supplied reply sender.
    ///
    /// Lets tests construct requests whose reply channel is already closed
    /// (by dropping the matching receiver before dispatch), so the
    /// cancellation path can be exercised without relying on `JoinHandle::abort`
    /// racing with the worker's `rx.recv`.
    #[cfg(test)]
    fn enqueue_for_test(
        &self,
        token_ids: Vec<Vec<u32>>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
    ) -> Result<(), BatchError> {
        self.sender
            .try_send(Item { token_ids, reply })
            .map_err(|_| {
                BatchError::QueueFull(QueueFullError {
                    batcher_name: self.name.as_ref().clone(),
                })
            })
    }
}

type EmbedFn = Arc<dyn Fn(Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, String> + Send + Sync + 'static>;

/// Running aggregate over Items already admitted into the in-progress batch.
/// Mirrors TEI's `current_batch` state but extended for our multi-text Items.
#[derive(Debug, Default)]
struct BatchAccum {
    /// Sum of `token_ids.len()` across all Items in the batch (count of texts).
    items: usize,
    /// Longest token sequence across all texts in the batch.
    max_len: usize,
    /// Sum of all token lengths (non-padded) across the batch.
    total_tokens: usize,
}

impl BatchAccum {
    fn push(&mut self, item: &Item) {
        self.items += item.n_texts();
        self.max_len = max(self.max_len, item.max_seq_len());
        self.total_tokens += item.total_tokens();
    }

    /// TEI-style fit check for a candidate Item.  Returns true iff adding
    /// this item keeps the padded/ragged batch under budget.
    ///
    /// Gate is strict `<`: the padded product `new_max * new_items` must
    /// be strictly less than `max_batch_tokens`. A value exactly equal to
    /// the budget is treated as overflow and deferred.
    fn fits(&self, item: &Item, cfg: &BatcherConfig) -> bool {
        let new_items = self.items + item.n_texts();
        // Honour item-count fairness cap.
        if new_items > cfg.max_batch_items {
            return false;
        }
        let new_max = max(self.max_len, item.max_seq_len());
        let total_if_added = if cfg.padded_model {
            new_max.saturating_mul(new_items)
        } else {
            self.total_tokens + item.total_tokens()
        };
        total_if_added < cfg.max_batch_tokens
    }
}

async fn run_worker(
    mut rx: mpsc::Receiver<Item>,
    embed_fn: EmbedFn,
    name: Arc<String>,
    cfg: BatcherConfig,
) {
    let mut carry: Option<Item> = None;
    loop {
        let first = match carry.take() {
            Some(c) => c,
            None => match rx.recv().await {
                Some(i) => i,
                None => break,
            },
        };
        let mut accum = BatchAccum::default();
        accum.push(&first);
        let mut batch = vec![first];
        let deadline = Instant::now() + cfg.wait;
        loop {
            // Short-circuit: if any further item is impossible under the
            // current accum (we're already saturated on items or padded
            // tokens), stop waiting for more.
            if accum.items >= cfg.max_batch_items {
                break;
            }
            if cfg.padded_model && accum.max_len.saturating_mul(accum.items) >= cfg.max_batch_tokens
            {
                break;
            }
            if !cfg.padded_model && accum.total_tokens >= cfg.max_batch_tokens {
                break;
            }
            let rem = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or(Duration::ZERO);
            match tokio::time::timeout(rem, rx.recv()).await {
                Ok(Some(item)) => {
                    if item.reply.is_closed() {
                        // Client disconnected before the batch window
                        // closed — skip without charging its tokens.
                        crate::metrics::record_cancelled(&name);
                        continue;
                    }
                    if accum.fits(&item, &cfg) {
                        accum.push(&item);
                        batch.push(item);
                    } else {
                        // Overflow: defer to the next batch (don't drop).
                        crate::metrics::record_carry(&name);
                        carry = Some(item);
                        break;
                    }
                }
                _ => break,
            }
        }
        // Second check: sender may have closed during the coalesce window.
        // Best-effort; accum/batch lengths don't matter post-dispatch.
        let cancelled_at_dispatch = batch.iter().filter(|it| it.reply.is_closed()).count();
        for _ in 0..cancelled_at_dispatch {
            crate::metrics::record_cancelled(&name);
        }
        batch.retain(|it| !it.reply.is_closed());
        if batch.is_empty() {
            continue;
        }
        // Token-budget observability (per dispatched batch). `accum`
        // reflects the intended admit set — including any items later
        // filtered by the cancellation retain above — which gives a
        // faithful view of budget pressure. For non-padded models the
        // padded value degenerates to raw, so the ratio is always 0.
        let raw_tokens = accum.total_tokens;
        let padded_tokens = if cfg.padded_model {
            accum.max_len.saturating_mul(accum.items)
        } else {
            raw_tokens
        };
        crate::metrics::record_batch_tokens(&name, padded_tokens);
        crate::metrics::record_padding_waste(&name, padded_tokens, raw_tokens);
        dispatch_batch(batch, embed_fn.clone(), name.clone()).await;
    }
}

async fn dispatch_batch(items: Vec<Item>, embed_fn: EmbedFn, name: Arc<String>) {
    let counts: Vec<usize> = items.iter().map(|i| i.token_ids.len()).collect();
    let mut ids: Vec<Vec<u32>> = Vec::new();
    let mut replies = Vec::with_capacity(items.len());
    for Item {
        token_ids: t,
        reply,
    } in items
    {
        ids.extend(t);
        replies.push(reply);
    }
    let total = ids.len();
    let start = Instant::now();

    let fan_err = |replies: Vec<oneshot::Sender<_>>, msg: String| {
        for r in replies {
            let _ = r.send(Err(msg.clone()));
        }
    };

    match tokio::task::spawn_blocking(move || embed_fn(ids)).await {
        Ok(Ok(mut vecs)) => {
            if vecs.len() != total {
                fan_err(
                    replies,
                    format!("embed_fn returned {} vectors for {total} texts", vecs.len()),
                );
                return;
            }
            crate::metrics::record_inference(&name, start.elapsed(), total);
            for (reply, &n) in replies.into_iter().zip(counts.iter()) {
                let _ = reply.send(Ok(vecs.drain(..n).collect()));
            }
        }
        Ok(Err(msg)) => fan_err(replies, msg),
        Err(e) => fan_err(replies, format!("embed task panicked: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    /// Install a Prometheus recorder the first time it's needed and cache
    /// its handle. Subsequent calls return the same handle; the global
    /// `metrics` recorder can only be installed once per process.
    fn test_prometheus_handle() -> &'static PrometheusHandle {
        static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
        HANDLE.get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("install Prometheus recorder for tests")
        })
    }

    /// Build a batcher tuned for item-count-style tests (pre-B3 semantics):
    /// token budget is set large enough that the max_items cap is what
    /// binds, and non-padded accounting is used so mixing short+long doesn't
    /// trigger the padded-product cap. Matches the intent of tests written
    /// against the old `with_name(max_batch=N, ...)` signature.
    fn item_cap_batcher<F>(
        name: &str,
        embed_fn: F,
        max_items: usize,
        wait_ms: u64,
        max_queue: usize,
    ) -> DynamicBatcher
    where
        F: Fn(Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, String> + Send + Sync + 'static,
    {
        // Very large token budget so it never binds; non-padded so the
        // gate reduces to total_tokens < budget, which also won't bind
        // for these fixtures (single-token ids).
        DynamicBatcher::with_tokens(
            name,
            embed_fn,
            /*max_batch_tokens*/ usize::MAX,
            /*max_batch_items*/ max_items,
            /*padded_model*/ false,
            wait_ms,
            max_queue,
        )
    }

    /// Test helper: build a batcher that logs each dispatched batch's
    /// token-id vectors (one `Vec<Vec<u32>>` per batch call) and returns
    /// deterministic fixed-size vectors.
    fn log_batcher(
        name: &str,
        log: Arc<Mutex<Vec<Vec<Vec<u32>>>>>,
        max_items: usize,
        wait_ms: u64,
        max_queue: usize,
    ) -> DynamicBatcher {
        item_cap_batcher(
            name,
            move |ids: Vec<Vec<u32>>| {
                log.lock().unwrap().push(ids.clone());
                Ok(ids.iter().map(|_| vec![1.0f32, 2.0, 3.0, 4.0]).collect())
            },
            max_items,
            wait_ms,
            max_queue,
        )
    }

    /// Distinct single-token ids, useful when tests want to assert that a
    /// specific item round-tripped without collisions.
    fn tok(ids: &[u32]) -> Vec<Vec<u32>> {
        ids.iter().map(|&i| vec![i]).collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batcher_passes_token_ids_verbatim() {
        // RED for B1: the batcher must forward pre-tokenized input_ids
        // (Vec<Vec<u32>>) to the embed closure unchanged.
        let got: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![]));
        let g = got.clone();
        let b = item_cap_batcher(
            "t_tok",
            move |ids: Vec<Vec<u32>>| {
                g.lock().unwrap().push(ids.clone());
                Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            32,
            50,
            16,
        );
        let r = b
            .embed_tokens(vec![vec![1u32, 2, 3], vec![4, 5]])
            .await
            .unwrap();
        assert_eq!(r.len(), 2);
        let captured = got.lock().unwrap().clone();
        assert_eq!(captured.len(), 1, "expected exactly one batch call");
        assert_eq!(captured[0], vec![vec![1u32, 2, 3], vec![4, 5]]);
        b.shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_request_no_coalesce() {
        let log = Arc::new(Mutex::new(vec![]));
        let b = log_batcher("t1", log.clone(), 32, 50, 16);
        let r = b.embed_tokens(tok(&[1, 2])).await.unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].len(), 4);
        let calls = log.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], tok(&[1, 2]));
        drop(calls);
        b.shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_coalesce() {
        let log = Arc::new(Mutex::new(vec![]));
        let b = Arc::new(log_batcher("t2", log.clone(), 32, 100, 16));
        {
            let (b1, b2, b3) = (b.clone(), b.clone(), b.clone());
            let (r1, r2, r3) = tokio::join!(
                b1.embed_tokens(tok(&[1, 2])),
                b2.embed_tokens(tok(&[3, 4, 5])),
                b3.embed_tokens(tok(&[6])),
            );
            assert!(r1.is_ok());
            assert!(r2.is_ok());
            assert!(r3.is_ok());
        }
        let calls = log.lock().unwrap();
        assert!(
            calls.len() <= 2,
            "expected <=2 batches, got {}",
            calls.len()
        );
        // Gather the single-token values across all dispatched batches and
        // verify every request's token showed up exactly once.
        let all: Vec<u32> = calls
            .iter()
            .flat_map(|batch| batch.iter().map(|v| v[0]))
            .collect();
        for t in [1u32, 2, 3, 4, 5, 6] {
            assert!(all.contains(&t), "missing token: {t}");
        }
        drop(calls);
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batch_size_cap_splits() {
        let log = Arc::new(Mutex::new(vec![]));
        let b = Arc::new(log_batcher("t3", log.clone(), 4, 100, 16));
        {
            let (b1, b2, b3) = (b.clone(), b.clone(), b.clone());
            let _results = tokio::join!(
                b1.embed_tokens(tok(&[1, 2])),
                b2.embed_tokens(tok(&[3, 4])),
                b3.embed_tokens(tok(&[5, 6])),
            );
        }
        let calls = log.lock().unwrap();
        assert!(
            calls.len() >= 2,
            "expected >=2 batches, got {}",
            calls.len()
        );
        assert!(calls.iter().map(|v| v.len()).sum::<usize>() >= 4);
        drop(calls);
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_full_returns_error() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let blocked = Arc::new(AtomicBool::new(true));
        let blocked_cl = blocked.clone();
        let b = Arc::new(item_cap_batcher(
            "t4",
            move |ids: Vec<Vec<u32>>| {
                while blocked_cl.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            32,
            1,
            1,
        ));
        {
            let b1 = b.clone();
            let first = tokio::spawn(async move { b1.embed_tokens(tok(&[1])).await });
            tokio::time::sleep(Duration::from_millis(30)).await;
            let b2 = b.clone();
            let filler = tokio::spawn(async move { b2.embed_tokens(tok(&[2])).await });
            tokio::time::sleep(Duration::from_millis(20)).await;
            let overflow = b.embed_tokens(tok(&[3])).await;
            assert!(
                matches!(overflow, Err(BatchError::QueueFull(_))),
                "got: {overflow:?}"
            );
            blocked.store(false, Ordering::SeqCst);
            let _ = first.await;
            let _ = filler.await;
        }
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(500))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn coalesce_overflow_defers_to_next_batch() {
        // Regression: previously the 2nd request got "item exceeded max_batch after coalesce"
        // and was dropped. Now it should be deferred into the next batch and still succeed.
        let log = Arc::new(Mutex::new(vec![]));
        let b = Arc::new(log_batcher("t_defer", log.clone(), 4, 100, 16));
        {
            let (b1, b2) = (b.clone(), b.clone());
            let (r1, r2) = tokio::join!(b1.embed_tokens(tok(&[1, 2, 3])), async {
                tokio::time::sleep(Duration::from_millis(15)).await;
                b2.embed_tokens(tok(&[4, 5])).await
            },);
            let v1 = r1.expect("first request must succeed");
            let v2 = r2.expect("second request must succeed (previously dropped)");
            assert_eq!(v1.len(), 3);
            assert_eq!(v2.len(), 2);
        }
        let calls = log.lock().unwrap();
        assert_eq!(
            calls.len(),
            2,
            "expected exactly 2 batches, got {}",
            calls.len()
        );
        assert_eq!(calls[0], tok(&[1, 2, 3]));
        assert_eq!(calls[1], tok(&[4, 5]));
        drop(calls);
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inference_error_fails_all_coalesced() {
        let b = Arc::new(item_cap_batcher(
            "t5",
            |_: Vec<Vec<u32>>| Err("model died".to_string()),
            32,
            50,
            16,
        ));
        {
            let (b1, b2) = (b.clone(), b.clone());
            let (r1, r2) = tokio::join!(b1.embed_tokens(tok(&[1])), b2.embed_tokens(tok(&[2])));
            for r in [r1, r2] {
                let Err(BatchError::Inference(msg)) = r else {
                    panic!("expected Inference, got: {r:?}")
                };
                assert!(msg.contains("model died"), "unexpected: {msg}");
            }
        }
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_items_are_skipped() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        let b = Arc::new(item_cap_batcher(
            "t_cancel",
            move |ids: Vec<Vec<u32>>| {
                cc.fetch_add(ids.len(), Ordering::SeqCst);
                Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            32,
            50,
            16,
        ));

        // Deterministic cancellation: build a oneshot pair, drop the receiver
        // immediately so `reply.is_closed()` is true before the worker ever
        // sees the item. No timing assumptions — the worker MUST see the
        // closed reply by the time it dispatches.
        let (cancelled_tx, cancelled_rx) = oneshot::channel();
        drop(cancelled_rx);
        assert!(
            cancelled_tx.is_closed(),
            "precondition: dropping rx should close tx"
        );
        b.enqueue_for_test(tok(&[99]), cancelled_tx)
            .expect("enqueue cancelled item");

        // Live request. Awaiting its result guarantees the worker has
        // processed the queue past the cancelled item; whether the two
        // coalesce into one batch or dispatch separately, the cancelled
        // item is always dropped by `reply.is_closed()` checks.
        let r = b.embed_tokens(tok(&[1])).await.unwrap();
        assert_eq!(r.len(), 1);

        // Inference should have run ONLY for the live token (1 total).
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "expected only the live item to be embedded, got {} total",
            call_count.load(Ordering::SeqCst)
        );
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_items_increment_metric() {
        let handle = test_prometheus_handle();
        // Unique model name so the counter can be grepped in Prometheus
        // exposition text without colliding with other tests.
        let model = "t_cancel_metric";
        let b = Arc::new(item_cap_batcher(
            model,
            |ids: Vec<Vec<u32>>| Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect()),
            32,
            50,
            16,
        ));

        // Snapshot the counter before we do anything.
        let before = read_cancelled_counter(&handle.render(), model);

        // Two cancelled items + one live item.
        for _ in 0..2 {
            let (tx, rx) = oneshot::channel();
            drop(rx);
            b.enqueue_for_test(tok(&[99]), tx).expect("enqueue");
        }
        let r = b.embed_tokens(tok(&[1])).await.unwrap();
        assert_eq!(r.len(), 1);

        let after = read_cancelled_counter(&handle.render(), model);
        assert_eq!(
            after - before,
            2,
            "expected counter to increment by 2, went {before} -> {after}"
        );

        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    /// Parse a specific `embed_batcher_cancelled_items_total{model="..."} N`
    /// line out of a Prometheus text-exposition render. Returns 0 if the
    /// metric isn't present yet (counter hasn't been touched for this model).
    fn read_cancelled_counter(rendered: &str, model: &str) -> u64 {
        let needle = format!("model=\"{model}\"");
        rendered
            .lines()
            .filter(|l| l.starts_with("embed_batcher_cancelled_items_total"))
            .find(|l| l.contains(&needle))
            .and_then(|l| l.rsplit_once(' '))
            .and_then(|(_, v)| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------
    // B3: padded-model token-budget accounting (TEI core/src/queue.rs
    // formula port). These tests exercise the actual budget arithmetic.
    // -----------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn padded_accounting_prevents_mixing_long_with_many_short() {
        // max_batch_tokens = 1000, padded_model = true.
        // First: 500-tok Item. Then 10× 50-tok Items arrive.
        // Naive sum: 500 + 10*50 = 1000 → all would fit.
        // Padded correct: max(500, 50) * (1+1) = 1000 ≥ 1000 → first
        // short defers; and each subsequent short, standalone in the
        // next batch, seeds its own accum with 50*1 = 50 < 1000 and can
        // coalesce more 50s but not the original 500.
        let log: Arc<Mutex<Vec<Vec<usize>>>> = Arc::new(Mutex::new(vec![]));
        let l = log.clone();
        let b = Arc::new(DynamicBatcher::with_tokens(
            "t_pad",
            move |ids: Vec<Vec<u32>>| {
                l.lock()
                    .unwrap()
                    .push(ids.iter().map(|i| i.len()).collect());
                Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            /*max_batch_tokens*/ 1000,
            /*max_batch_items*/ 100,
            /*padded_model*/ true,
            /*wait_ms*/ 50,
            /*max_queue*/ 16,
        ));
        let b_first = b.clone();
        let first = tokio::spawn(async move { b_first.embed_tokens(vec![vec![0u32; 500]]).await });
        tokio::time::sleep(Duration::from_millis(5)).await;
        let mut rest = vec![];
        for _ in 0..10 {
            let bc = b.clone();
            rest.push(tokio::spawn(async move {
                bc.embed_tokens(vec![vec![0u32; 50]]).await
            }));
        }
        let _ = first.await;
        for h in rest {
            let _ = h.await;
        }
        let batches = log.lock().unwrap();
        assert!(
            batches[0] == vec![500],
            "first batch must hold only the 500-token item, got {:?}",
            batches[0]
        );
        assert!(
            batches.len() >= 2,
            "10 short items must dispatch in separate batch(es), got {} batch(es)",
            batches.len()
        );
        drop(batches);
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_padded_accounting_sums_tokens() {
        // Same fixture, padded_model = false. The gate becomes
        // `total_tokens + entry_tokens < 1000`, so 500 + 10*50 = 1000
        // is NOT strictly less than 1000 → the 10th short still defers,
        // but the first 9 shorts (500 + 9*50 = 950 < 1000) join the
        // 500-tok item.  Assert that the first batch holds more than just
        // the 500 (i.e. non-padded accounting is looser than padded).
        let log: Arc<Mutex<Vec<Vec<usize>>>> = Arc::new(Mutex::new(vec![]));
        let l = log.clone();
        let b = Arc::new(DynamicBatcher::with_tokens(
            "t_nopad",
            move |ids: Vec<Vec<u32>>| {
                l.lock()
                    .unwrap()
                    .push(ids.iter().map(|i| i.len()).collect());
                Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            /*max_batch_tokens*/ 1000,
            /*max_batch_items*/ 100,
            /*padded_model*/ false,
            /*wait_ms*/ 80,
            /*max_queue*/ 32,
        ));
        let b_first = b.clone();
        let first = tokio::spawn(async move { b_first.embed_tokens(vec![vec![0u32; 500]]).await });
        tokio::time::sleep(Duration::from_millis(5)).await;
        let mut rest = vec![];
        for _ in 0..10 {
            let bc = b.clone();
            rest.push(tokio::spawn(async move {
                bc.embed_tokens(vec![vec![0u32; 50]]).await
            }));
        }
        let _ = first.await;
        for h in rest {
            let _ = h.await;
        }
        let batches = log.lock().unwrap();
        let total_items: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total_items, 11, "all 11 items must dispatch");
        assert!(
            batches[0].len() > 1,
            "non-padded accounting should let short items join the 500-tok batch, got batches[0] = {:?}",
            batches[0]
        );
        // First batch should hold the 500-tok item plus some shorts.
        assert!(batches[0].contains(&500));
        drop(batches);
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oversized_single_item_still_dispatches_alone() {
        // A 1500-tok Item submitted to a 1000-tok budget must still run —
        // seeding a fresh batch never consults the budget, otherwise the
        // request could never make progress.
        let log: Arc<Mutex<Vec<Vec<usize>>>> = Arc::new(Mutex::new(vec![]));
        let l = log.clone();
        let b = DynamicBatcher::with_tokens(
            "t_solo",
            move |ids: Vec<Vec<u32>>| {
                l.lock()
                    .unwrap()
                    .push(ids.iter().map(|i| i.len()).collect());
                Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            /*max_batch_tokens*/ 1000,
            /*max_batch_items*/ 100,
            /*padded_model*/ true,
            /*wait_ms*/ 20,
            /*max_queue*/ 4,
        );
        let r = b.embed_tokens(vec![vec![0u32; 1500]]).await.unwrap();
        assert_eq!(r.len(), 1);
        let batches = log.lock().unwrap();
        assert_eq!(*batches, vec![vec![1500usize]]);
        drop(batches);
        b.shutdown(Duration::from_millis(200)).await;
    }

    // Direct unit tests for the pure BatchAccum logic — fast, no tokio.
    #[test]
    fn batch_accum_padded_gate_is_strict_less_than() {
        let cfg = BatcherConfig {
            max_batch_tokens: 1000,
            max_batch_items: 100,
            padded_model: true,
            wait: Duration::from_millis(1),
        };
        // Seed batch with a single 500-token Item.
        let seed = Item {
            token_ids: vec![vec![0u32; 500]],
            reply: oneshot::channel().0,
        };
        let mut accum = BatchAccum::default();
        accum.push(&seed);

        // Candidate: one 50-token Item.  new_max=500, new_items=2 →
        // padded product 1000, gate <1000 → must NOT fit.
        let cand50 = Item {
            token_ids: vec![vec![0u32; 50]],
            reply: oneshot::channel().0,
        };
        assert!(
            !accum.fits(&cand50, &cfg),
            "padded gate must be strict `<`: product==budget should not fit"
        );
    }

    #[test]
    fn batch_accum_non_padded_sums_tokens() {
        let cfg = BatcherConfig {
            max_batch_tokens: 1000,
            max_batch_items: 100,
            padded_model: false,
            wait: Duration::from_millis(1),
        };
        let seed = Item {
            token_ids: vec![vec![0u32; 500]],
            reply: oneshot::channel().0,
        };
        let mut accum = BatchAccum::default();
        accum.push(&seed);

        // 499 tokens: 500+499 = 999 < 1000 → fits.
        let cand_fit = Item {
            token_ids: vec![vec![0u32; 499]],
            reply: oneshot::channel().0,
        };
        assert!(accum.fits(&cand_fit, &cfg));

        // 500 tokens: 500+500 = 1000 NOT < 1000 → does not fit.
        let cand_miss = Item {
            token_ids: vec![vec![0u32; 500]],
            reply: oneshot::channel().0,
        };
        assert!(!accum.fits(&cand_miss, &cfg));
    }

    #[test]
    fn batch_accum_max_items_cap_binds() {
        let cfg = BatcherConfig {
            max_batch_tokens: usize::MAX,
            max_batch_items: 2,
            padded_model: false,
            wait: Duration::from_millis(1),
        };
        let two = Item {
            token_ids: vec![vec![1u32], vec![2u32]],
            reply: oneshot::channel().0,
        };
        let mut accum = BatchAccum::default();
        accum.push(&two);
        // Already at 2 items. Adding any text would exceed cap.
        let one = Item {
            token_ids: vec![vec![3u32]],
            reply: oneshot::channel().0,
        };
        assert!(!accum.fits(&one, &cfg));
    }

    // -----------------------------------------------------------------
    // B4: token-budget observability metrics. One scenario drives all
    // three (batch_tokens histogram, padding_waste_ratio histogram,
    // carry_events_total counter) via the Prometheus text exposition.
    // -----------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn token_budget_metrics_are_recorded() {
        let handle = test_prometheus_handle();
        // Unique model name to avoid cross-test interference.
        let name = "t_budget_metrics";
        // max_batch_tokens=1000 padded. First 500 tokens, then 10× 50-tok
        // forces at least one carry and one short-second-batch dispatch.
        let b = Arc::new(DynamicBatcher::with_tokens(
            name,
            |ids: Vec<Vec<u32>>| Ok(ids.iter().map(|_| vec![0.0f32; 4]).collect()),
            /*max_batch_tokens*/ 1000,
            /*max_batch_items*/ 100,
            /*padded_model*/ true,
            /*wait_ms*/ 50,
            /*max_queue*/ 16,
        ));
        let b_first = b.clone();
        let first = tokio::spawn(async move { b_first.embed_tokens(vec![vec![0u32; 500]]).await });
        tokio::time::sleep(Duration::from_millis(5)).await;
        let mut rest = vec![];
        for _ in 0..10 {
            let bc = b.clone();
            rest.push(tokio::spawn(async move {
                bc.embed_tokens(vec![vec![0u32; 50]]).await
            }));
        }
        let _ = first.await;
        for h in rest {
            let _ = h.await;
        }
        // Allow final dispatch to flush.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let text = handle.render();
        // Three metrics must appear with this model name.
        assert!(
            text.contains(&format!("embed_batch_tokens_count{{model=\"{name}\"}}")),
            "missing batch_tokens: {text}"
        );
        assert!(
            text.contains(&format!(
                "embed_batch_padding_waste_ratio_count{{model=\"{name}\"}}"
            )),
            "missing padding_waste: {text}"
        );
        assert!(
            text.contains(&format!("embed_carry_events_total{{model=\"{name}\"}}")),
            "missing carry: {text}"
        );
        Arc::try_unwrap(b)
            .ok()
            .expect("still has clones")
            .shutdown(Duration::from_millis(200))
            .await;
    }
}
