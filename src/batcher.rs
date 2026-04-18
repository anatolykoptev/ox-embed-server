//! Dynamic batching: coalesces concurrent embed calls, dispatches one embed_fn per batch.
#![allow(dead_code)]
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
pub struct QueueFullError { pub batcher_name: String }
impl fmt::Display for QueueFullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "batcher '{}' queue is full", self.batcher_name)
    }
}
impl std::error::Error for QueueFullError {}

#[derive(Debug)]
struct Item {
    texts: Vec<String>,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
}

#[derive(Debug)]
pub struct DynamicBatcher {
    name: Arc<String>,
    sender: mpsc::Sender<Item>,
    worker: JoinHandle<()>,
}

impl DynamicBatcher {
    /// Create a batcher and start its worker.  `embed_fn` runs in `spawn_blocking`.
    pub fn with_name<F>(name: &str, embed_fn: F, max_batch: usize, wait_ms: u64, max_queue: usize) -> Self
    where
        F: Fn(Vec<String>) -> Result<Vec<Vec<f32>>, String> + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel::<Item>(max_queue);
        let arc_name = Arc::new(name.to_string());
        let handle = tokio::spawn(run_worker(
            rx, Arc::new(embed_fn), arc_name.clone(),
            max_batch, Duration::from_millis(wait_ms),
        ));
        DynamicBatcher { name: arc_name, sender: tx, worker: handle }
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BatchError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.sender.try_send(Item { texts, reply: reply_tx }).is_err() {
            crate::metrics::record_queue_rejected(&self.name);
            return Err(BatchError::QueueFull(QueueFullError { batcher_name: self.name.as_ref().clone() }));
        }
        match reply_rx.await {
            Ok(Ok(v))  => Ok(v),
            Ok(Err(m)) => Err(BatchError::Inference(m)),
            Err(_)     => Err(BatchError::Shutdown),
        }
    }

    pub async fn shutdown(self, timeout: Duration) {
        let DynamicBatcher { sender, worker, .. } = self;
        drop(sender);
        let _ = tokio::time::timeout(timeout, worker).await;
    }
}

type EmbedFn = Arc<dyn Fn(Vec<String>) -> Result<Vec<Vec<f32>>, String> + Send + Sync + 'static>;

async fn run_worker(mut rx: mpsc::Receiver<Item>, embed_fn: EmbedFn, name: Arc<String>, max_batch: usize, wait: Duration) {
    let mut carry: Option<Item> = None;
    loop {
        let first = match carry.take() {
            Some(c) => c,
            None => match rx.recv().await { Some(i) => i, None => break },
        };
        let mut batch = vec![first];
        let mut cum = batch[0].texts.len();
        let deadline = Instant::now() + wait;
        loop {
            if cum >= max_batch { break }
            let rem = deadline.checked_duration_since(Instant::now()).unwrap_or(Duration::ZERO);
            match tokio::time::timeout(rem, rx.recv()).await {
                Ok(Some(item)) if cum + item.texts.len() <= max_batch => {
                    // Client disconnected before the batch window closed — skip this item
                    // without charging its tokens to the batch budget. Best-effort: the
                    // sender may still close between this check and dispatch, and that's fine.
                    if item.reply.is_closed() { continue; }
                    cum += item.texts.len();
                    batch.push(item);
                }
                Ok(Some(item)) => {
                    // Overflow: defer this item to the next batch instead of dropping it.
                    carry = Some(item);
                    break;
                }
                _ => break,
            }
        }
        // Second check: sender may have closed during the coalesce window.
        // Best-effort; see above.
        batch.retain(|it| !it.reply.is_closed());
        if batch.is_empty() { continue; }
        dispatch_batch(batch, embed_fn.clone(), name.clone()).await;
    }
}

async fn dispatch_batch(items: Vec<Item>, embed_fn: EmbedFn, name: Arc<String>) {
    let counts: Vec<usize> = items.iter().map(|i| i.texts.len()).collect();
    let mut texts: Vec<String> = Vec::new();
    let mut replies = Vec::with_capacity(items.len());
    for Item { texts: t, reply } in items { texts.extend(t); replies.push(reply); }
    let total = texts.len();
    let start = Instant::now();

    let fan_err = |replies: Vec<oneshot::Sender<_>>, msg: String| {
        for r in replies { let _ = r.send(Err(msg.clone())); }
    };

    match tokio::task::spawn_blocking(move || embed_fn(texts)).await {
        Ok(Ok(mut vecs)) => {
            if vecs.len() != total {
                fan_err(replies, format!("embed_fn returned {} vectors for {total} texts", vecs.len()));
                return;
            }
            crate::metrics::record_inference(&name, start.elapsed(), total);
            for (reply, &n) in replies.into_iter().zip(counts.iter()) {
                let _ = reply.send(Ok(vecs.drain(..n).collect()));
            }
        }
        Ok(Err(msg)) => fan_err(replies, msg),
        Err(e)       => fan_err(replies, format!("embed task panicked: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn log_batcher(name: &str, log: Arc<Mutex<Vec<Vec<String>>>>, max_batch: usize, wait_ms: u64, max_queue: usize) -> DynamicBatcher {
        DynamicBatcher::with_name(name, move |t| {
            log.lock().unwrap().push(t.clone());
            Ok(t.iter().map(|_| vec![1.0f32, 2.0, 3.0, 4.0]).collect())
        }, max_batch, wait_ms, max_queue)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_request_no_coalesce() {
        let log = Arc::new(Mutex::new(vec![]));
        let b = log_batcher("t1", log.clone(), 32, 50, 16);
        let r = b.embed(vec!["a".into(), "b".into()]).await.unwrap();
        assert_eq!(r.len(), 2); assert_eq!(r[0].len(), 4);
        let calls = log.lock().unwrap();
        assert_eq!(calls.len(), 1); assert_eq!(calls[0], vec!["a", "b"]);
        drop(calls); b.shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_coalesce() {
        let log = Arc::new(Mutex::new(vec![]));
        let b = Arc::new(log_batcher("t2", log.clone(), 32, 100, 16));
        {
            let (b1, b2, b3) = (b.clone(), b.clone(), b.clone());
            let (r1, r2, r3) = tokio::join!(
                b1.embed(vec!["a".into(), "b".into()]),
                b2.embed(vec!["c".into(), "d".into(), "e".into()]),
                b3.embed(vec!["f".into()]),
            );
            assert!(r1.is_ok()); assert!(r2.is_ok()); assert!(r3.is_ok());
        }
        let calls = log.lock().unwrap();
        assert!(calls.len() <= 2, "expected <=2 batches, got {}", calls.len());
        let all: Vec<String> = calls.iter().flat_map(|v| v.iter().cloned()).collect();
        for t in ["a","b","c","d","e","f"] { assert!(all.contains(&t.to_string()), "missing: {t}"); }
        drop(calls);
        Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batch_size_cap_splits() {
        let log = Arc::new(Mutex::new(vec![]));
        let b = Arc::new(log_batcher("t3", log.clone(), 4, 100, 16));
        {
            let (b1, b2, b3) = (b.clone(), b.clone(), b.clone());
            let _results = tokio::join!(
                b1.embed(vec!["a".into(), "b".into()]),
                b2.embed(vec!["c".into(), "d".into()]),
                b3.embed(vec!["e".into(), "f".into()]),
            );
        }
        let calls = log.lock().unwrap();
        assert!(calls.len() >= 2, "expected >=2 batches, got {}", calls.len());
        assert!(calls.iter().map(|v| v.len()).sum::<usize>() >= 4);
        drop(calls);
        Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_full_returns_error() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let blocked = Arc::new(AtomicBool::new(true));
        let blocked_cl = blocked.clone();
        let b = Arc::new(DynamicBatcher::with_name("t4", move |texts| {
            while blocked_cl.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(texts.iter().map(|_| vec![0.0f32; 4]).collect())
        }, 32, 1, 1));
        {
            let b1 = b.clone();
            let first = tokio::spawn(async move { b1.embed(vec!["first".into()]).await });
            tokio::time::sleep(Duration::from_millis(30)).await;
            let b2 = b.clone();
            let filler = tokio::spawn(async move { b2.embed(vec!["fill".into()]).await });
            tokio::time::sleep(Duration::from_millis(20)).await;
            let overflow = b.embed(vec!["overflow".into()]).await;
            assert!(matches!(overflow, Err(BatchError::QueueFull(_))), "got: {overflow:?}");
            blocked.store(false, Ordering::SeqCst);
            let _ = first.await;
            let _ = filler.await;
        }
        Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(500)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn coalesce_overflow_defers_to_next_batch() {
        // Regression: previously the 2nd request got "item exceeded max_batch after coalesce"
        // and was dropped. Now it should be deferred into the next batch and still succeed.
        let log = Arc::new(Mutex::new(vec![]));
        let b = Arc::new(log_batcher("t_defer", log.clone(), 4, 100, 16));
        {
            let (b1, b2) = (b.clone(), b.clone());
            let (r1, r2) = tokio::join!(
                b1.embed(vec!["a".into(), "b".into(), "c".into()]),
                async {
                    tokio::time::sleep(Duration::from_millis(15)).await;
                    b2.embed(vec!["d".into(), "e".into()]).await
                },
            );
            let v1 = r1.expect("first request must succeed");
            let v2 = r2.expect("second request must succeed (previously dropped)");
            assert_eq!(v1.len(), 3);
            assert_eq!(v2.len(), 2);
        }
        let calls = log.lock().unwrap();
        assert_eq!(calls.len(), 2, "expected exactly 2 batches, got {}", calls.len());
        assert_eq!(calls[0], vec!["a", "b", "c"]);
        assert_eq!(calls[1], vec!["d", "e"]);
        drop(calls);
        Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inference_error_fails_all_coalesced() {
        let b = Arc::new(DynamicBatcher::with_name("t5", |_| Err("model died".to_string()), 32, 50, 16));
        {
            let (b1, b2) = (b.clone(), b.clone());
            let (r1, r2) = tokio::join!(b1.embed(vec!["a".into()]), b2.embed(vec!["b".into()]));
            for r in [r1, r2] {
                let Err(BatchError::Inference(msg)) = r else { panic!("expected Inference, got: {r:?}") };
                assert!(msg.contains("model died"), "unexpected: {msg}");
            }
        }
        Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_items_are_skipped() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        let b = Arc::new(DynamicBatcher::with_name(
            "t_cancel",
            move |t: Vec<String>| {
                cc.fetch_add(t.len(), Ordering::SeqCst);
                Ok(t.iter().map(|_| vec![0.0f32; 4]).collect())
            },
            32, 50, 16,
        ));
        // Spawn a request, abort its JoinHandle before the batch window closes.
        let b1 = b.clone();
        let aborted = tokio::spawn(async move { b1.embed(vec!["cancelled".into()]).await });
        // Give it a moment to enter the worker's inner loop, then abort.
        tokio::time::sleep(Duration::from_millis(5)).await;
        aborted.abort();
        // Wait for the batch window (50ms) to elapse so the worker dispatches.
        tokio::time::sleep(Duration::from_millis(80)).await;
        // Second, healthy, request.
        let r = b.embed(vec!["alive".into()]).await.unwrap();
        assert_eq!(r.len(), 1);
        // Inference should have run ONLY for the second "alive" text (1 total).
        // Pre-fix: count == 2.
        // Post-fix: count == 1.
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "expected only the live item to be embedded, got {} total",
            call_count.load(Ordering::SeqCst)
        );
        Arc::try_unwrap(b).ok().expect("still has clones").shutdown(Duration::from_millis(200)).await;
    }
}
