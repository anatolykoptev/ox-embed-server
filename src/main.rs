mod api;
mod api_health;
mod api_rerank;
mod api_splade;
mod arena;
mod batcher;
mod cache;
mod cache_flow;
mod config;
mod evictable_pool;
#[allow(dead_code)] // ipc items used by supervisor submodule; unused from main.rs directly
mod ipc;
mod metrics;
mod mlock;
mod model;
mod model_reranker;
mod model_splade;
mod onnx_cache;
mod otel;
mod pool;
mod proc;
mod supervisor;
mod token_cache;
mod types;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;

use crate::batcher::DynamicBatcher;
use crate::cache::EmbeddingCache;
use crate::config::Config;
use crate::model::EmbedModel;
use crate::model_reranker::RerankerModel;
use crate::model_splade::SpladeModel;
use crate::token_cache::TokenCache;
use crate::types::{AppState, ModelEntry, RerankerEntry, SpladeEntry};

/// Waits for SIGTERM or SIGINT, then cancels the token and sleeps for drain_timeout
/// to allow in-flight HTTP requests to complete before axum closes the listener.
async fn shutdown_signal(token: CancellationToken, drain_timeout: Duration) {
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = term.recv() => tracing::info!("SIGTERM received, starting graceful shutdown"),
        _ = int.recv()  => tracing::info!("SIGINT received, starting graceful shutdown"),
    }
    token.cancel();
    // Give in-flight HTTP requests drain_timeout to complete naturally.
    // After this future returns, axum stops accepting new connections; when
    // the last handler finishes, Arc<AppState> drops → batcher workers exit.
    tracing::info!(
        secs = drain_timeout.as_secs(),
        "draining in-flight requests"
    );
    tokio::time::sleep(drain_timeout).await;
    tracing::info!("drain complete");
}

/// Cleanly join every model's batcher worker after axum's HTTP drain has
/// returned. By this point no handler holds an `Arc<AppState>` clone, so
/// `Arc::try_unwrap` on the local `state` and on each `Arc<DynamicBatcher>`
/// should succeed, letting us invoke `DynamicBatcher::shutdown(self, …)` —
/// which drops the channel and awaits the `JoinHandle`, guaranteeing the
/// worker finishes its current batch instead of being cut mid-forward-pass.
///
/// Defensive: if the strong count is still > 1 (somebody leaked a clone),
/// we log a warn and skip rather than block or panic. Budget is `timeout`
/// total across all batchers (they drain concurrently via `JoinSet`).
///
/// Production-only code path — the test-suite exercises `DynamicBatcher::shutdown`
/// directly; this function wires it into the SIGTERM flow (follow-up task #20).
async fn drain_batchers(state: Arc<AppState>, timeout: Duration) {
    let mut app_state = match Arc::try_unwrap(state) {
        Ok(s) => s,
        Err(arc) => {
            tracing::warn!(
                strong = Arc::strong_count(&arc),
                "AppState still shared after HTTP drain, skipping batcher shutdown"
            );
            return;
        }
    };

    // Collect owned DynamicBatcher instances (consuming each Arc). Both
    // embed-model batchers and reranker batchers drain through the same
    // `DynamicBatcher::shutdown(timeout)` path — the batcher doesn't
    // care which kind of inference the adapter closure runs inside
    // `spawn_blocking`.
    let mut owned: Vec<DynamicBatcher> =
        Vec::with_capacity(app_state.models.len() + app_state.rerankers.len());
    for (name, entry) in app_state.models.iter_mut() {
        let Some(arc) = entry.batcher.take() else {
            continue;
        };
        match Arc::try_unwrap(arc) {
            Ok(b) => owned.push(b),
            Err(still_shared) => {
                tracing::warn!(
                    model = %name,
                    strong = Arc::strong_count(&still_shared),
                    "batcher Arc still shared, skipping shutdown for this model"
                );
            }
        }
    }
    for (name, entry) in app_state.rerankers.iter_mut() {
        let Some(arc) = entry.batcher.take() else {
            continue;
        };
        match Arc::try_unwrap(arc) {
            Ok(b) => owned.push(b),
            Err(still_shared) => {
                tracing::warn!(
                    reranker = %name,
                    strong = Arc::strong_count(&still_shared),
                    "reranker batcher Arc still shared, skipping shutdown"
                );
            }
        }
    }

    if owned.is_empty() {
        tracing::info!("no batchers to drain");
        return;
    }

    tracing::info!(
        count = owned.len(),
        secs = timeout.as_secs(),
        "draining batcher workers"
    );

    // Drain all batchers concurrently; each respects its own `timeout`.
    let mut set = tokio::task::JoinSet::new();
    for b in owned {
        set.spawn(async move {
            b.shutdown(timeout).await;
        });
    }
    while let Some(res) = set.join_next().await {
        if let Err(e) = res {
            tracing::warn!(error = %e, "batcher drain task panicked");
        }
    }
    tracing::info!("batcher drain complete");
}

#[tokio::main]
async fn main() {
    // Phase H.18 — OTEL when OTEL_EXPORTER_OTLP_ENDPOINT is set, plain
    // JSON logs otherwise. The handle is held until the end of main so
    // the batch exporter gets `shutdown()` on graceful termination.
    // RUST_LOG / OTEL_LOG_LEVEL still drive filtering inside otel::init.
    let otel_provider = otel::init();

    // Initialize ort runtime explicitly (required for load-dynamic).
    if !ort::init().commit() {
        eprintln!("ort init failed (environment already configured?)");
    }
    tracing::info!("ort runtime initialized");

    // glibc `malloc_trim(0)` background task — port of the TEI pattern at
    // `huggingface/text-embeddings-inference router/src/main.rs:222-229`.
    //
    // glibc's malloc keeps freed pages inside its mmap arenas instead of
    // returning them to the OS. For ML inference workloads with large
    // allocation spikes (per-batch ORT scratch, tokenizer buffers), this
    // causes resident-set drift — process RSS climbs even when working
    // set shrinks. `malloc_trim(0)` forces glibc to release all unused
    // trailing pages back to the kernel. Complements (does not replace)
    // the shared CPU arena registered above.
    //
    // 1000 ms cadence (was 100 ms, ported from TEI without measurement).
    //
    // TEI targets serverless burst-and-idle; embed-server is long-running
    // steady state. At 100 ms the madvise → page-fault cycle produces
    // ~39 GB of block I/O per day (measured on ARM Neoverse-N1 prod).
    // 1000 ms reduces that 10× with negligible latency impact: maximum
    // RSS drift between trims is ≤ 1 batch cycle (~256 MiB temporary),
    // which fits within the 0.42 GiB RSS slack observed in prod.
    //
    // `malloc_trim` itself is a fast no-op when there's nothing to
    // release. Linux-only — `malloc_trim` is a glibc-specific extension,
    // absent on macOS/Windows.
    #[cfg(target_os = "linux")]
    {
        tokio::spawn(async {
            let mut tick = tokio::time::interval(Duration::from_millis(1000));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                // SAFETY: `malloc_trim` is documented thread-safe by glibc
                // and takes no pointers; the only argument is a pad size in
                // bytes (0 = release all unused trailing pages). Returns 1
                // when memory was released, 0 otherwise.
                let released = unsafe { libc::malloc_trim(0) } != 0;
                metrics::record_malloc_trim(released);
            }
        });
        tracing::info!("glibc malloc_trim(0) background task started (1000ms cadence)");
    }

    // Install Prometheus recorder BEFORE any gauge!()/counter!()/histogram!()
    // call. arena::register_shared_cpu_arena() below calls set_arena_gauges()
    // — without recorder installed first, those gauge writes are silently
    // dropped (FU-28).
    let version = std::env::var("EMBED_VERSION").unwrap_or_else(|_| "dev".into());
    let prom_handle = std::sync::Arc::new(metrics::init(&version));

    // Register a shared CPU arena allocator with kSameAsRequested extend
    // strategy — see arena.rs. Critical for memory stability under variable
    // batch sizes; without it, default kNextPowerOfTwo arena per session
    // grows unboundedly to 8GB OOM.
    if let Err(e) = arena::register_shared_cpu_arena() {
        tracing::warn!(error = %e, "shared arena registration failed; sessions will use per-session BFCArena");
    } else {
        tracing::info!("shared CPU arena registered (kSameAsRequested)");
    }

    let cfg = Config::from_env().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });

    // Warn operators about per-model env vars that are silently unsupported.
    // RERANKER_SESSION_POOL_SIZE_* and SPLADE_SESSION_POOL_SIZE_* patterns look
    // analogous to EMBED_SESSION_POOL_SIZE_* but are not yet wired — see PR #74
    // follow-up. Scanning at startup surfaces the mistake before the operator
    // wastes time debugging why the pool size didn't change.
    {
        let ignored: Vec<String> = std::env::vars()
            .filter_map(|(k, _)| {
                if k.starts_with("RERANKER_SESSION_POOL_SIZE_")
                    || k.starts_with("SPLADE_SESSION_POOL_SIZE_")
                {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();
        if !ignored.is_empty() {
            tracing::warn!(
                vars = %ignored.join(", "),
                "per-model overrides not yet supported for reranker/splade — see PR #74 follow-up; these env vars have no effect"
            );
        }
    }

    // Warn if EMBED_MAX_WAITERS_<KEY> is set for a model key that doesn't
    // match any loaded model (embed + rerank + splade). This catches typos
    // like EMBED_MAX_WAITERS_JINA_CODEV2 (missing underscore) early.
    {
        let known_keys: std::collections::HashSet<String> = cfg
            .models
            .iter()
            .map(|m| crate::config::model_env_key(&m.name))
            .chain(
                cfg.rerankers
                    .iter()
                    .map(|r| crate::config::model_env_key(&r.name)),
            )
            .chain(
                cfg.splades
                    .iter()
                    .map(|s| crate::config::model_env_key(&s.name)),
            )
            .collect();
        let unknown_waiters: Vec<String> = std::env::vars()
            .filter_map(|(k, _)| {
                if let Some(suffix) = k.strip_prefix("EMBED_MAX_WAITERS_") {
                    if !known_keys.contains(suffix) {
                        Some(k)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        if !unknown_waiters.is_empty() {
            tracing::warn!(
                vars = %unknown_waiters.join(", "),
                known = ?known_keys,
                "EMBED_MAX_WAITERS_<KEY> set for unknown model key(s); check for typos (key must be uppercase with dashes replaced by underscores)"
            );
        }
    }

    // When EMBED_MULTI_PROCESS=1 the supervisor must NOT load ONNX sessions —
    // doing so duplicates ~2.4 GiB RSS that the workers already hold.
    // ModelEntry.model is therefore Option<Arc<EmbedModel>>: Some in legacy
    // single-process mode, None in multi-process mode (metadata-only entries).
    let mut model_entries: HashMap<String, ModelEntry> = HashMap::new();
    if !cfg.multi_process {
        // First pass: load models, stash per-model knobs alongside the Arc.
        // We can't merge the two loops because raw_models is iterated in
        // arbitrary HashMap order in the second pass, but we need the
        // per-model batch_max_seq and max_len from ModelDef. Collect into a
        // Vec to preserve the association.
        let mut loaded_models: Vec<(
            String,
            Arc<EmbedModel>,
            usize, /* batch_max_seq */
            usize, /* max_len (for effective_seq metric) */
        )> = Vec::new();
        for def in &cfg.models {
            tracing::info!(
                model = %def.name,
                dir = %def.dir,
                batch_max_seq = def.batch_max_seq,
                warmup_seq_len = ?def.warmup_seq_len,
                "loading model"
            );
            let m = EmbedModel::load(
                def,
                cfg.intra_threads,
                cfg.auto_truncate,
                cfg.embed_pool_size,
                cfg.idle_evict_secs,
            )
            .unwrap_or_else(|e| {
                eprintln!("failed to load model '{}': {e}", def.name);
                std::process::exit(1);
            });
            // Pre-warm at every configured embed batch shape (default `[1, 8]`
            // — covers trivial single-text callers AND the downstream consumer's
            // texts_per_req=8 default). Best-effort: per-shape errors log a
            // warn and the next shape proceeds. Override via
            // `EMBED_WARMUP_BATCH_SIZES`.
            //
            // Use per-model warmup_seq_len (defaults to Some(max_len)) to
            // ensure memory_pattern plans against the correct worst-case
            // scratch tensor for this specific model.
            if let Err(e) = m.warmup(&def.name, &cfg.embed_warmup_batch_sizes, def.warmup_seq_len) {
                tracing::error!(model = %def.name, error = %e, "embed warmup failed (non-fatal)");
            }
            loaded_models.push((
                def.name.clone(),
                Arc::new(m),
                def.batch_max_seq,
                def.max_len,
            ));
        }

        tracing::info!(
            models = loaded_models.len(),
            default = %cfg.default_model,
            "all models loaded"
        );

        for (name, model_arc, model_batch_max_seq, model_max_len) in loaded_models {
            let batcher = if cfg.batching_enabled {
                let m = model_arc.clone();
                // ONNX BERT-style encoders always pad to max(seq_len), so
                // padded_model=true is the right accounting for our stack.
                // Kept as a batcher parameter so tests can exercise the
                // non-padded branch directly.
                let b = batcher::DynamicBatcher::with_tokens_and_max_len(
                    &name,
                    move |token_ids| m.embed_tokens(&token_ids),
                    cfg.batch_max_tokens,
                    cfg.batch_max,
                    model_batch_max_seq,
                    /*padded_model*/ true,
                    cfg.batch_wait_ms,
                    cfg.max_queue_size,
                    model_max_len,
                );
                Some(Arc::new(b))
            } else {
                None
            };
            model_entries.insert(
                name,
                ModelEntry {
                    model: Some(model_arc),
                    batcher,
                },
            );
        }
    } else {
        // Multi-process mode: build metadata-only entries (no ONNX sessions).
        // Inference is handled exclusively by the worker pool; the supervisor
        // only needs to know which model names exist for request validation.
        tracing::info!(
            models = cfg.models.len(),
            "multi-process mode: skipping in-process model loading (metadata-only entries)"
        );
        for def in &cfg.models {
            model_entries.insert(
                def.name.clone(),
                ModelEntry {
                    model: None,
                    batcher: None,
                },
            );
        }
    }

    // Load every configured reranker the same way embedding models are
    // loaded, before building AppState. An empty `cfg.rerankers` is a
    // valid no-op — the server boots serving only `/v1/embeddings`.
    // When EMBED_MULTI_PROCESS=1, reranker sessions are skipped (worker
    // handles inference); metadata-only entries keep the name lookup working.
    let mut reranker_entries: HashMap<String, RerankerEntry> = HashMap::new();
    if !cfg.multi_process {
        for def in &cfg.rerankers {
            tracing::info!(reranker = %def.name, dir = %def.dir, "loading reranker");
            let m = RerankerModel::load(
                &def.name,
                &def.dir,
                def.max_len,
                def.padded_model,
                cfg.reranker_intra_threads,
                cfg.reranker_pool_size,
            )
            .unwrap_or_else(|e| {
                eprintln!("failed to load reranker '{}': {e}", def.name);
                std::process::exit(1);
            });
            // Pre-warm every session in the pool at every configured batch
            // shape so the FIRST production request at each shape doesn't
            // pay graph compile + arena alloc cost (~3s observed on cold
            // gte-multi-rerank vs ~1.5s steady state). Default shape list is
            // `[1, 5]` — batch=1 covers the static fast-path single-pair
            // calls, batch=5 covers the downstream consumer's D7 sub-query fanout default.
            // Override via `RERANK_WARMUP_BATCH_SIZES`. Best-effort:
            // warmup failure is logged but does not abort boot — the
            // server still serves correctly without it.
            if let Err(e) = m.warmup(&cfg.rerank_warmup_batch_sizes, cfg.embed_warmup_seq_len) {
                tracing::error!(reranker = %def.name, error = %e, "reranker warmup failed (non-fatal)");
            }
            let model_arc = Arc::new(m);

            // Adapter bridging `RerankerModel::score_pairs -> Vec<f32>` into
            // `DynamicBatcher`'s `Fn(Vec<Vec<u32>>) -> Vec<Vec<f32>>` contract.
            // We wrap each scalar score as a 1-element Vec so the batcher's
            // per-item "one vector per text" semantics still holds (the E3
            // handler unwraps each inner Vec and takes element 0). The
            // `into_iter`/`.map` form avoids the pointless copy that
            // `.iter().map(|&s| vec![s])` would produce.
            let batcher = if cfg.batching_enabled {
                let rr = model_arc.clone();
                let b = batcher::DynamicBatcher::with_tokens(
                    &def.name,
                    move |token_ids| {
                        rr.score_pairs(&token_ids)
                            .map(|scores| scores.into_iter().map(|s| vec![s]).collect())
                    },
                    cfg.batch_max_tokens,
                    // Use reranker-specific cap (defaults to 4× batch_max).
                    // Embed batchers still use cfg.batch_max — see the embed
                    // model loop above.
                    cfg.reranker_batch_max,
                    cfg.batch_max_seq,
                    def.padded_model,
                    cfg.batch_wait_ms,
                    cfg.max_queue_size,
                );
                Some(Arc::new(b))
            } else {
                None
            };

            reranker_entries.insert(
                def.name.clone(),
                RerankerEntry {
                    model: Some(model_arc),
                    batcher,
                },
            );
        }
    } else {
        // Multi-process: metadata-only reranker entries.
        for def in &cfg.rerankers {
            reranker_entries.insert(
                def.name.clone(),
                RerankerEntry {
                    model: None,
                    batcher: None,
                },
            );
        }
    }

    // SPLADE loading loop. Mirrors the reranker block exactly except
    // there's no batcher integration in v1 — `SpladeEntry::batcher` is
    // always `None` for now. Fail loudly on load errors (same fail-at-
    // boot stance as the embedding/reranker loops).
    // When EMBED_MULTI_PROCESS=1, SPLADE sessions are skipped; metadata-only
    // entries keep name resolution working.
    let mut splade_entries: HashMap<String, SpladeEntry> = HashMap::new();
    if !cfg.multi_process {
        for def in &cfg.splades {
            tracing::info!(splade = %def.name, dir = %def.dir, "loading splade");
            let m = SpladeModel::load(
                &def.name,
                &def.dir,
                def.max_len,
                cfg.splade_intra_threads,
                cfg.splade_pool_size,
            )
            .unwrap_or_else(|e| {
                eprintln!("failed to load splade '{}': {e}", def.name);
                std::process::exit(1);
            });
            // Pre-warm every SPLADE session. SPLADE inference is intrinsically
            // batch=1 (single text in, sparse vector out — see model_splade.rs),
            // so the shape list for SPLADE typically has one entry; default
            // is `[1]`. Override via `SPLADE_WARMUP_BATCH_SIZES` only if a
            // future SPLADE batched API lands.
            if let Err(e) = m.warmup(&cfg.splade_warmup_batch_sizes) {
                tracing::error!(splade = %def.name, error = %e, "splade warmup failed (non-fatal)");
            }
            splade_entries.insert(
                def.name.clone(),
                SpladeEntry {
                    model: Some(Arc::new(m)),
                    // v1: no dynamic batcher. Follow-up will populate this
                    // once we observe SPLADE traffic shape and decide
                    // whether per-batch padding amortisation is worth the
                    // adapter complexity.
                    batcher: None,
                },
            );
        }
    } else {
        // Multi-process: metadata-only SPLADE entries.
        for def in &cfg.splades {
            splade_entries.insert(
                def.name.clone(),
                SpladeEntry {
                    model: None,
                    batcher: None,
                },
            );
        }
    }

    let inproc_embed = model_entries.values().filter(|e| e.model.is_some()).count();
    let inproc_rerank = reranker_entries
        .values()
        .filter(|e| e.model.is_some())
        .count();
    let inproc_splade = splade_entries
        .values()
        .filter(|e| e.model.is_some())
        .count();
    tracing::info!(
        multi_process = cfg.multi_process,
        batching_enabled = cfg.batching_enabled,
        metadata_models = model_entries.len(),
        in_process_models = inproc_embed,
        metadata_rerankers = reranker_entries.len(),
        in_process_rerankers = inproc_rerank,
        metadata_splades = splade_entries.len(),
        in_process_splades = inproc_splade,
        "model registry built"
    );

    let drain_timeout = Duration::from_secs(cfg.drain_timeout_s);
    let shutdown_token = CancellationToken::new();

    // Process-local response cache sized from CACHE_MAX_ENTRIES (default
    // 10_000). Setting CACHE_MAX_ENTRIES=0 produces a disabled shell
    // (get/insert are no-ops) — the documented runtime kill-switch.
    let cache = Arc::new(EmbeddingCache::new(cfg.cache_max_entries));
    // Stamp the gauge with 0 so /metrics exposes `embed_cache_size` from startup,
    // even before the first cache miss populates it.
    crate::metrics::set_cache_size(0);
    // Pre-touch the ready-probe counter so /metrics shows
    // `embed_ready_probe_total{result=...}` from startup, not absent.
    crate::metrics::ready_probe_touch();
    tracing::info!(
        cache_max_entries = cfg.cache_max_entries,
        cache_enabled = cache.is_enabled(),
        "response cache ready"
    );

    // Per-pair tokenizer cache (H.7). Default TOKEN_CACHE_MAX_ENTRIES=0
    // (disabled) — existing deployments see no behaviour change. A positive
    // value amortizes ~50ms tokenizer calls when the downstream consumer's D7 sub-query
    // rewrites re-score the same (query, doc) pairs.
    let token_cache = Arc::new(TokenCache::new(cfg.token_cache_max_entries));
    // Pre-warm hit+miss counters for every loaded reranker model so
    // /metrics shows the series from startup (before the first request).
    // This mirrors the embedding-cache gauge stamp above.
    for def in &cfg.rerankers {
        crate::metrics::record_token_cache_hit(&def.name, 0);
        crate::metrics::record_token_cache_miss(&def.name, 0);
    }
    tracing::info!(
        token_cache_max_entries = cfg.token_cache_max_entries,
        token_cache_enabled = token_cache.is_enabled(),
        "token cache ready"
    );

    // Global rerank concurrency cap. Load-shed at HTTP edge per TEI's
    // `Infer::try_acquire_permit` pattern — 429 + Retry-After: 1 BEFORE
    // tokenizer CPU is spent on requests that will queue behind capacity.
    //
    // Auto-default for burst headroom: `pool × intra × 8` truly-parallel
    // inference slots. The permit is held for the full request lifetime
    // (tokenize + batcher wait + inference, often 1–3s per call), so
    // capping at exactly `pool × intra` (e.g. 4 for prod's 2×2) leaves
    // zero headroom for tokenize / queue burst — production saw ~1%
    // 429s on a chat-50 sweep (22/1779 calls). 8× covers realistic
    // bursts without becoming an unlimited firehose. Floor of 16 keeps
    // tiny pool=1+intra=1 dev setups from getting an absurd 8-permit cap.
    //
    // Override via `MAX_CONCURRENT_RERANK_REQUESTS` env. Set to `0` to
    // disable the semaphore entirely (legacy unlimited behaviour).
    let auto_cap = cfg
        .reranker_pool_size
        .saturating_mul(cfg.reranker_intra_threads)
        .saturating_mul(8)
        .max(16);
    let env_override = std::env::var("MAX_CONCURRENT_RERANK_REQUESTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let rerank_semaphore = match env_override {
        Some(0) => {
            tracing::info!("rerank semaphore disabled via MAX_CONCURRENT_RERANK_REQUESTS=0");
            None
        }
        Some(n) => {
            tracing::info!(cap = n, source = "env", "rerank semaphore enabled");
            Some(Arc::new(tokio::sync::Semaphore::new(n)))
        }
        None => {
            tracing::info!(
                cap = auto_cap,
                source = "auto",
                pool = cfg.reranker_pool_size,
                intra = cfg.reranker_intra_threads,
                "rerank semaphore enabled with auto headroom (pool × intra × 8)"
            );
            Some(Arc::new(tokio::sync::Semaphore::new(auto_cap)))
        }
    };

    // Spawn worker pool if multi-process mode is enabled.
    // All workers (embed + rerank + splade) launch in parallel via tokio::spawn —
    // each spawn awaits its UDS socket independently (cold model load ~5-15s per
    // model). Sequential await would add 30-50s to startup and trip the deployment system's
    // smoke-test deadline (canary smoke timeout = 120s).
    let worker_pool: Option<Arc<crate::supervisor::WorkerPool>> = if cfg.multi_process {
        tracing::info!(
            "multi-process mode enabled — spawning worker pool (in-process sessions skipped)"
        );
        let pool = crate::supervisor::WorkerPool::new();

        // Collect spawn specs for embed + rerank + splade.
        //
        // Per-worker metrics port scheme:
        //   Base port from EMBED_WORKER_METRICS_PORT_BASE (default 19200).
        //   Each worker gets base + index (across embed+rerank+splade in
        //   declaration order). Unset EMBED_WORKER_METRICS_PORT_BASE → metrics
        //   HTTP disabled for all workers (back-compat).
        let metrics_port_base: Option<u16> = std::env::var("EMBED_WORKER_METRICS_PORT_BASE")
            .ok()
            .and_then(|s| s.trim().parse::<u16>().ok());

        // Read global pool size raw value once so per-model resolver can fall
        // back to it without re-reading the env var inside the loop.
        let global_embed_pool_raw = std::env::var("EMBED_SESSION_POOL_SIZE").ok();
        let mut specs: Vec<crate::supervisor::SpawnSpec> = Vec::new();
        let mut worker_index: u16 = 0;
        for model_def in &cfg.models {
            let pool_size = crate::config::resolve_embed_pool_size_for_model(
                &model_def.name,
                global_embed_pool_raw.as_deref(),
            );
            let mut env_extra: Vec<(String, String)> = Vec::new();
            if let Some(port) = worker_metrics_port(metrics_port_base, worker_index) {
                env_extra.push(("EMBED_WORKER_METRICS_PORT".into(), port.to_string()));
            }
            worker_index = worker_index.saturating_add(1);
            specs.push(crate::supervisor::SpawnSpec {
                model: model_def.name.clone(),
                kind: crate::supervisor::WorkerKind::Embed,
                worker_bin: cfg.worker_bin_path.clone(),
                socket_dir: cfg.worker_socket_dir.clone(),
                pool_size,
                intra_threads: cfg.intra_threads,
                env_extra,
            });
        }
        for r_def in &cfg.rerankers {
            let mut env_extra: Vec<(String, String)> = Vec::new();
            if let Some(port) = worker_metrics_port(metrics_port_base, worker_index) {
                env_extra.push(("EMBED_WORKER_METRICS_PORT".into(), port.to_string()));
            }
            worker_index = worker_index.saturating_add(1);
            specs.push(crate::supervisor::SpawnSpec {
                model: r_def.name.clone(),
                kind: crate::supervisor::WorkerKind::Rerank,
                worker_bin: cfg.worker_bin_path.clone(),
                socket_dir: cfg.worker_socket_dir.clone(),
                pool_size: cfg.reranker_pool_size.max(1),
                intra_threads: cfg.reranker_intra_threads.max(1),
                env_extra,
            });
        }
        for s_def in &cfg.splades {
            let mut env_extra: Vec<(String, String)> = Vec::new();
            if let Some(port) = worker_metrics_port(metrics_port_base, worker_index) {
                env_extra.push(("EMBED_WORKER_METRICS_PORT".into(), port.to_string()));
            }
            worker_index = worker_index.saturating_add(1);
            specs.push(crate::supervisor::SpawnSpec {
                model: s_def.name.clone(),
                kind: crate::supervisor::WorkerKind::Splade,
                worker_bin: cfg.worker_bin_path.clone(),
                socket_dir: cfg.worker_socket_dir.clone(),
                pool_size: cfg.splade_pool_size.max(1),
                intra_threads: cfg.splade_intra_threads.max(1),
                env_extra,
            });
        }

        // Resolve inter-worker spawn stagger to smooth cold-load I/O peak.
        //
        // EMBED_WORKER_SPAWN_DELAY_MS (default 2000):
        //   milliseconds to wait *between* successive tokio::spawn calls.
        //   First worker spawns immediately; each subsequent worker waits this
        //   long before its spawn, giving the previous worker's ONNX read a
        //   head-start on pagecache warm-up.
        //   Set to 0 to disable (parallel cold-load, original behaviour).
        //   the deployment system's smoke timeout is 120s; 4 workers × 2s = 6s overhead — well within budget.
        let spawn_stagger =
            crate::supervisor::util::resolve_spawn_stagger_ms("EMBED_WORKER_SPAWN_DELAY_MS", 2000);
        let total_workers = specs.len();

        // Fan out spawn — each future independently awaits its worker's UDS socket.
        // Workers are staggered by spawn_stagger to prevent simultaneous ONNX
        // cold-reads from causing an I/O storm under host RAM pressure.
        let mut handles: Vec<_> = Vec::with_capacity(total_workers);
        for (position, spec) in specs.into_iter().enumerate() {
            // Stagger: first worker spawns immediately (position=0); each
            // subsequent worker waits `spawn_stagger` before its spawn,
            // giving the previous worker's ONNX read a pagecache head-start.
            // `spawn_stagger` is `Option<Duration>` — `None` means disabled
            // (EMBED_WORKER_SPAWN_DELAY_MS=0 or absent with default=0).
            if let Some(delay) = spawn_stagger.filter(|_| position > 0) {
                tokio::time::sleep(delay).await;
            }
            let name = spec.model.clone();
            let kind = spec.kind;
            let pos_display = position + 1;
            let delay_ms_display = spawn_stagger.map(|d| d.as_millis()).unwrap_or(0);
            tracing::info!(
                model = %name,
                kind = %kind.as_str(),
                position = pos_display,
                total = total_workers,
                delay_ms = delay_ms_display,
                "spawning worker"
            );
            handles.push(tokio::spawn(async move {
                let result = crate::supervisor::WorkerSupervisor::launch(spec).await;
                (name, kind, result)
            }));
        }

        // Collect results — any failure aborts startup.
        for h in handles {
            let (name, kind, result) = h.await.unwrap_or_else(|e| {
                tracing::error!(error = ?e, "join handle for worker supervisor panicked");
                std::process::exit(1);
            });
            match result {
                Ok(supervisor) => pool.add(supervisor).await,
                Err(e) => {
                    tracing::error!(
                        model = %name,
                        kind = %kind.as_str(),
                        error = ?e,
                        "worker supervisor launch failed"
                    );
                    std::process::exit(1);
                }
            }
        }

        Some(Arc::new(pool))
    } else {
        None
    };

    // Spawn per-worker RSS gauge poller (multi-process mode only).
    //
    // Reads /proc/<pid>/status VmRSS every 15 s and writes to
    // `embed_worker_rss_bytes{model}`. Wires directly into the existing
    // WorkerPool -- no extra state needed.
    //
    // 15 s cadence is coarse enough to have negligible CPU cost while
    // still detecting BFCArena ratchet within one scrape interval
    // (Prometheus default = 15 s, so worst-case lag = 30 s).
    if let Some(ref rss_pool) = worker_pool {
        let rss_pool_clone = Arc::clone(rss_pool);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let pairs = rss_pool_clone.worker_pids().await;
                for (model, pid) in pairs {
                    match proc::read_proc_status_rss(pid) {
                        Ok(bytes) => {
                            crate::metrics::worker_rss_set(&model, bytes as f64);
                        }
                        Err(e) => {
                            tracing::debug!(
                                model = %model,
                                pid,
                                error = %e,
                                "RSS read failed; worker may have just restarted"
                            );
                        }
                    }
                }
            }
        });
    }

    let state = Arc::new(AppState {
        models: model_entries,
        rerankers: reranker_entries,
        splades: splade_entries,
        default_model: cfg.default_model,
        shutdown: shutdown_token.clone(),
        drain_timeout,
        cache,
        token_cache,
        rerank_semaphore,
        embed_max_input_array: cfg.embed_max_input_array,
        rerank_max_input_docs: cfg.rerank_max_input_docs,
        worker_pool,
        ready_probe_timeout_ms: cfg.ready_probe_timeout_ms,
    });

    let metrics_handle = prom_handle.clone();
    // Clone the Arc for the router (`.with_state` consumes it) — we retain
    // the original binding so we can drain batcher workers once axum's HTTP
    // drain returns.
    let router_state = state.clone();
    let app = Router::new()
        .route("/health", axum::routing::get(api_health::health))
        .route("/ready", axum::routing::get(api_health::ready))
        .route(
            "/metrics",
            axum::routing::get(move || {
                let h = metrics_handle.clone();
                async move {
                    (
                        [(
                            axum::http::header::CONTENT_TYPE,
                            "text/plain; version=0.0.4",
                        )],
                        h.render(),
                    )
                }
            }),
        )
        .route("/v1/embeddings", axum::routing::post(api::embeddings))
        .route("/v1/rerank", axum::routing::post(api_rerank::rerank))
        // SPLADE / sparse embeddings — HuggingFace TEI convention path.
        // No `/v1/` prefix because TEI itself doesn't use one for sparse;
        // this maximises drop-in compat with TEI-aware tooling.
        .route(
            "/embed_sparse",
            axum::routing::post(api_splade::sparse_embeddings),
        )
        // Phase H.18 — every request gets a root span linked to the
        // upstream caller's trace via W3C traceparent (the downstream consumer sets it
        // on every outbound /v1/* call). /health and /metrics also get
        // spans, but at default sampling (5 %) they're effectively
        // free; can exclude via env-driven sampler if it ever matters.
        .layer(axum::middleware::from_fn(otel::trace_request))
        .with_state(router_state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!(addr = %addr, "embed-server listening");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_token, drain_timeout))
        .await
        .unwrap();

    // HTTP listener has closed and all handlers have returned — no `State<Arc<AppState>>`
    // clones remain. Now cleanly join every batcher worker so no batch is cut
    // mid-forward-pass. Uses the same drain_timeout budget (task #20).
    drain_batchers(state, drain_timeout).await;

    // Phase H.18 — flush the OTEL batch exporter so spans for the last
    // requests handled before SIGTERM make it to Jaeger instead of
    // dropping with the process. No-op when OTEL was disabled at boot.
    if let Some(p) = otel_provider {
        otel::shutdown(p);
    }
}

/// Compute the metrics port for a worker at `index` given an optional `base`
/// port.
///
/// Returns `None` in two cases:
/// - `base` is `None` (feature disabled — `EMBED_WORKER_METRICS_PORT_BASE`
///   not set).
/// - `base + index` would overflow `u16` (port number > 65535). Logs an
///   error so operators can detect misconfiguration at startup rather than
///   at scrape time.
fn worker_metrics_port(base: Option<u16>, index: u16) -> Option<u16> {
    let b = base?;
    match b.checked_add(index) {
        Some(port) => Some(port),
        None => {
            tracing::error!(
                base,
                index,
                "EMBED_WORKER_METRICS_PORT_BASE + worker index overflows u16 (max port 65535); metrics port disabled for this worker"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::worker_metrics_port;

    #[test]
    fn worker_metrics_port_none_base_returns_none() {
        assert_eq!(worker_metrics_port(None, 0), None);
        assert_eq!(worker_metrics_port(None, 40), None);
    }

    #[test]
    fn worker_metrics_port_normal_range() {
        assert_eq!(worker_metrics_port(Some(19200), 0), Some(19200));
        assert_eq!(worker_metrics_port(Some(19200), 3), Some(19203));
    }

    #[test]
    fn worker_metrics_port_exact_max_u16() {
        assert_eq!(worker_metrics_port(Some(65535), 0), Some(65535));
        assert_eq!(worker_metrics_port(Some(65534), 1), Some(65535));
    }

    #[test]
    fn worker_metrics_port_overflow_returns_none() {
        // base=65500, index=40 overflows: 65540 > 65535
        assert_eq!(worker_metrics_port(Some(65500), 40), None);
        // base=65535, index=1: 65536 overflows
        assert_eq!(worker_metrics_port(Some(65535), 1), None);
    }
}
