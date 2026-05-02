mod api;
mod api_rerank;
mod api_splade;
mod arena;
mod batcher;
mod cache;
mod cache_flow;
mod config;
mod metrics;
mod model;
mod model_reranker;
mod model_splade;
mod pool;
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
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort::logging=warn".parse().unwrap()),
        )
        .init();

    // Initialize ort runtime explicitly (required for load-dynamic).
    if !ort::init().commit() {
        eprintln!("ort init failed (environment already configured?)");
    }
    tracing::info!("ort runtime initialized");

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

    let version = std::env::var("EMBED_VERSION").unwrap_or_else(|_| "dev".into());
    let prom_handle = std::sync::Arc::new(metrics::init(&version));

    let mut raw_models: HashMap<String, Arc<EmbedModel>> = HashMap::new();
    for def in &cfg.models {
        tracing::info!(model = %def.name, dir = %def.dir, "loading model");
        let m = EmbedModel::load(def, cfg.intra_threads, cfg.auto_truncate).unwrap_or_else(|e| {
            eprintln!("failed to load model '{}': {e}", def.name);
            std::process::exit(1);
        });
        raw_models.insert(def.name.clone(), Arc::new(m));
    }

    tracing::info!(
        models = raw_models.len(),
        default = %cfg.default_model,
        "all models loaded"
    );

    let mut model_entries: HashMap<String, ModelEntry> = HashMap::new();
    for (name, model_arc) in raw_models {
        let batcher = if cfg.batching_enabled {
            let m = model_arc.clone();
            // ONNX BERT-style encoders always pad to max(seq_len), so
            // padded_model=true is the right accounting for our stack.
            // Kept as a batcher parameter so tests can exercise the
            // non-padded branch directly.
            let b = batcher::DynamicBatcher::with_tokens(
                &name,
                move |token_ids| m.embed_tokens(&token_ids),
                cfg.batch_max_tokens,
                cfg.batch_max,
                /*padded_model*/ true,
                cfg.batch_wait_ms,
                cfg.max_queue_size,
            );
            Some(Arc::new(b))
        } else {
            None
        };
        model_entries.insert(
            name,
            ModelEntry {
                model: model_arc,
                batcher,
            },
        );
    }

    // Load every configured reranker the same way embedding models are
    // loaded, before building AppState. An empty `cfg.rerankers` is a
    // valid no-op — the server boots serving only `/v1/embeddings`.
    let mut reranker_entries: HashMap<String, RerankerEntry> = HashMap::new();
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
        // Pre-warm every session in the pool so the FIRST production
        // request doesn't pay graph compile + arena alloc cost (~3s
        // observed on cold gte-multi-rerank vs ~1.5s steady state).
        // Best-effort: warmup failure is logged but does not abort boot
        // — the server still serves correctly without it.
        if let Err(e) = m.warmup() {
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
                model: model_arc,
                batcher,
            },
        );
    }

    // SPLADE loading loop. Mirrors the reranker block exactly except
    // there's no batcher integration in v1 — `SpladeEntry::batcher` is
    // always `None` for now. Fail loudly on load errors (same fail-at-
    // boot stance as the embedding/reranker loops).
    let mut splade_entries: HashMap<String, SpladeEntry> = HashMap::new();
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
        splade_entries.insert(
            def.name.clone(),
            SpladeEntry {
                model: Arc::new(m),
                // v1: no dynamic batcher. Follow-up will populate this
                // once we observe SPLADE traffic shape and decide
                // whether per-batch padding amortisation is worth the
                // adapter complexity.
                batcher: None,
            },
        );
    }

    tracing::info!(
        batching_enabled = cfg.batching_enabled,
        models = model_entries.len(),
        rerankers = reranker_entries.len(),
        splades = splade_entries.len(),
        "app state ready"
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
    tracing::info!(
        cache_max_entries = cfg.cache_max_entries,
        cache_enabled = cache.is_enabled(),
        "response cache ready"
    );

    // Per-pair tokenizer cache (H.7). Default TOKEN_CACHE_MAX_ENTRIES=0
    // (disabled) — existing deployments see no behaviour change. A positive
    // value amortizes ~50ms tokenizer calls when memdb-go's D7 sub-query
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

    // Optional global rerank concurrency cap. Reads
    // MAX_CONCURRENT_RERANK_REQUESTS — when unset/0, semaphore is None
    // and the handler accepts unlimited concurrent rerank calls (legacy
    // behaviour). When set, requests over the cap get HTTP 429 with
    // Retry-After: 1 BEFORE tokenizer CPU is spent — load-shed at the
    // edge per TEI's `Infer::try_acquire_permit` pattern.
    let rerank_semaphore = std::env::var("MAX_CONCURRENT_RERANK_REQUESTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map(|n| {
            tracing::info!(cap = n, "rerank semaphore enabled");
            Arc::new(tokio::sync::Semaphore::new(n))
        });

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
    });

    let metrics_handle = prom_handle.clone();
    // Clone the Arc for the router (`.with_state` consumes it) — we retain
    // the original binding so we can drain batcher workers once axum's HTTP
    // drain returns.
    let router_state = state.clone();
    let app = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
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
}
