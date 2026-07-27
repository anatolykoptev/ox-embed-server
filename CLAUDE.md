# embed-server — Unified Rust ONNX Embedding + Rerank + SPLADE Sidecar

**Rust / axum** · Docker · Prometheus `/metrics` · branch `main`

axum HTTP on `:8082` serving 4 ONNX models. Since 2026-05-12 runs as supervisor + N worker child processes (`EMBED_MULTI_PROCESS=1`), one worker per model with isolated BFCArena.

| Class    | Model                                                   | Endpoint             | API shape |
|----------|---------------------------------------------------------|----------------------|-----------|
| Dense    | `multilingual-e5-large` (1024d), `jina-code-v2` (768d)  | `POST /v1/embeddings`| OpenAI    |
| Reranker | `gte-multi-rerank`                                      | `POST /v1/rerank`    | Cohere    |
| Sparse   | `splade-v3-distilbert`                                  | `POST /embed_sparse` | TEI       |

## Documentation map

- **Architecture (multi-process)**: `docs/architecture/multi-process.md` — process model, IPC protocol, supervisor lifecycle, memory cost, observability. Diagrams: `docs/architecture/embed-server.c4` (LikeC4).
- **Runbook** (operations, symptom → response): `docs/runbook.md`
- **Bugs** (historical workarounds): `docs/BUGS.md` (BUG-004 jina-code-v2 OOM — resolved by multi-process Phase 2)
- **Roadmap / plans**: `docs/ROADMAP.md`, `docs/superpowers/plans/2026-05-12-multi-process-refactor.md`
- **Benchmarks**: `docs/benchmarks/` (ARM Neoverse-N1 baselines)

## API

- `POST /v1/embeddings` — OpenAI-compat. 503 + `Retry-After: 1` on queue full, 503 + `Retry-After: 5` during shutdown. Response cache in front.
- `POST /v1/rerank` — Cohere-shape. Documents accepted as plain string or `{"text":...}` (untagged serde). No cache.
- `POST /embed_sparse` — TEI convention (no `/v1/` prefix). Body: `{"input":["text",...]}` (singular `input`).
- `GET /health` — `ok`.
- `GET /metrics` — Prometheus. Multi-process specific: `embed_worker_restart_total{model}` (pre-touched to 0). Full series list: see `metrics::init` in `src/metrics.rs`.

## Environment — key flags

Multi-process (live prod):
- `EMBED_MULTI_PROCESS=1` — supervisor spawns workers. Set to `0` for monolith rollback.
- `EMBED_WORKER_BIN=/usr/local/bin/embed-worker` — shipped in image.
- `EMBED_WORKER_SOCKET_DIR=/tmp/embed-workers` — UDS socket directory.
- `EMBED_WORKER_SPAWN_DELAY_MS=2000` — stagger between successive worker spawns to smooth cold-load I/O peak (first worker spawns immediately, each subsequent waits this long). `0` = disable (parallel cold-load, original behaviour). 4 workers × 2s = 6s overhead, well within dozor 120s smoke timeout. PR #79.

Models (live prod values):
- `EMBED_PORT=8082`
- `EMBED_MODELS="multilingual-e5-large:/models:1024:256:1:false,jina-code-v2:/models-jina:768:512:0:false"` — format `name:dir:dim:max_len:pad_id:has_tti[:model_file]`
- `EMBED_DEFAULT_MODEL=multilingual-e5-large`
- `RERANKER_MODELS="gte-multi-rerank:/models-gte-rerank:256:true"` — format `name:dir:max_len:padded`
- `SPLADE_MODELS="splade-v3-distilbert:/models-splade:256"` — format `name:dir:max_len`

ORT tuning (prod-validated):
- `EMBED_INTRA_THREADS=2`, `EMBED_SESSION_POOL_SIZE=2` — `pool_size * intra_threads ≤ cores` rule on 4-core ARM Neoverse-N1.
- `EMBED_MEMORY_PATTERN_JINA_CODE_V2=false` — required for jina (variable seq + ALiBi). Other models keep default `true`.
- `EMBED_ARENA_MAX_MEM_BYTES=6442450944` (6 GiB) — BFCArena ceiling per worker.
- `ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so` — required by `ort` with `load-dynamic`.

Batcher (in-process path; pre-existing):
- `BATCH_MAX=32`, `BATCH_MAX_TOKENS=16384`, `BATCH_MAX_SEQ=256`, `BATCH_WAIT_MS=30`, `MAX_QUEUE_SIZE=256`.

Full env reference with per-variable rationale: see `compose/memdb.yml` inline comments (each env line is annotated with the incident or PR that set its value).

## Local CI

GitHub Actions runs only release-please. Cargo gates are local — run `make ci` before pushing.

| Target | What it runs |
|--------|--------------|
| `make fmt` | `cargo fmt --all -- --check` |
| `make clippy` | `cargo clippy --locked --all-targets --workspace -- -D warnings` |
| `make test` | `cargo nextest run --locked --all-targets --workspace` |
| `make build` | `cargo build --release --locked` |
| `make ci` | lint + test + build (full gate, ~2 min warm) |

`--locked` mandatory on all targets — catches `Cargo.lock` drift that `cargo check` misses (incident 2026-05-02).

Integration tests with real models require: `EMBED_MODELS=...` + `RERANKER_MODELS=...` + `SPLADE_MODELS=...` + `ORT_DYLIB_PATH=...` env. Run `--test-threads=1` (parallel OOMs 4-core 24 GB).

## Deploy

Auto-deploy: push to `main` → dozor webhook (`~/.dozor/deploy-repos.yaml`, repo `anatolykoptev/ox-embed-server`) → rebuild image → `docker compose up -d --no-deps --force-recreate embed-server`. Smoke timeout 120 s (workers warm up in ~3.4 s parallel).

Manual:
```bash
cd ~/deploy/krolik-server
docker compose build --no-cache embed-server   # ~3 min cold deps, ~40 s warm
docker compose up -d --no-deps --force-recreate embed-server
```

Releases: release-please on push to `main`. Conventional commits → auto PR + tag. Do not tag manually.

## Gotchas

- `ort` + `load-dynamic` → `libonnxruntime.so` from `ORT_DYLIB_PATH` at startup.
- `model_optimized.onnx` (graph-fused) is **slower** than `model_quantized.onnx` on ARM Neoverse-N1 for BERT-family. Don't switch `MODEL_FILE` without benchmarking.
- `ort-sys` downloads the ORT binary on first compile (~30 s cold).
- Batcher `carry: Option<Item>` — items overflowing `BATCH_MAX` defer to next batch. Tracked as `embed_carry_events_total`.
- Multi-process worker test parallelism: **`--test-threads=1` mandatory** for `cargo nextest run --test multi_process_*` — each test spawns its own embed-server with full model set, parallel OOMs on 24 GB ARM.
- Worker child process must call `arena::register_shared_cpu_arena()` before any `Session::builder()` — supervisor's registration doesn't carry across fork. (Handled in `src/bin/worker.rs`.)
- Per-request UDS conn (not persistent pool) — see `docs/architecture/multi-process.md` for cancel-safety rationale.
