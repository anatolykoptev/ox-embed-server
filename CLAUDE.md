# embed-server — Unified Rust ONNX Embedding + Rerank + SPLADE Sidecar

**Rust / axum** · Docker · Prometheus `/metrics` · branch `main`

axum HTTP server serving 4 ONNX models for dense embedding, reranking, and sparse lexical expansion. Runs as supervisor + N worker child processes (`EMBED_MULTI_PROCESS=1`), one worker per model with isolated BFCArena.

| Class    | Model                                                   | Endpoint             | API shape |
|----------|---------------------------------------------------------|----------------------|-----------|
| Dense    | `multilingual-e5-large` (1024d), `jina-code-v2` (768d)  | `POST /v1/embeddings`| OpenAI    |
| Reranker | `gte-multi-rerank`                                      | `POST /v1/rerank`    | Cohere    |
| Sparse   | `splade-v3-distilbert`                                  | `POST /embed_sparse` | TEI       |

## Documentation map

- **Architecture (multi-process)**: `docs/architecture/multi-process.md` — process model, IPC protocol, supervisor lifecycle, memory cost, observability. Diagrams: `docs/architecture/embed-server.c4` (LikeC4).
- **Bugs** (historical workarounds): `docs/BUGS.md`
- **Roadmap**: `docs/ROADMAP.md`
- **Benchmarks**: `docs/benchmarks/` (ARM Neoverse-N1 baselines)

## API

- `POST /v1/embeddings` — OpenAI-compat. 429 + `Retry-After: 1` on queue full, 503 + `Retry-After: 5` during shutdown. Response cache in front.
- `POST /v1/rerank` — Cohere-shape. Documents accepted as plain string or `{"text":...}` (untagged serde). No cache.
- `POST /embed_sparse` — TEI convention (no `/v1/` prefix). Body: `{"input":["text",...]}` (singular `input`).
- `GET /health` — `ok`.
- `GET /metrics` — Prometheus. Multi-process specific: `embed_worker_restart_total{model}` (pre-touched to 0). Full series list: see `metrics::init` in `src/metrics.rs`.

## Environment — key flags

Multi-process:
- `EMBED_MULTI_PROCESS=1` — supervisor spawns workers. Set to `0` for monolith rollback.
- `EMBED_WORKER_BIN=/usr/local/bin/embed-worker` — shipped in image.
- `EMBED_WORKER_SOCKET_DIR=/tmp/embed-workers` — UDS socket directory.

Models (format `name:dir:dim:max_len:pad_id:has_tti[:model_file]`):
- `EMBED_MODELS="multilingual-e5-large:/models:1024:256:1:false,jina-code-v2:/models-jina:768:512:0:false"`
- `EMBED_DEFAULT_MODEL=multilingual-e5-large`
- `RERANKER_MODELS="gte-multi-rerank:/models-gte-rerank:256:true"` — format `name:dir:max_len:padded`
- `SPLADE_MODELS="splade-v3-distilbert:/models-splade:256"` — format `name:dir:max_len`

ORT tuning:
- `EMBED_INTRA_THREADS=2`, `EMBED_SESSION_POOL_SIZE=2` — `pool_size * intra_threads ≤ cores` rule.
- `EMBED_MEMORY_PATTERN_JINA_CODE_V2=false` — required for jina (variable seq + ALiBi). Other models keep default `true`.
- `EMBED_ARENA_MAX_MEM_BYTES=6442450944` (6 GiB) — BFCArena ceiling per worker.
- `ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so` — required by `ort` with `load-dynamic`.

Batcher:
- `BATCH_MAX=32`, `BATCH_MAX_TOKENS=16384`, `BATCH_MAX_SEQ=256`, `BATCH_WAIT_MS=30`, `MAX_QUEUE_SIZE=256`.
- `BATCH_LENGTH_RATIO_THRESHOLD` — (default `0.0` = disabled) when > 0.0, items whose `max_seq_len` exceeds `accum.max_len * threshold` are carried to the next batch to reduce padding waste on padded models.

## CI

Two lanes, both on GitHub-hosted `ubuntu-24.04-arm` (public repo → free + unlimited, and arm64 is what pillow runs).

**`preflight`** — every PR: gitleaks → osv-scanner → fmt → clippy → deny → nextest → release build → **mutants `--in-diff`**.

**`nightly`** (03:00 UTC) — full mutation scope sharded 4×, then a ratchet job; plus `deny`/`osv-scanner` against that day's advisory feeds (a dep does not have to change to become vulnerable), plus the full suite with `--no-fail-fast`.

Local gates:

| Target | What it runs |
|--------|--------------|
| `make fmt` | `cargo fmt --all -- --check` |
| `make clippy` | `cargo clippy --locked --all-targets --workspace -- -D warnings` |
| `make deny` / `secrets` / `vulns` | cargo-deny · gitleaks · osv-scanner |
| `make audit` | deny + secrets + vulns |
| `make test` | `cargo nextest run --locked --all-targets --workspace` |
| `make build` | `cargo build --release --locked` |
| `make ci` | lint + audit + test + build (full pre-push gate) |
| `make mutants-diff` | cargo-mutants on this branch's changed lines — **what preflight gates on** |
| `make mutants` | cargo-mutants over the full scope (hours; nightly's job) |

`--locked` mandatory on all targets — catches `Cargo.lock` drift.

### Mutation testing

`fmt`, `clippy` and `nextest` all pass on a test that asserts nothing. cargo-mutants breaks the source one edit at a time and requires that some test go red; a surviving ("MISSED") mutant is a line no test actually checks. This repo has shipped that exact defect more than once — `heartbeat_kills_wedged_worker` had zero callers of the function it claimed to prove, and `record_carry_lost` had only its negative case.

Scope, test tool and timeouts live in **`.cargo/mutants.toml`** — one file, read by both lanes, so they cannot drift. Scope is deliberately partial: model-free logic with real unit tests. Anything needing a live ORT session (`model.rs`, `model_reranker/**`, `model_splade.rs`, `arena.rs`, `onnx_cache.rs`) is excluded because its tests early-return without `EMBED_MODELS`, so every mutant would report MISSED for an environmental reason and the gate would mean nothing. **Widening that list as coverage grows is the goal, not the exception.**

`.cargo/mutants-baseline.txt` is the ratchet: nightly fails if the missed count rises above it, and tells you to lower it when it falls. Raising it requires a justification in the same PR.

Integration tests with real models require: `EMBED_MODELS=...` + `RERANKER_MODELS=...` + `SPLADE_MODELS=...` + `ORT_DYLIB_PATH=...` env. Run `--test-threads=1` (parallel OOMs on 4-core 24 GB).

## Releases

release-please on push to `main`. Conventional commits → auto PR + tag. Do not tag manually.

## Gotchas

- `ort` + `load-dynamic` → `libonnxruntime.so` from `ORT_DYLIB_PATH` at startup.
- `model_optimized.onnx` (graph-fused) is **slower** than `model_quantized.onnx` on ARM Neoverse-N1 for BERT-family. Don't switch `MODEL_FILE` without benchmarking.
- `ort-sys` downloads the ORT binary on first compile (~30 s cold).
- Batcher `carry: Option<Item>` — items overflowing `BATCH_MAX` defer to next batch. Tracked as `embed_carry_events_total`.
- Multi-process worker test parallelism: **`--test-threads=1` mandatory** for `cargo nextest run --test multi_process_*` — each test spawns its own embed-server with full model set, parallel OOMs.
- Worker child process must call `arena::register_shared_cpu_arena()` before any `Session::builder()` — supervisor's registration doesn't carry across fork. (Handled in `src/bin/worker.rs`.)
- Per-request UDS conn (not persistent pool) — see `docs/architecture/multi-process.md` for cancel-safety rationale.
