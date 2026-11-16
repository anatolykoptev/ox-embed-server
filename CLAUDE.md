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
- `X-Embed-Priority: bulk` (optional header, all three POST endpoints) — marks background work. On the supervisor wait for a worker slot (`pool_size` permits per model) interactive requests, the default, are admitted first; a bulk request waiting longer than `EMBED_BULK_AGING_MS` (default 2000) is served as interactive, so it cannot starve. Idle server: no effect. Any other value, or no header, is interactive. Metrics: `embed_supervisor_wait_duration_seconds{model,priority,size,outcome}` (outcome = `granted` or `cancelled`, so abandoned waits are not dropped from the tail; size = inputs per request: 1, 2-4, 5-16, 17-64, 65+) and gauge `embed_supervisor_waiters{model,priority}`.
  Internal batch callers (background scoring, precompute, bulk ingest) should send `X-Embed-Priority: bulk`; interactive callers send nothing. Heartbeat and `/ready` probes use an internal `probe` priority and are not mixed into caller latency.
- `GET /health` — `ok`.
- `GET /metrics` — Prometheus. Multi-process specific: `embed_worker_restart_total{model}` (pre-touched to 0). Full series list: see `metrics::init` in `src/metrics.rs`.

## Environment — key flags

Multi-process:
- `EMBED_MULTI_PROCESS=1` — supervisor spawns workers. Set to `0` for monolith rollback.
- `EMBED_WORKER_BIN=/usr/local/bin/embed-worker` — shipped in image.
- `EMBED_WORKER_SOCKET_DIR=/tmp/embed-workers` — UDS socket directory.
- `EMBED_IPC_TIMEOUT_SECS` (default `60`) — whole-IPC deadline per request (permit wait + connect + write + read). `EMBED_IPC_TIMEOUT_SECS_<MODEL>` overrides per model. A wedged worker fails with 503 + `Retry-After: 5` instead of hanging.
- `MAX_CONCURRENT_EMBED_REQUESTS` — global `/v1/embeddings` concurrency cap; `MAX_CONCURRENT_EMBED_REQUESTS_<MODEL>` per-model; `0` disables. Unset → per-model `pool × intra × 8`, floor 16. Over cap → 429 + `Retry-After: 1`.

Models (format `name:dir:dim:max_len:pad_id:has_tti[:model_file]`):
- `EMBED_MODELS="multilingual-e5-large:/models:1024:256:1:false,jina-code-v2:/models-jina:768:512:0:false"`
- `EMBED_DEFAULT_MODEL=multilingual-e5-large`
- `RERANKER_MODELS="gte-multi-rerank:/models-gte-rerank:256:true"` — format `name:dir:max_len:padded`
- `SPLADE_MODELS="splade-v3-distilbert:/models-splade:256"` — format `name:dir:max_len`

ORT tuning (the values here are defaults and examples; a deployment overrides them per model, so read the live setting, not this list):
- `EMBED_INTRA_THREADS` (default `2`), `EMBED_SESSION_POOL_SIZE` — `pool_size * intra_threads ≤ cores` rule. Per-model overrides exist (e.g. `RERANKER_INTRA_THREADS`); a deployment may run a model, e5 for instance, with a different thread count than the default.
- `EMBED_MEMORY_PATTERN_<MODEL>=false` — required for jina (variable seq + ALiBi). The code default is `true`; a deployment may set it `false` for any model, and the embedder pad rule below follows each model's setting.
- `EMBED_ARENA_MAX_MEM_BYTES` (default 6 GiB) — BFCArena ceiling **per worker**; `EMBED_ARENA_MAX_MEM_BYTES_<MODEL>` overrides it for one worker.
- `ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so` — required by `ort` with `load-dynamic`.

Batcher:
- `BATCH_MAX=32`, `BATCH_MAX_TOKENS=16384`, `BATCH_MAX_SEQ=256`, `BATCH_WAIT_MS=30`, `MAX_QUEUE_SIZE=256`.
- `BATCH_LENGTH_RATIO_THRESHOLD` — (default `0.0` = disabled) when > 0.0, items whose `max_seq_len` exceeds `accum.max_len * threshold` are carried to the next batch to reduce padding waste on padded models.

Padding / sub-batching (worker path):
- Embedders pad each forward pass to the longest sequence rounded up to a multiple of 4 tokens when the model's `memory_pattern` is off (`pool::SEQ_PAD_MULTIPLE`), or of 16 when it is on (`pool::SEQ_PAD_MULTIPLE_PATTERN_ON`: ORT keeps one plan per distinct shape, so the shape count must stay bounded). Neither is a power of two any more. The live multiple is `embed_seq_pad_multiple{model}`. `EMBED_SUBBATCH_RATIO[_<MODEL>]` (default `2.0`, `<= 1.0` disables) splits a request into length-homogeneous passes.
- `RERANKER_SUBBATCH_RATIO` (default `2.0`, `<= 1.0` disables) does the same for `/v1/rerank`: pairs whose lengths differ by more than the ratio run in separate forward passes. Scores are returned in input order. Observable as `embed_subbatch_groups{model}`.

Vector semantics (both switches change the vectors a model returns, so both are per-model opt-ins, off by default):
- `EMBED_INPUT_PREFIX_<MODEL>=e5` prepends `query: ` (`input_type=query`) or `passage: ` (default) before inference; a text that already starts with either prefix is left alone, so clients that prefix themselves keep working. Without it `input_type` only partitions the response cache. Turn it on only after every store built from this server has been re-embedded: vectors with and without the prefix are not comparable.
- `EMBED_BATCH_INVARIANT_<MODEL>=1` (or the global `EMBED_BATCH_INVARIANT`) runs every text of a request in its own forward pass. The int8 graph takes each activation scale from the whole tensor, so by default a text's vector depends on the texts embedded with it (and the response cache keeps whichever vector came first). A lone text is unchanged by the switch; multi-text requests cost more, see `docs/benchmarks/2026-11-16-embed-batch-drift.md`.

## CI

Two lanes, both on GitHub-hosted `ubuntu-24.04-arm` (public repo → free + unlimited, and arm64 is what production runs).

**`preflight`** — every PR: gitleaks → osv-scanner → fmt → clippy → deny → nextest → release build → **mutants `--in-diff`**.

**`nightly`** (03:00 UTC) — `deny` + `osv-scanner` against that day's advisory feeds (a dep does not have to change to become vulnerable), plus the full suite with `--no-fail-fast`.

**`nightly` / weekly sweep** (Sundays 04:00 UTC, same workflow) — the full mutation scope sharded 16×, then a ratchet job. Weekly rather than nightly because one mutant costs 150-360s of rebuild: ~700 in-scope mutants is ~20 machine-hours per sweep. The per-PR `--in-diff` run is what protects new code daily; this lane measures and ratchets the standing debt.

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

`.cargo/mutants-baseline.txt` is the ratchet: the weekly sweep fails if the missed count rises above it, and tells you to lower it when it falls. Raising it requires a justification in the same PR. A non-numeric value is a hard error, not a pass.

**The recorded baseline is 604 missed of 882 — and that number is an ARTIFACT, not a measurement of this codebase.** It was measured while `additional_cargo_test_args` said `--lib` without `--bins`. `src/main.rs` declares six modules that `src/lib.rs` does not expose (`api_health`, `batcher`, `cache`, `cache_flow`, `token_cache`, `types`), so their tests compile into the `bin/embed-server` target — 366 tests that the gate never invoked, against the 144 it did. Six of the sixteen files in `examine_globs` are in that bucket, and their mutants were all reported MISSED for a mechanical reason.

`--bins` was added once the effect was visible: two tests that assert a gauge's value failed to kill the mutant that deletes that gauge. Until the next weekly sweep re-measures, **604 is a valid upper bound but a loose one** — adding tests can only lower the true count, so the ratchet cannot fire falsely, but it will not catch a small regression either. Lower the file to whatever the next sweep reports; do not compare against 604 as if it described the code.

Integration tests with real models require: `EMBED_MODELS=...` + `RERANKER_MODELS=...` + `SPLADE_MODELS=...` + `ORT_DYLIB_PATH=...` env. Run `--test-threads=1` (parallel OOMs on 4-core 24 GB).

## Releases

release-please on push to `main`. Conventional commits → auto PR + tag. Do not tag manually.

## Gotchas

- `ort` + `load-dynamic` → `libonnxruntime.so` from `ORT_DYLIB_PATH` at startup.
- `model_optimized.onnx` (graph-fused) is **slower** than `model_quantized.onnx` on ARM Neoverse-N1 for BERT-family. Don't switch `MODEL_FILE` without benchmarking.
- `ort-sys` downloads the ORT binary on first compile (~30 s cold).
- Batcher `carry: Option<Item>` — items overflowing `BATCH_MAX` defer to next batch. Tracked as `embed_carry_events_total`.
- Multi-process worker test parallelism: **`--test-threads=1` mandatory** for `cargo nextest run --test multi_process_*` — each test spawns its own embed-server with full model set, parallel OOMs.
- Model-backed tests (`multi_process_*_e2e`, `worker_e5_inference`, `worker_client_roundtrip`, `supervisor_dispatch`, `model::pad_call_site_tests`) run on the tiny fixtures in `tests/fixtures/tiny` (random weights: assert shape, determinism, batch independence, never relevance). Needs `ORT_DYLIB_PATH="$(scripts/fetch-ort.sh)"`. Missing runtime/fixture: SKIP locally, **panic when `CI=true`**. `EMBED_TEST_FIXTURES=<dir>` redirects the fixture root. Regenerate with `scripts/gen_test_fixtures.py`.
- krolik's cargo wrapper shares ONE target dir across all worktrees of a repo, and cargo's mtime fingerprints do not tell worktrees apart: a build or test run can silently use another worktree's artifacts. Set `CARGO_TARGET_DIR` to a private dir when the result must be about YOUR tree.
- Worker child process must call `arena::register_shared_cpu_arena()` before any `Session::builder()` — supervisor's registration doesn't carry across fork. (Handled in `src/bin/worker.rs`.)
- Per-request UDS conn (not persistent pool) — see `docs/architecture/multi-process.md` for cancel-safety rationale.
