# embed-server Runbook

Unified Rust ONNX sidecar at `embed-server:8082` serving 4 models:
- `multilingual-e5-large` (1024 dim, embed)
- `jina-code-v2` (768 dim, embed)
- `gte-multi-rerank` (cross-encoder, rerank)
- `splade-v3-distilbert` (sparse, splade)

Since 2026-05-12: **multi-process** via `EMBED_MULTI_PROCESS=1`. Supervisor (`embed-server` PID 1) spawns one `embed-worker` child per model. Workers communicate over UDS sockets in `/tmp/embed-workers/`. See `CLAUDE.md` process-model section.

## Normal state

- `docker ps | grep embed-server` → `Up X (healthy)`.
- `docker top embed-server` shows 5 processes: 1× `embed-server` (PID 1) + 4× `embed-worker` (one per model).
- Logs contain `multi-process mode enabled — spawning worker pool` + 4× `worker handle ready`.
- `/metrics` has `embed_build_info{version="phase-2-multi-process"}`, `embed_worker_restart_total{model=...} 0` (pre-touched), per-model `embed_requests_total{model=...,status="ok"}` growing.
- `embed_queue_rejected_total` ≈ 0 (in-process queue still present; less relevant when worker_pool routes hot path).
- No `worker_pool ... dispatch failed` errors in logs.

## Multi-process specific symptoms

### `embed_worker_restart_total{model=X}` > 0

Worker crashed and watchdog respawned. Check exit reason in logs:
```bash
docker logs embed-server 2>&1 | grep "worker exited" | tail -5
```
Look for `code`, `signal` fields. `signal=Some(6)` = SIGABRT (panic in worker, `panic=abort` profile). `signal=Some(9)` or `code=None` with high RSS = OOM-kill. Counter > 5/hour for any model = systemic issue, escalate.

### `worker_pool ... dispatch failed: response id X != request id Y`

**Was the rerank regression fixed in PR #62** (per-request UDS conn, cancel-safe). If returns: rollback to image preceding PR #62 NOT possible — code path removed. Instead: check if downstream client (memdb-go) has aggressive request timeouts triggering cancellation cascade. Increase client timeout or reduce ONNX inference latency.

### Worker startup timeout (60s `did not create socket within 60s`)

Worker process started but failed to bind UDS. Likely causes:
1. Missing model file at `EMBED_MODELS` path.
2. ORT dylib mismatch (`ORT_DYLIB_PATH` wrong).
3. Arena registration failed — check `shared arena registration failed` warn in logs.

Watchdog will retry indefinitely with exponential backoff (2s→60s). Per CLAUDE.md "no silent errors" — failure is logged with full error context.

### Rolling back multi-process

```bash
# In ~/deploy/krolik-server/compose/memdb.yml, set:
EMBED_MULTI_PROCESS: "0"
# Then:
docker compose up -d --no-deps --force-recreate embed-server
```
Behaviour returns byte-identical to pre-2026-05-12 monolith path. Workers will not spawn; in-process `EmbedModel`/`RerankerModel`/`SpladeModel` handle inference.

## Symptom → response

### High latency (p90 > 2 s sustained)

1. Scrape `/metrics` — compare `embed_inference_duration_seconds_bucket`
   (pure ONNX) vs `embed_request_duration_seconds_bucket` (queue + ONNX).
   Queue-heavy difference points to a hot caller.
2. `embed_queue_depth{model}` > 200 → caller is bursting. Options: raise
   `MAX_QUEUE_SIZE`, tune `BATCH_MAX` (cautiously — >8 hurts on ARM), or
   rate-limit the caller.
3. CPU check: `docker stats embed-server` or
   `top -p $(docker inspect -f '{{.State.Pid}}' embed-server)`. Pegged at
   400 % under load is expected (4 vCPU × intra_op_threads=4).

### 503 rate climbing

- Queue-full 503: raise `MAX_QUEUE_SIZE` (compose env, recreate — no rebuild).
- Shutdown 503: check `docker logs embed-server | grep SIGTERM` — only
  emits during graceful shutdown. If you see it at runtime, orchestrator
  is sending SIGTERM unexpectedly.

### Container unhealthy / flapping

1. `docker logs embed-server --tail 50` — look for model-load errors.
2. Typical issue: `model_quantized.onnx not found in /models-jina` →
   volume mount wrong. Current prod mounts:
   - `/home/krolik/deploy/krolik-server/models/multilingual-e5-large:/models:ro`
   - `/home/krolik/deploy/krolik-server/models/jina-code-v2:/models-jina:ro`
3. If inference errors in logs: likely tokenizer/model mismatch. Check
   model dir contents match `EMBED_MODELS` definition.

### Regression after a change

Run `rtk proxy python3 bench.py --url http://<ip>:8082/v1/embeddings --model <model>`.
Compare to `docs/benchmarks/2026-04-16-baseline.md`.

- Sequential regressed: suspect model/ORT session options.
- Concurrent regressed but c=1 fine: batcher issue — try
  `BATCHING_ENABLED=false` temporarily.

## Rollbacks

All via env only (no rebuild):

### Disable batcher, keep other improvements

```bash
# Edit compose: BATCHING_ENABLED="false"
cd ~/deploy/krolik-server && docker compose up -d --no-deps --force-recreate embed-server
```

Reverts to legacy lock path (one inference at a time via Mutex<Session>).
Loses coalescing + backpressure but keeps metrics + /health + graceful SIGTERM.

### Full revert to single-model (pre-migration)

```bash
# Edit compose/memdb.yml embed-server:
#   EMBED_MODELS: "multilingual-e5-large:/models:1024:256:1:false"
# Remove jina-code-v2 volume mount.
docker compose up -d --no-deps --force-recreate embed-server
```

Then re-deploy the old Python sidecar from the archive:
`git clone git@github.com:anatolykoptev/ox-embed-jina.git ~/src/embed-jina &&
git -C ~/src/embed-jina checkout retired-2026-04-17` — this restores the
full pre-retirement state (batcher, metrics, graceful shutdown all intact).
No rebuild of embed-server needed.

### Revert to earlier commit

Every per-task commit is rebuild-safe. Checkpoint commits:

- `77d83bb` — pre-R2 baseline (bench harness added)
- `07da0f9` — + metrics
- `334a95d` — + batcher (unit tests only)
- `53711d6` — + batcher wired into handler
- `29a9082` — + Dockerfile optimization
- `607f62c` — + graceful SIGTERM

```bash
cd ~/src/embed-server
git checkout <sha> -- src/
cd ~/deploy/krolik-server && docker compose build embed-server && docker compose up -d --force-recreate embed-server
```

## Prometheus alert recipes

Not auto-deployed. Add manually:

```yaml
groups:
- name: embed-server
  rules:
  - alert: EmbedServerHighLatency
    expr: histogram_quantile(0.9, rate(embed_request_duration_seconds_bucket[5m])) > 2
    for: 5m
  - alert: EmbedServerQueueRejections
    expr: rate(embed_queue_rejected_total[5m]) > 0
    for: 2m
  - alert: EmbedServerHighErrorRate
    expr: rate(embed_requests_total{status="error"}[5m]) / ignoring(status) rate(embed_requests_total[5m]) > 0.01
    for: 5m
  - alert: EmbedServerModelMissing
    expr: count(embed_build_info) < 1
    for: 1m
```

## Known quirks

- `read_only: true` + `tmpfs: /tmp` — writes land in tmpfs only.
- Container IP changes on each `--force-recreate`. Callers inside the
  network resolve via `embed-server` hostname.
- Healthcheck uses `curl -sf http://localhost:8082/health` — fails if
  embed-server crashed mid-inference but TCP listener still alive.
- Deploy rebuild: ~40 s code-change, ~2 s no-change, ~3 min cold. If a
  build is taking 40 minutes, check `# syntax=docker/dockerfile:1.4` is
  the first line and `--mount=type=cache,target=/...` directives intact.

## Static-shape reranker fast-path (Phase H.20 + 2026-05-02 multi-shape)

The reranker loader scans its model dir for static-shape ONNX siblings
and routes `score_pairs` calls whose batch size exactly matches an
exported shape through the matching session pool. Exact-match only —
no pad-up, no fallback shape coercion. Misses fall through to the
dynamic pool unchanged.

**Filename convention** (`src/model_reranker/load.rs`):

| File                               | Activates             |
|------------------------------------|-----------------------|
| `model_quantized_static_b1.onnx`   | batch=1 fast path     |
| `model_quantized_static_b5.onnx`   | batch=5 fast path     |
| `model_quantized_static_b<N>.onnx` | batch=N fast path     |
| `model_quantized_static.onnx`      | batch=1 (PR #27 legacy, kept for backwards compat) |

If both `model_quantized_static_b1.onnx` and `model_quantized_static.onnx`
exist, the explicit `_b1` file wins and a `warn` log flags the
duplicate.

### Adding a new shape

1. Run `scripts/export_static_modernbert.sh` on a dev box (NOT on
   krolik — the export saturates CPU and disk for tens of minutes):
   ```bash
   BATCH_SIZES="1 5" ./scripts/export_static_modernbert.sh
   ```
2. rsync the resulting files into krolik's model dir:
   ```bash
   rsync -avh ./gte-reranker-modernbert-base-static/model_quantized_static_b*.onnx \
     krolik:/home/krolik/deploy/krolik-server/models/gte-reranker-modernbert-base/
   ```
3. Recreate the embed-server container:
   ```bash
   ssh krolik 'cd ~/deploy/krolik-server && docker compose up -d --no-deps --force-recreate embed-server'
   ```
4. Confirm the new shapes loaded:
   ```bash
   ssh krolik 'docker logs embed-server --since 1m | grep "static-shape ONNX"'
   ```
   One info line per shape: `loading static-shape ONNX fast-path
   session pool batch=N path=...`.

### Validating impact

Use `bench.py` to compare before/after — pin the same `(model, size,
docs_per_req)` knobs across the two runs. PR #27's prod measurement
showed ~1.04× uplift over dynamic on quantized int8 ModernBERT at
batch=1, vs 1.74× standalone — the int8 quant/dequant cost absorbs
most of the static-graph win. Validate at batch=5 before promising
the optimization to memdb-go. A negative result is fine; revert is
`rm` on the static file.

### Memory budget

Each ModernBERT static session ~255 MiB; pool size 2 = ~510 MiB per
shape. Default {b=1} = ~1 GiB total. With {b=1, b=5} = ~1.5 GiB.
Container shared CPU arena cap is 3 GiB (Phase H.16), with embed +
SPLADE consuming the rest. Don't enable more than {1, 5} without
re-doing the budget math.

## Future work

- Model swap candidate: `nomic-ai/CodeRankEmbed` (137M, Apache 2.0, code
  SOTA on CodeSearchNet). No GGUF/ONNX yet — needs `optimum` conversion.
- Avoided candidates (research Apr 2026):
  - `Salesforce/SFR-Embedding-Code-400M_R` — **CC-BY-NC-4.0**, blocked.
  - `jinaai/jina-code-embeddings-0.5b` — license ambiguity post-Elastic
    acquisition; verify before use.
  - Qwen3-Embedding-0.6B — too slow on Neoverse-N1 w/o AVX/SVE
    (10-15 s per request; 6-10× slower than jina-code-v2).
