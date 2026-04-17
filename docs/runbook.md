# embed-server Runbook

Unified Rust ONNX sidecar at `embed-server:8082` serving
`multilingual-e5-large` (1024 dim) and `jina-code-v2` (768 dim).

## Normal state

- `docker ps | grep embed-server` → `Up X (healthy)`.
- Logs contain `batching_enabled=true` and `all models loaded models=2`.
- `/metrics` has `embed_build_info{version=...}` and per-model
  `embed_requests_total{model=...,status="ok"}` growing.
- `embed_queue_rejected_total` ≈ 0.
- `embed_queue_depth{model}` small (single-digit) under steady load.

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

## Future work

- Model swap candidate: `nomic-ai/CodeRankEmbed` (137M, Apache 2.0, code
  SOTA on CodeSearchNet). No GGUF/ONNX yet — needs `optimum` conversion.
- Avoided candidates (research Apr 2026):
  - `Salesforce/SFR-Embedding-Code-400M_R` — **CC-BY-NC-4.0**, blocked.
  - `jinaai/jina-code-embeddings-0.5b` — license ambiguity post-Elastic
    acquisition; verify before use.
  - Qwen3-Embedding-0.6B — too slow on Neoverse-N1 w/o AVX/SVE
    (10-15 s per request; 6-10× slower than jina-code-v2).
