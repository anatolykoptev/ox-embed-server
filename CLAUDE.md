# embed-server — ONNX Embedding Server

**Rust** | Docker container | Provides OpenAI-compatible `/v1/embeddings` API

## Structure

| File | Role |
|------|------|
| `src/main.rs` | Axum server, routes |
| `src/api.rs` | `/v1/embeddings` handler |
| `src/model.rs` | ONNX model loading + inference |
| `src/pool.rs` | Token pooling (mean/cls) |
| `src/config.rs` | Environment config |

## API

`POST /v1/embeddings` — OpenAI-compatible. Input: text or array of texts. Output: embeddings.

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_PATH` | required | Path to ONNX model directory |
| `PORT` | `8083` | Listen port |

## Deploy

```bash
cd ~/deploy/krolik-server
docker compose build --no-cache embed-server && docker compose up -d --no-deps --force-recreate embed-server
```

## Gotchas

- Uses `ort` with `load-dynamic` — ONNX runtime loaded at startup, not linked
- Model: `multilingual-e5-large` (1024 dim) for MemDB, `jina-code-v2` (768 dim) via separate `embed-jina`
