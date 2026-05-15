# Changelog

## [Unreleased]

### Features

* **worker:** `MAX_WAITERS` env var makes waiter-queue cap configurable (default: `8×pool_size`, floor 16). Resolves queue-overflow errors under large bulk-indexing bursts without requiring a pool_size bump.


## [0.6.0](https://github.com/anatolykoptev/ox-embed-server/compare/embed-server-v0.5.0...embed-server-v0.6.0) (2026-05-02)


### Features

* **otel:** distributed tracing via OTLP gRPC to Jaeger (Phase H.18) ([#25](https://github.com/anatolykoptev/ox-embed-server/issues/25)) ([fec365f](https://github.com/anatolykoptev/ox-embed-server/commit/fec365f170175996ad3f70e4c7d5dd5cd20458f9))
* **rerank:** Phase H — RERANKER_BATCH_MAX, semaphore, ModernBERT mount, token cache ([#21](https://github.com/anatolykoptev/ox-embed-server/issues/21)) ([f2092cd](https://github.com/anatolykoptev/ox-embed-server/commit/f2092cd167fd975c80424dbdd0a706907f030e64))
* **rerank:** static-shape ONNX fast-path for batch=1 calls (Phase H.20) ([#27](https://github.com/anatolykoptev/ox-embed-server/issues/27)) ([c5ac856](https://github.com/anatolykoptev/ox-embed-server/commit/c5ac856570cf19acc6f22996d2442562a72157cf))


### Bug Fixes

* **arena:** cap shared CPU arena at 3 GiB + Phase 3B spin knob + Phase 2 baseline data ([#23](https://github.com/anatolykoptev/ox-embed-server/issues/23)) ([df7b0ca](https://github.com/anatolykoptev/ox-embed-server/commit/df7b0ca2ca14acf76a853545d38109d3c5291b80))
* **arena:** re-enable memory_pattern + Phase H.17 root-cause autopsy ([#24](https://github.com/anatolykoptev/ox-embed-server/issues/24)) ([fab82e5](https://github.com/anatolykoptev/ox-embed-server/commit/fab82e5d98898f78dd0c4d0203630034f8e2ae94))
* disable ONNX memory pattern for all models — fixes unbounded arena growth ([#19](https://github.com/anatolykoptev/ox-embed-server/issues/19)) ([bc71be0](https://github.com/anatolykoptev/ox-embed-server/commit/bc71be0eaa6e6a640e27eeeec09e9664e53f346b))
* **otel:** re-attach EnvFilter to fmt subscriber (Phase H.18 hot-fix) ([#26](https://github.com/anatolykoptev/ox-embed-server/issues/26)) ([eec5cf3](https://github.com/anatolykoptev/ox-embed-server/commit/eec5cf348dc289dd67a4d521a6d38f47a72731c4))
* shared CPU arena allocator with kSameAsRequested extend strategy ([#20](https://github.com/anatolykoptev/ox-embed-server/issues/20)) ([65b85c0](https://github.com/anatolykoptev/ox-embed-server/commit/65b85c0cd8481a791951a629bfc41126cd34aeb0))

## [0.3.0](https://github.com/anatolykoptev/ox-embed-server/compare/embed-server-v0.2.0...embed-server-v0.3.0) (2026-04-18)


### Features

* **embed:** Phase A throughput warm-up (cancel-check, ORT opt, auto-truncate) ([c210bc8](https://github.com/anatolykoptev/ox-embed-server/commit/c210bc84b00dd154c97a0b01cd564b73e23f2ae0))
* **embed:** Phase B token-budget batcher (2x+ throughput on mixed loads) ([1dd856b](https://github.com/anatolykoptev/ox-embed-server/commit/1dd856b4e89b0060c23eba999544f9ede05052ab))


### Performance Improvements

* **api:** tokenize off tokio runtime ([604ed27](https://github.com/anatolykoptev/ox-embed-server/commit/604ed27f35d934830c8be8ca2fa8c923a61b3d3c))

## [0.2.0](https://github.com/anatolykoptev/ox-embed-server/compare/embed-server-v0.1.0...embed-server-v0.2.0) (2026-04-18)


### Features

* DynamicBatcher (tokio mpsc + oneshot, bounded, TDD) ([334a95d](https://github.com/anatolykoptev/ox-embed-server/commit/334a95d61b546b2a20bdf5fdde96d799cbcb4778))
* graceful SIGTERM (CancellationToken + reject-new + batcher drain) ([607f62c](https://github.com/anatolykoptev/ox-embed-server/commit/607f62c7326e48597b53bceded8e3cf44a8675f6))
* implement ONNX embedding inference with OpenAI-compatible API ([725e8fb](https://github.com/anatolykoptev/ox-embed-server/commit/725e8fb63bdd15b099a48466d33b3524f2473fe1))
* Prometheus /metrics endpoint with per-model labels ([07da0f9](https://github.com/anatolykoptev/ox-embed-server/commit/07da0f9c5fa5ede8fc975d2e8049c91f52a11c40))
* scaffold embed-server Rust project ([6ba5256](https://github.com/anatolykoptev/ox-embed-server/commit/6ba525668bafb8c0d7495f960a14b275dd004548))
* wire DynamicBatcher into api.rs behind BATCHING_ENABLED flag ([53711d6](https://github.com/anatolykoptev/ox-embed-server/commit/53711d62e1463b0195bf07c736980c6dc674ebaa))


### Bug Fixes

* **batcher:** defer coalesce-overflow items instead of dropping them ([3598b48](https://github.com/anatolykoptev/ox-embed-server/commit/3598b48779653a8a2ebbf916b47ed41b8d989733))
* simplify Dockerfile, add .dockerignore ([75ab1e7](https://github.com/anatolykoptev/ox-embed-server/commit/75ab1e7af7e06f16e79cbab6edd47f48d481572c))
* use f64 precision for pooling/norm, tokenizers 0.22, attention_mask from tokenizer ([b594abe](https://github.com/anatolykoptev/ox-embed-server/commit/b594abeaeada05a8616fd6367ea206466c063f95))
