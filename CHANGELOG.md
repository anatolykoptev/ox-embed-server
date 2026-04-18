# Changelog

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
