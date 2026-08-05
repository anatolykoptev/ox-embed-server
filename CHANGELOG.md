# Changelog

## [0.8.1](https://github.com/anatolykoptev/ox-embed-server/compare/embed-server-v0.8.0...embed-server-v0.8.1) (2026-08-05)


### Bug Fixes

* **mutants:** correct the scope config and the Makefile pathspec ([#154](https://github.com/anatolykoptev/ox-embed-server/issues/154)) ([6e1ee7a](https://github.com/anatolykoptev/ox-embed-server/commit/6e1ee7a989e65cc5cfaab0999b9499c1a82c551c))

## [0.8.0](https://github.com/anatolykoptev/ox-embed-server/compare/embed-server-v0.7.0...embed-server-v0.8.0) (2026-08-04)


### Features

* **ci,metrics:** mutation testing + secret/vulnerability gates + embed_build_info pkg_version ([#143](https://github.com/anatolykoptev/ox-embed-server/issues/143)) ([f904540](https://github.com/anatolykoptev/ox-embed-server/commit/f9045402c174de2642ddca56608d210b07e80501))


### Bug Fixes

* **deps:** clear 10 advisories — openssl, opentelemetry, serial_test ([#144](https://github.com/anatolykoptev/ox-embed-server/issues/144)) ([2b2e8dc](https://github.com/anatolykoptev/ox-embed-server/commit/2b2e8dc64a0b9b30060684cdaefa86037f1b8608))
* **heartbeat,ready:** stop killing healthy workers; count error replies as failures ([#146](https://github.com/anatolykoptev/ox-embed-server/issues/146)) ([7b217d3](https://github.com/anatolykoptev/ox-embed-server/commit/7b217d32c21980cc8118fd3914920e9eef446439))

## [0.7.0](https://github.com/anatolykoptev/ox-embed-server/compare/embed-server-v0.6.0...embed-server-v0.7.0) (2026-07-27)


### Features

* **api_rerank:** cap /v1/rerank documents array (close BUG-004 recurrence gap) ([#52](https://github.com/anatolykoptev/ox-embed-server/issues/52)) ([fb66e80](https://github.com/anatolykoptev/ox-embed-server/commit/fb66e807b3e01a133c8750df5e7a40b83b207821))
* **api,batcher:** cap /v1/embeddings input array + observe first-item-oversize ([#49](https://github.com/anatolykoptev/ox-embed-server/issues/49)) ([4b5b72e](https://github.com/anatolykoptev/ox-embed-server/commit/4b5b72ea729f30b106fea64ad8bd7f5e3dfddf84))
* **arena:** per-run BFCArena shrinkage on memory_pattern=false models (Phase A fragmentation fix) ([#54](https://github.com/anatolykoptev/ox-embed-server/issues/54)) ([e50a937](https://github.com/anatolykoptev/ox-embed-server/commit/e50a9376bd18f06bf12d81c909ba99d4d3572aa8))
* **batcher,config:** per-model seq cap + warmup_seq_len + solo overflow counter ([#41](https://github.com/anatolykoptev/ox-embed-server/issues/41)) ([3ebae23](https://github.com/anatolykoptev/ox-embed-server/commit/3ebae239e80a22f0f2f2b21cacea90b0ee52216f))
* **config:** optional 7th segment in EMBED_MODELS for custom ONNX filename ([#86](https://github.com/anatolykoptev/ox-embed-server/issues/86)) ([d1d4a52](https://github.com/anatolykoptev/ox-embed-server/commit/d1d4a5237d92a5992463e67580dfc69accfb9638))
* **config:** per-model pool size + arena override + worker metrics ([#74](https://github.com/anatolykoptev/ox-embed-server/issues/74)) ([96ca713](https://github.com/anatolykoptev/ox-embed-server/commit/96ca7139f55e2266d7f8021532af72443afcf1fa))
* **docker:** add sccache to builder stage for cross-build cache hits ([#76](https://github.com/anatolykoptev/ox-embed-server/issues/76)) ([e12efe6](https://github.com/anatolykoptev/ox-embed-server/commit/e12efe6902106168ec2be35cd85b7f66189b6b43))
* **embed:** per-worker RSS gauge metric (embed_worker_rss_bytes{model}) ([#81](https://github.com/anatolykoptev/ox-embed-server/issues/81)) ([84626f3](https://github.com/anatolykoptev/ox-embed-server/commit/84626f31f4214e1fe7861e734fe484710ac40951))
* **embed:** session pool + intra=2 default + power-of-2 seq pad (Phase 1 perf) ([#38](https://github.com/anatolykoptev/ox-embed-server/issues/38)) ([2f68f48](https://github.com/anatolykoptev/ox-embed-server/commit/2f68f48f06a77d03f739e099e62edbcafee5cac6))
* **jina:** precompute ALiBi const + Slice to eliminate 1.258 GiB per-call scratch ([#46](https://github.com/anatolykoptev/ox-embed-server/issues/46)) ([1077095](https://github.com/anatolykoptev/ox-embed-server/commit/1077095c97733c8aa1ee7f637349b8cf9660a2c8))
* **main:** glibc malloc_trim(0) background task on Linux ([#37](https://github.com/anatolykoptev/ox-embed-server/issues/37)) ([226fcfd](https://github.com/anatolykoptev/ox-embed-server/commit/226fcfd105d80383ef12477ab308340485e04a98))
* **metrics:** add observability counters for silent failures ([#96](https://github.com/anatolykoptev/ox-embed-server/issues/96), [#100](https://github.com/anatolykoptev/ox-embed-server/issues/100), [#101](https://github.com/anatolykoptev/ox-embed-server/issues/101)) ([#138](https://github.com/anatolykoptev/ox-embed-server/issues/138)) ([71e1d60](https://github.com/anatolykoptev/ox-embed-server/commit/71e1d60d7db6ace94f0b9da3fac2844bf3743da8))
* **metrics:** forensic arena + inference metrics to localise 1.258 GiB OOM ([#39](https://github.com/anatolykoptev/ox-embed-server/issues/39)) ([10a2a3d](https://github.com/anatolykoptev/ox-embed-server/commit/10a2a3dac225eb6c82c64b16462a2732f2707bc2))
* **metrics:** make jina-code-v2 queue-wait observable (split from inference time) ([#84](https://github.com/anatolykoptev/ox-embed-server/issues/84)) ([402901b](https://github.com/anatolykoptev/ox-embed-server/commit/402901bd85289a4853ff7c81d06e537e1b0cab60))
* **model:** mlock model weights to prevent swap-out (closes cold-load page-fault gap) ([#78](https://github.com/anatolykoptev/ox-embed-server/issues/78)) ([2790f47](https://github.com/anatolykoptev/ox-embed-server/commit/2790f471c53227298abf94a635d1d8946a2e2803))
* **pool:** port EvictablePool from ox-whisper — opt-in ONNX session idle eviction ([#48](https://github.com/anatolykoptev/ox-embed-server/issues/48)) ([6d78720](https://github.com/anatolykoptev/ox-embed-server/commit/6d787207ddfc7b3c49fb9bb8c8a56f5598fce100))
* **refactor:** Phase 1 — IPC scaffold + worker binary (multi-process) ([#56](https://github.com/anatolykoptev/ox-embed-server/issues/56)) ([841a447](https://github.com/anatolykoptev/ox-embed-server/commit/841a4473ad2fcb95e391893b95b00062553de82d))
* **refactor:** Phase 2 — supervisor cutover (embed routing via worker_pool) ([#57](https://github.com/anatolykoptev/ox-embed-server/issues/57)) ([54dd4ad](https://github.com/anatolykoptev/ox-embed-server/commit/54dd4add1a7d7a502aff25adc64f0d1dba717220))
* **refactor:** Wave 2.4b — rerank + splade routing through worker_pool ([#58](https://github.com/anatolykoptev/ox-embed-server/issues/58)) ([cdc727d](https://github.com/anatolykoptev/ox-embed-server/commit/cdc727d81b35a2c0be5efb42b8814cd26bd024b4))
* **scripts:** static-shape ONNX export + equivalence gate for code-rank-embed (Phase 1) ([#87](https://github.com/anatolykoptev/ox-embed-server/issues/87)) ([82cdb95](https://github.com/anatolykoptev/ox-embed-server/commit/82cdb95d68d17074ee0cadb6d5387a7506e8cf07))
* **security:** add cargo-deny + adopt nextest ([#55](https://github.com/anatolykoptev/ox-embed-server/issues/55)) ([a5f8e4e](https://github.com/anatolykoptev/ox-embed-server/commit/a5f8e4e64ea7476dd638146b0288daf1144004d7))
* **supervisor:** EMBED_WORKER_SPAWN_DELAY_MS + rebase on main + fmt/clippy gate fixes ([#124](https://github.com/anatolykoptev/ox-embed-server/issues/124)) ([ad50fd7](https://github.com/anatolykoptev/ox-embed-server/commit/ad50fd75e05efd68e985012860d11aac3ee2a9a9))
* Wave 3.2 — skip in-process model loading when EMBED_MULTI_PROCESS=1 ([#68](https://github.com/anatolykoptev/ox-embed-server/issues/68)) ([8439130](https://github.com/anatolykoptev/ox-embed-server/commit/8439130233733026b51514270e1a60848fe91ad6))
* **worker:** configurable MAX_WAITERS env for queue overflow tuning ([#72](https://github.com/anatolykoptev/ox-embed-server/issues/72)) ([9935eee](https://github.com/anatolykoptev/ox-embed-server/commit/9935eee2d6572f3ac70d62b7bb965ddd0cabc31c))
* **worker:** per-model EMBED_MAX_WAITERS_&lt;KEY&gt;, queue-depth gauge, [model] log prefix ([d8dfad6](https://github.com/anatolykoptev/ox-embed-server/commit/d8dfad6b5f052b4df59ad30352550911e37ab95c))
* **worker:** per-model EMBED_MAX_WAITERS_&lt;KEY&gt;, queue-depth gauge, worker log prefix ([40b337f](https://github.com/anatolykoptev/ox-embed-server/commit/40b337f94aac1ad4fc224c363bbb15853bfef3db))


### Bug Fixes

* add /ready endpoint with real inference probe ([#89](https://github.com/anatolykoptev/ox-embed-server/issues/89)) ([#133](https://github.com/anatolykoptev/ox-embed-server/issues/133)) ([f793dff](https://github.com/anatolykoptev/ox-embed-server/commit/f793dffb986b0abf0ae06fa4a090767976539147))
* add heartbeat liveness probe to detect wedged workers ([#90](https://github.com/anatolykoptev/ox-embed-server/issues/90)) ([#134](https://github.com/anatolykoptev/ox-embed-server/issues/134)) ([29f1818](https://github.com/anatolykoptev/ox-embed-server/commit/29f18184d4a09c3a3ce1cac7135999dc63aaf9db))
* **arena+cache:** V2 API + LRU policy + DisableCpuMemArena (FU-26+FU-27) ([#34](https://github.com/anatolykoptev/ox-embed-server/issues/34)) ([8c0db55](https://github.com/anatolykoptev/ox-embed-server/commit/8c0db558eefea511869b38aeccb32b5c05a7a033))
* **arena:** per-model memory_pattern knob — disable for jina-code-v2 ([#44](https://github.com/anatolykoptev/ox-embed-server/issues/44)) ([b7b2cfc](https://github.com/anatolykoptev/ox-embed-server/commit/b7b2cfcfc83e1879260503e9f50378a1ac925df4))
* **arena:** runtime assert that arena registered before any Session::builder() ([#45](https://github.com/anatolykoptev/ox-embed-server/issues/45)) ([a9449c5](https://github.com/anatolykoptev/ox-embed-server/commit/a9449c5e3220b2327d7aa5b0af47caa231b1ba07))
* **batcher:** abort worker on shutdown timeout to drop carry item ([#91](https://github.com/anatolykoptev/ox-embed-server/issues/91)) ([#109](https://github.com/anatolykoptev/ox-embed-server/issues/109)) ([7a57d62](https://github.com/anatolykoptev/ox-embed-server/commit/7a57d6254a008865312d59dbfff7549edcdb7121))
* **batcher:** off-by-one in queue-full tests caused infinite hang ([#123](https://github.com/anatolykoptev/ox-embed-server/issues/123)) ([#125](https://github.com/anatolykoptev/ox-embed-server/issues/125)) ([2429420](https://github.com/anatolykoptev/ox-embed-server/commit/24294209d96acce7b41d70a605d2612c72ef2ca5))
* **ci:** replace actions/cache with Swatinem/rust-cache (target dir corruption) ([#142](https://github.com/anatolykoptev/ox-embed-server/issues/142)) ([64f3f62](https://github.com/anatolykoptev/ox-embed-server/commit/64f3f6272e64e73da131455609a64ee2a99b98dd))
* **docker:** stub all three Cargo targets in dep-only Layer 1 ([#59](https://github.com/anatolykoptev/ox-embed-server/issues/59)) ([413bc7b](https://github.com/anatolykoptev/ox-embed-server/commit/413bc7bdcaa1d146a8cf4cbd75eba1e9d54932e5))
* **docker:** touch src/lib.rs to invalidate stub artifact in Layer 2 ([#60](https://github.com/anatolykoptev/ox-embed-server/issues/60)) ([866f85e](https://github.com/anatolykoptev/ox-embed-server/commit/866f85e2a3c089d13aa7ffdd1abe640a214a3f51))
* **heartbeat:** dispatch correct probe kind per worker ([#90](https://github.com/anatolykoptev/ox-embed-server/issues/90)) ([#135](https://github.com/anatolykoptev/ox-embed-server/issues/135)) ([d2c2842](https://github.com/anatolykoptev/ox-embed-server/commit/d2c2842fd822818d323fec7b55f8c26273204f0c))
* **ipc:** per-request UDS conn — cancel-safe WorkerClient ([#62](https://github.com/anatolykoptev/ox-embed-server/issues/62)) ([831f13a](https://github.com/anatolykoptev/ox-embed-server/commit/831f13a5f2bf6b62ee9f4aeb05fcff4033eaaaaf))
* **memory:** re-enable memory_pattern + bound warmup + per-batch max_seq cap ([#36](https://github.com/anatolykoptev/ox-embed-server/issues/36)) ([20dfd76](https://github.com/anatolykoptev/ox-embed-server/commit/20dfd7652b890a2ecc4e6d201359d75d38bfc382))
* **metrics:** classify worker errors via reason label, stop leaking raw messages ([#69](https://github.com/anatolykoptev/ox-embed-server/issues/69)) ([61bb511](https://github.com/anatolykoptev/ox-embed-server/commit/61bb511dee69ee79140c92e8f0b4f717dfc0d90a))
* **metrics:** install Prometheus recorder before arena registration (FU-28) ([#35](https://github.com/anatolykoptev/ox-embed-server/issues/35)) ([6836cea](https://github.com/anatolykoptev/ox-embed-server/commit/6836ceac65d47c910e2356a834fa8011c48c6bee))
* **ort:** with_intra_op_spinning(false) + with_inter_threads(1) on all model kinds ([#131](https://github.com/anatolykoptev/ox-embed-server/issues/131)) ([5737442](https://github.com/anatolykoptev/ox-embed-server/commit/57374425d508878185cbfe69981d0cd662840d10))
* **pool,api:** recover item on poisoned EvictablePool drop + 503 on dispatch timeout ([#94](https://github.com/anatolykoptev/ox-embed-server/issues/94), [#97](https://github.com/anatolykoptev/ox-embed-server/issues/97)) ([#137](https://github.com/anatolykoptev/ox-embed-server/issues/137)) ([94af820](https://github.com/anatolykoptev/ox-embed-server/commit/94af8206dcf03aae7e0384eb0ce8e03b8f0841dc))
* **worker:** bind /metrics on 0.0.0.0 by default; expose EMBED_WORKER_METRICS_BIND env ([#77](https://github.com/anatolykoptev/ox-embed-server/issues/77)) ([05d3c16](https://github.com/anatolykoptev/ox-embed-server/commit/05d3c164a665b1816a753eae031432c159075754))
* **worker:** bounded-queue admission via acquire().await ([#70](https://github.com/anatolykoptev/ox-embed-server/issues/70)) ([6bf2bfa](https://github.com/anatolykoptev/ox-embed-server/commit/6bf2bfa7315b7046ec4e491f0e82aec8fc47e7b4))
* **worker:** classify post-[#70](https://github.com/anatolykoptev/ox-embed-server/issues/70) errors + bound waiter queue ([#71](https://github.com/anatolykoptev/ox-embed-server/issues/71)) ([7713a8f](https://github.com/anatolykoptev/ox-embed-server/commit/7713a8fc5b539c7abdffc87590a37a9df7f06e15))
* **worker:** fail startup on arena registration failure ([#92](https://github.com/anatolykoptev/ox-embed-server/issues/92)) ([#110](https://github.com/anatolykoptev/ox-embed-server/issues/110)) ([a35f84f](https://github.com/anatolykoptev/ox-embed-server/commit/a35f84f56ed16a28bf5533c23263e118c1051fd6))
* **worker:** restore warmup + auto-bind /metrics (regression from PR [#68](https://github.com/anatolykoptev/ox-embed-server/issues/68) Wave 3.2) ([#75](https://github.com/anatolykoptev/ox-embed-server/issues/75)) ([881b349](https://github.com/anatolykoptev/ox-embed-server/commit/881b3494f32a228bfa225ff0186084c472a24e0b))


### Performance Improvements

* **deps:** bump tokenizers 0.22.2 → 0.23.1 ([#141](https://github.com/anatolykoptev/ox-embed-server/issues/141)) ([86e9eed](https://github.com/anatolykoptev/ox-embed-server/commit/86e9eedab5d5abe3ee7e5691332693fa3d8b5051))
* hot-path quick wins (supervisor tokenize elim, tti conditional, clone reduce) ([#65](https://github.com/anatolykoptev/ox-embed-server/issues/65)) ([a6acf8b](https://github.com/anatolykoptev/ox-embed-server/commit/a6acf8bdde882163d0f7c957496c6d6396c82cc6))
* **main:** spawn embed/rerank/splade workers in parallel via tokio::spawn ([#61](https://github.com/anatolykoptev/ox-embed-server/issues/61)) ([bfcc4a9](https://github.com/anatolykoptev/ox-embed-server/commit/bfcc4a924a1df884aa7425a90d179024fcef650d))
* **pool,batcher:** SIMD mean_pool_normalize + length-ratio carry gate ([#128](https://github.com/anatolykoptev/ox-embed-server/issues/128), [#129](https://github.com/anatolykoptev/ox-embed-server/issues/129)) ([#132](https://github.com/anatolykoptev/ox-embed-server/issues/132)) ([02db471](https://github.com/anatolykoptev/ox-embed-server/commit/02db471d7ff61cf3d0ae46a142010445b03315c0))
* **splade:** thread-local sparse buffer + select_nth top-k ([#67](https://github.com/anatolykoptev/ox-embed-server/issues/67)) ([57c5bdb](https://github.com/anatolykoptev/ox-embed-server/commit/57c5bdb7c5db05181d15d67688762cad4719ec1c))

## [Unreleased]

### Features

* **worker:** `EMBED_MAX_WAITERS` env var makes waiter-queue cap configurable (default: `8×pool_size`, floor 16). Resolves queue-overflow errors under large bulk-indexing bursts without requiring a pool_size bump.


## [0.6.0](https://github.com/ox/embed-server/compare/embed-server-v0.5.0...embed-server-v0.6.0) (2026-05-02)


### Features

* **otel:** distributed tracing via OTLP gRPC to Jaeger (Phase H.18) ([#25](https://github.com/ox/embed-server/issues/25)) ([fec365f](https://github.com/ox/embed-server/commit/fec365f170175996ad3f70e4c7d5dd5cd20458f9))
* **rerank:** Phase H — RERANKER_BATCH_MAX, semaphore, ModernBERT mount, token cache ([#21](https://github.com/ox/embed-server/issues/21)) ([f2092cd](https://github.com/ox/embed-server/commit/f2092cd167fd975c80424dbdd0a706907f030e64))
* **rerank:** static-shape ONNX fast-path for batch=1 calls (Phase H.20) ([#27](https://github.com/ox/embed-server/issues/27)) ([c5ac856](https://github.com/ox/embed-server/commit/c5ac856570cf19acc6f22996d2442562a72157cf))


### Bug Fixes

* **arena:** cap shared CPU arena at 3 GiB + Phase 3B spin knob + Phase 2 baseline data ([#23](https://github.com/ox/embed-server/issues/23)) ([df7b0ca](https://github.com/ox/embed-server/commit/df7b0ca2ca14acf76a853545d38109d3c5291b80))
* **arena:** re-enable memory_pattern + Phase H.17 root-cause autopsy ([#24](https://github.com/ox/embed-server/issues/24)) ([fab82e5](https://github.com/ox/embed-server/commit/fab82e5d98898f78dd0c4d0203630034f8e2ae94))
* disable ONNX memory pattern for all models — fixes unbounded arena growth ([#19](https://github.com/ox/embed-server/issues/19)) ([bc71be0](https://github.com/ox/embed-server/commit/bc71be0eaa6e6a640e27eeeec09e9664e53f346b))
* **otel:** re-attach EnvFilter to fmt subscriber (Phase H.18 hot-fix) ([#26](https://github.com/ox/embed-server/issues/26)) ([eec5cf3](https://github.com/ox/embed-server/commit/eec5cf348dc289dd67a4d521a6d38f47a72731c4))
* shared CPU arena allocator with kSameAsRequested extend strategy ([#20](https://github.com/ox/embed-server/issues/20)) ([65b85c0](https://github.com/ox/embed-server/commit/65b85c0cd8481a791951a629bfc41126cd34aeb0))

## [0.3.0](https://github.com/ox/embed-server/compare/embed-server-v0.2.0...embed-server-v0.3.0) (2026-04-18)


### Features

* **embed:** Phase A throughput warm-up (cancel-check, ORT opt, auto-truncate) ([c210bc8](https://github.com/ox/embed-server/commit/c210bc84b00dd154c97a0b01cd564b73e23f2ae0))
* **embed:** Phase B token-budget batcher (2x+ throughput on mixed loads) ([1dd856b](https://github.com/ox/embed-server/commit/1dd856b4e89b0060c23eba999544f9ede05052ab))


### Performance Improvements

* **api:** tokenize off tokio runtime ([604ed27](https://github.com/ox/embed-server/commit/604ed27f35d934830c8be8ca2fa8c923a61b3d3c))

## [0.2.0](https://github.com/ox/embed-server/compare/embed-server-v0.1.0...embed-server-v0.2.0) (2026-04-18)


### Features

* DynamicBatcher (tokio mpsc + oneshot, bounded, TDD) ([334a95d](https://github.com/ox/embed-server/commit/334a95d61b546b2a20bdf5fdde96d799cbcb4778))
* graceful SIGTERM (CancellationToken + reject-new + batcher drain) ([607f62c](https://github.com/ox/embed-server/commit/607f62c7326e48597b53bceded8e3cf44a8675f6))
* implement ONNX embedding inference with OpenAI-compatible API ([725e8fb](https://github.com/ox/embed-server/commit/725e8fb63bdd15b099a48466d33b3524f2473fe1))
* Prometheus /metrics endpoint with per-model labels ([07da0f9](https://github.com/ox/embed-server/commit/07da0f9c5fa5ede8fc975d2e8049c91f52a11c40))
* scaffold embed-server Rust project ([6ba5256](https://github.com/ox/embed-server/commit/6ba525668bafb8c0d7495f960a14b275dd004548))
* wire DynamicBatcher into api.rs behind BATCHING_ENABLED flag ([53711d6](https://github.com/ox/embed-server/commit/53711d62e1463b0195bf07c736980c6dc674ebaa))


### Bug Fixes

* **batcher:** defer coalesce-overflow items instead of dropping them ([3598b48](https://github.com/ox/embed-server/commit/3598b48779653a8a2ebbf916b47ed41b8d989733))
* simplify Dockerfile, add .dockerignore ([75ab1e7](https://github.com/ox/embed-server/commit/75ab1e7af7e06f16e79cbab6edd47f48d481572c))
* use f64 precision for pooling/norm, tokenizers 0.22, attention_mask from tokenizer ([b594abe](https://github.com/ox/embed-server/commit/b594abeaeada05a8616fd6367ea206466c063f95))
