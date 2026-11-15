//! Latency benchmark for the worker inference path.
//!
//! Drives `StandaloneReranker::score` / `StandaloneEmbedder::infer` — exactly
//! what the `embed-worker` process calls per request — over a deterministic
//! corpus, and prints p50/p95/min wall-clock per scenario. Tokenization is
//! included (the worker does it too).
//!
//! ```text
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.{so,dylib} \
//! EMBED_MODELS="multilingual-e5-large:/models/e5:1024:256:1:false" \
//! RERANKER_MODELS="gte-multi-rerank:/models/gte:256:true" \
//!   cargo run --release --example bench_infer -- rerank gte-multi-rerank \
//!     --threads 2 --iters 8
//! ```
//!
//! `kind` is `rerank` or `embed`; the second argument is the model name as it
//! appears in the matching `*_MODELS` variable. Flags: `--threads N` (ORT
//! intra-op threads, default 2), `--pool N` (session pool, default 1),
//! `--iters N` (timed iterations, default 8), `--warmup N` (default 2),
//! `--only <substring>` (run scenarios whose label contains it), `--seed N`
//! (a different deterministic corpus), `--dump FILE` (write the scores /
//! vectors for offline comparison between two builds), `--alone` (with
//! `--dump`, rerank only: score each pair in its own forward pass, the
//! least-padded reference), `--par K` (embed only: split the request over K
//! threads against a `--pool K` embedder).
//!
//! The corpus is built from this repository's own docs plus a short built-in
//! Russian passage, so token density is realistic for mixed EN/RU traffic and
//! no private text is involved.

use std::time::Instant;

use embed_server::config::Config;
use embed_server::model::{StandaloneEmbedder, StandaloneReranker};

const EN_SOURCES: [&str; 3] = [
    include_str!("../README.md"),
    include_str!("../CHANGELOG.md"),
    include_str!("../docs/ROADMAP.md"),
];

const RU_PASSAGE: &str = "Перекрёстный энкодер оценивает пару из запроса и документа совместно, \
поэтому его качество выше, чем у двух независимых векторов, но стоимость растёт линейно с числом \
документов. Для ускорения на процессоре обычно квантуют веса до восьми бит, ограничивают длину \
последовательности и группируют тексты близкой длины, чтобы не считать лишние токены заполнения. \
Исследовательский конвейер сначала отбирает кандидатов дешёвым векторным поиском, затем пересортировывает \
небольшой хвост точной моделью. Задержка ответа складывается из токенизации, прямого прохода по сети \
и ожидания свободного потока; на четырёх ядрах два параллельных запроса делят одни и те же ядра, \
поэтому выигрыш даёт только сокращение числа обрабатываемых токенов. Измерения нужно повторять на \
том же железе, где работает сервис, и сравнивать медианы, а не единичные запуски.";

const QUERY_EN: &str = "how does the reranker batch documents and limit sequence length";
const QUERY_RU: &str = "как ускорить ранжирование документов на процессоре без потери качества";

/// Deterministic LCG so every run (and every commit under test) sees the same texts.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn sentences() -> Vec<String> {
    let mut out = Vec::new();
    for src in EN_SOURCES {
        for s in src.split(['.', '\n']) {
            let s = s.trim().trim_matches(|c: char| !c.is_alphanumeric());
            if s.chars().count() >= 40 && s.is_ascii() && !s.contains('|') && !s.contains("```") {
                out.push(s.to_string());
            }
        }
    }
    for s in RU_PASSAGE.split(". ") {
        let s = s.trim();
        if !s.is_empty() {
            out.push(s.to_string());
        }
    }
    out
}

/// `n` texts of roughly `chars` characters each. `chars == 0` → mixed
/// 80..1400; `chars == 1` → bimodal, alternating 200 and 1000 (snippet next to
/// a page chunk).
fn corpus(pool: &[String], n: usize, chars: usize, seed: u64) -> Vec<String> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| {
            let target = match chars {
                0 => 80 + (rng.next() as usize % 1320),
                1 => [200, 1000][i % 2],
                c => c,
            };
            let mut text = String::new();
            while text.chars().count() < target {
                let s = &pool[rng.next() as usize % pool.len()];
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(s);
                text.push('.');
            }
            text.chars().take(target).collect()
        })
        .collect()
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn arg_val(args: &[String], flag: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: bench_infer <rerank|embed> <model> [flags]"
    );
    let (kind, model) = (args[0].as_str(), args[1].as_str());
    let threads = arg_val(&args, "--threads", 2);
    let pool_size = arg_val(&args, "--pool", 1);
    let iters = arg_val(&args, "--iters", 8);
    let warmup = arg_val(&args, "--warmup", 2);
    let par = arg_val(&args, "--par", 1).max(1);
    // `--seed N` selects a different deterministic corpus (default 0).
    let seed = arg_val(&args, "--seed", 0) as u64 * 7919;
    let only = args
        .iter()
        .position(|a| a == "--only")
        .and_then(|i| args.get(i + 1))
        .cloned();
    // `--dump FILE`: also write one line per scenario — `label<TAB>v,v,...` —
    // with the scores (rerank) or the first coordinate of each vector (embed)
    // for a fixed query, so two builds/settings can be compared offline.
    let dump = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut dump_lines = String::new();

    // Same init order as src/bin/worker.rs.
    if !ort::init().commit() {
        eprintln!("ort init: environment already configured");
    }
    embed_server::arena::register_shared_cpu_arena_for_model(model)
        .map_err(|e| anyhow::anyhow!("arena: {e}"))?;
    let cfg = Config::from_env().map_err(|e| anyhow::anyhow!("config: {e}"))?;

    let pool = sentences();
    // (label, n, chars)
    let scenarios: Vec<(String, usize, usize)> = match kind {
        "rerank" => [
            (16, 200),
            (16, 1000),
            (32, 200),
            (32, 1000),
            (32, 0),
            (32, 1),
        ]
        .iter()
        .map(|&(n, c)| (label(n, c), n, c))
        .collect(),
        "embed" => [(32, 0), (32, 1000), (128, 120), (512, 120)]
            .iter()
            .map(|&(n, c)| (label(n, c), n, c))
            .collect(),
        other => anyhow::bail!("unknown kind {other}"),
    };

    let reranker = (kind == "rerank")
        .then(|| StandaloneReranker::load(model, &cfg, threads, pool_size))
        .transpose()
        .map_err(|e| anyhow::anyhow!("load reranker: {e}"))?;
    let embedder = (kind == "embed")
        .then(|| StandaloneEmbedder::load(model, &cfg, threads, pool_size))
        .transpose()
        .map_err(|e| anyhow::anyhow!("load embedder: {e}"))?;

    println!("kind,model,threads,pool,scenario,iters,min_s,p50_s,p95_s");
    for (name, n, chars) in scenarios {
        if only.as_deref().is_some_and(|o| !name.contains(o)) {
            continue;
        }
        let docs = corpus(&pool, n, chars, 0x5EED + n as u64 + chars as u64 + seed);
        let mut samples = Vec::with_capacity(iters);
        for i in 0..warmup + iters {
            let t = Instant::now();
            if let Some(r) = &reranker {
                // Alternate EN/RU query between iterations: cache-neutral, realistic.
                let q = if i % 2 == 0 { QUERY_EN } else { QUERY_RU };
                let scores = r
                    .score(q.to_string(), docs.clone(), 0)
                    .map_err(|e| anyhow::anyhow!("score: {e}"))?;
                anyhow::ensure!(scores.len() == n, "score count");
            } else if let Some(e) = &embedder {
                // `--par K` splits the request into K contiguous chunks run on
                // K threads against one embedder — what a pool of K sessions
                // allows when one request may use several of them.
                let chunk = n.div_ceil(par);
                let total: usize = std::thread::scope(|s| {
                    docs.chunks(chunk)
                        .map(|part| s.spawn(move || e.infer(part.to_vec(), 0).map(|r| r.0.len())))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .expect("infer thread")
                                .map_err(|e| anyhow::anyhow!("infer: {e}"))
                        })
                        .sum::<anyhow::Result<usize>>()
                })?;
                anyhow::ensure!(total == n, "vector count");
            }
            let dt = t.elapsed().as_secs_f64();
            if i >= warmup {
                samples.push(dt);
            }
        }
        if dump.is_some() {
            let values: Vec<f32> = if let Some(r) = &reranker {
                if args.iter().any(|a| a == "--alone") {
                    // One pair per forward pass: no batch-mates, no padding —
                    // the least-contaminated reference for the quality check.
                    let mut out = Vec::with_capacity(docs.len());
                    for d in &docs {
                        out.extend(
                            r.score(QUERY_EN.to_string(), vec![d.clone()], 0)
                                .map_err(|e| anyhow::anyhow!("score: {e}"))?,
                        );
                    }
                    out
                } else {
                    r.score(QUERY_EN.to_string(), docs.clone(), 0)
                        .map_err(|e| anyhow::anyhow!("score: {e}"))?
                }
            } else if let Some(e) = &embedder {
                // Whole vectors, flattened: cosine is computed offline.
                e.infer(docs.clone(), 0)
                    .map_err(|e| anyhow::anyhow!("infer: {e}"))?
                    .0
                    .into_iter()
                    .flatten()
                    .collect()
            } else {
                vec![]
            };
            let joined: Vec<String> = values.iter().map(|v| format!("{v:.6}")).collect();
            dump_lines.push_str(&format!("{name}\t{}\n", joined.join(",")));
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        println!(
            "{kind},{model},{threads},{pool_size},{name},{iters},{:.3},{:.3},{:.3}",
            samples[0],
            pct(&samples, 0.5),
            pct(&samples, 0.95)
        );
    }
    if let Some(path) = dump {
        std::fs::write(&path, dump_lines)?;
    }
    Ok(())
}

fn label(n: usize, chars: usize) -> String {
    if chars == 0 {
        format!("{n}docs_mixed")
    } else if chars == 1 {
        format!("{n}docs_bimodal")
    } else {
        format!("{n}docs_{chars}ch")
    }
}
