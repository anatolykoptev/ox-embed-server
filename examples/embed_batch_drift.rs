//! How much does a text's vector move with the texts embedded beside it?
//!
//! Embeds a target set of texts alone (one request each, the reference), then
//! again inside requests of different shapes, and prints the cosine between
//! the two vectors per scenario. Runs the same `StandaloneEmbedder::infer`
//! path the worker serves.
//!
//! ```text
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
//! EMBED_MODELS="multilingual-e5-large:/models/e5:1024:256:1:false" \
//! EMBED_MEMORY_PATTERN_MULTILINGUAL_E5_LARGE=false \
//!   cargo run --release --example embed_batch_drift -- multilingual-e5-large
//! ```
//!
//! Flags: `--targets N` (default 48), `--threads N` (default 2).
//! `EMBED_SUBBATCH_RATIO=0` runs every request as one forward pass;
//! `EMBED_BATCH_INVARIANT=1` runs every text in its own pass.
use embed_server::config::Config;
use embed_server::model::StandaloneEmbedder;

const EN_SOURCES: [&str; 3] = [
    include_str!("../README.md"),
    include_str!("../CHANGELOG.md"),
    include_str!("../docs/ROADMAP.md"),
];

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
    out
}

/// A text of about `chars` characters, built from random sentences.
fn text(pool: &[String], rng: &mut Lcg, chars: usize) -> String {
    let mut t = String::new();
    while t.chars().count() < chars {
        if !t.is_empty() {
            t.push(' ');
        }
        t.push_str(&pool[rng.next() as usize % pool.len()]);
        t.push('.');
    }
    t.chars().take(chars).collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn report(label: &str, cs: &mut [f64]) {
    cs.sort_by(|a, b| a.total_cmp(b));
    let mean = cs.iter().sum::<f64>() / cs.len() as f64;
    let p05 = cs[((cs.len() as f64 - 1.0) * 0.05).round() as usize];
    println!(
        "{label:<46} n={:<4} mean={mean:.5} p05={p05:.5} min={:.5}",
        cs.len(),
        cs[0]
    );
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
    anyhow::ensure!(!args.is_empty(), "usage: embed_batch_drift <model> [flags]");
    let model = args[0].as_str();
    let n_targets = arg_val(&args, "--targets", 48);
    let threads = arg_val(&args, "--threads", 2);
    if !ort::init().commit() {
        eprintln!("ort init: environment already configured");
    }
    embed_server::arena::register_shared_cpu_arena_for_model(model)
        .map_err(|e| anyhow::anyhow!("arena: {e}"))?;
    let cfg = Config::from_env().map_err(|e| anyhow::anyhow!("config: {e}"))?;
    let emb = StandaloneEmbedder::load(model, &cfg, threads, 1)
        .map_err(|e| anyhow::anyhow!("load: {e}"))?;
    let infer = |texts: Vec<String>| -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(emb
            .infer(texts, 0)
            .map_err(|e| anyhow::anyhow!("infer: {e}"))?
            .0)
    };

    let pool = sentences();
    let mut rng = Lcg(0xD1F7);
    // Targets: mixed lengths, 80..1100 characters (about 20..250 tokens).
    let targets: Vec<String> = (0..n_targets)
        .map(|_| {
            let c = 80 + (rng.next() as usize % 1020);
            text(&pool, &mut rng, c)
        })
        .collect();
    let mates = |rng: &mut Lcg, k: usize, lo: usize, hi: usize| -> Vec<String> {
        (0..k)
            .map(|_| {
                let c = lo + (rng.next() as usize % (hi - lo + 1));
                text(&pool, rng, c)
            })
            .collect()
    };

    // Reference: each target alone.
    let mut alone = Vec::new();
    for t in &targets {
        alone.push(infer(vec![t.clone()])?.remove(0));
    }
    // Determinism control: the same single request twice.
    let mut c: Vec<f64> = Vec::new();
    for (t, a) in targets.iter().zip(&alone).take(12) {
        c.push(cos(&infer(vec![t.clone()])?[0], a));
    }
    report("control: same request twice", &mut c);

    // `shape` builds the request around target `t`; the target is at `pos`.
    type Shape<'a> = Box<dyn FnMut(&str) -> (Vec<String>, usize) + 'a>;
    let run = |label: &str, mut shape: Shape<'_>| -> anyhow::Result<Vec<Vec<f32>>> {
        let mut cs = Vec::new();
        let mut kept = Vec::new();
        for (t, a) in targets.iter().zip(&alone) {
            let (req, pos) = shape(t);
            let v = infer(req)?.swap_remove(pos);
            cs.push(cos(&v, a));
            kept.push(v);
        }
        report(label, &mut cs);
        Ok(kept)
    };

    let mut r1 = Lcg(1);
    let mut r2 = Lcg(2);
    let mut r3 = Lcg(3);
    let mut r4 = Lcg(4);
    let mut r5 = Lcg(5);
    let mut r6 = Lcg(6);
    let mut r7 = Lcg(7);
    run(
        "[t, t] duplicate",
        Box::new(|t| (vec![t.to_string(), t.to_string()], 0)),
    )?;
    run(
        "[t, 20-char text]",
        Box::new(|t| (vec![t.to_string(), "short filler text".to_string()], 0)),
    )?;
    run(
        "[t, ~1100-char text]",
        Box::new(|t| {
            let mut v = vec![t.to_string()];
            v.extend(mates(&mut r1, 1, 1090, 1100));
            (v, 0)
        }),
    )?;
    run(
        "B=4 random mates (80..1100 chars)",
        Box::new(|t| {
            let mut v = vec![t.to_string()];
            v.extend(mates(&mut r2, 3, 80, 1100));
            (v, 0)
        }),
    )?;
    let b8a = run(
        "B=8 random mates, draw A",
        Box::new(|t| {
            let mut v = vec![t.to_string()];
            v.extend(mates(&mut r3, 7, 80, 1100));
            (v, 0)
        }),
    )?;
    let b8b = run(
        "B=8 random mates, draw B",
        Box::new(|t| {
            let mut v = vec![t.to_string()];
            v.extend(mates(&mut r4, 7, 80, 1100));
            (v, 0)
        }),
    )?;
    let mut ab: Vec<f64> = b8a.iter().zip(&b8b).map(|(x, y)| cos(x, y)).collect();
    report("B=8 draw A vs draw B (no alone reference)", &mut ab);
    run(
        "B=32 random mates",
        Box::new(|t| {
            let mut v = vec![t.to_string()];
            v.extend(mates(&mut r5, 31, 80, 1100));
            (v, 0)
        }),
    )?;
    run(
        "B=128 short mates (60..200 chars)",
        Box::new(|t| {
            let mut v = vec![t.to_string()];
            v.extend(mates(&mut r6, 127, 60, 200));
            (v, 0)
        }),
    )?;
    run(
        "target as the last of B=16 random mates",
        Box::new(|t| {
            let mut v = mates(&mut r7, 15, 80, 1100);
            v.push(t.to_string());
            (v, 15)
        }),
    )?;
    Ok(())
}
