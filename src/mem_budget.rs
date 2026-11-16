//! Startup check: do the configured arena ceilings fit inside the container?
//!
//! Every worker registers an ORT arena that may grow up to its
//! `EMBED_ARENA_MAX_MEM_BYTES[_<MODEL>]` ceiling, and the supervisor registers
//! one more from the global value. The ceilings live in the compose file, and
//! so does the container memory limit, and nothing related the two. When the
//! ceilings alone add up to the limit, a few workers growing together make the
//! kernel OOM-kill inside the cgroup, which takes the supervisor and every
//! worker down at once (a full cold reload).
//!
//! The check is advisory: a violation logs one `error` line and sets
//! gauges, it never stops the server (refusing to start would turn a
//! misconfiguration into an outage).
//!
//! Metrics: `embed_arena_ceilings_sum_bytes`, `embed_memory_limit_bytes`
//! (0 = unknown or unlimited), `embed_memory_budget_headroom_bytes`,
//! `embed_memory_budget_ratio` (`(sum + headroom) / limit`, only set when the
//! limit is known) and `embed_memory_budget_exceeded` (1 when the sum plus
//! headroom reaches the limit).

use std::path::Path;

const GIB: u64 = 1024 * 1024 * 1024;

/// Memory the arenas do not cover: model weights (about 2 GB of duplicated
/// ONNX buffers across the workers), tokenizers, the response cache (about
/// 0.4 GB at 100k entries) and supervisor heap. Override with
/// `EMBED_MEMORY_HEADROOM_BYTES`.
pub const DEFAULT_HEADROOM_BYTES: u64 = 3 * GIB;

/// cgroup v2 limit of the container; the literal `max` means unlimited.
const CGROUP_V2_MEMORY_MAX: &str = "/sys/fs/cgroup/memory.max";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// `sum + headroom` stays below the limit.
    Fits,
    /// `sum + headroom` reaches or passes the limit.
    Exceeded,
    /// No readable limit (cgroup v1, unlimited, or not in a container).
    NoLimit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    /// `(who, ceiling in bytes)`; the supervisor's own arena plus one per worker.
    pub ceilings: Vec<(String, u64)>,
    pub headroom: u64,
    pub limit: Option<u64>,
}

impl Budget {
    pub fn sum(&self) -> u64 {
        self.ceilings
            .iter()
            .fold(0u64, |a, (_, c)| a.saturating_add(*c))
    }

    pub fn required(&self) -> u64 {
        self.sum().saturating_add(self.headroom)
    }

    pub fn verdict(&self) -> Verdict {
        match self.limit {
            None => Verdict::NoLimit,
            Some(limit) if self.required() < limit => Verdict::Fits,
            Some(_) => Verdict::Exceeded,
        }
    }

    pub fn ratio(&self) -> Option<f64> {
        self.limit
            .filter(|l| *l > 0)
            .map(|l| self.required() as f64 / l as f64)
    }
}

/// Parse the content of `memory.max`: a byte count, or `max` for unlimited.
pub fn parse_memory_max(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok()
}

fn read_limit(path: &Path) -> Option<u64> {
    parse_memory_max(&std::fs::read_to_string(path).ok()?)
}

/// Ceilings the supervisor and its workers will register. Single-process mode
/// has one shared arena (the supervisor's); multi-process adds one per worker.
pub fn collect_ceilings(
    model_names: &[String],
    multi_process: bool,
    get: &dyn Fn(&str) -> Option<String>,
) -> Vec<(String, u64)> {
    let mut out = Vec::with_capacity(model_names.len() + 1);
    let mut push = |who: &str, key: &str| {
        let bytes = match crate::arena::resolve_max_mem_bytes(key, get) {
            Ok(b) => b as u64,
            Err(e) => {
                // The worker would abort on the same value; count the default
                // so the check still reports something.
                tracing::warn!(error = %e, who, "arena ceiling unparsable; budget check assumes the default");
                6 * GIB
            }
        };
        out.push((who.to_string(), bytes));
    };
    push("supervisor", "");
    if multi_process {
        for name in model_names {
            push(name, name);
        }
    }
    out
}

pub fn headroom_from(get: &dyn Fn(&str) -> Option<String>) -> u64 {
    match get("EMBED_MEMORY_HEADROOM_BYTES") {
        None => DEFAULT_HEADROOM_BYTES,
        Some(raw) => match crate::arena::try_parse_bytes_with_suffix(raw.trim()) {
            Ok(b) => b as u64,
            Err(e) => {
                tracing::warn!(error = %e, "EMBED_MEMORY_HEADROOM_BYTES unparsable; using the default");
                DEFAULT_HEADROOM_BYTES
            }
        },
    }
}

fn mib(b: u64) -> u64 {
    b / (1024 * 1024)
}

fn report(budget: &Budget) {
    let sum = budget.sum();
    crate::metrics::set_memory_budget(
        sum,
        budget.limit,
        budget.headroom,
        budget.ratio(),
        budget.verdict() == Verdict::Exceeded,
    );
    let detail = budget
        .ceilings
        .iter()
        .map(|(n, c)| format!("{n}={}MiB", mib(*c)))
        .collect::<Vec<_>>()
        .join(",");
    match budget.verdict() {
        Verdict::Exceeded => tracing::error!(
            ceilings_sum_mib = mib(sum),
            headroom_mib = mib(budget.headroom),
            limit_mib = mib(budget.limit.unwrap_or(0)),
            ceilings = %detail,
            "MEMORY BUDGET EXCEEDED: arena ceilings plus headroom reach the container memory limit; \
             workers growing together will be OOM-killed as a group. Lower EMBED_ARENA_MAX_MEM_BYTES_<MODEL> or raise the limit"
        ),
        Verdict::Fits => tracing::info!(
            ceilings_sum_mib = mib(sum),
            headroom_mib = mib(budget.headroom),
            limit_mib = mib(budget.limit.unwrap_or(0)),
            ceilings = %detail,
            "memory budget ok: arena ceilings plus headroom fit the container limit"
        ),
        Verdict::NoLimit => tracing::info!(
            ceilings_sum_mib = mib(sum),
            ceilings = %detail,
            "memory budget not checked: no readable cgroup v2 memory.max limit"
        ),
    }
}

/// Run the check once at startup. Never fails and never blocks startup.
pub fn check_at_startup(model_names: &[String], multi_process: bool) {
    let get = |k: &str| std::env::var(k).ok();
    let budget = Budget {
        ceilings: collect_ceilings(model_names, multi_process, &get),
        headroom: headroom_from(&get),
        limit: read_limit(Path::new(CGROUP_V2_MEMORY_MAX)),
    };
    report(&budget);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const MIB: u64 = 1024 * 1024;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    fn budget(sum_gib: u64, headroom_gib: u64, limit_gib: Option<u64>) -> Budget {
        Budget {
            ceilings: vec![("w".into(), sum_gib * GIB)],
            headroom: headroom_gib * GIB,
            limit: limit_gib.map(|g| g * GIB),
        }
    }

    #[test]
    fn parse_memory_max_handles_unlimited_and_bytes() {
        assert_eq!(parse_memory_max("max\n"), None);
        assert_eq!(parse_memory_max("17179869184\n"), Some(17_179_869_184));
        assert_eq!(parse_memory_max(""), None);
        assert_eq!(parse_memory_max("garbage"), None);
    }

    #[test]
    fn verdict_boundary_is_strict() {
        // sum + headroom == limit leaves nothing: exceeded.
        assert_eq!(budget(13, 3, Some(16)).verdict(), Verdict::Exceeded);
        assert_eq!(budget(12, 3, Some(16)).verdict(), Verdict::Fits);
        assert_eq!(budget(14, 3, Some(16)).verdict(), Verdict::Exceeded);
        assert_eq!(budget(1, 1, None).verdict(), Verdict::NoLimit);
    }

    #[test]
    fn ratio_is_required_over_limit() {
        let r = budget(6, 2, Some(16)).ratio().unwrap();
        assert!((r - 0.5).abs() < 1e-9, "{r}");
        assert_eq!(budget(6, 2, None).ratio(), None);
    }

    /// The per-model ceilings of the production compose, which add up to the
    /// 16 GiB container limit. The check must flag exactly this layout.
    #[test]
    fn production_layout_is_flagged() {
        let get = env(&[
            ("EMBED_ARENA_MAX_MEM_BYTES", "536870912"),
            (
                "EMBED_ARENA_MAX_MEM_BYTES_MULTILINGUAL_E5_LARGE",
                "4294967296",
            ),
            ("EMBED_ARENA_MAX_MEM_BYTES_JINA_CODE_V2", "3758096384"),
            ("EMBED_ARENA_MAX_MEM_BYTES_CODE_RANK_EMBED", "4294967296"),
            ("EMBED_ARENA_MAX_MEM_BYTES_GTE_MULTI_RERANK", "2147483648"),
            (
                "EMBED_ARENA_MAX_MEM_BYTES_SPLADE_V3_DISTILBERT",
                "2147483648",
            ),
        ]);
        let names: Vec<String> = [
            "multilingual-e5-large",
            "jina-code-v2",
            "code-rank-embed",
            "gte-multi-rerank",
            "splade-v3-distilbert",
        ]
        .map(String::from)
        .to_vec();
        let ceilings = collect_ceilings(&names, true, &get);
        let b = Budget {
            ceilings,
            headroom: DEFAULT_HEADROOM_BYTES,
            limit: Some(17_179_869_184),
        };
        assert_eq!(b.sum(), 16 * GIB);
        assert_eq!(b.verdict(), Verdict::Exceeded);
    }

    #[test]
    fn single_process_counts_only_the_shared_arena() {
        let get = env(&[
            ("EMBED_ARENA_MAX_MEM_BYTES", "1073741824"),
            ("EMBED_ARENA_MAX_MEM_BYTES_E5", "4294967296"),
        ]);
        let names = vec!["e5".to_string()];
        let single = collect_ceilings(&names, false, &get);
        assert_eq!(single, vec![("supervisor".to_string(), GIB)]);
        let multi = collect_ceilings(&names, true, &get);
        assert_eq!(multi.iter().map(|(_, c)| *c).sum::<u64>(), 5 * GIB);
    }

    #[test]
    fn unparsable_ceiling_does_not_panic() {
        let get = env(&[("EMBED_ARENA_MAX_MEM_BYTES_E5", "2KB")]);
        let c = collect_ceilings(&["e5".to_string()], true, &get);
        assert_eq!(c[1], ("e5".to_string(), 6 * GIB));
    }

    #[test]
    fn headroom_env_overrides_and_falls_back() {
        assert_eq!(headroom_from(&env(&[])), DEFAULT_HEADROOM_BYTES);
        assert_eq!(
            headroom_from(&env(&[("EMBED_MEMORY_HEADROOM_BYTES", "512MiB")])),
            512 * MIB
        );
        assert_eq!(
            headroom_from(&env(&[("EMBED_MEMORY_HEADROOM_BYTES", "lots")])),
            DEFAULT_HEADROOM_BYTES
        );
    }
}
