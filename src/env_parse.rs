//! One place where an environment value that cannot be used is reported.
//!
//! Every knob in this crate that falls back to a default on garbage goes
//! through [`note_fallback`] (directly, or through the typed helpers here), so
//! a bad value always produces ONE loud `warn!` naming the key and the value,
//! and increments `embed_config_fallback_total{env_var=<key>}`. Without the
//! counter a zero reads as "no misconfiguration" while a typo is silently
//! running on the default.
//!
//! None of the keys read through this module is a secret (ports, counts,
//! timeouts, booleans, ratios, a bind address), so the offending value is
//! logged. It is truncated and `Debug`-escaped regardless, so a value pasted
//! into the wrong variable cannot flood or forge log lines.
//!
//! `tests::every_env_read_is_accounted_for` scans `src/` and fails when a new
//! `std::env::var` read is neither routed through here nor explicitly marked
//! (`// env-strict:` fails loudly, `// env-raw:` is an unparsed string, and
//! `// parsed-by: <fn>` names the parser that notes the fallback).

use std::fmt::Display;
use std::str::FromStr;

/// Longest slice of an offending value that is written to the log.
const MAX_LOGGED_VALUE: usize = 64;

/// Read an environment variable, treating non-UTF-8 as unset. Pair it with a
/// parser that calls [`note_fallback`]; mark the call with `// parsed-by: <fn>`.
pub fn raw(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Report that `key` was set to `value` which cannot be used, and that
/// `default` is used instead. One `warn!` line plus one counter increment.
pub fn note_fallback(key: &str, value: &str, expected: &str, default: &dyn Display) {
    let shown: String = value.chars().take(MAX_LOGGED_VALUE).collect();
    tracing::warn!(
        env_key = key,
        value = ?shown,
        expected,
        default = %default,
        "{key} has an unusable value; using the default"
    );
    crate::metrics::record_config_fallback(key);
}

/// Parse `raw` as `T`. Unset or blank → `default`, silently (an empty line in
/// an env file is not a typo). Unparseable, or rejected by `valid` →
/// [`note_fallback`] and `default`.
pub fn parse_or<T>(
    key: &str,
    raw: Option<&str>,
    default: T,
    valid: impl Fn(&T) -> bool,
    expected: &str,
) -> T
where
    T: FromStr + Display,
{
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return default;
    };
    match s.parse::<T>() {
        Ok(v) if valid(&v) => v,
        _ => {
            note_fallback(key, s, expected, &default);
            default
        }
    }
}

/// A knob whose unset state is itself meaningful (`None`): a value that does
/// not parse is noted and treated as unset. Blank → `None` silently.
pub fn parse_optional<T: FromStr>(key: &str, raw: &str, expected: &str) -> Option<T> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    match s.parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            note_fallback(key, s, expected, &"unset");
            None
        }
    }
}

/// `parse_or` for a count that must be `> 0`.
pub fn positive<T>(key: &str, raw: Option<&str>, default: T) -> T
where
    T: FromStr + Display + PartialOrd + Default,
{
    parse_or(
        key,
        raw,
        default,
        |v| *v > T::default(),
        "a positive integer",
    )
}

/// `parse_or` for any value of the type (zero is meaningful).
pub fn any<T>(key: &str, raw: Option<&str>, default: T, expected: &str) -> T
where
    T: FromStr + Display,
{
    parse_or(key, raw, default, |_| true, expected)
}

/// Parse a boolean knob. `truthy` / `falsy` are compared case-insensitively
/// after trimming. Unset or blank → `default` silently; any other value →
/// [`note_fallback`] and `default`.
pub fn flag(key: &str, raw: Option<&str>, default: bool, truthy: &[&str], falsy: &[&str]) -> bool {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return default;
    };
    if truthy.iter().any(|t| s.eq_ignore_ascii_case(t)) {
        return true;
    }
    if falsy.iter().any(|f| s.eq_ignore_ascii_case(f)) {
        return false;
    }
    note_fallback(key, s, &format!("one of {truthy:?} / {falsy:?}"), &default);
    default
}

/// `ORT_OPT_LEVEL` as `0..=3`. One parser for the embed, reranker and SPLADE
/// loaders, which share the knob. Unset → 3; garbage or out of range → noted,
/// 3.
pub fn ort_opt_level() -> u8 {
    ort_opt_level_from(raw("ORT_OPT_LEVEL").as_deref()) // parsed-by: ort_opt_level_from
}

/// Pure half of [`ort_opt_level`].
pub fn ort_opt_level_from(raw: Option<&str>) -> u8 {
    parse_or("ORT_OPT_LEVEL", raw, 3u8, |v| *v <= 3, "an integer 0..=3")
}

/// Per-model variable families: `<PREFIX><MODEL_KEY>` where the suffix is
/// `config::model_env_key(model)`. A set variable whose suffix matches no
/// loaded model is silently ignored by its reader, so startup reports it
/// ([`unknown_per_model_vars`]). `tests::per_model_prefixes_cover_the_code`
/// derives the real families from `src/` and fails if this list lags.
pub const PER_MODEL_PREFIXES: &[&str] = &[
    "BATCH_MAX_SEQ_",
    "EMBED_ARENA_MAX_MEM_BYTES_",
    "EMBED_ARENA_SHRINK_",
    "EMBED_BATCH_INVARIANT_",
    "EMBED_INPUT_PREFIX_",
    "EMBED_IPC_TIMEOUT_SECS_",
    "EMBED_MAX_WAITERS_",
    "EMBED_MEMORY_PATTERN_",
    "EMBED_SESSION_POOL_SIZE_",
    "EMBED_SUBBATCH_RATIO_",
    "EMBED_WARMUP_SEQ_LEN_",
    "EMBED_WORKER_NICE_",
    "MAX_CONCURRENT_EMBED_REQUESTS_",
    "ONNX_OPT_CACHE_DIR_",
];

/// Variables in `vars` that look like a per-model override (they match a
/// family in [`PER_MODEL_PREFIXES`] ignoring case and `-`/`_`) but whose exact
/// name is `<PREFIX><known key>` for no loaded model, so no reader will ever
/// see them.
pub fn unknown_per_model_vars(
    vars: impl IntoIterator<Item = String>,
    known_keys: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut out: Vec<String> = vars
        .into_iter()
        .filter(|k| {
            let norm = k.to_ascii_uppercase().replace('-', "_");
            PER_MODEL_PREFIXES.iter().any(|p| {
                norm.strip_prefix(p).is_some()
                    && !(k.starts_with(p) && known_keys.contains(&k[p.len()..]))
            })
        })
        .collect();
    out.sort();
    out
}

/// Startup scan: warn once per ignored per-model variable and count it.
pub fn report_unknown_per_model_vars(known_keys: &std::collections::HashSet<String>) {
    for var in unknown_per_model_vars(std::env::vars().map(|(k, _)| k), known_keys) {
        let mut known: Vec<&String> = known_keys.iter().collect();
        known.sort();
        tracing::warn!(
            env_key = %var,
            known_model_keys = ?known,
            "per-model variable matches no loaded model and is ignored; the suffix is the model \
             name upper-cased with '-' replaced by '_'"
        );
        crate::metrics::record_config_fallback(&var);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    /// Render the counter for the calls made inside `f`.
    fn counted(f: impl FnOnce()) -> String {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, f);
        handle.render()
    }

    fn fallback_line(key: &str, n: u32) -> String {
        format!("embed_config_fallback_total{{env_var=\"{key}\"}} {n}")
    }

    #[test]
    fn garbage_falls_back_and_is_counted() {
        let mut got = 0usize;
        let out = counted(|| got = positive("SOME_KNOB", Some("abc"), 7usize));
        assert_eq!(got, 7, "garbage must keep the default");
        assert!(
            out.contains(&fallback_line("SOME_KNOB", 1)),
            "a typo'd value must increment embed_config_fallback_total:\n{out}"
        );
    }

    #[test]
    fn zero_is_rejected_by_positive_but_allowed_by_any() {
        let mut p = 0usize;
        let mut a = 99usize;
        let out = counted(|| {
            p = positive("POS", Some("0"), 5usize);
            a = any("ANY", Some("0"), 5usize, "an integer");
        });
        assert_eq!(p, 5);
        assert_eq!(a, 0, "zero is a legal value for `any`");
        assert!(out.contains(&fallback_line("POS", 1)), "{out}");
        assert!(
            !out.contains("env_var=\"ANY\""),
            "valid zero must not count:\n{out}"
        );
    }

    #[test]
    fn unset_blank_and_valid_are_silent() {
        let out = counted(|| {
            assert_eq!(positive("K", None, 3usize), 3);
            assert_eq!(positive("K", Some("  "), 3usize), 3);
            assert_eq!(positive("K", Some(" 9 "), 3usize), 9);
            assert!(!flag("F", None, false, &["1"], &["0"]));
            assert!(flag("F", Some("TRUE"), false, &["true"], &["false"]));
        });
        assert!(!out.contains("embed_config_fallback_total"), "{out}");
    }

    #[test]
    fn flag_counts_unrecognised_and_keeps_default() {
        let mut got = true;
        let out = counted(|| got = flag("FLAG_KNOB", Some("ye"), false, &["1"], &["0"]));
        assert!(!got);
        assert!(out.contains(&fallback_line("FLAG_KNOB", 1)), "{out}");
    }

    #[test]
    fn opt_level_range_is_enforced() {
        let out = counted(|| {
            assert_eq!(ort_opt_level_from(Some("2")), 2);
            assert_eq!(ort_opt_level_from(None), 3);
            assert_eq!(ort_opt_level_from(Some("5")), 3);
            assert_eq!(ort_opt_level_from(Some("fast")), 3);
        });
        assert!(
            out.contains(&fallback_line("ORT_OPT_LEVEL", 2)),
            "5 and \"fast\" are unusable, 2 and unset are not:\n{out}"
        );
    }

    #[test]
    fn unknown_per_model_vars_flags_typos_not_known_keys() {
        let known: HashSet<String> = ["JINA_CODE_V2".to_string()].into();
        let vars = [
            "EMBED_MAX_WAITERS_JINA_CODE_V2",  // exact: fine
            "EMBED_MAX_WAITERS_JINA_CODEV2",   // typo
            "EMBED_ARENA_SHRINK_jina_code_v2", // wrong case
            "BATCH_MAX_SEQ_JINA-CODE-V2",      // hyphens
            "EMBED_MAX_WAITERS",               // global: not per-model
            "PATH",
        ]
        .map(String::from);
        assert_eq!(
            unknown_per_model_vars(vars, &known),
            vec![
                "BATCH_MAX_SEQ_JINA-CODE-V2".to_string(),
                "EMBED_ARENA_SHRINK_jina_code_v2".to_string(),
                "EMBED_MAX_WAITERS_JINA_CODEV2".to_string(),
            ]
        );
    }

    /// The startup scan counts each ignored per-model variable once.
    #[test]
    #[serial_test::serial]
    fn report_unknown_per_model_vars_counts_the_typo() {
        const VAR: &str = "EMBED_MAX_WAITERS_NO_SUCH_MODEL_FOR_TEST";
        let known: HashSet<String> = ["JINA_CODE_V2".to_string()].into();
        unsafe { std::env::set_var(VAR, "4") };
        let out = counted(|| report_unknown_per_model_vars(&known));
        unsafe { std::env::remove_var(VAR) };
        assert!(
            out.contains(&fallback_line(VAR, 1)),
            "a per-model variable for no loaded model must be counted:\n{out}"
        );
    }

    /// `main` must actually run the scan: it is the only caller, and a
    /// deleted call fails nothing else (the scan is observability, not
    /// behaviour).
    #[test]
    fn unknown_var_scan_is_wired_into_startup() {
        let main_rs = production_lines(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs"));
        assert!(
            main_rs
                .iter()
                .any(|l| l.contains("env_parse::report_unknown_per_model_vars(")),
            "src/main.rs no longer calls env_parse::report_unknown_per_model_vars"
        );
    }

    // ── Code-derived guards ─────────────────────────────────────────────────
    fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read src dir") {
            let p = e.expect("dir entry").path();
            if p.is_dir() {
                rs_files(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }

    fn src_files() -> Vec<PathBuf> {
        let mut v = Vec::new();
        rs_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut v);
        v.sort();
        v
    }

    /// Every line of `path` up to its trailing test module (the first
    /// column-0 `#[cfg(test)]`; every file here keeps its tests last).
    fn lines_of(path: &Path) -> Vec<String> {
        let text = std::fs::read_to_string(path).expect("read source");
        text.lines()
            .take_while(|l| *l != "#[cfg(test)]")
            .map(str::to_string)
            .collect()
    }

    /// `lines_of`, minus files that are test code or this module itself
    /// (whose own `std::env::var` is the one allowed raw read).
    fn production_lines(path: &Path) -> Vec<String> {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == "tests.rs" || name == "env_parse.rs" {
            return Vec::new();
        }
        lines_of(path)
    }

    fn is_fn_start(l: &str) -> bool {
        let t = l.trim_start();
        [
            "fn ",
            "pub fn ",
            "pub(crate) fn ",
            "pub(super) fn ",
            "async fn ",
        ]
        .iter()
        .any(|p| t.starts_with(p))
    }

    /// Body of `fn <name>` in any non-test source (from its signature to the
    /// next `fn` at the same or lower indent). Includes this module, so a
    /// `parsed-by: positive` marker can be followed into the helper.
    fn fn_body(name: &str) -> Option<String> {
        // `fn name(` or the generic `fn name<T>(`.
        let plain = format!("fn {name}(");
        let generic = format!("fn {name}<");
        for path in src_files() {
            let lines = lines_of(&path);
            if let Some(i) = lines
                .iter()
                .position(|l| l.contains(&plain) || l.contains(&generic))
            {
                let indent = lines[i].len() - lines[i].trim_start().len();
                let mut body = lines[i].clone();
                for l in &lines[i + 1..] {
                    let ind = l.len() - l.trim_start().len();
                    if !l.trim().is_empty() && ind <= indent && is_fn_start(l) {
                        break;
                    }
                    body.push('\n');
                    body.push_str(l);
                }
                return Some(body);
            }
        }
        None
    }

    /// Identifiers directly followed by `(` in `body` (calls and macros).
    fn called_names(body: &str) -> Vec<String> {
        let b = body.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_alphabetic() || b[i] == b'_' {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                if i < b.len() && b[i] == b'(' {
                    out.push(body[start..i].to_string());
                }
            } else {
                i += 1;
            }
        }
        out
    }

    /// Does `fn <name>` report an unusable value, directly or through a
    /// helper it calls (followed up to `depth` levels)?
    fn notes_fallback(name: &str, depth: u8) -> bool {
        let Some(body) = fn_body(name) else {
            return false;
        };
        if body.contains("note_fallback(") || body.contains("record_config_fallback(") {
            return true;
        }
        depth > 0
            && called_names(&body)
                .iter()
                .filter(|c| c.as_str() != name && c.as_str() != "raw")
                .any(|c| notes_fallback(c, depth - 1))
    }

    /// A marker may sit on the read's line, or on a comment line above it
    /// (3 lines) or just below it (rustfmt moves trailing comments inward).
    fn marker_near<'a>(lines: &'a [String], i: usize, marker: &str) -> Option<&'a str> {
        let lo = i.saturating_sub(3);
        let hi = (i + 2).min(lines.len() - 1);
        (lo..=hi)
            .filter_map(|j| lines[j].split(marker).nth(1))
            .next()
    }

    /// The parse sites are enumerated from the code: any `std::env::var` read
    /// outside this module must carry a marker saying why it cannot silently
    /// fall back, and every `env_parse::raw` read must name a parser that
    /// notes its fallback. A new unmarked read fails here.
    #[test]
    fn every_env_read_is_accounted_for() {
        let mut problems = Vec::new();
        let mut direct = 0usize;
        let mut raw_reads = 0usize;
        for path in src_files() {
            let lines = production_lines(&path);
            for (i, line) in lines.iter().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                let loc = format!("{}:{}", path.display(), i + 1);
                if code.contains("env::var(") || code.contains("env::var_os(") {
                    direct += 1;
                    if marker_near(&lines, i, "// env-strict:").is_none()
                        && marker_near(&lines, i, "// env-raw:").is_none()
                    {
                        problems.push(format!(
                            "{loc}: raw env read without `// env-strict:` / `// env-raw:` \
                             marker; route it through crate::env_parse"
                        ));
                    }
                }
                if code.contains("env_parse::raw(") {
                    raw_reads += 1;
                    match marker_near(&lines, i, "// parsed-by:") {
                        None => problems.push(format!(
                            "{loc}: env_parse::raw without `// parsed-by: <fn>`"
                        )),
                        Some(p) => {
                            let p = p.trim();
                            if fn_body(p).is_none() {
                                problems.push(format!("{loc}: parsed-by fn `{p}` not found"));
                            } else if !notes_fallback(p, 3) {
                                problems
                                    .push(format!("{loc}: parser `{p}` never notes a fallback"));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            direct > 0 && raw_reads > 0,
            "scanner found no env reads ({direct} direct, {raw_reads} raw): it is broken"
        );
        assert!(problems.is_empty(), "\n{}", problems.join("\n"));
    }

    /// Every `<PREFIX>{suffix}` per-model family that `src/` formats must be
    /// in `PER_MODEL_PREFIXES`, or the startup scan cannot see its typos.
    #[test]
    fn per_model_prefixes_cover_the_code() {
        let mut found = HashSet::new();
        for path in src_files() {
            for line in lines_of(&path) {
                let code = line.split("//").next().unwrap_or("");
                let mut rest = code;
                while let Some(p) = rest.find('"') {
                    rest = &rest[p + 1..];
                    let Some(end) = rest.find('"') else { break };
                    let lit = &rest[..end];
                    rest = &rest[end + 1..];
                    // "PREFIX_{key}", "PREFIX_{suffix}" or "PREFIX_{}"
                    if let Some((prefix, hole)) = lit.split_once('{')
                        && hole.ends_with('}')
                        && hole[..hole.len() - 1]
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c == '_')
                        && prefix.len() > 1
                        && prefix.ends_with('_')
                        && prefix
                            .chars()
                            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                    {
                        found.insert(prefix.to_string());
                    }
                }
            }
        }
        assert!(
            found.len() >= 8,
            "scanner found only {found:?}: it is broken"
        );
        for p in &found {
            assert!(
                PER_MODEL_PREFIXES.contains(&p.as_str()),
                "per-model family {p} is read in src/ but missing from PER_MODEL_PREFIXES"
            );
        }
    }
}
