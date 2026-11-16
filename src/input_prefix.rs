//! Instruction prefixes for asymmetric embedding models.
//!
//! `multilingual-e5-*` is trained on `query: ` / `passage: ` inputs; without
//! them retrieval quality drops. Applying the prefix changes every vector the
//! model returns, so it is a per-model opt-in (`EMBED_INPUT_PREFIX_<MODEL>=e5`)
//! and off by default: stores built from un-prefixed vectors stay valid until
//! an operator rebuilds them and flips the switch.
//!
//! The prefix is idempotent: a text that already starts with `query: ` or
//! `passage: ` is left alone, so callers that prefix client-side keep working
//! unchanged with the switch on.

const E5_QUERY: &str = "query: ";
const E5_PASSAGE: &str = "passage: ";

/// Which prefix scheme a model expects. `Off` sends texts to the model as given.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InputPrefix {
    #[default]
    Off,
    /// `query: ` for `input_type=query`, `passage: ` for documents.
    E5,
}

impl InputPrefix {
    /// Parse the value of `EMBED_INPUT_PREFIX_<MODEL>`. Unset or empty means
    /// off; an unknown value is an error (a typo here would silently keep every
    /// vector un-prefixed while the operator believes otherwise).
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw.map(str::trim) {
            None | Some("") | Some("off") | Some("none") => Ok(Self::Off),
            Some("e5") => Ok(Self::E5),
            Some(other) => Err(format!(
                "unknown input prefix scheme '{other}' (expected 'e5', 'off' or 'none')"
            )),
        }
    }

    /// The texts to feed the model for a request of the given kind.
    pub fn apply(self, is_query: bool, texts: &[String]) -> Vec<String> {
        let prefix = match self {
            Self::Off => return texts.to_vec(),
            Self::E5 if is_query => E5_QUERY,
            Self::E5 => E5_PASSAGE,
        };
        texts
            .iter()
            .map(|t| {
                if t.starts_with(E5_QUERY) || t.starts_with(E5_PASSAGE) {
                    t.clone()
                } else {
                    format!("{prefix}{t}")
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn off_passes_texts_through_unchanged() {
        assert_eq!(InputPrefix::Off.apply(true, &v(&["a"])), v(&["a"]));
        assert_eq!(InputPrefix::Off.apply(false, &v(&["a"])), v(&["a"]));
    }

    #[test]
    fn e5_prefixes_by_input_type() {
        assert_eq!(
            InputPrefix::E5.apply(true, &v(&["a", "b"])),
            v(&["query: a", "query: b"])
        );
        assert_eq!(InputPrefix::E5.apply(false, &v(&["a"])), v(&["passage: a"]));
    }

    #[test]
    fn e5_leaves_already_prefixed_texts_alone() {
        // A caller that prefixes client-side must not be prefixed twice.
        let got = InputPrefix::E5.apply(false, &v(&["query: q", "passage: p", "plain"]));
        assert_eq!(got, v(&["query: q", "passage: p", "passage: plain"]));
    }

    #[test]
    fn parse_accepts_known_and_rejects_typos() {
        assert_eq!(InputPrefix::parse(None), Ok(InputPrefix::Off));
        assert_eq!(InputPrefix::parse(Some("")), Ok(InputPrefix::Off));
        assert_eq!(InputPrefix::parse(Some("off")), Ok(InputPrefix::Off));
        assert_eq!(InputPrefix::parse(Some("e5")), Ok(InputPrefix::E5));
        assert_eq!(InputPrefix::parse(Some(" e5 ")), Ok(InputPrefix::E5));
        assert!(InputPrefix::parse(Some("E5x")).is_err());
        assert!(InputPrefix::parse(Some("true")).is_err());
    }
}
