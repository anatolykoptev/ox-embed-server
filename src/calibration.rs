//! Per-model reranker score calibration.
//!
//! A cross-encoder's raw logits live on a model-specific scale, so a model swap
//! behind a stable served name silently moves every client-side threshold set on
//! the sigmoid score. An optional `<model dir>/calibration.json` holds a
//! piecewise-linear map from this model's logit to the logit scale clients were
//! tuned against; the server applies it just before the sigmoid.
//!
//! File absent → identity. File present but invalid → startup error: a silent
//! identity would hide a broken calibration behind plausible-looking scores.

use std::path::Path;

use serde::Deserialize;

/// File name looked up inside a reranker's model directory.
pub const CALIBRATION_FILE: &str = "calibration.json";

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid {path}: {reason}")]
    Invalid { path: String, reason: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCalibration {
    schema: u32,
    id: String,
    knots_x: Vec<f64>,
    knots_y: Vec<f64>,
}

/// A validated monotone piecewise-linear map in logit space.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    id: String,
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// `slopes[i]` is the slope of segment `[i, i+1]`: finite and > 0, checked at load.
    slopes: Vec<f64>,
}

impl Calibration {
    /// Build from raw knots, enforcing the file schema's invariants.
    pub fn new(id: String, xs: Vec<f64>, ys: Vec<f64>) -> Result<Self, String> {
        if id.trim().is_empty() {
            return Err("id must not be empty".into());
        }
        if xs.len() != ys.len() {
            return Err(format!(
                "knots_x has {} entries but knots_y has {}",
                xs.len(),
                ys.len()
            ));
        }
        if xs.len() < 2 {
            return Err(format!("need at least 2 knots, got {}", xs.len()));
        }
        if let Some(i) = xs.iter().chain(ys.iter()).position(|v| !v.is_finite()) {
            return Err(format!("non-finite knot value at flat index {i}"));
        }
        if let Some(i) = xs.windows(2).position(|w| w[1] <= w[0]) {
            return Err(format!(
                "knots_x not strictly increasing at index {}",
                i + 1
            ));
        }
        // Strictly increasing: a flat segment would tie documents the model ranks apart.
        if let Some(i) = ys.windows(2).position(|w| w[1] <= w[0]) {
            return Err(format!(
                "knots_y not strictly increasing at index {}",
                i + 1
            ));
        }
        // Finite knots can still overflow in the arithmetic: reject any segment
        // whose rise or slope is not finite, so evaluation cannot create NaN/inf.
        let mut slopes = Vec::with_capacity(xs.len() - 1);
        for i in 0..xs.len() - 1 {
            let (dx, dy) = (xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
            let slope = dy / dx;
            if !dy.is_finite() || !dx.is_finite() || !slope.is_finite() || slope <= 0.0 {
                return Err(format!(
                    "segment {i}..{} has a non-finite or non-positive slope",
                    i + 1
                ));
            }
            slopes.push(slope);
        }
        Ok(Self { id, xs, ys, slopes })
    }

    /// Identifier of the fit, surfaced in responses and metrics.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Number of knots.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Map one logit. Linear between knots, linear extrapolation from the end
    /// segments. NaN passes through. A finite input can still overflow when it
    /// is extrapolated far past the end knots; the result may then be non-finite
    /// (see `apply` for the policy).
    pub fn map(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        let n = self.xs.len();
        // Segment [i-1, i] containing x; clamp so ends reuse the edge segment.
        let i = self.xs.partition_point(|&k| k <= x).clamp(1, n - 1);
        self.ys[i - 1] + self.slopes[i - 1] * (x - self.xs[i - 1])
    }

    /// Calibrate a slice of logits in place. Policy: a finite logit whose
    /// calibrated value is not finite as f32 (extrapolation overflow) is left
    /// as the raw logit, and counted in the return value so the caller can
    /// export it. Non-finite inputs pass through. Returns the fallback count.
    pub fn apply(&self, scores: &mut [f32]) -> usize {
        let mut fallbacks = 0;
        for s in scores.iter_mut() {
            if !s.is_finite() {
                continue;
            }
            let y = self.map(f64::from(*s)) as f32;
            if y.is_finite() {
                *s = y;
            } else {
                fallbacks += 1;
            }
        }
        fallbacks
    }

    /// Load `<dir>/calibration.json`. `Ok(None)` when the file does not exist;
    /// any other failure (unreadable, malformed, invariant violation) is an error.
    pub fn load_from_dir(dir: &Path) -> Result<Option<Self>, CalibrationError> {
        let path = dir.join(CALIBRATION_FILE);
        let shown = path.display().to_string();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(CalibrationError::Io {
                    path: shown,
                    source,
                });
            }
        };
        let invalid = |reason: String| CalibrationError::Invalid {
            path: shown.clone(),
            reason,
        };
        let raw: RawCalibration =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        if raw.schema != 1 {
            return Err(invalid(format!("unsupported schema {}", raw.schema)));
        }
        Self::new(raw.id, raw.knots_x, raw.knots_y)
            .map(Some)
            .map_err(invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cal(xs: &[f64], ys: &[f64]) -> Calibration {
        Calibration::new("t".into(), xs.to_vec(), ys.to_vec()).unwrap()
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("calib-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn map_known_answers_interior_and_knots() {
        // Segments: slope 2 on [0,1], slope 0.5 on [1,3].
        let c = cal(&[0.0, 1.0, 3.0], &[0.0, 2.0, 3.0]);
        assert_eq!(c.map(0.0), 0.0);
        assert_eq!(c.map(0.5), 1.0);
        assert_eq!(c.map(1.0), 2.0);
        assert_eq!(c.map(2.0), 2.5);
        assert_eq!(c.map(3.0), 3.0);
    }

    #[test]
    fn map_extrapolates_linearly_from_end_segments() {
        let c = cal(&[0.0, 1.0, 3.0], &[0.0, 2.0, 3.0]);
        assert_eq!(c.map(-1.0), -2.0, "left end uses the first segment slope");
        assert_eq!(c.map(5.0), 4.0, "right end uses the last segment slope");
    }

    #[test]
    fn map_nan_passes_through() {
        let c = cal(&[0.0, 1.0], &[0.0, 1.0]);
        assert!(c.map(f64::NAN).is_nan());
    }

    #[test]
    fn apply_maps_each_score_in_place() {
        let c = cal(&[0.0, 1.0], &[10.0, 12.0]);
        let mut s = [0.0_f32, 0.5, 1.0];
        assert_eq!(c.apply(&mut s), 0);
        assert_eq!(s, [10.0, 11.0, 12.0]);
    }

    #[test]
    fn new_rejects_each_invariant_violation() {
        let ok_y = vec![0.0, 1.0];
        let bad = [
            (vec![0.0], vec![0.0], "fewer than 2 knots"),
            (vec![0.0, 1.0], vec![0.0], "length mismatch"),
            (
                vec![0.0, 0.0],
                ok_y.clone(),
                "x not strictly increasing (equal)",
            ),
            (vec![1.0, 0.0], ok_y.clone(), "x decreasing"),
            (vec![0.0, 1.0], vec![1.0, 0.0], "y decreasing"),
            (vec![0.0, 1.0], vec![1.0, 1.0], "y flat"),
            (vec![0.0, f64::NAN], ok_y.clone(), "NaN x"),
            (vec![0.0, 1.0], vec![0.0, f64::INFINITY], "inf y"),
        ];
        for (xs, ys, why) in bad {
            assert!(
                Calibration::new("t".into(), xs, ys).is_err(),
                "must reject: {why}"
            );
        }
        assert!(Calibration::new(" ".into(), vec![0.0, 1.0], ok_y).is_err());
    }

    #[test]
    fn new_rejects_flat_y_segments() {
        assert!(
            Calibration::new("t".into(), vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 1.0]).is_err(),
            "y1 == y0 must be rejected: strict monotonicity"
        );
    }

    #[test]
    fn new_rejects_overflowing_rise() {
        // y1 - y0 overflows to inf; evaluating at x0 would give inf * 0 = NaN.
        assert!(Calibration::new("t".into(), vec![0.0, 1.0], vec![-1e308, 1e308]).is_err());
    }

    #[test]
    fn new_rejects_infinite_slope() {
        // dx is subnormal, so the slope overflows to inf.
        assert!(Calibration::new("t".into(), vec![0.0, 1e-320], vec![0.0, 1.0]).is_err());
    }

    #[test]
    fn apply_falls_back_to_raw_logit_when_extrapolation_overflows() {
        // Valid: slope 1e37 is finite. A far logit extrapolates past f32::MAX.
        let c = cal(&[0.0, 1.0], &[0.0, 1e37]);
        let mut s = [0.5_f32, 100.0, f32::NAN];
        let fb = c.apply(&mut s);
        assert_eq!(fb, 1, "only the overflowing logit falls back");
        assert_eq!(s[1], 100.0, "raw logit kept");
        assert!(
            s[0].is_finite() && s[0] != 0.5,
            "in-range logit still mapped"
        );
        assert!(s[2].is_nan(), "NaN passes through untouched");
    }

    #[test]
    fn load_missing_file_is_identity_none() {
        let d = tmp_dir("missing");
        assert_eq!(Calibration::load_from_dir(&d).unwrap(), None);
    }

    #[test]
    fn load_valid_file() {
        let d = tmp_dir("valid");
        std::fs::write(
            d.join(CALIBRATION_FILE),
            r#"{"schema":1,"id":"fit-a","knots_x":[-1.0,1.0],"knots_y":[-2.0,2.0]}"#,
        )
        .unwrap();
        let c = Calibration::load_from_dir(&d).unwrap().expect("present");
        assert_eq!(c.id(), "fit-a");
        assert_eq!(c.len(), 2);
        assert_eq!(c.map(0.5), 1.0);
    }

    #[test]
    fn load_invalid_files_are_errors_not_identity() {
        let cases = [
            ("garbage", "not json"),
            (
                "schema",
                r#"{"schema":2,"id":"x","knots_x":[0,1],"knots_y":[0,1]}"#,
            ),
            ("noschema", r#"{"id":"x","knots_x":[0,1],"knots_y":[0,1]}"#),
            (
                "extra",
                r#"{"schema":1,"id":"x","knots_x":[0,1],"knots_y":[0,1],"z":1}"#,
            ),
            (
                "nonmono",
                r#"{"schema":1,"id":"x","knots_x":[0,1],"knots_y":[1,0]}"#,
            ),
            ("empty", ""),
        ];
        for (tag, body) in cases {
            let d = tmp_dir(tag);
            std::fs::write(d.join(CALIBRATION_FILE), body).unwrap();
            assert!(
                Calibration::load_from_dir(&d).is_err(),
                "{tag}: invalid file must be an error"
            );
        }
    }
}
