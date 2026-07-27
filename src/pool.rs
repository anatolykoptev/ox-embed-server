use ndarray::{ArrayViewD, Axis, s};

/// Build padded i64 tensors for input_ids, attention_mask, and optionally
/// token_type_ids from pre-tokenized `Vec<u32>` ids (one per sequence).
///
/// `attention_mask[i, j] = 1` iff `j < token_ids[i].len()` (synthesised
/// mask: 1 over real positions, 0 over pad positions). `token_type_ids`
/// are always zeros — single-input embedding has no segment-B.
///
/// `with_tti` controls whether `token_type_ids` are allocated:
///
/// - `true`  → returns `Some(vec![0i64; total])` (BERT-family models that
///   expose a `token_type_ids` input).
/// - `false` → returns `None`, skipping the allocation entirely.
///
/// # tti (token_type_ids) is BERT-specific; XLM-RoBERTa / RoBERTa / DistilBERT
/// models don't use it. Caller passes false for ~3-of-4 prod models, avoiding
/// a per-request 64 KiB zeroed Vec.
pub fn build_tensors_from_ids(
    token_ids: &[Vec<u32>],
    batch: usize,
    max_seq: usize,
    pad_id: u32,
    with_tti: bool,
) -> (Vec<i64>, Vec<i64>, Option<Vec<i64>>) {
    let total = batch * max_seq;
    let mut ids = vec![pad_id as i64; total];
    let mut mask = vec![0i64; total];
    // tti (token_type_ids) is BERT-specific; XLM-RoBERTa / RoBERTa / DistilBERT
    // models don't use it. Caller passes false for ~3-of-4 prod models, avoiding
    // a per-request 64 KiB zeroed Vec.
    let tti = if with_tti {
        Some(vec![0i64; total])
    } else {
        None
    };

    for (i, seq) in token_ids.iter().enumerate() {
        let len = seq.len().min(max_seq);
        let offset = i * max_seq;
        for j in 0..len {
            ids[offset + j] = seq[j] as i64;
            mask[offset + j] = 1i64;
        }
    }

    (ids, mask, tti)
}

/// Convert i64 attention mask to f32 for mean pooling.
pub fn mask_i64_to_f32(mask_i64: &[i64]) -> Vec<f32> {
    mask_i64.iter().map(|&v| v as f32).collect()
}

/// Mean-pool over non-padded tokens and L2-normalize each vector.
///
/// `raw` has shape [batch, seq_len, dim].
/// `mask` has shape [batch, seq_len] with 1.0 for real tokens.
///
/// # SIMD-friendly accumulation
///
/// The original triple-nested loop indexed `raw[[i, j, k]]` via `ArrayViewD`,
/// which recomputes dyn-stride offsets on every element access. This rewrite:
///
/// 1. Iterates rows via `axis_iter(Axis(0))` — avoids per-element index math.
/// 2. Exploits the prefix-1s mask invariant (see `build_tensors_from_ids`:
///    mask is `1` for `0..len` then `0` for the pad tail) to find the
///    real-token count once per sequence and skip the per-position `m > 0.0`
///    branch in the hot loop. A `debug_assert!` guards the invariant.
/// 3. Uses `to_slice()` on each token row when the last axis is contiguous
///    (the normal ONNX [B, S, D] row-major layout), so the inner `acc += row`
///    add loop iterates `&[f32]` slices and LLVM auto-vectorizes it into
///    NEON `faddp`/`fmla` lanes on ARM Neoverse-N1. Falls back to
///    `iter()`-based accumulation for non-contiguous views.
///
/// Numerical parity: f32 accumulation in `acc` (same as the original `vec`),
/// f64 for `count` and `sum_sq` (same as the original).
pub fn mean_pool_normalize(
    raw: &ArrayViewD<'_, f32>,
    mask: &ndarray::Array2<f32>,
    batch: usize,
    max_seq: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let shape = raw.shape();
    if shape.len() != 3 || shape[0] != batch || shape[2] != dim {
        return Err(format!(
            "unexpected output shape: {:?}, expected [{batch}, {max_seq}, {dim}]",
            shape
        ));
    }

    let mut result = Vec::with_capacity(batch);

    for (row_view, mask_row) in raw.axis_iter(Axis(0)).zip(mask.axis_iter(Axis(0))) {
        // Prefix-1s invariant: mask is `1` for `0..real_tokens` then `0`.
        // `build_tensors_from_ids` is the only mask producer and guarantees
        // this. Find the cut once, skip the per-position branch in the hot
        // loop. The debug_assert catches a future mask producer that breaks
        // the prefix assumption — in release it would silently under-pool
        // (sum only the prefix up to the first 0), which is wrong but not
        // unsafe; the assert makes it loud in dev.
        let real_tokens = mask_row.iter().position(|&m| m == 0.0).unwrap_or(max_seq);
        debug_assert!(
            !mask_row.iter().skip(real_tokens).any(|&m| m > 0.0),
            "mask has non-prefix 1s after position {real_tokens}; \
             build_tensors_from_ids produces prefix-1s masks"
        );

        let mut acc = vec![0.0f32; dim];
        for j in 0..real_tokens {
            let token_vec = row_view.slice(s![j, ..]);
            if let Some(slc) = token_vec.to_slice() {
                // Contiguous last axis (normal ONNX [B,S,D] row-major):
                // slice add auto-vectorizes to NEON fmla lanes.
                for (a, &b) in acc.iter_mut().zip(slc) {
                    *a += b;
                }
            } else {
                // Non-contiguous view (e.g. transposed output): fall back
                // to iterator-based add. Correctness over speed.
                for (a, &b) in acc.iter_mut().zip(token_vec.iter()) {
                    *a += b;
                }
            }
        }

        // Average over non-padded tokens. `count` is an exact integer
        // cast to f64 (no drift); matches the original `count += 1.0` loop.
        let count = real_tokens as f64;
        if count > 0.0 {
            let inv = (1.0 / count) as f32;
            for v in &mut acc {
                *v *= inv;
            }
        }

        // L2 normalize (accumulate in f64 to match Go's math.Sqrt(float64)).
        let sum_sq: f64 = acc.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            let inv = (1.0 / norm) as f32;
            for v in &mut acc {
                *v *= inv;
            }
        }

        result.push(acc);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Build a 3D f32 array and return it as a dynamic view, the shape
    /// `mean_pool_normalize` expects. The array is leaked (test-only,
    /// short-lived process) so the view's lifetime is `'static`.
    fn raw_view(data: &[f32], batch: usize, seq: usize, dim: usize) -> ArrayViewD<'static, f32> {
        let arr = Box::leak(Box::new(
            Array3::from_shape_vec((batch, seq, dim), data.to_vec())
                .expect("shape mismatch in test fixture"),
        ));
        arr.view().into_dyn()
    }

    /// Reference implementation — the original triple-nested loop — used
    /// to assert the rewrite is numerically identical.
    fn reference_mean_pool_normalize(
        raw: &ArrayViewD<'_, f32>,
        mask: &Array2<f32>,
        batch: usize,
        max_seq: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(batch);
        for i in 0..batch {
            let mut vec = vec![0.0f32; dim];
            let mut count = 0.0f64;
            for j in 0..max_seq {
                let m = mask[[i, j]];
                if m > 0.0 {
                    count += 1.0;
                    for k in 0..dim {
                        vec[k] += raw[[i, j, k]];
                    }
                }
            }
            if count > 0.0 {
                let inv = (1.0 / count) as f32;
                for v in &mut vec {
                    *v *= inv;
                }
            }
            let sum_sq: f64 = vec.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let norm = sum_sq.sqrt();
            if norm > 0.0 {
                let inv = (1.0 / norm) as f32;
                for v in &mut vec {
                    *v *= inv;
                }
            }
            result.push(vec);
        }
        result
    }

    #[test]
    fn single_sequence_full_mask_matches_reference() {
        // batch=1, seq=2, dim=3, no padding
        let raw = raw_view(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 2, 3);
        let mask = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let got = mean_pool_normalize(&raw, &mask, 1, 2, 3).unwrap();
        let want = reference_mean_pool_normalize(&raw, &mask, 1, 2, 3);
        assert_eq!(got, want);
        // Sanity: mean of [1,2,3] and [4,5,6] = [2.5, 3.5, 4.5], then L2-normalize.
        let expected_norm = (2.5f32 * 2.5 + 3.5 * 3.5 + 4.5 * 4.5).sqrt();
        let want_vec = vec![
            2.5 / expected_norm,
            3.5 / expected_norm,
            4.5 / expected_norm,
        ];
        for (g, w) in got[0].iter().zip(&want_vec) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn padded_sequence_pools_only_real_tokens() {
        // batch=1, seq=3, dim=2; real tokens at j=0,1; pad at j=2.
        // raw[0] = [1, 2], raw[1] = [3, 4], raw[2] = [100, 100] (must be ignored).
        let raw = raw_view(&[1.0, 2.0, 3.0, 4.0, 100.0, 100.0], 1, 3, 2);
        let mask = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 0.0]).unwrap();
        let got = mean_pool_normalize(&raw, &mask, 1, 3, 2).unwrap();
        let want = reference_mean_pool_normalize(&raw, &mask, 1, 3, 2);
        assert_eq!(got, want, "must match reference (which checks m > 0.0)");
        // mean of [1,2] and [3,4] = [2, 3]; norm = sqrt(13).
        let norm = (4.0f32 + 9.0).sqrt();
        let want_vec = vec![2.0 / norm, 3.0 / norm];
        for (g, w) in got[0].iter().zip(&want_vec) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn all_padded_sequence_yields_zero_not_nan() {
        // real_tokens = 0 → no averaging, no normalization → zero vec.
        let raw = raw_view(&[1.0, 2.0, 3.0, 4.0], 1, 2, 2);
        let mask = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let got = mean_pool_normalize(&raw, &mask, 1, 2, 2).unwrap();
        assert_eq!(
            got,
            vec![vec![0.0f32, 0.0]],
            "all-pad must be zero, not NaN"
        );
        // Verify no NaN/inf leaked in.
        for v in &got[0] {
            assert!(v.is_finite(), "all-pad produced non-finite {v}");
        }
    }

    #[test]
    fn mixed_batch_different_real_lengths_matches_reference() {
        // batch=2, seq=3, dim=2.
        // seq 0: real=2 (mask [1,1,0]) — pools rows 0,1.
        // seq 1: real=3 (mask [1,1,1]) — pools rows 0,1,2.
        let raw = raw_view(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            2,
            3,
            2,
        );
        let mask = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let got = mean_pool_normalize(&raw, &mask, 2, 3, 2).unwrap();
        let want = reference_mean_pool_normalize(&raw, &mask, 2, 3, 2);
        assert_eq!(got, want);
    }

    #[test]
    fn shape_mismatch_returns_error() {
        // raw claims [1,2,3] but caller passes batch=2.
        let raw = raw_view(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 2, 3);
        let mask = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let err = mean_pool_normalize(&raw, &mask, 2, 2, 3).unwrap_err();
        assert!(err.contains("unexpected output shape"), "got: {err}");
    }

    #[test]
    fn non_contiguous_raw_view_still_correct() {
        // Build a [2, 2, 2] array then take a sliced view that breaks
        // last-axis contiguity, forcing the fallback path. We slice
        // `s![0..2, 0..2, 0..2]` from a [2, 2, 4] source — the last axis
        // stride is 4 floats, not 2, so `to_slice()` returns None.
        let source =
            Array3::from_shape_vec((2, 2, 4), (0..16).map(|x| x as f32).collect::<Vec<_>>())
                .unwrap();
        let sliced: ndarray::ArrayView3<f32> = source.slice(s![.., .., 0..2]);
        let raw: ArrayViewD<f32> = sliced.into_dyn();
        let mask = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let got = mean_pool_normalize(&raw, &mask, 2, 2, 2).unwrap();
        let want = reference_mean_pool_normalize(&raw, &mask, 2, 2, 2);
        assert_eq!(
            got, want,
            "non-contiguous fallback must match reference exactly"
        );
    }

    #[test]
    fn large_batch_dim_1024_matches_reference() {
        // Production-shaped: batch=4, seq=8 (4 real), dim=1024.
        // Catches any SIMD-path numerical drift the small tests would miss.
        let batch = 4;
        let seq = 8;
        let dim = 1024;
        let mut data = Vec::with_capacity(batch * seq * dim);
        for i in 0..(batch * seq * dim) {
            // Pseudo-random but deterministic — full-range f32 values.
            data.push(((i as u64).wrapping_mul(2654435761) >> 11) as f32 / 1_000_000.0);
        }
        let raw = raw_view(&data, batch, seq, dim);
        let mut mask_data = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            for j in 0..seq {
                mask_data.push(if j < 4 { 1.0 } else { 0.0 });
            }
        }
        let mask = Array2::from_shape_vec((batch, seq), mask_data).unwrap();
        let got = mean_pool_normalize(&raw, &mask, batch, seq, dim).unwrap();
        let want = reference_mean_pool_normalize(&raw, &mask, batch, seq, dim);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            for (k, (gv, wv)) in g.iter().zip(w).enumerate() {
                assert!(
                    (gv - wv).abs() < 1e-5,
                    "drift at batch={i} dim={k}: got {gv}, want {wv}"
                );
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "mask has non-prefix 1s")]
    fn debug_assert_fires_on_non_prefix_mask() {
        // Documents the prefix-1s invariant: a mask like [1, 0, 1] triggers
        // the debug_assert in debug builds. In release the assert is a no-op
        // and the function would silently under-pool (sum only the prefix
        // up to the first 0) — the assert makes this loud in dev.
        let raw = raw_view(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 3, 2);
        let mask = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 1.0]).unwrap();
        let _ = mean_pool_normalize(&raw, &mask, 1, 3, 2);
    }
}
