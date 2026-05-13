use ndarray::ArrayViewD;

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

    for i in 0..batch {
        let mut vec = vec![0.0f32; dim];
        let mut count = 0.0f64; // f64 to match Go's float64 accumulation

        for j in 0..max_seq {
            let m = mask[[i, j]];
            if m > 0.0 {
                count += 1.0;
                for k in 0..dim {
                    vec[k] += raw[[i, j, k]];
                }
            }
        }

        // Average over non-padded tokens.
        if count > 0.0 {
            let inv = (1.0 / count) as f32;
            for v in &mut vec {
                *v *= inv;
            }
        }

        // L2 normalize (accumulate in f64 to match Go's math.Sqrt(float64)).
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

    Ok(result)
}
