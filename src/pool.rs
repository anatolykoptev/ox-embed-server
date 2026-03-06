use ndarray::ArrayViewD;
use tokenizers::Encoding;

/// Build padded i64 tensors for input_ids, attention_mask, token_type_ids.
pub fn build_tensors(
    encodings: &[Encoding],
    batch: usize,
    max_seq: usize,
    pad_id: u32,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let total = batch * max_seq;
    let mut ids = vec![pad_id as i64; total];
    let mut mask = vec![0i64; total];
    let tti = vec![0i64; total]; // always zeros

    for (i, enc) in encodings.iter().enumerate() {
        let token_ids = enc.get_ids();
        let len = token_ids.len().min(max_seq);
        let offset = i * max_seq;
        for j in 0..len {
            ids[offset + j] = token_ids[j] as i64;
            mask[offset + j] = 1;
        }
    }

    (ids, mask, tti)
}

/// Build f32 attention mask for mean pooling.
pub fn build_mask_f32(
    encodings: &[Encoding],
    batch: usize,
    max_seq: usize,
    _pad_id: u32,
) -> Vec<f32> {
    let total = batch * max_seq;
    let mut mask = vec![0.0f32; total];

    for (i, enc) in encodings.iter().enumerate() {
        let len = enc.get_ids().len().min(max_seq);
        let offset = i * max_seq;
        for j in 0..len {
            mask[offset + j] = 1.0;
        }
    }
    mask
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
        let mut count = 0.0f32;

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
            for v in &mut vec {
                *v /= count;
            }
        }

        // L2 normalize.
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        result.push(vec);
    }

    Ok(result)
}
