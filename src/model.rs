use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::config::ModelDef;
use crate::pool;

/// Wraps an ONNX session + tokenizer for a single embedding model.
pub struct EmbedModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub dim: usize,
    max_len: usize,
    pad_id: u32,
    has_token_type_ids: bool,
}

impl EmbedModel {
    /// Load model from a directory containing model_quantized.onnx
    /// and tokenizer.json.
    pub fn load(def: &ModelDef) -> Result<Self, String> {
        let dir = Path::new(&def.dir);

        let onnx_path = dir.join("model_quantized.onnx");
        if !onnx_path.exists() {
            return Err(format!("ONNX file not found: {}", onnx_path.display()));
        }

        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(format!(
                "tokenizer.json not found: {}",
                tok_path.display()
            ));
        }

        tracing::info!(path = %onnx_path.display(), "creating ONNX session");
        let session = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| format!("set threads: {e}"))?
            .commit_from_file(&onnx_path)
            .map_err(|e| format!("load ONNX {}: {e}", onnx_path.display()))?;
        tracing::info!("ONNX session created");

        tracing::info!(path = %tok_path.display(), "loading tokenizer");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        tracing::info!("tokenizer loaded");

        tracing::info!(
            model = %def.name,
            dim = def.dim,
            max_len = def.max_len,
            pad_id = def.pad_id,
            has_tti = def.has_token_type_ids,
            "loaded model"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: def.dim,
            max_len: def.max_len,
            pad_id: def.pad_id,
            has_token_type_ids: def.has_token_type_ids,
        })
    }

    /// Embed a batch of texts, returning one vector per text.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("tokenize: {e}"))?;

        // Find actual max length in batch, cap at model max.
        let max_seq = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.max_len);

        let batch = encodings.len();
        let (ids, mask, tti) =
            pool::build_tensors(&encodings, batch, max_seq, self.pad_id);

        let ids_arr = Array2::from_shape_vec([batch, max_seq], ids)
            .map_err(|e| format!("ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec([batch, max_seq], mask)
            .map_err(|e| format!("mask shape: {e}"))?;

        let ids_tensor = Tensor::from_array(ids_arr)
            .map_err(|e| format!("ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr)
            .map_err(|e| format!("mask tensor: {e}"))?;

        let inputs = if self.has_token_type_ids {
            let tti_arr = Array2::from_shape_vec([batch, max_seq], tti)
                .map_err(|e| format!("tti shape: {e}"))?;
            let tti_tensor = Tensor::from_array(tti_arr)
                .map_err(|e| format!("tti tensor: {e}"))?;
            ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => tti_tensor,
            }
        } else {
            ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            }
        };

        let mut session = self.session.lock().map_err(|e| format!("lock: {e}"))?;
        let outputs = session
            .run(inputs)
            .map_err(|e| format!("inference: {e}"))?;

        // Output shape: [batch, seq_len, dim]
        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let mask_arr_f = Array2::from_shape_vec(
            [batch, max_seq],
            pool::build_mask_f32(&encodings, batch, max_seq, self.pad_id),
        )
        .map_err(|e| format!("mask_f shape: {e}"))?;

        pool::mean_pool_normalize(&raw, &mask_arr_f, batch, max_seq, self.dim)
    }
}
