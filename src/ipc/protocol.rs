use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerRequest {
    Embed(EmbedRequest),
    Rerank(RerankRequest),
    Splade(SpladeRequest),
}

impl WorkerRequest {
    pub fn request_id(&self) -> u64 {
        match self {
            Self::Embed(r) => r.request_id,
            Self::Rerank(r) => r.request_id,
            Self::Splade(r) => r.request_id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Embed(_) => "embed",
            Self::Rerank(_) => "rerank",
            Self::Splade(_) => "splade",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedRequest {
    pub request_id: u64,
    pub model: String,
    pub texts: Vec<String>,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankRequest {
    pub request_id: u64,
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpladeRequest {
    pub request_id: u64,
    pub model: String,
    pub texts: Vec<String>,
    pub max_seq_len: u32,
    /// Maximum sparse entries per output. 0 means unlimited (pass full vocab
    /// size to the model; the model will apply its own default).
    pub top_k: u32,
    /// Drop sparse entries with weight <= this threshold. 0.0 disables filtering.
    pub min_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerResponse {
    Embed(EmbedResponseOk),
    Rerank(RerankResponseOk),
    Splade(SpladeResponseOk),
    Err { request_id: u64, message: String },
}

impl WorkerResponse {
    pub fn request_id(&self) -> u64 {
        match self {
            Self::Embed(r) => r.request_id,
            Self::Rerank(r) => r.request_id,
            Self::Splade(r) => r.request_id,
            Self::Err { request_id, .. } => *request_id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Embed(_) => "embed",
            Self::Rerank(_) => "rerank",
            Self::Splade(_) => "splade",
            Self::Err { .. } => "err",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedResponseOk {
    pub request_id: u64,
    pub vectors: Vec<Vec<f32>>,
    pub dims: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankResponseOk {
    pub request_id: u64,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpladeResponseOk {
    pub request_id: u64,
    pub sparse: Vec<Vec<(u32, f32)>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<
        T: serde::Serialize + for<'de> serde::Deserialize<'de> + PartialEq + std::fmt::Debug,
    >(
        v: T,
    ) {
        let bytes = postcard::to_allocvec(&v).unwrap();
        let decoded: T = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn roundtrip_embed() {
        roundtrip(WorkerRequest::Embed(EmbedRequest {
            request_id: 1,
            model: "e5".into(),
            texts: vec!["hi".into()],
            max_seq_len: 128,
        }));
        roundtrip(WorkerResponse::Embed(EmbedResponseOk {
            request_id: 1,
            vectors: vec![vec![0.1, 0.2]],
            dims: 2,
        }));
    }

    #[test]
    fn roundtrip_rerank() {
        roundtrip(WorkerRequest::Rerank(RerankRequest {
            request_id: 2,
            model: "gte".into(),
            query: "q".into(),
            documents: vec!["d1".into(), "d2".into()],
            max_seq_len: 256,
        }));
        roundtrip(WorkerResponse::Rerank(RerankResponseOk {
            request_id: 2,
            scores: vec![0.9, 0.1],
        }));
    }

    #[test]
    fn roundtrip_splade() {
        roundtrip(WorkerRequest::Splade(SpladeRequest {
            request_id: 3,
            model: "splade".into(),
            texts: vec!["hi".into()],
            max_seq_len: 128,
            top_k: 256,
            min_weight: 0.01,
        }));
        roundtrip(WorkerResponse::Splade(SpladeResponseOk {
            request_id: 3,
            sparse: vec![vec![(42, 0.5), (100, 0.3)]],
        }));
    }

    #[test]
    fn roundtrip_err() {
        roundtrip(WorkerResponse::Err {
            request_id: 7,
            message: "boom".into(),
        });
    }

    #[test]
    fn request_id_accessors() {
        let r = WorkerRequest::Embed(EmbedRequest {
            request_id: 99,
            model: "x".into(),
            texts: vec![],
            max_seq_len: 0,
        });
        assert_eq!(r.request_id(), 99);
        assert_eq!(r.kind(), "embed");
        let resp = WorkerResponse::Rerank(RerankResponseOk {
            request_id: 99,
            scores: vec![],
        });
        assert_eq!(resp.request_id(), 99);
    }

    #[test]
    fn response_kind_all_variants() {
        assert_eq!(
            WorkerResponse::Embed(EmbedResponseOk {
                request_id: 1,
                vectors: vec![],
                dims: 0
            })
            .kind(),
            "embed"
        );
        assert_eq!(
            WorkerResponse::Rerank(RerankResponseOk {
                request_id: 1,
                scores: vec![]
            })
            .kind(),
            "rerank"
        );
        assert_eq!(
            WorkerResponse::Splade(SpladeResponseOk {
                request_id: 1,
                sparse: vec![]
            })
            .kind(),
            "splade"
        );
        assert_eq!(
            WorkerResponse::Err {
                request_id: 1,
                message: "boom".into()
            }
            .kind(),
            "err"
        );
    }
}
