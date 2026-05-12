use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    pub request_id: u64,
    pub model: String,
    pub texts: Vec<String>,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InferResponse {
    Ok {
        request_id: u64,
        vectors: Vec<Vec<f32>>,
        dims: u32,
    },
    Err {
        request_id: u64,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlMessage {
    Ping,
    Pong,
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_infer_request() {
        let req = InferRequest {
            request_id: 42,
            model: "jina-code-v2".into(),
            texts: vec!["fn main() {}".into()],
            max_seq_len: 512,
        };
        let bytes = bincode::serde::encode_to_vec(&req, bincode::config::standard()).unwrap();
        let (decoded, _): (InferRequest, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(req, decoded);
    }

    #[test]
    fn roundtrip_infer_response_ok() {
        let resp = InferResponse::Ok {
            request_id: 1,
            vectors: vec![vec![0.1, 0.2, 0.3]],
            dims: 3,
        };
        let bytes = bincode::serde::encode_to_vec(&resp, bincode::config::standard()).unwrap();
        let (decoded, _): (InferResponse, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn roundtrip_infer_response_err() {
        let resp = InferResponse::Err {
            request_id: 7,
            message: "boom".into(),
        };
        let bytes = bincode::serde::encode_to_vec(&resp, bincode::config::standard()).unwrap();
        let (decoded, _): (InferResponse, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn roundtrip_control_message_all_variants() {
        for msg in [
            ControlMessage::Ping,
            ControlMessage::Pong,
            ControlMessage::Shutdown,
        ] {
            let bytes = bincode::serde::encode_to_vec(&msg, bincode::config::standard()).unwrap();
            let (decoded, _): (ControlMessage, _) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(msg, decoded);
        }
    }
}
