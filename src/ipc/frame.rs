use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024; // 64 MiB safety cap

/// Writes a length-prefixed bincode-encoded message to the stream.
pub async fn write_frame<W, T>(stream: &mut W, value: &T) -> io::Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: serde::Serialize,
{
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if bytes.len() > MAX_FRAME_BYTES as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    stream.write_u32_le(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;
    stream.flush().await
}

/// Reads a length-prefixed bincode-encoded message from the stream.
pub async fn read_frame<R, T>(stream: &mut R) -> io::Result<T>
where
    R: AsyncReadExt + Unpin,
    T: for<'de> serde::Deserialize<'de>,
{
    let len = stream.read_u32_le().await?;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;
    let (value, _): (T, _) = bincode::serde::decode_from_slice(&buf, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::protocol::InferRequest;

    #[tokio::test]
    async fn frame_roundtrip_through_pipe() {
        // 8192 is enough — the payload here is < 100 bytes. Larger payloads would
        // deadlock without spawn (write blocks until reader drains).
        let (mut a, mut b) = tokio::io::duplex(8192);
        let req = InferRequest {
            request_id: 1,
            model: "e5".into(),
            texts: vec!["hello".into()],
            max_seq_len: 128,
        };
        write_frame(&mut a, &req).await.unwrap();
        let decoded: InferRequest = read_frame(&mut b).await.unwrap();
        assert_eq!(req, decoded);
    }

    #[tokio::test]
    async fn read_frame_rejects_oversize_header() {
        use tokio::io::AsyncWriteExt;
        let (mut a, mut b) = tokio::io::duplex(64);
        // Write a length-prefix one byte over the cap, then close to make read fail fast.
        a.write_u32_le(MAX_FRAME_BYTES + 1).await.unwrap();
        drop(a); // close write half
        let err = read_frame::<_, crate::ipc::protocol::InferRequest>(&mut b)
            .await
            .expect_err("should reject oversized frame");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
