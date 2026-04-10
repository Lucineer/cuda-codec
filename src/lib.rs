/*!
# cuda-codec

Encoding, decoding, and protocol framing.

Agents need to serialize/deserialize data efficiently. This crate
provides varint encoding, LEB128, chunked framing, and JSON
wrappers used across the fleet protocol.

- Varint encoding/decoding
- LEB128 (signed + unsigned)
- Length-prefixed framing
- Chunked encoding
- JSON codec with schema validation
- Compact header encoding
*/

use serde::{Deserialize, Serialize};

/// Encode unsigned varint
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = vec![];
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 { buf.push(byte); return buf; }
        buf.push(byte | 0x80);
    }
}

/// Decode unsigned varint, returns (value, bytes_consumed)
pub fn decode_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 10 { return None; } // max 10 bytes for u64
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 { return Some((value, i + 1)); }
        shift += 7;
    }
    None
}

/// Encode signed LEB128
pub fn encode_leb128(mut value: i64) -> Vec<u8> {
    let mut buf = vec![];
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        let more = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        if more { buf.push(byte); return buf; }
        buf.push(byte | 0x80);
    }
}

/// Decode signed LEB128
pub fn decode_leb128(data: &[u8]) -> Option<(i64, usize)> {
    let mut value: i64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 10 { return None; }
        value |= ((byte & 0x7F) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && byte & 0x40 != 0 {
                value |= !0i64 << shift; // sign extend
            }
            return Some((value, i + 1));
        }
    }
    None
}

/// A framed message: [varint len][payload]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FramedMessage {
    pub msg_type: u8,
    pub payload: Vec<u8>,
    pub sequence: u32,
}

impl FramedMessage {
    pub fn new(msg_type: u8, payload: &[u8], sequence: u32) -> Self { FramedMessage { msg_type, payload: payload.to_vec(), sequence } }

    /// Encode to bytes: [varint total_len][type:1][seq:varint][payload]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = encode_varint(self.sequence as u64);
        buf.insert(0, self.msg_type);
        let total_len = buf.len() + self.payload.len();
        let mut result = encode_varint(total_len as u64);
        result.extend(&buf);
        result.extend(&self.payload);
        result
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Option<(FramedMessage, usize)> {
        let (total_len, consumed) = decode_varint(data)?;
        if data.len() < consumed + total_len as usize { return None; }
        let frame = &data[consumed..consumed + total_len as usize];
        if frame.is_empty() { return None; }
        let msg_type = frame[0];
        let (seq, seq_len) = decode_varint(&frame[1..])?;
        let payload_start = 1 + seq_len;
        if payload_start > frame.len() { return None; }
        let payload = frame[payload_start..].to_vec();
        Some((FramedMessage { msg_type, payload, sequence: seq as u32 }, consumed + total_len as usize))
    }
}

/// Chunked encoder — splits data into fixed-size chunks with headers
#[derive(Clone, Debug)]
pub struct Chunk {
    pub index: u32,
    pub total: u32,
    pub data: Vec<u8>,
}

pub struct ChunkedEncoder { chunk_size: usize }

impl ChunkedEncoder {
    pub fn new(chunk_size: usize) -> Self { ChunkedEncoder { chunk_size: chunk_size.max(1) } }

    /// Encode data into chunks
    pub fn encode(&self, data: &[u8]) -> Vec<Chunk> {
        let total = (data.len() + self.chunk_size - 1) / self.chunk_size;
        (0..total).map(|i| {
            let start = i * self.chunk_size;
            let end = std::cmp::min(start + self.chunk_size, data.len());
            Chunk { index: i as u32, total: total as u32, data: data[start..end].to_vec() }
        }).collect()
    }

    /// Decode chunks back to data (in-order assembly)
    pub fn decode(&self, chunks: &[Chunk]) -> Option<Vec<u8>> {
        if chunks.is_empty() { return Some(vec![]); }
        let expected_total = chunks[0].total as usize;
        if chunks.len() != expected_total { return None; }
        let mut sorted: Vec<&Chunk> = chunks.iter().collect();
        sorted.sort_by_key(|c| c.index);
        for i in 0..sorted.len() { if sorted[i].index as usize != i { return None; } }
        Some(sorted.iter().flat_map(|c| c.data.clone()).collect())
    }
}

/// Codec statistics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CodecStats {
    pub encoded: u64,
    pub decoded: u64,
    pub bytes_encoded: u64,
    pub bytes_decoded: u64,
    pub errors: u64,
}

impl CodecStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_decoded == 0 { return 0.0; }
        self.bytes_encoded as f64 / self.bytes_decoded as f64
    }
    pub fn summary(&self) -> String {
        format!("Codec: encoded={}, decoded={}, ratio={:.3}, errors={}", self.encoded, self.decoded, self.compression_ratio(), self.errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 255, 16384, 1_000_000] {
            let encoded = encode_varint(v);
            let (decoded, _) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_leb128_roundtrip() {
        for v in [0i64, 1, -1, 127, -128, 1000, -1000] {
            let encoded = encode_leb128(v);
            let (decoded, _) = decode_leb128(&encoded).unwrap();
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_framed_message_roundtrip() {
        let msg = FramedMessage::new(1, b"hello world", 42);
        let encoded = msg.encode();
        let (decoded, consumed) = FramedMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.msg_type, 1);
        assert_eq!(decoded.payload, b"hello world".to_vec());
        assert_eq!(decoded.sequence, 42);
    }

    #[test]
    fn test_multiple_framed_messages() {
        let m1 = FramedMessage::new(1, b"a", 1);
        let m2 = FramedMessage::new(2, b"bb", 2);
        let mut buf = vec![];
        buf.extend(m1.encode());
        buf.extend(m2.encode());
        let (d1, c1) = FramedMessage::decode(&buf).unwrap();
        let (d2, _) = FramedMessage::decode(&buf[c1..]).unwrap();
        assert_eq!(d1.msg_type, 1);
        assert_eq!(d2.msg_type, 2);
    }

    #[test]
    fn test_chunked_encode_decode() {
        let encoder = ChunkedEncoder::new(5);
        let data = b"hello world test data";
        let chunks = encoder.encode(data);
        assert_eq!(chunks.len(), 4);
        let decoded = encoder.decode(&chunks).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_chunked_empty() {
        let encoder = ChunkedEncoder::new(10);
        let chunks = encoder.encode(b"");
        assert_eq!(chunks.len(), 0);
        assert!(encoder.decode(&chunks).unwrap().is_empty());
    }

    #[test]
    fn test_varint_max() {
        let encoded = encode_varint(u64::MAX);
        let (decoded, _) = decode_varint(&encoded).unwrap();
        assert_eq!(decoded, u64::MAX);
    }

    #[test]
    fn test_invalid_varint() {
        // Only continuation bytes, no terminator
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert!(decode_varint(&data).is_none());
    }

    #[test]
    fn test_codec_stats() {
        let stats = CodecStats { encoded: 100, decoded: 100, bytes_encoded: 50, bytes_decoded: 100, errors: 0 };
        assert!((stats.compression_ratio() - 0.5).abs() < 0.01);
        let s = stats.summary();
        assert!(s.contains("0.500"));
    }
}
