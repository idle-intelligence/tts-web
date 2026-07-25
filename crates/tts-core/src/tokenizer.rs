use candle_core::{Error, Result};

const EXPECTED_VOCAB_SIZE: usize = 4000;

struct Piece {
    text: String,
    score: f32,
    kind: u64,
}

fn read_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result = 0u64;
    let mut shift = 0;
    while *pos < data.len() {
        let b = data[*pos];
        *pos += 1;
        result |= ((b & 0x7f) as u64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    result
}

fn decode_piece(data: &[u8]) -> Piece {
    let mut pos = 0;
    let mut text = String::new();
    let mut score = 0.0f32;
    let mut kind = 1u64;
    while pos < data.len() {
        let key = read_varint(data, &mut pos);
        let field_num = key >> 3;
        let wire_type = key & 0x7;
        match (field_num, wire_type) {
            (1, 2) => {
                let len = read_varint(data, &mut pos) as usize;
                text = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
                pos += len;
            }
            (2, 5) => {
                let bytes: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
                score = f32::from_le_bytes(bytes);
                pos += 4;
            }
            (3, 0) => {
                kind = read_varint(data, &mut pos);
            }
            (_, 0) => {
                read_varint(data, &mut pos);
            }
            (_, 1) => pos += 8,
            (_, 2) => {
                let len = read_varint(data, &mut pos) as usize;
                pos += len;
            }
            (_, 5) => pos += 4,
            _ => break,
        }
    }
    Piece { text, score, kind }
}

fn decode_model_proto(bytes: &[u8]) -> Vec<Piece> {
    let mut pos = 0;
    let mut pieces = Vec::new();
    while pos < bytes.len() {
        let key = read_varint(bytes, &mut pos);
        let field_num = key >> 3;
        let wire_type = key & 0x7;
        match (field_num, wire_type) {
            (1, 2) => {
                let len = read_varint(bytes, &mut pos) as usize;
                pieces.push(decode_piece(&bytes[pos..pos + len]));
                pos += len;
            }
            (_, 0) => {
                read_varint(bytes, &mut pos);
            }
            (_, 1) => pos += 8,
            (_, 2) => {
                let len = read_varint(bytes, &mut pos) as usize;
                pos += len;
            }
            (_, 5) => pos += 4,
            _ => break,
        }
    }
    pieces
}

const UNK_SCORE: f32 = -100.0;

pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    vocab: std::collections::HashMap<String, u32>,
    byte_ids: [u32; 256],
    unk_id: u32,
    max_piece_len: usize,
}

impl Tokenizer {
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self> {
        let raw_pieces = decode_model_proto(bytes);

        let mut pieces = Vec::with_capacity(raw_pieces.len());
        let mut scores = Vec::with_capacity(raw_pieces.len());
        let mut vocab = std::collections::HashMap::new();
        let mut byte_ids = [u32::MAX; 256];
        let mut unk_id = 0u32;
        let mut max_piece_len = 0usize;

        for (i, p) in raw_pieces.iter().enumerate() {
            let id = i as u32;
            pieces.push(p.text.clone());
            scores.push(p.score);

            match p.kind {
                2 => unk_id = id,
                1 | 4 | 6 => {
                    vocab.insert(p.text.clone(), id);
                    max_piece_len = max_piece_len.max(p.text.len());
                }
                _ => {}
            }

            if p.kind == 6
                && let Some(byte) = parse_byte_token(&p.text)
            {
                byte_ids[byte as usize] = id;
            }
        }

        if pieces.len() != EXPECTED_VOCAB_SIZE {
            return Err(Error::Msg(format!(
                "sentencepiece model has {} pieces, expected {EXPECTED_VOCAB_SIZE}",
                pieces.len()
            )));
        }

        Ok(Self {
            pieces,
            scores,
            vocab,
            byte_ids,
            unk_id,
            max_piece_len,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> &str {
        &self.pieces[id as usize]
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // sentencepiece skips the dummy prefix entirely for empty input.
        if text.is_empty() {
            return Vec::new();
        }
        let mut normalized = String::with_capacity(text.len() + 3);
        normalized.push('\u{2581}');
        for c in text.chars() {
            if c == ' ' {
                normalized.push('\u{2581}');
            } else {
                normalized.push(c);
            }
        }
        self.viterbi(normalized.as_bytes())
    }

    fn viterbi(&self, text: &[u8]) -> Vec<u32> {
        let n = text.len();

        #[derive(Clone, Copy)]
        struct Best {
            score: f32,
            len: usize,
            id: u32,
        }

        let mut best = vec![
            Best {
                score: f32::NEG_INFINITY,
                len: 0,
                id: 0,
            };
            n + 1
        ];
        best[0] = Best { score: 0.0, len: 0, id: 0 };

        for i in 0..n {
            if best[i].score == f32::NEG_INFINITY {
                continue;
            }
            let max_len = (n - i).min(self.max_piece_len);
            for len in 1..=max_len {
                let sub = &text[i..i + len];
                let Ok(sub_str) = std::str::from_utf8(sub) else {
                    continue;
                };
                if let Some(&id) = self.vocab.get(sub_str) {
                    let new_score = best[i].score + self.scores[id as usize];
                    if new_score > best[i + len].score {
                        best[i + len] = Best { score: new_score, len, id };
                    }
                }
            }
            if best[i + 1].score == f32::NEG_INFINITY {
                let byte = text[i];
                let byte_id = self.byte_ids[byte as usize];
                let (id, fallback_score) = if byte_id != u32::MAX {
                    (byte_id, self.scores[byte_id as usize])
                } else {
                    (self.unk_id, UNK_SCORE)
                };
                best[i + 1] = Best {
                    score: best[i].score + fallback_score,
                    len: 1,
                    id,
                };
            }
        }

        let mut ids = Vec::new();
        let mut p = n;
        while p > 0 {
            ids.push(best[p].id);
            p -= best[p].len;
        }
        ids.reverse();
        ids
    }
}

fn parse_byte_token(s: &str) -> Option<u8> {
    let hex = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_byte_token() {
        assert_eq!(parse_byte_token("<0xC3>"), Some(0xC3));
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("hello"), None);
    }

    #[test]
    fn decodes_a_single_piece() {
        // field 1 (piece, string "ab"), field 2 (score=1.5f32 LE), field 3 (type=1)
        let mut piece_bytes = Vec::new();
        piece_bytes.push((1 << 3) | 2);
        piece_bytes.push(2);
        piece_bytes.extend_from_slice(b"ab");
        piece_bytes.push((2 << 3) | 5);
        piece_bytes.extend_from_slice(&1.5f32.to_le_bytes());
        piece_bytes.push(3 << 3);
        piece_bytes.push(1);

        let mut model_bytes = Vec::new();
        model_bytes.push((1 << 3) | 2);
        model_bytes.push(piece_bytes.len() as u8);
        model_bytes.extend_from_slice(&piece_bytes);

        let pieces = decode_model_proto(&model_bytes);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].text, "ab");
        assert_eq!(pieces[0].score, 1.5);
        assert_eq!(pieces[0].kind, 1);
    }
}
