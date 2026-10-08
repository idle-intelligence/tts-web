//! Validates `tts_core::tokenizer::Tokenizer` against golden fixtures produced
//! by the reference Python `sentencepiece` library.
//!
//! See `scripts/pocket-tts/gen_tokenizer_fixtures.py` for how
//! `tests/fixtures/golden.json` was generated.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde::Deserialize;
use tts_core::tokenizer::Tokenizer;

const FIXTURES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[derive(Deserialize)]
struct Case {
    name: String,
    text: String,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct LanguageFixture {
    cases: Vec<Case>,
}

/// Escape control/whitespace characters so mismatched text is legible in
/// failure output.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            ' ' => out.push(' '),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{{{:04x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn pieces_string(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| tokenizer.piece(id))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn golden_fixtures() {
    let golden_path = format!("{FIXTURES_DIR}/golden.json");
    let golden_bytes = std::fs::read(&golden_path)
        .unwrap_or_else(|e| panic!("failed to read {golden_path}: {e}"));
    let golden: HashMap<String, LanguageFixture> = serde_json::from_slice(&golden_bytes)
        .unwrap_or_else(|e| panic!("failed to parse {golden_path}: {e}"));

    let mut languages: Vec<&String> = golden.keys().collect();
    languages.sort();

    let mut failures: Vec<String> = Vec::new();
    let mut total_cases = 0usize;

    for lang in languages {
        let fixture = &golden[lang];

        let model_path = format!("{FIXTURES_DIR}/tokenizers/{lang}.model");
        let model_bytes = std::fs::read(&model_path)
            .unwrap_or_else(|e| panic!("failed to read {model_path}: {e}"));
        let tokenizer = Tokenizer::from_model_bytes(&model_bytes)
            .unwrap_or_else(|e| panic!("failed to load tokenizer for {lang}: {e}"));

        assert_eq!(
            tokenizer.vocab_size(),
            4000,
            "{lang}: expected vocab_size 4000, got {}",
            tokenizer.vocab_size()
        );

        for case in &fixture.cases {
            total_cases += 1;
            let actual_ids = tokenizer.encode(&case.text);

            if actual_ids != case.ids {
                let expected_pieces = pieces_string(&tokenizer, &case.ids);
                let actual_pieces = pieces_string(&tokenizer, &actual_ids);
                failures.push(format!(
                    "[{lang}::{name}] text={text:?} (escaped: \"{escaped}\")\n  \
                     expected ids: {expected_ids:?}\n    actual ids: {actual_ids:?}\n  \
                     expected pieces: {expected_pieces}\n    actual pieces: {actual_pieces}",
                    name = case.name,
                    text = case.text,
                    escaped = escape(&case.text),
                    expected_ids = case.ids,
                    actual_ids = actual_ids,
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} golden tokenizer cases failed:\n\n{}",
            failures.len(),
            total_cases,
            failures.join("\n\n")
        );
    }
}
