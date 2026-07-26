//! Tests for `tts_generate`'s CLI argument parsing.
//!
//! Compiles the example file as a module so `parse_args` can be exercised
//! directly, without needing the 134 MB model file present.

#[path = "../examples/tts_generate.rs"]
#[allow(dead_code)]
mod tts_generate;

fn args(v: &[&str]) -> Vec<String> {
    std::iter::once("tts_generate")
        .chain(v.iter().copied())
        .map(String::from)
        .collect()
}

#[test]
fn missing_required_args_produces_clear_error() {
    let err = tts_generate::parse_args(&args(&[])).unwrap_err();
    assert!(err.contains("--model"), "unexpected error: {err}");

    let err = tts_generate::parse_args(&args(&["--model", "m.gguf"])).unwrap_err();
    assert!(err.contains("--tokenizer"), "unexpected error: {err}");

    let err = tts_generate::parse_args(&args(&[
        "--model",
        "m.gguf",
        "--tokenizer",
        "tok.model",
    ]))
    .unwrap_err();
    assert!(err.contains("--text"), "unexpected error: {err}");
}

#[test]
fn all_args_present_parses_correctly() {
    let parsed = tts_generate::parse_args(&args(&[
        "--model",
        "model.gguf",
        "--tokenizer",
        "tokenizer.model",
        "--text",
        "Hello, world.",
        "--voice",
        "alba.safetensors",
        "--output",
        "/tmp/out.wav",
        "--temperature",
        "0.9",
    ]))
    .expect("should parse");

    assert_eq!(parsed.model_path, "model.gguf");
    assert_eq!(parsed.tokenizer_path, "tokenizer.model");
    assert_eq!(parsed.text, "Hello, world.");
    assert_eq!(parsed.voice_path.as_deref(), Some("alba.safetensors"));
    assert_eq!(parsed.output_path, "/tmp/out.wav");
    assert_eq!(parsed.temperature, 0.9);
}

#[test]
fn required_args_without_voice_or_output_use_defaults() {
    let parsed = tts_generate::parse_args(&args(&[
        "--model",
        "model.gguf",
        "--tokenizer",
        "tokenizer.model",
        "--text",
        "Hi",
    ]))
    .expect("should parse");

    assert_eq!(parsed.voice_path, None);
    assert_eq!(parsed.output_path, "/tmp/test_tts.wav");
    assert_eq!(parsed.temperature, 0.7);
}

#[test]
fn unknown_flag_produces_clear_error() {
    let err = tts_generate::parse_args(&args(&[
        "--model",
        "model.gguf",
        "--tokenizer",
        "tokenizer.model",
        "--text",
        "Hi",
        "--bogus-flag",
    ]))
    .unwrap_err();
    assert!(err.contains("--bogus-flag"), "unexpected error: {err}");
}
