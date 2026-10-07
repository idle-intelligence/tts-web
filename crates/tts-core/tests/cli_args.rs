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
    // --model and --tokenizer are now optional (they default to downloading
    // by --language), so only --text is still required.
    let err = tts_generate::parse_args(&args(&[])).unwrap_err();
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

    assert_eq!(parsed.model_path.as_deref(), Some("model.gguf"));
    assert_eq!(parsed.tokenizer_path.as_deref(), Some("tokenizer.model"));
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
    assert_eq!(parsed.language, "english");
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

#[test]
fn model_tokenizer_omitted_downloads_by_language() {
    let parsed = tts_generate::parse_args(&args(&[
        "--language",
        "italian",
        "--voice",
        "giovanni",
        "--text",
        "Ciao",
    ]))
    .expect("should parse");

    assert_eq!(parsed.model_path, None);
    assert_eq!(parsed.tokenizer_path, None);
    assert_eq!(parsed.language, "italian");
    assert_eq!(parsed.voice_path.as_deref(), Some("giovanni"));
}

#[test]
fn unknown_language_produces_clear_error_listing_valid_ones() {
    let err = tts_generate::parse_args(&args(&["--language", "klingon", "--text", "Hi"])).unwrap_err();
    assert!(err.contains("klingon"), "unexpected error: {err}");
    for lang in tts_generate::LANGUAGES {
        assert!(err.contains(lang), "error should list {lang}: {err}");
    }
}

#[test]
fn unknown_voice_name_produces_clear_error_listing_valid_ones() {
    let err = tts_generate::parse_args(&args(&["--voice", "nobody", "--text", "Hi"])).unwrap_err();
    assert!(err.contains("nobody"), "unexpected error: {err}");
    for voice in tts_generate::VOICES {
        assert!(err.contains(voice), "error should list {voice}: {err}");
    }
}

#[test]
fn voice_value_that_looks_like_a_path_skips_name_validation() {
    // "nobody.safetensors" is not a known voice name, but it ends in
    // .safetensors so it's treated as a path override, not a name lookup.
    let parsed = tts_generate::parse_args(&args(&[
        "--voice",
        "nobody.safetensors",
        "--text",
        "Hi",
    ]))
    .expect("path-like voice value should bypass name validation");
    assert_eq!(parsed.voice_path.as_deref(), Some("nobody.safetensors"));
}

#[test]
fn list_flag_does_not_require_text() {
    let parsed = tts_generate::parse_args(&args(&["--list"])).expect("--list should parse alone");
    assert!(parsed.list);
}

#[test]
fn default_voice_per_language_matches_web_demo() {
    assert_eq!(tts_generate::default_voice_for_language("english"), "alba");
    assert_eq!(tts_generate::default_voice_for_language("french"), "estelle");
    assert_eq!(tts_generate::default_voice_for_language("german"), "juergen");
    assert_eq!(tts_generate::default_voice_for_language("spanish"), "lola");
    assert_eq!(tts_generate::default_voice_for_language("portuguese"), "rafael");
    assert_eq!(tts_generate::default_voice_for_language("italian"), "giovanni");
}

#[test]
fn urls_use_the_expected_hosts_and_pinned_voice_revision() {
    assert_eq!(
        tts_generate::model_url("italian"),
        "https://huggingface.co/idle-intelligence/pocket-tts-gguf/resolve/main/languages/italian/pocket-tts-q8_0.gguf"
    );
    assert_eq!(
        tts_generate::tokenizer_url("italian"),
        "https://huggingface.co/idle-intelligence/pocket-tts-gguf/resolve/main/languages/italian/tokenizer.model"
    );
    let voice = tts_generate::voice_url("italian", "giovanni");
    assert!(
        voice.starts_with("https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/4e1e0a3e611c51c0b4ed8174fc10f32a54644303/"),
        "voice URL must stay pinned to the exact revision the GGUFs were quantized from: {voice}"
    );
    assert!(voice.ends_with("/languages/italian/embeddings/giovanni.safetensors"));
}

/// Generates a short clip for every language (default voice) and checks it
/// reaches EOS with finite samples. Downloads ~134MB GGUF + tokenizer + voice
/// per language on first run (cached after that). #[ignore] by default —
/// run with `cargo test -p tts-core --release --test cli_args -- --ignored`.
#[test]
#[ignore]
fn every_language_generates_to_eos_with_finite_samples() {
    for language in tts_generate::LANGUAGES {
        let voice = tts_generate::default_voice_for_language(language);
        let args = tts_generate::Args {
            model_path: None,
            tokenizer_path: None,
            text: "Hello, this is a short test.".to_string(),
            voice_path: None,
            output_path: "/tmp/test_tts_ignored.wav".to_string(),
            temperature: 0.7,
            language: language.to_string(),
            safetensors: false,
            list: false,
        };

        let t0 = std::time::Instant::now();
        let result = tts_generate::generate(&args)
            .unwrap_or_else(|e| panic!("generation failed for {language} ({voice}): {e}"));
        let elapsed = t0.elapsed().as_secs_f32();

        assert!(result.eos_reached, "{language} ({voice}) never reached EOS");
        assert!(!result.samples.is_empty(), "{language} ({voice}) produced no samples");
        assert!(
            result.samples.iter().all(|s| s.is_finite()),
            "{language} ({voice}) produced non-finite samples"
        );

        let duration_s = result.samples.len() as f64 / result.sample_rate as f64;
        eprintln!(
            "{language} ({voice}): {} steps, {:.2}s audio, {:.2}s wall",
            result.total_steps, duration_s, elapsed
        );
    }
}
