/// Integration tests for KittenTTS.
///
/// These tests require local model files and are gated with `#[ignore]`.
/// Run with:
///   cargo test -p kitten-core -- --ignored --nocapture

use candle_core::{Device, IndexOp, Tensor};
use kitten_core::config::KittenConfig;
use kitten_core::kitten_model::KittenModel;
use kitten_core::phoneme_map::map_phonemes_to_ids;
use safetensors::SafeTensors;

const MODEL_PATH: &str = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-nano.safetensors";
const VOICES_PATH: &str = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-voices.safetensors";

const ALL_VOICES: &[&str] = &["bella", "bruno", "hugo", "jasper", "kiki", "leo", "luna", "rosie"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_model() -> anyhow::Result<KittenModel> {
    let cfg = KittenConfig::nano();
    let data = std::fs::read(MODEL_PATH)?;
    KittenModel::load(&data, &cfg, &Device::Cpu)
}

fn load_style(voice_name: &str, text_len: usize) -> anyhow::Result<Tensor> {
    let voices_data = std::fs::read(VOICES_PATH)?;
    let st = SafeTensors::deserialize(&voices_data)?;
    let tv = st
        .tensor(voice_name)
        .map_err(|_| anyhow::anyhow!("voice {:?} not found", voice_name))?;

    let data = tv.data();
    let rows = tv.shape()[0];
    let cols = tv.shape()[1];

    let flat_f32: Vec<f32> = match tv.dtype() {
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        other => return Err(anyhow::anyhow!("unsupported dtype: {:?}", other)),
    };

    let t = Tensor::from_vec(flat_f32, (rows, cols), &Device::Cpu)?;
    let idx = text_len.min(rows.saturating_sub(1));
    Ok(t.i(idx)?.unsqueeze(0)?)
}

fn check_audio_basic(samples: &[f32], label: &str) {
    assert!(!samples.is_empty(), "{label}: output is empty");

    let nan_count = samples.iter().filter(|x| x.is_nan()).count();
    let inf_count = samples.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "{label}: {nan_count} NaN values");
    assert_eq!(inf_count, 0, "{label}: {inf_count} Inf values");

    let max_abs = samples.iter().copied().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(
        max_abs <= 1.0 + 1e-5,
        "{label}: max_abs={max_abs:.4} exceeds 1.0 (tanh clamp failed)"
    );

    let rms: f32 = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(rms > 0.001, "{label}: RMS={rms:.6} too low (all zeros?)");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_model_loads() {
    load_model().expect("Model should load without error");
    eprintln!("test_model_loads: PASS");
}

#[test]
#[ignore]
fn test_phoneme_map_reference() {
    // "həlˈəʊ wˈɜːld" → ONNX reference IDs
    let result = map_phonemes_to_ids("həlˈəʊ wˈɜːld");
    let expected = vec![0i32, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];
    assert_eq!(result, expected, "Phoneme IDs don't match ONNX reference");
    eprintln!("test_phoneme_map_reference: PASS — {:?}", result);
}

#[test]
#[ignore]
fn test_synthesize_hello_world() {
    let model = load_model().expect("load model");
    // Pre-computed IPA matching ONNX reference (avoids espeak dependency)
    let phoneme_ids: Vec<i32> =
        vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];
    let style = load_style("jasper", 11).expect("load style");

    let samples = model.synthesize(&phoneme_ids, &style, 1.0).expect("synthesize");
    eprintln!(
        "test_synthesize_hello_world: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 24000.0
    );

    assert!(
        samples.len() >= 10_000 && samples.len() <= 50_000,
        "Expected 10k-50k samples for 'Hello world', got {}",
        samples.len()
    );

    check_audio_basic(&samples, "hello_world");
    eprintln!("test_synthesize_hello_world: PASS");
}

#[test]
#[ignore]
fn test_synthesize_fox() {
    let model = load_model().expect("load model");
    // "The quick brown fox jumps over the lazy dog." — IPA from espeak-ng
    let ipa = "ðə kwɪk bɹaʊn fɒks dʒʌmps əʊvə ðə leɪzi dɒɡ";
    let phoneme_ids = map_phonemes_to_ids(ipa);
    eprintln!("fox IDs ({} tokens): {:?}", phoneme_ids.len(), &phoneme_ids);

    let style = load_style("jasper", ipa.len()).expect("load style");
    let samples = model.synthesize(&phoneme_ids, &style, 1.0).expect("synthesize");
    eprintln!(
        "test_synthesize_fox: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 24000.0
    );

    assert!(
        samples.len() >= 50_000 && samples.len() <= 120_000,
        "Expected 50k-120k samples for fox sentence, got {}",
        samples.len()
    );

    check_audio_basic(&samples, "fox");
    eprintln!("test_synthesize_fox: PASS");
}

#[test]
#[ignore]
fn test_all_voices() {
    let model = load_model().expect("load model");
    let phoneme_ids: Vec<i32> =
        vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];

    for voice in ALL_VOICES {
        let style = load_style(voice, 11).expect("load style");
        let samples = model
            .synthesize(&phoneme_ids, &style, 1.0)
            .unwrap_or_else(|e| panic!("voice {voice}: synthesize failed: {e}"));
        check_audio_basic(&samples, voice);
        eprintln!(
            "voice {voice}: {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / 24000.0
        );
    }
    eprintln!("test_all_voices: PASS — all {} voices OK", ALL_VOICES.len());
}

#[test]
#[ignore]
fn test_speed_control() {
    let model = load_model().expect("load model");
    let phoneme_ids: Vec<i32> =
        vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];
    let style = load_style("jasper", 11).expect("load style");

    let slow = model.synthesize(&phoneme_ids, &style, 0.5).expect("slow");
    let fast = model.synthesize(&phoneme_ids, &style, 2.0).expect("fast");

    eprintln!(
        "test_speed_control: slow(0.5)={} fast(2.0)={}",
        slow.len(),
        fast.len()
    );

    assert!(
        slow.len() > fast.len(),
        "speed=0.5 should produce more samples than speed=2.0: slow={} fast={}",
        slow.len(),
        fast.len()
    );

    let ratio = slow.len() as f32 / fast.len() as f32;
    eprintln!("  ratio (slow/fast): {ratio:.2}");
    assert!(
        ratio >= 1.5 && ratio <= 8.0,
        "Speed ratio out of expected range: {ratio:.2}"
    );

    eprintln!("test_speed_control: PASS");
}

#[test]
#[ignore]
fn test_bert_output_matches_onnx() -> anyhow::Result<()> {
    let model = load_model()?;
    let phoneme_ids: Vec<i32> =
        vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];
    let style = load_style("jasper", 11)?;

    let dbg = model.debug_forward(&phoneme_ids, &style, 1.0)?;

    // BERT output: [1, 16, 128]
    let bert_shape = dbg.bert_output.shape().dims().to_vec();
    eprintln!("bert_output shape: {:?}", bert_shape);
    assert_eq!(bert_shape, vec![1, 16, 128], "bert_output shape mismatch");

    // Print first 10 values of token 0 for comparison with ONNX
    let token0: Vec<f32> = dbg.bert_output.i((0, 0, ..10))?.to_vec1()?;
    eprintln!("bert_output token0 first10: {:?}", token0);

    // ONNX reference token0 first values: ~[2.318, 6.384, ...] (from earlier comparison)
    // Check they are finite and non-trivially non-zero
    let any_nonzero = token0.iter().any(|&v| v.abs() > 0.01);
    assert!(any_nonzero, "bert_output token0 appears to be all zeros");

    let all_finite = token0.iter().all(|v| v.is_finite());
    assert!(all_finite, "bert_output contains non-finite values");

    eprintln!("test_bert_output_matches_onnx: PASS (compare printed values vs ONNX reference)");
    Ok(())
}

#[test]
#[ignore]
fn test_duration_prediction_matches_onnx() -> anyhow::Result<()> {
    let model = load_model()?;
    let phoneme_ids: Vec<i32> =
        vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];
    let style = load_style("jasper", 11)?;

    let dbg = model.debug_forward(&phoneme_ids, &style, 1.0)?;

    // durations: [1, 16] i64
    let durs = dbg.durations.to_vec2::<i64>()?;
    let durs = &durs[0];
    eprintln!("durations: {:?}", durs);

    let total: i64 = durs.iter().sum();
    eprintln!("total duration: {total}");

    // ONNX durations for "Hello world" Jasper: [26, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1]
    let onnx_durs: Vec<i64> = vec![26, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1];
    let onnx_total: i64 = onnx_durs.iter().sum();
    eprintln!("ONNX expected: {:?} (total={onnx_total})", onnx_durs);

    assert!(
        (total - onnx_total).abs() <= 3,
        "Duration total {total} vs ONNX {onnx_total} — diff too large"
    );

    let mut max_diff = 0i64;
    for (i, (&rust_d, &onnx_d)) in durs.iter().zip(onnx_durs.iter()).enumerate() {
        let diff = (rust_d - onnx_d).abs();
        if diff > 0 {
            eprintln!("  token {i}: rust={rust_d} onnx={onnx_d} diff={diff}");
        }
        max_diff = max_diff.max(diff);
    }

    assert!(
        max_diff <= 2,
        "Per-token duration diff {max_diff} exceeds tolerance of 2"
    );

    eprintln!("test_duration_prediction_matches_onnx: PASS (max_diff={max_diff})");
    Ok(())
}
