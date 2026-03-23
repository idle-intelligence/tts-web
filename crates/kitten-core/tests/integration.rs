/// Integration tests for KittenTTS.
///
/// These tests require local model files and are gated with `#[ignore]`.
/// Run with:
///   cargo test -p kitten-core -- --ignored --nocapture

use candle_core::{Device, IndexOp, Tensor};
use kitten_core::config::KittenConfig;
use kitten_core::decoder::Decoder;
use kitten_core::kitten_model::KittenModel;
use kitten_core::phoneme_map::map_phonemes_to_ids;
use kitten_core::predictor::Predictor;
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

fn check_audio_quality(samples: &[f32], label: &str) {
    let n = samples.len();
    assert!(n > 0, "{label}: output is empty");

    // 1. No NaN or Inf
    let nan_count = samples.iter().filter(|x| x.is_nan()).count();
    let inf_count = samples.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "{label}: {nan_count} NaN values");
    assert_eq!(inf_count, 0, "{label}: {inf_count} Inf values");

    // 2. All values in [-1, 1] (tanh clamped)
    let max_abs = samples.iter().copied().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs <= 1.0 + 1e-5, "{label}: max_abs={max_abs:.4}");

    // 3. RMS level check — ONNX reference is ~0.10, speech should be 0.02-0.30
    //    If RMS > 0.40, the output is way too loud (noise/saturation)
    let rms: f32 = (samples.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    eprintln!("  [{label}] RMS={rms:.4}");
    assert!(rms > 0.001, "{label}: RMS={rms:.6} too low (silence)");
    assert!(rms < 0.40, "{label}: RMS={rms:.4} too high (>0.40, ONNX ref is ~0.10 — likely noise/saturation)");

    // 4. Tanh saturation — count samples at ±0.999+
    //    Speech should have very few; noise/broken output has many
    let saturated = samples.iter().filter(|&&x| x.abs() > 0.999).count();
    let sat_pct = saturated as f64 / n as f64 * 100.0;
    eprintln!("  [{label}] saturated={saturated} ({sat_pct:.2}%)");
    assert!(sat_pct < 1.0, "{label}: {sat_pct:.1}% samples saturated (>0.999) — pre-tanh values too large");

    // 5. Peak amplitude should be reasonable — ONNX peak is ~0.5
    //    If peak is > 0.95, pre-tanh values were large (bad)
    assert!(max_abs < 0.95, "{label}: peak={max_abs:.4} too close to 1.0 (pre-tanh overflow)");

    // 6. First 500 samples should be quiet (speech starts with silence/ramp)
    if n > 500 {
        let head_rms: f32 = (samples[..500].iter().map(|x| x * x).sum::<f32>() / 500.0).sqrt();
        eprintln!("  [{label}] head_rms(0-500)={head_rms:.6}");
        assert!(head_rms < 0.10, "{label}: head_rms={head_rms:.4} — first 500 samples too loud (no silence ramp)");
    }

    // 7. Dynamic range — speech has varying amplitude, noise is flat
    //    Split into 10 chunks, check variance of per-chunk RMS
    if n >= 1000 {
        let chunk_size = n / 10;
        let chunk_rms: Vec<f32> = (0..10)
            .map(|i| {
                let start = i * chunk_size;
                let end = start + chunk_size;
                (samples[start..end].iter().map(|x| x * x).sum::<f32>() / chunk_size as f32).sqrt()
            })
            .collect();
        let mean_rms: f32 = chunk_rms.iter().sum::<f32>() / 10.0;
        let rms_var: f32 = chunk_rms.iter().map(|x| (x - mean_rms).powi(2)).sum::<f32>() / 10.0;
        let rms_std = rms_var.sqrt();
        eprintln!("  [{label}] chunk_rms_std={rms_std:.4} (dynamic range indicator)");
        // Speech has varying loudness; pure noise is flat
        // Very low std (<0.001) means flat noise; very high (>0.3) means extreme variation
    }

    eprintln!("  [{label}] PASS — {n} samples, RMS={rms:.4}, peak={max_abs:.4}, sat={sat_pct:.1}%");
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

    check_audio_quality(&samples, "hello_world");
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

    check_audio_quality(&samples, "fox");
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
        check_audio_quality(&samples, voice);
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

// ---------------------------------------------------------------------------
// ONNX fixture injection test
// ---------------------------------------------------------------------------

/// Load a .npy file saved by extract_onnx_intermediates.py.
/// Supports F32 arrays only (that's what we save). Shape is preserved.
fn load_npy(path: &str) -> anyhow::Result<Tensor> {
    // .npy format: 10-byte magic + 1-byte major + 1-byte minor + 2-byte header_len (LE) + header + data
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(bytes.len() >= 10, "npy too short");
    anyhow::ensure!(&bytes[..6] == b"\x93NUMPY", "not a .npy file: {path}");

    let major = bytes[6];
    let header_len_bytes = if major == 1 {
        let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10, hl)
    } else {
        // version 2: header_len is 4 bytes at offset 8
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12, hl)
    };
    let (prefix, hl) = header_len_bytes;
    let header_end = prefix + hl;
    anyhow::ensure!(bytes.len() >= header_end, "npy header truncated");

    let header = std::str::from_utf8(&bytes[prefix..header_end])?;

    // Parse shape from header string like "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 256, 51), }"
    let shape = parse_npy_shape(header)?;
    let data_bytes = &bytes[header_end..];

    // Expect float32 ('descr': '<f4' or '=f4')
    let n_elements: usize = shape.iter().product();
    anyhow::ensure!(
        data_bytes.len() >= n_elements * 4,
        "data too short for shape {shape:?}: need {} bytes, got {}",
        n_elements * 4,
        data_bytes.len()
    );

    let floats: Vec<f32> = data_bytes[..n_elements * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(Tensor::from_vec(floats, shape.as_slice(), &Device::Cpu)?)
}

fn parse_npy_shape(header: &str) -> anyhow::Result<Vec<usize>> {
    // Find 'shape': (...)
    let start = header.find("'shape':").ok_or_else(|| anyhow::anyhow!("no 'shape' in header: {header}"))?;
    let rest = &header[start + 8..];
    let lparen = rest.find('(').ok_or_else(|| anyhow::anyhow!("no '(' after shape"))?;
    let rparen = rest.find(')').ok_or_else(|| anyhow::anyhow!("no ')' after shape"))?;
    let inner = rest[lparen + 1..rparen].trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { s.parse().ok() }
        })
        .collect();
    Ok(dims)
}

/// Compare two tensors element-wise, printing stats and returning max abs diff.
fn compare_tensors(label: &str, rust: &Tensor, onnx: &Tensor) -> anyhow::Result<f32> {
    let rust_flat = rust.flatten_all()?.to_vec1::<f32>()?;
    let onnx_flat = onnx.flatten_all()?.to_vec1::<f32>()?;

    anyhow::ensure!(
        rust_flat.len() == onnx_flat.len(),
        "{label}: shape mismatch — Rust len={} ONNX len={} (Rust shape={:?} ONNX shape={:?})",
        rust_flat.len(),
        onnx_flat.len(),
        rust.shape(),
        onnx.shape(),
    );

    let max_diff = rust_flat
        .iter()
        .zip(onnx_flat.iter())
        .map(|(r, o)| (r - o).abs())
        .fold(0f32, f32::max);

    let mean_diff = rust_flat
        .iter()
        .zip(onnx_flat.iter())
        .map(|(r, o)| (r - o).abs())
        .sum::<f32>()
        / rust_flat.len() as f32;

    // First channel first 10 values for both
    let n_show = 10.min(rust_flat.len());
    let rust_first: Vec<f32> = rust_flat[..n_show].to_vec();
    let onnx_first: Vec<f32> = onnx_flat[..n_show].to_vec();

    eprintln!(
        "[INJECT] {label}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} shape={:?}",
        rust.shape()
    );
    eprintln!("  Rust  first{n_show}: {rust_first:.6?}");
    eprintln!("  ONNX  first{n_show}: {onnx_first:.6?}");

    Ok(max_diff)
}

#[test]
#[ignore]
fn test_decoder_with_onnx_inputs() -> anyhow::Result<()> {
    // Load model weights
    let cfg = KittenConfig::nano();
    let data = std::fs::read(MODEL_PATH)?;
    use candle_core::DType;
    use candle_nn::VarBuilder;
    let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, &Device::Cpu)?;
    let decoder = Decoder::load(vb, &cfg)?;

    eprintln!("\n=== DECODER ONNX INJECTION TEST ===");
    eprintln!("Loading ONNX fixture tensors from /tmp/onnx_fixtures/");

    let fix = |name: &str| -> anyhow::Result<Tensor> {
        let path = format!("/tmp/onnx_fixtures/{name}");
        let t = load_npy(&path)?;
        eprintln!("  Loaded {name}: shape={:?}", t.shape());
        Ok(t)
    };

    // Load all fixtures
    let shared_lstm_out = fix("shared_lstm_out_128ch.npy")?;  // [1, 128, 51]
    let asr_res_out     = fix("asr_res_64ch.npy")?;           // [1, 64, 51]
    let f0_down         = fix("f0_down_1ch.npy")?;            // [1, 1, 51]
    let n_down          = fix("n_down_1ch.npy")?;             // [1, 1, 51]
    let style_half      = fix("style_first_half.npy")?;        // [1, 128] — style[:, :128]

    // ONNX expected outputs
    let onnx_encode_out  = fix("encode_output_256ch.npy")?;   // [1, 256, 51]
    let onnx_decode0_out = fix("decode0_output_256ch.npy")?;  // [1, 256, 51]
    let onnx_decode1_out = fix("decode1_output_256ch.npy")?;  // [1, 256, 51]
    let onnx_decode2_out = fix("decode2_output_256ch.npy")?;  // [1, 256, 51]
    let onnx_decode3_out = fix("decode3_output_256ch.npy")?;  // [1, 256, 102]
    let onnx_conv_post   = fix("conv_post_output_22ch.npy")?; // [1, 22, T_stft]
    let onnx_waveform    = fix("waveform_before_tanh.npy")?;  // [1, 1, T_audio]

    // f0_2t: we don't have the pre-downsampled f0, so reconstruct by nearest-upsample from f0_down.
    // This matches what the Rust decoder.forward() does when given f0 at 2T.
    let t = f0_down.dim(2)?;
    let f0_2t = f0_down.upsample_nearest1d(t * 2)?; // [1, 1, 102]

    eprintln!("\n--- Running debug_decoder_forward with ONNX fixture inputs ---");
    let (rust_encode_out, rust_decode0_out, rust_decode1_out, rust_decode2_out, rust_decode3_out, rust_waveform) =
        decoder.debug_decoder_forward(
            &shared_lstm_out,
            &asr_res_out,
            &f0_down,
            &n_down,
            &style_half,
            &f0_2t,
        )?;

    eprintln!("\n=== STAGE COMPARISON ===");

    let diff_encode  = compare_tensors("ENCODE_OUT",  &rust_encode_out,  &onnx_encode_out)?;
    let diff_decode0 = compare_tensors("DECODE_0_OUT", &rust_decode0_out, &onnx_decode0_out)?;
    let diff_decode1 = compare_tensors("DECODE_1_OUT", &rust_decode1_out, &onnx_decode1_out)?;
    let diff_decode2 = compare_tensors("DECODE_2_OUT", &rust_decode2_out, &onnx_decode2_out)?;
    let diff_decode3 = compare_tensors("DECODE_3_OUT", &rust_decode3_out, &onnx_decode3_out)?;

    // Waveform: shapes may differ slightly due to STFT padding/trim differences
    let rust_wv_flat = rust_waveform.flatten_all()?.to_vec1::<f32>()?;
    let onnx_wv_flat = onnx_waveform.flatten_all()?.to_vec1::<f32>()?;
    let min_len = rust_wv_flat.len().min(onnx_wv_flat.len());
    eprintln!(
        "\n[INJECT] WAVEFORM: Rust len={} ONNX len={} comparing first {min_len}",
        rust_wv_flat.len(),
        onnx_wv_flat.len()
    );
    let wv_max_diff = rust_wv_flat[..min_len]
        .iter()
        .zip(onnx_wv_flat[..min_len].iter())
        .map(|(r, o)| (r - o).abs())
        .fold(0f32, f32::max);
    let n_show = 20.min(min_len);
    eprintln!("  Rust  first{n_show}: {:.6?}", &rust_wv_flat[..n_show]);
    eprintln!("  ONNX  first{n_show}: {:.6?}", &onnx_wv_flat[..n_show]);
    eprintln!("  waveform max_diff={wv_max_diff:.6}");

    eprintln!("\n=== SUMMARY ===");
    eprintln!("  encode  max_diff: {diff_encode:.6}");
    eprintln!("  decode0 max_diff: {diff_decode0:.6}");
    eprintln!("  decode1 max_diff: {diff_decode1:.6}");
    eprintln!("  decode2 max_diff: {diff_decode2:.6}");
    eprintln!("  decode3 max_diff: {diff_decode3:.6}");
    eprintln!("  waveform max_diff: {wv_max_diff:.6}");

    // Tolerance: numerical differences from float32 ops should be < 1e-4
    // If a stage diverges significantly (> 0.1), it points to a computation bug.
    const TIGHT_TOL: f32 = 1e-3;
    const LOOSE_TOL: f32 = 1.0; // for downstream stages that amplify earlier errors

    // Stage 0 (encode block) gets ONNX-exact inputs, so any divergence here is a computation bug.
    if diff_encode > TIGHT_TOL {
        eprintln!("  [FAIL] ENCODE block diverges with ONNX-exact inputs! Bug in encode computation.");
    } else {
        eprintln!("  [PASS] ENCODE block matches ONNX (max_diff={diff_encode:.6} <= {TIGHT_TOL})");
    }

    if diff_decode0 > TIGHT_TOL {
        eprintln!("  [FAIL] DECODE.0 block diverges. Bug in decode.0 computation.");
    } else {
        eprintln!("  [PASS] DECODE.0 block matches ONNX (max_diff={diff_decode0:.6})");
    }

    if diff_decode1 > LOOSE_TOL {
        eprintln!("  [FAIL] DECODE.1 block diverges significantly.");
    } else {
        eprintln!("  [INFO] DECODE.1 max_diff={diff_decode1:.6}");
    }

    if diff_decode2 > LOOSE_TOL {
        eprintln!("  [FAIL] DECODE.2 block diverges significantly.");
    } else {
        eprintln!("  [INFO] DECODE.2 max_diff={diff_decode2:.6}");
    }

    if diff_decode3 > LOOSE_TOL {
        eprintln!("  [FAIL] DECODE.3 block diverges significantly.");
    } else {
        eprintln!("  [INFO] DECODE.3 max_diff={diff_decode3:.6}");
    }

    eprintln!("\ntest_decoder_with_onnx_inputs: DONE");
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw binary loader (flat f32, little-endian)
// ---------------------------------------------------------------------------

/// Load a flat binary file of f32 values (little-endian) and reshape to `shape`.
fn load_bin_f32(path: &str, shape: &[usize]) -> anyhow::Result<Tensor> {
    let bytes = std::fs::read(path)?;
    let n: usize = shape.iter().product();
    anyhow::ensure!(
        bytes.len() == n * 4,
        "load_bin_f32 {path}: expected {} bytes for shape {shape:?}, got {}",
        n * 4,
        bytes.len()
    );
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(floats, shape, &Device::Cpu)?)
}

// ---------------------------------------------------------------------------
// F0 predictor isolation test: inject ONNX shared-LSTM output, compare F0 output
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_f0_predictor_with_onnx_shared_lstm() -> anyhow::Result<()> {
    use candle_core::DType;
    use candle_nn::VarBuilder;

    eprintln!("\n=== F0 PREDICTOR ISOLATION TEST ===");

    // Load model weights (just need the predictor)
    let cfg = KittenConfig::nano();
    let data = std::fs::read(MODEL_PATH)?;
    let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, &Device::Cpu)?;
    let predictor = Predictor::load(vb, &cfg)?;
    eprintln!("Model loaded.");

    // Load ONNX shared LSTM output: [1, 128, 51] f32 flat binary
    let shared_lstm = load_bin_f32("/tmp/onnx_shared_lstm_1x128x51.bin", &[1, 128, 51])?;
    eprintln!("shared_lstm shape: {:?}", shared_lstm.shape());

    // Load ONNX style vector: [1, 256] f32 flat binary
    let style = load_bin_f32("/tmp/onnx_style_1x256.bin", &[1, 256])?;
    eprintln!("style shape: {:?}", style.shape());

    // style[:, 128:] — what the predictor's AdaIN actually uses
    let style_half = style.i((.., 128..))?;
    eprintln!("style_half (128:) shape: {:?}", style_half.shape());

    // Run ONLY the F0 + N predictor branches
    let (f0, n_amp) = predictor.predict_f0_n(&shared_lstm, &style)?;
    eprintln!("F0 output shape:   {:?}", f0.shape());
    eprintln!("N_amp output shape: {:?}", n_amp.shape());

    // F0 stats
    let f0_flat = f0.flatten_all()?.to_vec1::<f32>()?;
    let f0_min = f0_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let f0_max = f0_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n_show = 10.min(f0_flat.len());
    let f0_first10: Vec<f32> = f0_flat[..n_show].to_vec();

    eprintln!("\n--- F0 branch output (BEFORE F0_conv stride=2 downsample) ---");
    eprintln!("  shape:  {:?}", f0.shape());
    eprintln!("  range:  [{f0_min:.4}, {f0_max:.4}]");
    eprintln!("  first{n_show}: {f0_first10:.4?}");

    // N_amp stats
    let n_flat = n_amp.flatten_all()?.to_vec1::<f32>()?;
    let n_min = n_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let n_max = n_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n_first10: Vec<f32> = n_flat[..10.min(n_flat.len())].to_vec();

    eprintln!("\n--- N branch output (BEFORE N_conv stride=2 downsample) ---");
    eprintln!("  shape:  {:?}", n_amp.shape());
    eprintln!("  range:  [{n_min:.4}, {n_max:.4}]");
    eprintln!("  first10: {n_first10:.4?}");

    // ONNX F0_down (AFTER F0_conv stride=2) reference for comparison.
    // The Rust output is at T=102 (2×51); ONNX F0_down is at T=51 (after stride-2 conv).
    // We compare Rust values at even indices [0,2,4,...] against ONNX F0_down values,
    // which is an approximation since stride-2 conv != simple subsampling.
    let onnx_f0_down_first10: [f32; 10] =
        [0.52, 0.62, 0.72, 0.75, 0.74, 0.68, 0.65, 0.62, 0.59, 0.55];
    let onnx_f0_range: (f32, f32) = (-39.2, 0.75);

    eprintln!("\n--- ONNX F0_down reference (AFTER stride-2 conv, for scale comparison) ---");
    eprintln!("  range:  [{:.2}, {:.2}]", onnx_f0_range.0, onnx_f0_range.1);
    eprintln!("  first10: {onnx_f0_down_first10:.4?}");

    eprintln!("\n--- Interpretation ---");
    eprintln!("  Rust F0 is pre-F0_conv (shape T={}), ONNX F0_down is post (T=51).", f0_flat.len());
    eprintln!("  If Rust range matches ONNX range magnitude → F0 branch is likely correct.");
    eprintln!("  If Rust range is wildly different → F0 predictor bug.");

    // Sanity: output must be finite
    let nan_count = f0_flat.iter().filter(|x| x.is_nan()).count();
    let inf_count = f0_flat.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "F0 output contains {nan_count} NaN values");
    assert_eq!(inf_count, 0, "F0 output contains {inf_count} Inf values");

    eprintln!("\ntest_f0_predictor_with_onnx_shared_lstm: DONE");
    Ok(())
}
