/// Baseline tests using the original non-quantized BF16 model.
/// Run with: cargo test -p tts-core --test baseline_original -- --nocapture
///
/// These tests require the original model at /tmp/tts_original.safetensors
/// Download: curl -L -o /tmp/tts_original.safetensors \
///   https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main/tts_b6369a24.safetensors

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tts_core::config::TTSConfig;
use tts_core::flow_lm::Rng;
use tts_core::tts_model::TTSModel;

const MODEL_PATH: &str = "/tmp/tts_original.safetensors";

fn tensor_stats(name: &str, t: &Tensor) {
    match t.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
        Ok(data) => {
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let variance =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            let std = variance.sqrt();
            let nan_count = data.iter().filter(|x| x.is_nan()).count();
            let inf_count = data.iter().filter(|x| x.is_infinite()).count();
            eprintln!(
                "[{name}] shape={:?} min={min:.6} max={max:.6} mean={mean:.6} std={std:.6} nan={nan_count} inf={inf_count}",
                t.shape()
            );
        }
        Err(e) => eprintln!("[{name}] failed to compute stats: {e}"),
    }
}

fn model_path_or_skip() -> Option<&'static str> {
    if std::path::Path::new(MODEL_PATH).exists() {
        Some(MODEL_PATH)
    } else {
        eprintln!("SKIP: original model not found at {MODEL_PATH}");
        None
    }
}

fn load_original_model() -> (TTSModel, TTSConfig) {
    eprintln!("Loading ORIGINAL BF16 model from {MODEL_PATH}...");
    let bytes = std::fs::read(MODEL_PATH).expect("failed to read model file");
    eprintln!("  read {} MB", bytes.len() / (1024 * 1024));

    // For BF16 model: dequantize_and_remap just does key remapping (no INT8 conversion)
    let remapped = mimi_rs::dequantize::dequantize_and_remap(&bytes);
    eprintln!("  remapped: {} MB", remapped.len() / (1024 * 1024));

    // VarBuilder auto-converts BF16 -> F32
    let vb = VarBuilder::from_buffered_safetensors(remapped, DType::F32, &Device::Cpu)
        .expect("failed to create VarBuilder");

    let cfg = TTSConfig::v202601(0.7);
    let model = TTSModel::load(vb, &cfg).expect("failed to load TTSModel");
    eprintln!("  model loaded OK");
    (model, cfg)
}

struct FixedRng {
    values: Vec<f32>,
    index: usize,
}

impl FixedRng {
    fn new_seeded(len: usize) -> Self {
        let mut values = Vec::with_capacity(len);
        let mut state: u64 = 12345;
        let mut next_u01 = move || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32)
        };
        while values.len() < len {
            let u1 = next_u01().max(1e-10);
            let u2 = next_u01();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            values.push(r * theta.cos() * 0.7_f32.sqrt());
            if values.len() < len {
                values.push(r * theta.sin() * 0.7_f32.sqrt());
            }
        }
        Self { values, index: 0 }
    }
}

impl Rng for FixedRng {
    fn sample(&mut self) -> f32 {
        let v = self.values[self.index % self.values.len()];
        self.index += 1;
        v
    }
}

fn sample_token_ids() -> Vec<u32> {
    vec![1, 50, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500]
}

// ===========================================================================
// Test 1: Original model loading sanity
// ===========================================================================

#[test]
fn test_original_model_loads() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_original_model();

    eprintln!("\n=== ORIGINAL MODEL: Loading Sanity ===");

    let emb_std = &model.flow_lm.emb_std;
    assert_eq!(emb_std.dims(), &[32], "emb_std shape mismatch");
    tensor_stats("flow_lm.emb_std", emb_std);

    let emb_mean = &model.flow_lm.emb_mean;
    assert_eq!(emb_mean.dims(), &[32], "emb_mean shape mismatch");
    tensor_stats("flow_lm.emb_mean", emb_mean);

    let std_data = emb_std.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = std_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = std_data.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "emb_std contains NaN");
    assert_eq!(inf_count, 0, "emb_std contains Inf");

    let embed = &model.flow_lm.conditioner.embed;
    assert_eq!(embed.dims(), &[4001, 1024], "conditioner.embed shape mismatch");
    tensor_stats("flow_lm.conditioner.embed (sample)", &embed.narrow(0, 0, 10).unwrap());

    eprintln!("  ldim = {}", model.flow_lm.ldim);
    eprintln!("  dim = {}", model.flow_lm.dim);
    eprintln!("  sample_rate = {}", model.sample_rate());

    eprintln!("PASS: original_model_loads");
}

// ===========================================================================
// Test 2: Compare emb_std / emb_mean between original and quantized
// ===========================================================================

#[test]
fn test_original_vs_quantized_normalization_params() {
    if model_path_or_skip().is_none() {
        return;
    }

    eprintln!("\n=== ORIGINAL MODEL: Normalization params ===");

    let (model, _cfg) = load_original_model();

    let std_data = model.flow_lm.emb_std.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mean_data = model.flow_lm.emb_mean.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    eprintln!("Original emb_std:  {:?}", &std_data[..8]);
    eprintln!("Original emb_mean: {:?}", &mean_data[..8]);

    // These should match the INT8 model normalization params (they're not quantized)
    // INT8 model emb_std:  min=0.593750 max=1.343750 mean=0.968384 (from pipeline_debug test)
    let std_min = std_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let std_max = std_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(std_min > 0.0, "emb_std has non-positive values");
    assert!(std_min > 0.5 && std_max < 2.0, "emb_std out of expected range [{std_min}, {std_max}]");

    eprintln!("  emb_std range: [{std_min:.6}, {std_max:.6}]");
    eprintln!("PASS: original_vs_quantized_normalization_params");
}

// ===========================================================================
// Test 3: Single generation step with original model
// ===========================================================================

#[test]
fn test_original_single_generation_step() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_original_model();

    eprintln!("\n=== ORIGINAL MODEL: Single Generation Step ===");

    let token_ids = sample_token_ids();
    let mut state = model.init_flow_lm_state();
    model.prompt_text(&mut state, &token_ids).expect("prompt_text failed");

    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let bos_latent =
        Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();

    let mut rng = FixedRng::new_seeded(ldim * 10);

    let (latent, is_eos) = model
        .generate_step(&mut state, &bos_latent, &mut rng)
        .expect("generate_step failed");

    eprintln!("  latent shape: {:?}", latent.shape());
    eprintln!("  is_eos: {is_eos}");

    tensor_stats("original_latent", &latent);

    let data = latent.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();

    eprintln!("  nan_count: {nan_count}, inf_count: {inf_count}");
    assert_eq!(nan_count, 0, "original model latent contains NaN");
    assert_eq!(inf_count, 0, "original model latent contains Inf");

    eprintln!("PASS: original_single_generation_step");
}

// ===========================================================================
// Test 4: Mimi decode with original model
// ===========================================================================

#[test]
fn test_original_mimi_decode() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_original_model();

    eprintln!("\n=== ORIGINAL MODEL: Mimi Decode ===");

    let token_ids = sample_token_ids();
    let mut tts_state = model.init_flow_lm_state();
    model.prompt_text(&mut tts_state, &token_ids).expect("prompt_text failed");

    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let bos_latent =
        Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();

    let mut rng = FixedRng::new_seeded(ldim * 10);
    let (latent, _is_eos) = model
        .generate_step(&mut tts_state, &bos_latent, &mut rng)
        .expect("generate_step failed");

    tensor_stats("original_latent_before_decode", &latent);

    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu).expect("init_mimi_state failed");
    let audio = model.decode_latent(&latent, &mut mimi_state).expect("decode_latent failed");

    eprintln!("  audio shape: {:?}", audio.shape());
    tensor_stats("original_audio_chunk", &audio);

    let audio_data = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = audio_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = audio_data.iter().filter(|x| x.is_infinite()).count();

    eprintln!("  nan_count: {nan_count}, inf_count: {inf_count}");
    assert_eq!(nan_count, 0, "original model audio contains NaN");
    assert_eq!(inf_count, 0, "original model audio contains Inf");

    let pct_in_range = audio_data.iter().filter(|&&x| x.abs() <= 1.0).count() as f32
        / audio_data.len() as f32;
    eprintln!("  values within [-1, 1]: {:.1}%", pct_in_range * 100.0);

    let max_abs = audio_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("  max_abs: {max_abs:.6}");

    eprintln!("PASS: original_mimi_decode");
}

// ===========================================================================
// Test 5: Full pipeline - 5 steps with original model (detailed logging)
// ===========================================================================

#[test]
fn test_original_five_generation_steps() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_original_model();

    eprintln!("\n=== ORIGINAL MODEL: Five Generation Steps ===");

    let token_ids = sample_token_ids();
    let mut tts_state = model.init_flow_lm_state();
    model.prompt_text(&mut tts_state, &token_ids).expect("prompt_text failed");

    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu).expect("init_mimi_state failed");

    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let mut prev_latent =
        Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();

    let mut rng = FixedRng::new_seeded(ldim * 100);
    let mut total_audio_samples = 0usize;

    for step in 0..5 {
        eprintln!("\n  --- Original Model Step {step} ---");

        let (latent, is_eos) = model
            .generate_step(&mut tts_state, &prev_latent, &mut rng)
            .expect("generate_step failed");

        tensor_stats(&format!("original_latent[{step}]"), &latent);
        eprintln!("    is_eos: {is_eos}");

        let latent_data = latent.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan_count = latent_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = latent_data.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(nan_count, 0, "step {step}: latent contains NaN");
        assert_eq!(inf_count, 0, "step {step}: latent contains Inf");

        let audio = model
            .decode_latent(&latent, &mut mimi_state)
            .expect("decode_latent failed");

        tensor_stats(&format!("original_audio[{step}]"), &audio);

        let audio_data = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan_count = audio_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = audio_data.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(nan_count, 0, "step {step}: audio contains NaN");
        assert_eq!(inf_count, 0, "step {step}: audio contains Inf");

        let max_abs = audio_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("    max_abs_audio: {max_abs:.4}");
        assert!(max_abs < 100.0, "step {step}: audio blew up (max_abs={max_abs})");

        total_audio_samples += audio_data.len();
        eprintln!("    cumulative_audio_samples: {total_audio_samples}");

        prev_latent = latent;
    }

    let total_duration_secs = total_audio_samples as f32 / 24000.0;
    eprintln!("\n  Total audio samples: {total_audio_samples}");
    eprintln!("  Total duration: {total_duration_secs:.3}s");

    eprintln!("PASS: original_five_generation_steps");
}
