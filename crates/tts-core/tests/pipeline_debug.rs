/// Integration tests that validate each step of the TTS pipeline.
/// Run with: cargo test -p tts-core --test pipeline_debug -- --nocapture
///
/// These tests require the model file at:
///   /Users/tc/Code/idle-intelligence/tts-web/model_int8.safetensors
///
/// If the file is absent, tests are skipped (they return early with a message).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tts_core::config::TTSConfig;
use tts_core::flow_lm::Rng;
use tts_core::tts_model::TTSModel;

const MODEL_PATH: &str = "/Users/tc/Code/idle-intelligence/tts-web/model_int8.safetensors";

// ---------------------------------------------------------------------------
// Helper: print tensor statistics
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: check model file exists, skip if not
// ---------------------------------------------------------------------------

fn model_path_or_skip() -> Option<&'static str> {
    if std::path::Path::new(MODEL_PATH).exists() {
        Some(MODEL_PATH)
    } else {
        eprintln!("SKIP: model file not found at {MODEL_PATH}");
        None
    }
}

// ---------------------------------------------------------------------------
// Helper: load model (shared across tests)
// ---------------------------------------------------------------------------

fn load_model() -> (TTSModel, TTSConfig) {
    eprintln!("Loading model from {MODEL_PATH}...");
    let bytes = std::fs::read(MODEL_PATH).expect("failed to read model file");
    eprintln!("  read {} MB", bytes.len() / (1024 * 1024));

    let dequantized = mimi_rs::dequantize::dequantize_and_remap(&bytes);
    eprintln!("  dequantized: {} MB", dequantized.len() / (1024 * 1024));

    let vb =
        VarBuilder::from_buffered_safetensors(dequantized, DType::F32, &Device::Cpu)
            .expect("failed to create VarBuilder");

    let cfg = TTSConfig::v202601(0.7);
    let model = TTSModel::load(vb, &cfg).expect("failed to load TTSModel");
    eprintln!("  model loaded OK");
    (model, cfg)
}

// ---------------------------------------------------------------------------
// Simple deterministic RNG for tests
// ---------------------------------------------------------------------------

struct FixedRng {
    values: Vec<f32>,
    index: usize,
}

impl FixedRng {
    /// Seeded normal distribution using a simple LCG.
    fn new_seeded(len: usize) -> Self {
        // Generate deterministic pseudo-normal values via Box-Muller
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
            values.push(r * theta.cos() * 0.7_f32.sqrt()); // scale by sqrt(temp)
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

// ---------------------------------------------------------------------------
// Some sample token IDs (hand-crafted; real tokenization is done in JS/Python)
// We use token IDs from a small vocabulary window that are known valid.
// ---------------------------------------------------------------------------

/// A short sentence tokenized as plausible IDs (< 4001).
/// These are dummy IDs — the test just checks shapes and finiteness.
fn sample_token_ids() -> Vec<u32> {
    // IDs in range [1, 3999] — within the conditioner vocabulary
    vec![1, 50, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500]
}

// ===========================================================================
// Test 1: Model loading sanity
// ===========================================================================

#[test]
fn test_model_loads_correctly() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_model();

    eprintln!("\n=== Test 1: Model Loading Sanity ===");

    // Check emb_std shape [32]
    let emb_std = &model.flow_lm.emb_std;
    assert_eq!(emb_std.dims(), &[32], "emb_std shape mismatch");
    tensor_stats("flow_lm.emb_std", emb_std);

    // Check emb_mean shape [32]
    let emb_mean = &model.flow_lm.emb_mean;
    assert_eq!(emb_mean.dims(), &[32], "emb_mean shape mismatch");
    tensor_stats("flow_lm.emb_mean", emb_mean);

    // Check no NaN or Inf in emb_std
    let std_data = emb_std.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = std_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = std_data.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "emb_std contains NaN");
    assert_eq!(inf_count, 0, "emb_std contains Inf");

    // Check no NaN or Inf in emb_mean
    let mean_data = emb_mean.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = mean_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = mean_data.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "emb_mean contains NaN");
    assert_eq!(inf_count, 0, "emb_mean contains Inf");

    // Check conditioner embedding shape [4001, 1024]
    let embed = &model.flow_lm.conditioner.embed;
    assert_eq!(embed.dims(), &[4001, 1024], "conditioner.embed shape mismatch");
    tensor_stats("flow_lm.conditioner.embed (sample)", &embed.narrow(0, 0, 10).unwrap());

    // Check conditioner dim
    assert_eq!(model.flow_lm.conditioner.dim, 1024);

    eprintln!("  ldim = {}", model.flow_lm.ldim);
    eprintln!("  dim = {}", model.flow_lm.dim);
    eprintln!("  sample_rate = {}", model.sample_rate());

    eprintln!("PASS: model_loads_correctly");
}

// ===========================================================================
// Test 2: Transformer state initialization
// ===========================================================================

#[test]
fn test_transformer_state_init() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_model();

    eprintln!("\n=== Test 2: Transformer State Init ===");

    let state = model.init_flow_lm_state();
    let num_layers = state.flow_lm_state.transformer_state.layer_states.len();
    eprintln!("  num transformer layers: {num_layers}");
    assert_eq!(num_layers, 6, "expected 6 transformer layers");

    let seq_len = state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  initial seq_len: {seq_len}");
    assert_eq!(seq_len, 0, "fresh state should have seq_len=0");

    eprintln!("PASS: transformer_state_init");
}

// ===========================================================================
// Test 3: Token embedding
// ===========================================================================

#[test]
fn test_token_embedding() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_model();

    eprintln!("\n=== Test 3: Token Embedding ===");

    let token_ids = sample_token_ids();
    let n = token_ids.len();

    let embeddings = model
        .flow_lm
        .conditioner
        .embed_tokens(&token_ids)
        .expect("embed_tokens failed");

    eprintln!("  token_ids: {token_ids:?}");
    eprintln!("  embedding shape: {:?}", embeddings.shape());
    assert_eq!(embeddings.dims(), &[1, n, 1024], "embedding shape mismatch: expected [1, {n}, 1024]");

    tensor_stats("token_embeddings", &embeddings);

    let data = embeddings.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    let zero_count = data.iter().filter(|x| **x == 0.0).count();

    assert_eq!(nan_count, 0, "embeddings contain NaN");
    assert_eq!(inf_count, 0, "embeddings contain Inf");
    assert!(
        zero_count < data.len(),
        "embeddings are all zero — model may not have loaded correctly"
    );

    eprintln!("  zero_count={zero_count}/{}", data.len());
    eprintln!("PASS: token_embedding");
}

// ===========================================================================
// Test 4: Transformer forward pass (prompt_text)
// ===========================================================================

#[test]
fn test_transformer_forward() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_model();

    eprintln!("\n=== Test 4: Transformer Forward Pass ===");

    let token_ids = sample_token_ids();
    let n = token_ids.len();

    let mut state = model.init_flow_lm_state();

    let seq_before = state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  seq_len before prompt_text: {seq_before}");

    model.prompt_text(&mut state, &token_ids).expect("prompt_text failed");

    let seq_after = state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  seq_len after prompt_text: {seq_after}");
    assert_eq!(seq_after, n, "expected seq_len={n} after prompting with {n} tokens");

    // Check KV cache in all layers for NaN
    for (i, layer_state) in state.flow_lm_state.transformer_state.layer_states.iter().enumerate() {
        use mimi_rs::transformer::LayerAttentionState;
        match layer_state {
            LayerAttentionState::FlowLm(mha_state) => {
                eprintln!("  layer {i}: current_end={}", mha_state.current_end);
                assert_eq!(mha_state.current_end, n, "layer {i} current_end mismatch");
            }
            _ => panic!("expected FlowLm state in layer {i}"),
        }
    }

    eprintln!("PASS: transformer_forward");
}

// ===========================================================================
// Test 5: Single generation step
// ===========================================================================

#[test]
fn test_single_generation_step() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_model();

    eprintln!("\n=== Test 5: Single Generation Step ===");

    let token_ids = sample_token_ids();
    let mut state = model.init_flow_lm_state();
    model.prompt_text(&mut state, &token_ids).expect("prompt_text failed");

    let ldim = cfg.flow_lm.ldim;

    // BOS latent: NaN tensor [1, 1, ldim]
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let bos_latent =
        Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();

    let mut rng = FixedRng::new_seeded(ldim * 10);

    let (latent, is_eos) = model
        .generate_step(&mut state, &bos_latent, &mut rng)
        .expect("generate_step failed");

    eprintln!("  latent shape: {:?}", latent.shape());
    eprintln!("  is_eos: {is_eos}");
    assert_eq!(latent.dims(), &[1, 1, ldim], "latent shape mismatch: expected [1, 1, {ldim}]");

    tensor_stats("latent", &latent);

    let data = latent.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "latent contains NaN");
    assert_eq!(inf_count, 0, "latent contains Inf");

    eprintln!("PASS: single_generation_step");
}

// ===========================================================================
// Test 6: Mimi decode
// ===========================================================================

#[test]
fn test_mimi_decode() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_model();

    eprintln!("\n=== Test 6: Mimi Decode ===");

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

    tensor_stats("latent_before_decode", &latent);

    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu).expect("init_mimi_state failed");

    let audio = model.decode_latent(&latent, &mut mimi_state).expect("decode_latent failed");

    eprintln!("  audio shape: {:?}", audio.shape());
    tensor_stats("audio_chunk", &audio);

    let audio_data = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = audio_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = audio_data.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "audio contains NaN");
    assert_eq!(inf_count, 0, "audio contains Inf");

    // PCM values: most should be within a reasonable range (-10 to 10 for unnormalized)
    let extreme_count = audio_data.iter().filter(|&&x| x.abs() > 10.0).count();
    let pct_extreme = extreme_count as f32 / audio_data.len() as f32;
    eprintln!(
        "  values > ±10: {extreme_count}/{} ({:.1}%)",
        audio_data.len(),
        pct_extreme * 100.0
    );

    let pct_in_range = audio_data.iter().filter(|&&x| x.abs() <= 1.0).count() as f32
        / audio_data.len() as f32;
    eprintln!("  values within [-1, 1]: {:.1}%", pct_in_range * 100.0);

    eprintln!("PASS: mimi_decode");
}

// ===========================================================================
// Test 7: Full pipeline sanity (3 steps)
// ===========================================================================

#[test]
fn test_three_generation_steps() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_model();

    eprintln!("\n=== Test 7: Three Generation Steps ===");

    let token_ids = sample_token_ids();
    let mut tts_state = model.init_flow_lm_state();
    model.prompt_text(&mut tts_state, &token_ids).expect("prompt_text failed");

    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu).expect("init_mimi_state failed");

    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let mut prev_latent =
        Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();

    let mut rng = FixedRng::new_seeded(ldim * 100);

    for step in 0..3 {
        eprintln!("\n  --- Step {step} ---");

        let (latent, is_eos) = model
            .generate_step(&mut tts_state, &prev_latent, &mut rng)
            .expect("generate_step failed");

        tensor_stats(&format!("latent[{step}]"), &latent);
        eprintln!("    is_eos: {is_eos}");

        let latent_data = latent.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan_count = latent_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = latent_data.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(nan_count, 0, "step {step}: latent contains NaN");
        assert_eq!(inf_count, 0, "step {step}: latent contains Inf");

        let audio = model
            .decode_latent(&latent, &mut mimi_state)
            .expect("decode_latent failed");

        tensor_stats(&format!("audio[{step}]"), &audio);
        eprintln!("    audio shape: {:?}", audio.shape());

        let audio_data = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan_count = audio_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = audio_data.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(nan_count, 0, "step {step}: audio contains NaN");
        assert_eq!(inf_count, 0, "step {step}: audio contains Inf");

        // Check audio doesn't blow up
        let max_abs = audio_data
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        eprintln!("    max_abs_audio: {max_abs:.4}");
        assert!(max_abs < 100.0, "step {step}: audio blew up (max_abs={max_abs})");

        let seq_len = tts_state.flow_lm_state.transformer_state.current_seq_len();
        eprintln!("    seq_len after step: {seq_len}");

        prev_latent = latent;
    }

    eprintln!("PASS: three_generation_steps");
}

// ===========================================================================
// Test 8: Denormalization correctness
// ===========================================================================

#[test]
fn test_denormalization() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, cfg) = load_model();

    eprintln!("\n=== Test 8: Denormalization Correctness ===");

    // emb_std and emb_mean should be plausible statistics
    let std_data = model.flow_lm.emb_std.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mean_data = model.flow_lm.emb_mean.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let std_min = std_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let std_max = std_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_min = mean_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_max = mean_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    eprintln!("  emb_std: min={std_min:.6} max={std_max:.6}");
    eprintln!("  emb_mean: min={mean_min:.6} max={mean_max:.6}");

    // std should be positive
    assert!(std_min > 0.0, "emb_std has non-positive values: min={std_min}");

    // Simulate denormalization: latent ~ N(0,1), denorm = latent * std + mean
    let ldim = cfg.flow_lm.ldim;
    let unit_latent: Vec<f32> = (0..ldim).map(|i| (i as f32 / ldim as f32) * 2.0 - 1.0).collect();
    let latent_tensor =
        Tensor::from_vec(unit_latent, (1usize, 1usize, ldim), &Device::Cpu).unwrap();

    let denorm = latent_tensor
        .broadcast_mul(&model.flow_lm.emb_std)
        .unwrap()
        .broadcast_add(&model.flow_lm.emb_mean)
        .unwrap();

    tensor_stats("denorm_latent", &denorm);

    let denorm_data = denorm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let nan_count = denorm_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = denorm_data.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(nan_count, 0, "denorm contains NaN");
    assert_eq!(inf_count, 0, "denorm contains Inf");

    eprintln!("PASS: denormalization");
}

// ===========================================================================
// Test 9: Empty token list handling
// ===========================================================================

#[test]
fn test_empty_token_embedding() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_model();

    eprintln!("\n=== Test 9: Empty Token Embedding ===");

    let embeddings = model
        .flow_lm
        .conditioner
        .embed_tokens(&[])
        .expect("embed_tokens with empty list failed");

    eprintln!("  shape: {:?}", embeddings.shape());
    assert_eq!(embeddings.dims(), &[1, 0, 1024], "empty embedding shape mismatch");

    eprintln!("PASS: empty_token_embedding");
}

// ===========================================================================
// Test 10: DType sanity — all model tensors should be F32 after loading
// ===========================================================================

#[test]
fn test_dtype_sanity() {
    if model_path_or_skip().is_none() {
        return;
    }
    let (model, _cfg) = load_model();

    eprintln!("\n=== Test 10: DType Sanity ===");

    let tensors_to_check: Vec<(&str, &Tensor)> = vec![
        ("flow_lm.emb_std", &model.flow_lm.emb_std),
        ("flow_lm.emb_mean", &model.flow_lm.emb_mean),
        ("flow_lm.conditioner.embed", &model.flow_lm.conditioner.embed),
    ];

    for (name, tensor) in tensors_to_check {
        let dtype = tensor.dtype();
        eprintln!("  {name}: dtype={dtype:?}");
        assert_eq!(
            dtype,
            DType::F32,
            "{name} has wrong dtype: expected F32, got {dtype:?}"
        );
    }

    eprintln!("PASS: dtype_sanity");
}

// ===========================================================================
// Test 11: Voice KV cache loading — dtype, shapes, values, current_end
// ===========================================================================
//
// The WASM add_voice_ code expects:
//   - tensor name: "transformer.layers.{i}.self_attn/cache"
//   - shape: [2, batch, seq_len, num_heads, head_dim]
//   - dtype: F32 (the WASM code pattern-matches on TypedTensor::F32)
//   - current_end is set to seq_len (125) regardless of the file's current_end field
//
// If the file stores BF16 instead of F32, the WASM code bails with
// "expected f32 tensor" and voice loading silently fails.

const VOICE_PATH: &str = "/tmp/alba.safetensors";

#[test]
fn test_voice_cache_loading() {
    if !std::path::Path::new(VOICE_PATH).exists() {
        eprintln!("SKIP: voice file not found at {VOICE_PATH}");
        return;
    }

    eprintln!("\n=== Test 11: Voice KV Cache Loading ===");

    let bytes = std::fs::read(VOICE_PATH).expect("failed to read voice file");
    eprintln!("  read {} KB", bytes.len() / 1024);

    let st = safetensors::SafeTensors::deserialize(&bytes)
        .expect("failed to deserialize voice safetensors");

    let tensor_names = st.names();
    eprintln!("  tensor count: {}", tensor_names.len());
    for name in &tensor_names {
        eprintln!("    {name}");
    }

    // Expected: 6 layers × 2 tensors = 12 total
    assert_eq!(tensor_names.len(), 12, "expected 12 tensors (6 layers × 2)");

    let num_layers = 6;
    let expected_num_heads = 16usize;
    let expected_head_dim = 64usize;

    for i in 0..num_layers {
        let cache_name = format!("transformer.layers.{i}.self_attn/cache");
        let end_name = format!("transformer.layers.{i}.self_attn/current_end");

        // ---- cache tensor ----
        let cache = st.tensor(&cache_name)
            .unwrap_or_else(|_| panic!("missing tensor: {cache_name}"));

        let cache_dtype = cache.dtype();
        eprintln!("  layer {i} cache dtype: {cache_dtype:?}");

        // CRITICAL CHECK: dtype must be F32, not BF16.
        // If BF16, the WASM add_voice_ code will fail to pattern-match TypedTensor::F32
        // and bail — voice loading silently fails, model runs without voice conditioning.
        assert_eq!(
            cache_dtype,
            safetensors::Dtype::F32,
            "layer {i} cache has wrong dtype: expected F32, got {cache_dtype:?}. \
             This will cause voice loading to silently fail in WASM!"
        );

        // Shape check: [2, batch, seq_len, num_heads, head_dim]
        let shape = cache.shape();
        eprintln!("  layer {i} cache shape: {shape:?}");
        assert_eq!(shape.len(), 5, "layer {i} cache should be 5D");
        assert_eq!(shape[0], 2, "layer {i} first dim should be 2 (k/v split)");
        assert_eq!(shape[1], 1, "layer {i} batch dim should be 1");
        let seq_len = shape[2];
        assert_eq!(shape[3], expected_num_heads,
            "layer {i} num_heads mismatch: expected {expected_num_heads}, got {}", shape[3]);
        assert_eq!(shape[4], expected_head_dim,
            "layer {i} head_dim mismatch: expected {expected_head_dim}, got {}", shape[4]);

        // Value check: no NaN/Inf, non-trivial values
        let raw = cache.data();
        assert_eq!(raw.len(), 2 * 1 * seq_len * expected_num_heads * expected_head_dim * 4,
            "layer {i} cache byte length mismatch");

        let f32_values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let nan_count = f32_values.iter().filter(|x| x.is_nan()).count();
        let inf_count = f32_values.iter().filter(|x| x.is_infinite()).count();
        let min = f32_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = f32_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = f32_values.iter().sum::<f32>() / f32_values.len() as f32;
        eprintln!("  layer {i} cache values: min={min:.4} max={max:.4} mean={mean:.4} nan={nan_count} inf={inf_count}");

        assert_eq!(nan_count, 0, "layer {i} cache contains NaN");
        assert_eq!(inf_count, 0, "layer {i} cache contains Inf");
        assert!(max.abs() > 0.01, "layer {i} cache values look like zeros — likely wrong");

        // ---- current_end tensor ----
        let end_tensor = st.tensor(&end_name)
            .unwrap_or_else(|_| panic!("missing tensor: {end_name}"));

        let end_shape = end_tensor.shape();
        let end_dtype = end_tensor.dtype();
        eprintln!("  layer {i} current_end: dtype={end_dtype:?} shape={end_shape:?}");

        // WASM code does NOT use this tensor — it uses seq_len from the cache shape instead.
        // But we document what the file actually stores.
        let end_values: Vec<f32> = end_tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let end_min = end_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let end_max = end_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("  layer {i} current_end values: min={end_min} max={end_max} (file says {end_min}, WASM uses seq_len={seq_len})");

        // IMPORTANT: check if file's current_end matches seq_len.
        // If file stores 0 but seq_len is 125, the WASM code is correct to use seq_len.
        // If file stores a non-zero value != seq_len, something is wrong.
        let file_current_end = end_max as usize; // take max in case it's a scalar stored as array
        if file_current_end != seq_len && file_current_end != 0 {
            eprintln!(
                "  WARNING: layer {i} file current_end={file_current_end} != seq_len={seq_len}. \
                 WASM code uses seq_len — verify this is correct."
            );
        }
    }

    eprintln!("\n  Summary: seq_len=125, num_heads=16, head_dim=64");
    eprintln!("  WASM voice loading uses current_end=seq_len=125 (ignores file's current_end field)");
    eprintln!("PASS: voice_cache_loading");
}

// ===========================================================================
// Test 12: Voice-primed generation — 3 steps starting from voice KV cache
// ===========================================================================
//
// This simulates what the WASM code does:
// 1. Load the alba voice KV cache into transformer state
// 2. Run prompt_text with real-ish token IDs
// 3. Generate 3 steps and check audio amplitude
//
// With voice conditioning, audio amplitude should be noticeably higher
// than the no-voice case (test 7 produced max_abs ≈ 0.05–0.10).

#[test]
fn test_voice_primed_generation() {
    if model_path_or_skip().is_none() {
        return;
    }
    if !std::path::Path::new(VOICE_PATH).exists() {
        eprintln!("SKIP: voice file not found at {VOICE_PATH}");
        return;
    }

    eprintln!("\n=== Test 12: Voice-Primed Generation ===");

    let (model, cfg) = load_model();

    // Load voice cache into a TTSState by constructing StreamingMHAState manually,
    // mirroring what the WASM add_voice_ / resize_tts_state code does.
    let voice_bytes = std::fs::read(VOICE_PATH).expect("failed to read voice file");
    let st = safetensors::SafeTensors::deserialize(&voice_bytes)
        .expect("failed to deserialize voice safetensors");

    let num_layers = 6;

    // Build layer states from the voice KV cache
    let mut layer_states = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let cache_name = format!("transformer.layers.{i}.self_attn/cache");
        let cache_tensor = st.tensor(&cache_name)
            .unwrap_or_else(|_| panic!("missing voice tensor: {cache_name}"));

        let shape = cache_tensor.shape();
        // shape: [2, batch, seq_len, num_heads, head_dim]
        let seq_len = shape[2];
        let num_heads = shape[3];
        let head_dim = shape[4];

        assert_eq!(cache_tensor.dtype(), safetensors::Dtype::F32,
            "layer {i} voice cache is not F32 — voice loading will fail in WASM");

        // Load into candle tensor: raw bytes → F32 tensor [2, 1, seq_len, num_heads, head_dim]
        let raw = cache_tensor.data().to_vec();
        let total_elements = 2 * 1 * seq_len * num_heads * head_dim;
        let f32_data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(f32_data.len(), total_elements,
            "layer {i} element count mismatch");

        let cache_t = Tensor::from_vec(
            f32_data,
            (2usize, 1usize, seq_len, num_heads, head_dim),
            &Device::Cpu,
        ).expect("failed to create cache tensor");

        // Split into k and v: [batch, seq_len, num_heads, head_dim] each
        // cache_t shape: [2, 1, seq_len, num_heads, head_dim]
        // After narrow+squeeze on dim 0: [1, seq_len, num_heads, head_dim] = [B, T, H, D]
        // This matches StreamingMHAState.k_chunks element shape.
        let k = cache_t.narrow(0, 0, 1).unwrap()
            .squeeze(0).unwrap()
            .contiguous().unwrap();
        let v = cache_t.narrow(0, 1, 1).unwrap()
            .squeeze(0).unwrap()
            .contiguous().unwrap();
        // k/v shape: [1, seq_len, num_heads, head_dim] = [B, T, H, D]

        // k shape: [1, seq_len, num_heads, head_dim] = [B, T, H, D]
        eprintln!("  layer {i}: k shape={:?} v shape={:?} current_end={seq_len}", k.shape(), v.shape());

        tensor_stats(&format!("voice_k[{i}]"), &k);

        use mimi_rs::transformer::{LayerAttentionState, StreamingMHAState};
        layer_states.push(LayerAttentionState::FlowLm(
            StreamingMHAState::with_kv(k, v, seq_len)
        ));
    }

    use mimi_rs::transformer::StreamingTransformerState;
    use tts_core::flow_lm::FlowLMState;
    use tts_core::tts_model::TTSState;

    let mut tts_state = TTSState {
        flow_lm_state: FlowLMState {
            transformer_state: StreamingTransformerState { layer_states },
        },
    };

    let seq_after_voice = tts_state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  seq_len after loading voice cache: {seq_after_voice}");
    assert_eq!(seq_after_voice, 125, "expected seq_len=125 after voice cache load");

    // Now run prompt_text
    let token_ids = sample_token_ids();
    model.prompt_text(&mut tts_state, &token_ids).expect("prompt_text failed");
    let seq_after_text = tts_state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  seq_len after prompt_text: {seq_after_text}");
    assert_eq!(seq_after_text, 125 + token_ids.len(),
        "seq_len should be voice_seq + text_tokens");

    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu).expect("init_mimi_state failed");
    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let mut prev_latent =
        Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();
    let mut rng = FixedRng::new_seeded(ldim * 100);

    let mut max_audio_abs = 0.0f32;

    for step in 0..3 {
        eprintln!("\n  --- Voice-primed Step {step} ---");

        let (latent, is_eos) = model
            .generate_step(&mut tts_state, &prev_latent, &mut rng)
            .expect("generate_step failed");

        tensor_stats(&format!("voice_latent[{step}]"), &latent);
        eprintln!("    is_eos: {is_eos}");

        let latent_data = latent.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan_count = latent_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = latent_data.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(nan_count, 0, "voice step {step}: latent NaN");
        assert_eq!(inf_count, 0, "voice step {step}: latent Inf");

        let audio = model
            .decode_latent(&latent, &mut mimi_state)
            .expect("decode_latent failed");

        tensor_stats(&format!("voice_audio[{step}]"), &audio);

        let audio_data = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nan_count = audio_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = audio_data.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(nan_count, 0, "voice step {step}: audio NaN");
        assert_eq!(inf_count, 0, "voice step {step}: audio Inf");

        let step_max = audio_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        max_audio_abs = max_audio_abs.max(step_max);

        let pct_in_range = audio_data.iter().filter(|&&x| x.abs() <= 1.0).count() as f32
            / audio_data.len() as f32;
        eprintln!("    max_abs={step_max:.4} pct_in_[-1,1]={:.1}%", pct_in_range * 100.0);

        prev_latent = latent;
    }

    eprintln!("\n  max_audio_abs across 3 voice-primed steps: {max_audio_abs:.4}");
    eprintln!("  (no-voice baseline was ~0.05–0.10; with voice expect higher amplitude)");

    // Audio should not blow up
    assert!(max_audio_abs < 100.0, "voice-primed audio blew up: max_abs={max_audio_abs}");

    eprintln!("PASS: voice_primed_generation");
}
