/// Deep intermediate tensor dump test.
///
/// Runs a SINGLE generation step with voice conditioning and dumps tensor values
/// at every pipeline stage so they can be compared against a reference implementation.
///
/// Run with:
///   cargo test -p tts-core --test intermediate_dump -- --nocapture 2>&1 | tee /tmp/intermediate_dump.txt
///
/// Requirements:
///   MODEL: /Users/tc/Code/idle-intelligence/tts-web/model_int8.safetensors
///   VOICE: /tmp/alba.safetensors

use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::{Module, VarBuilder};
use tts_core::config::TTSConfig;
use tts_core::flow_lm::{FlowLMState, Rng};
use tts_core::tts_model::{TTSModel, TTSState};

const MODEL_PATH: &str = "/Users/tc/Code/idle-intelligence/tts-web/model_int8.safetensors";
const VOICE_PATH: &str = "/tmp/alba.safetensors";
const WAV_OUTPUT: &str = "/tmp/test_tts_debug.wav";

// Fixed token IDs for "Hello, this is a test of the text to speech system."
// (Real sentencepiece encoding of the prepared text)
const TOKEN_IDS: &[u32] = &[
    2994, 262, 285, 277, 267, 1115, 272, 265, 2009, 266, 260, 3476, 260, 848, 263,
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dump_tensor(label: &str, t: &Tensor, n_preview: usize) {
    let shape = t.shape().clone();
    match t.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
        Ok(data) => {
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            let std = var.sqrt();
            let nan_count = data.iter().filter(|x| x.is_nan()).count();
            let inf_count = data.iter().filter(|x| x.is_infinite()).count();
            let preview: Vec<f32> = data.iter().take(n_preview).cloned().collect();
            eprintln!(
                "DUMP [{label}] shape={shape:?} min={min:.6} max={max:.6} mean={mean:.6} std={std:.6} nan={nan_count} inf={inf_count}"
            );
            eprintln!("  first_{n_preview}: {preview:.6?}");
        }
        Err(e) => eprintln!("DUMP [{label}] ERROR: {e}"),
    }
}

fn dump_vec(label: &str, data: &[f32], n_preview: usize) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std = var.sqrt();
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    let preview: Vec<f32> = data.iter().take(n_preview).cloned().collect();
    eprintln!(
        "DUMP [{label}] len={} min={min:.6} max={max:.6} mean={mean:.6} std={std:.6} nan={nan_count} inf={inf_count}",
        data.len()
    );
    eprintln!("  first_{n_preview}: {preview:.6?}");
}

struct SeededRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
    generated: Vec<f32>,
}

impl SeededRng {
    fn new(temperature: f32) -> Self {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self { inner: rng, distr, generated: Vec::new() }
    }
    fn get_generated(&self) -> &[f32] {
        &self.generated
    }
}

impl Rng for SeededRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        let v: f32 = self.inner.sample(self.distr);
        self.generated.push(v);
        v
    }
}

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let num_samples = samples.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size = num_samples * 4;
    let chunk_size = 36 + data_size;
    f.write_all(b"RIFF")?;
    f.write_all(&chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&18u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?;
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The actual deep debug test
// ---------------------------------------------------------------------------

#[test]
fn test_intermediate_tensor_dump() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: model not found at {MODEL_PATH}");
        return;
    }
    if !std::path::Path::new(VOICE_PATH).exists() {
        eprintln!("SKIP: voice not found at {VOICE_PATH}");
        return;
    }

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("=== INTERMEDIATE TENSOR DUMP TEST ===");
    eprintln!("{}\n", "=".repeat(70));

    // -----------------------------------------------------------------------
    // Stage 0: Load model
    // -----------------------------------------------------------------------
    eprintln!("--- STAGE 0: Model Loading ---");
    let bytes = std::fs::read(MODEL_PATH).expect("failed to read model");
    eprintln!("DUMP [raw_model] bytes={}", bytes.len());

    let dequantized = mimi_rs::dequantize::dequantize_and_remap(&bytes);
    eprintln!("DUMP [dequantized_model] bytes={}", dequantized.len());

    let vb = VarBuilder::from_buffered_safetensors(dequantized, DType::F32, &Device::Cpu)
        .expect("VarBuilder failed");

    let cfg = TTSConfig::v202601(0.7);
    let model = TTSModel::load(vb, &cfg).expect("model load failed");
    eprintln!("DUMP [model_loaded] OK");

    // Dump key model parameters
    dump_tensor("emb_std", &model.flow_lm.emb_std, 32);
    dump_tensor("emb_mean", &model.flow_lm.emb_mean, 32);

    let bos_data = model.flow_lm.bos_emb.flatten_all()
        .and_then(|f| f.to_vec1::<f32>())
        .unwrap_or_default();
    dump_vec("bos_emb", &bos_data, 32);

    // -----------------------------------------------------------------------
    // Stage 1: Load voice KV cache
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 1: Voice KV Cache Loading ---");
    let voice_bytes = std::fs::read(VOICE_PATH).expect("failed to read voice");
    let tensors = candle_core::safetensors::load_buffer(&voice_bytes, &Device::Cpu)
        .expect("safetensors load failed");

    let num_layers = 6usize;
    let mut layer_states = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let cache_name = format!("transformer.layers.{i}.self_attn/cache");
        let cache = tensors.get(&cache_name)
            .unwrap_or_else(|| panic!("missing: {cache_name}"));

        let k = cache.narrow(0, 0, 1).unwrap().squeeze(0).unwrap().contiguous().unwrap();
        let v = cache.narrow(0, 1, 1).unwrap().squeeze(0).unwrap().contiguous().unwrap();
        let seq_len = k.dim(1).unwrap();

        eprintln!("DUMP [voice_layer_{i}] seq_len={seq_len} k_shape={:?}", k.shape());
        if i == 0 {
            dump_tensor(&format!("voice_k[{i}]_first_token"), &k.narrow(1, 0, 1).unwrap(), 10);
            dump_tensor(&format!("voice_k[{i}]_last_token"), &k.narrow(1, seq_len-1, 1).unwrap(), 10);
        }

        use mimi_rs::transformer::{LayerAttentionState, StreamingMHAState};
        layer_states.push(LayerAttentionState::FlowLm(
            StreamingMHAState::with_kv(k, v, seq_len)
        ));
    }

    use mimi_rs::transformer::StreamingTransformerState;
    let voice_state = TTSState {
        flow_lm_state: FlowLMState {
            transformer_state: StreamingTransformerState { layer_states },
        },
    };
    eprintln!("DUMP [voice_state_loaded] seq_len={}",
        voice_state.flow_lm_state.transformer_state.current_seq_len());

    // -----------------------------------------------------------------------
    // Stage 2: Token embedding
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 2: Token Embedding ---");
    eprintln!("DUMP [token_ids] {:?}", TOKEN_IDS);

    let text_embeddings = model.flow_lm.conditioner.embed_tokens(TOKEN_IDS)
        .expect("embed_tokens failed");
    dump_tensor("text_embeddings", &text_embeddings, 10);

    // First token embedding
    let first_emb = text_embeddings.narrow(1, 0, 1).unwrap();
    dump_tensor("text_embeddings[0]", &first_emb, 10);

    // -----------------------------------------------------------------------
    // Stage 3: prompt_text (text → transformer forward)
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 3: prompt_text (text conditioner forward pass) ---");
    let mut tts_state = voice_state.clone();
    model.prompt_text(&mut tts_state, TOKEN_IDS).expect("prompt_text failed");

    let seq_after_text = tts_state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("DUMP [seq_after_prompt_text] seq_len={seq_after_text}");

    // Dump KV cache state after prompt_text (first layer, last few tokens)
    {
        use mimi_rs::transformer::LayerAttentionState;
        for (i, ls) in tts_state.flow_lm_state.transformer_state.layer_states.iter().enumerate() {
            match ls {
                LayerAttentionState::FlowLm(mha) => {
                    eprintln!("DUMP [kv_after_prompt_text_layer_{i}] current_end={}", mha.current_end);
                    if i == 0 {
                        // Get all accumulated k/v chunks
                        let all_chunks: Vec<&Tensor> = mha.k_chunks.iter().collect();
                        if !all_chunks.is_empty() {
                            let k_all = if all_chunks.len() == 1 {
                                all_chunks[0].clone()
                            } else {
                                Tensor::cat(&all_chunks, 1).unwrap()
                            };
                            eprintln!("DUMP [k_all_layer_0] shape={:?}", k_all.shape());
                            // Last token (text token 14, index = 125+14=139 in combined)
                            let last_tok = k_all.dim(1).unwrap() - 1;
                            let last_k = k_all.narrow(1, last_tok, 1).unwrap();
                            dump_tensor("k_all_layer_0_last_token", &last_k, 10);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stage 4: BOS latent (NaN → replace_nan_with_bos)
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 4: BOS Latent (NaN initial) ---");
    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let bos_latent = Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu).unwrap();
    eprintln!("DUMP [bos_latent_before_replace] shape={:?} (all NaN)", bos_latent.shape());

    // Manually replicate replace_nan_with_bos to see intermediate
    let bos_emb_data = model.flow_lm.bos_emb.flatten_all().and_then(|f| f.to_vec1::<f32>()).unwrap();
    let flat_bos = bos_latent.flatten_all().and_then(|f| f.to_vec1::<f32>()).unwrap();
    let mut replaced_data: Vec<f32> = flat_bos.iter().enumerate().map(|(i, &v)| {
        if v.is_nan() { bos_emb_data[i % ldim] } else { v }
    }).collect();
    dump_vec("bos_latent_after_replace_nan", &replaced_data, 32);

    // -----------------------------------------------------------------------
    // Stage 5: input_linear projection of BOS latent
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 5: input_linear (ldim→d_model) ---");
    let replaced_tensor = Tensor::from_vec(replaced_data.clone(), (1usize, 1usize, ldim), &Device::Cpu)
        .unwrap();
    let input_projected = model.flow_lm.input_linear.forward(&replaced_tensor)
        .expect("input_linear forward failed");
    dump_tensor("input_linear_output", &input_projected, 10);

    // -----------------------------------------------------------------------
    // Stage 6: backbone (transformer forward on single step)
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 6: Generate Step 0 (full generate_step call) ---");
    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu).expect("init_mimi_state failed");
    let mut rng = SeededRng::new(0.7);

    let (latent_step0, is_eos_step0) = model.generate_step(
        &mut tts_state,
        &bos_latent,
        &mut rng,
    ).expect("generate_step failed");

    eprintln!("DUMP [generate_step_0_is_eos] {is_eos_step0}");
    dump_tensor("latent_step0_raw", &latent_step0, 32);

    // Dump noise values used
    dump_vec("rng_noise_step0", rng.get_generated(), 32);

    // -----------------------------------------------------------------------
    // Stage 7: Denormalization (latent * emb_std + emb_mean)
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 7: Denormalization (latent * emb_std + emb_mean) ---");
    let denorm = latent_step0
        .broadcast_mul(&model.flow_lm.emb_std).unwrap()
        .broadcast_add(&model.flow_lm.emb_mean).unwrap();
    dump_tensor("denorm_latent", &denorm, 32);

    // -----------------------------------------------------------------------
    // Stage 8: DummyQuantizer
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 8: DummyQuantizer (quantizer_dim→output_dim) ---");
    // denorm is [B, T, C] = [1, 1, 32]; need [B, C, T] for conv
    let transposed = denorm.transpose(1, 2).unwrap().contiguous().unwrap();
    eprintln!("DUMP [transposed_for_quantizer] shape={:?}", transposed.shape());
    let quantized = model.mimi.quantizer.forward(&transposed).expect("quantizer failed");
    dump_tensor("quantizer_output", &quantized, 10);

    // -----------------------------------------------------------------------
    // Stage 9: Mimi decode_from_latent
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 9: decode_latent (mimi decoder) ---");
    let audio_chunk = model.decode_latent(&latent_step0, &mut mimi_state)
        .expect("decode_latent failed");

    dump_tensor("audio_chunk_step0", &audio_chunk, 20);

    let audio_data = audio_chunk.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Last 10 samples
    let last_10: Vec<f32> = audio_data.iter().rev().take(10).cloned().collect();
    eprintln!("DUMP [audio_chunk_step0_last_10] {last_10:.6?}");

    // -----------------------------------------------------------------------
    // Stage 10: Steps 1 and 2 with real prev_latent
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 10: Steps 1 and 2 ---");
    let mut prev_latent = latent_step0.clone();
    let mut all_audio = audio_data.clone();

    for step in 1..3 {
        eprintln!("\n  -- Step {step} --");

        let (latent, is_eos) = model.generate_step(
            &mut tts_state,
            &prev_latent,
            &mut rng,
        ).expect("generate_step failed");

        eprintln!("DUMP [generate_step_{step}_is_eos] {is_eos}");
        dump_tensor(&format!("latent_step{step}"), &latent, 32);

        let audio = model.decode_latent(&latent, &mut mimi_state).expect("decode_latent failed");
        dump_tensor(&format!("audio_step{step}"), &audio, 10);

        let audio_vec = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        all_audio.extend_from_slice(&audio_vec);

        let seq_len = tts_state.flow_lm_state.transformer_state.current_seq_len();
        eprintln!("DUMP [seq_len_after_step_{step}] {seq_len}");

        prev_latent = latent;
    }

    // -----------------------------------------------------------------------
    // Stage 11: Final audio stats and WAV write
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 11: Final Audio Stats ---");
    dump_vec("final_audio_3steps", &all_audio, 10);

    let sample_rate = model.sample_rate() as u32;
    write_wav(WAV_OUTPUT, &all_audio, sample_rate).expect("wav write failed");
    eprintln!("DUMP [wav_written] path={WAV_OUTPUT} samples={} sr={sample_rate}", all_audio.len());

    // -----------------------------------------------------------------------
    // Stage 12: Consistency checks
    // -----------------------------------------------------------------------
    eprintln!("\n--- STAGE 12: Consistency Checks ---");

    let nan_count = all_audio.iter().filter(|x| x.is_nan()).count();
    let inf_count = all_audio.iter().filter(|x| x.is_infinite()).count();
    let max_abs = all_audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let pct_in_range = all_audio.iter().filter(|&&x| x.abs() <= 1.0).count() as f32 / all_audio.len() as f32;

    eprintln!("DUMP [final_checks] nan={nan_count} inf={inf_count} max_abs={max_abs:.4} pct_in_[-1,1]={:.1}%", pct_in_range * 100.0);

    assert_eq!(nan_count, 0, "audio contains NaN");
    assert_eq!(inf_count, 0, "audio contains Inf");
    assert!(max_abs < 100.0, "audio blew up: max_abs={max_abs}");
    assert!(max_abs > 0.001, "audio is silent: max_abs={max_abs}");

    eprintln!("\n=== INTERMEDIATE DUMP COMPLETE ===");
    eprintln!("WAV saved to: {WAV_OUTPUT}");
}
