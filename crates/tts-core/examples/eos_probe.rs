/// Per-step EOS-logit probe for the flow LM only (no Mimi decode).
///
/// Used to isolate whether a Rust/official divergence in generation length
/// comes from the flow LM's own numerics or from Mimi/audio-path code, by
/// running only `FlowLM` (no `MimiModel`) against unquantized safetensors
/// weights and dumping the raw EOS logit at every generation step
/// (`self.out_eos(transformer_out)`, `flow_lm.py:157`), for direct diffing
/// against an equivalent per-step dump from the official PyTorch package.
///
/// Deliberately bypasses `TTSModel::load` / `MimiModel::load`: the official
/// pocket-tts safetensors checkpoint's `mimi.downsample` conv has a
/// non-depthwise, channel-reducing shape ([32, 512, 32], i.e. 512→32
/// channels) that mimi-rs's non-GGUF `MimiModel::load` does not expect
/// (it assumes a same-width [dimension, dimension, kernel] conv); loading
/// the full `TTSModel` from this checkpoint's safetensors fails with a
/// shape-mismatch error before generation ever starts. That mismatch is in
/// the shared `mimi-rs` crate, not in tts-core, and is out of scope here —
/// this probe only needs the flow LM, so it never touches Mimi.
///
/// Usage:
///   TTS_DEBUG_EOS=1 cargo run --example eos_probe -p tts-core --release -- \
///     --model languages/french/model.safetensors \
///     --tokenizer languages/french/tokenizer.model \
///     --voice languages/french/embeddings/estelle.safetensors \
///     --language french \
///     --text "Bonjour, ceci est un test du système de synthèse vocale." \
///     --max-steps 60
use candle_core::{Device, Result as CResult, Tensor};
use candle_nn::VarBuilder;
use mimi_rs::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use tts_core::flow_lm::{FlowLM, FlowLMState};
use tts_core::generation::EosGate;
use tts_core::text_config::TextConfig;
use tts_core::tokenizer::Tokenizer;
use tts_core::tts_model::prepare_text_prompt;

struct ZeroRng;
impl tts_core::flow_lm::Rng for ZeroRng {
    fn sample(&mut self) -> f32 {
        0.0
    }
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn run() -> CResult<()> {
    let argv: Vec<String> = std::env::args().collect();
    let model_path = arg(&argv, "--model").expect("--model is required");
    let tokenizer_path = arg(&argv, "--tokenizer").expect("--tokenizer is required");
    let voice_path = arg(&argv, "--voice").expect("--voice is required");
    let language = arg(&argv, "--language").unwrap_or_else(|| "english".to_string());
    let text = arg(&argv, "--text").expect("--text is required");
    let max_steps: usize = arg(&argv, "--max-steps")
        .map(|s| s.parse().unwrap())
        .unwrap_or(80);

    let tensors = candle_core::safetensors::load(&model_path, &Device::Cpu)?;
    let cfg = tts_core::config::TTSConfig::v202601_for_safetensors_keys(tensors.keys(), 0.0)?;
    let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F32, &Device::Cpu);
    let flow_lm = FlowLM::load(vb.pp("flow_lm"), &cfg.flow_lm)?;
    eprintln!("flow_lm loaded, num_layers={}", cfg.flow_lm.num_layers);

    // --- Voice KV cache (same format read by tts_generate.rs) ---
    let voice_bytes = std::fs::read(&voice_path).expect("read voice file");
    let voice_tensors = candle_core::safetensors::load_buffer(&voice_bytes, &Device::Cpu)?;
    let num_layers = cfg.flow_lm.num_layers;
    let mut layer_states = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let cache_name = format!("transformer.layers.{i}.self_attn/cache");
        let cache = voice_tensors.get(&cache_name).expect("missing voice cache tensor");
        let k = cache.narrow(0, 0, 1)?.squeeze(0)?;
        let v = cache.narrow(0, 1, 1)?.squeeze(0)?;
        let seq_len = k.dim(1)?;
        layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState::with_kv(
            k.contiguous()?,
            v.contiguous()?,
            seq_len,
        )));
    }
    let mut state = FlowLMState {
        transformer_state: StreamingTransformerState { layer_states },
    };

    // --- Text prep + prompt ---
    let text_config = TextConfig::for_language(&language);
    let (prepared_text, frames_after_eos) =
        prepare_text_prompt(&text, cfg.model_recommended_frames_after_eos, &text_config);
    eprintln!("prepared_text={prepared_text:?} frames_after_eos={frames_after_eos}");
    let tokenizer_bytes = std::fs::read(&tokenizer_path).expect("read tokenizer");
    let tokenizer = Tokenizer::from_model_bytes(&tokenizer_bytes)?;
    let token_ids = tokenizer.encode(&prepared_text);
    eprintln!("token_ids ({} tokens): {:?}", token_ids.len(), token_ids);

    // Mirrors TTSModel::run_backbone_and_increment: with no audio latents yet,
    // feed text embeddings straight through (skip input_linear, which can't
    // reshape a zero-length sequence).
    let text_embeddings = flow_lm.conditioner.embed_tokens(&token_ids)?;
    let _ = flow_lm
        .transformer
        .forward(&text_embeddings, &mut state.transformer_state)?;
    eprintln!("prompt_text done");

    // --- Generation loop: flow LM only, temperature 0 ---
    let mut rng = ZeroRng;
    let mut gate = EosGate::new(frames_after_eos);
    let nan_data: Vec<f32> = vec![f32::NAN; flow_lm.ldim];
    let mut prev_latent = Tensor::from_vec(nan_data, (1usize, 1usize, flow_lm.ldim), &Device::Cpu)?;
    let empty_text = Tensor::zeros((1, 0, flow_lm.conditioner.dim), text_embeddings.dtype(), text_embeddings.device())?;

    for step in 0..max_steps {
        let (latent, is_eos) = flow_lm.sample_next_latent(
            &prev_latent,
            &empty_text,
            &mut state,
            1,
            &mut rng,
            -4.0,
        )?;
        let latent_data = latent.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = latent_data.iter().fold(0f32, |a, &b| a.max(b.abs()));
        eprintln!("STEP {step} is_eos={is_eos} latent_max_abs={max_abs:.6}");
        if !gate.accept(step, is_eos) {
            eprintln!("STOP at step {step}");
            break;
        }
        prev_latent = latent;
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
