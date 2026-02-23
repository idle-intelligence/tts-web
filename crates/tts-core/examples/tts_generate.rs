/// End-to-end TTS generation example.
///
/// Usage:
///   cargo run --example tts_generate -- \
///     --model /path/to/model_int8.safetensors \
///     --voice /path/to/alba.safetensors \
///     --output /tmp/test_tts.wav \
///     [--temperature 0.7]
///
/// Writes a Float32 PCM WAV at 24000 Hz (mono).

use candle_core::{Device, Result as CResult, Tensor};
use mimi_rs::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use tts_core::flow_lm::{FlowLMState, Rng};
use tts_core::tts_model::{TTSState, prepare_text_prompt};

// ---------------------------------------------------------------------------
// CLI arg parsing (manual, no external deps)
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    voice_path: Option<String>,
    output_path: String,
    temperature: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = None;
    let mut voice_path = None;
    let mut output_path = None;
    let mut temperature = 0.7f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args[i].clone());
            }
            "--voice" => {
                i += 1;
                voice_path = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                output_path = Some(args[i].clone());
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("temperature must be f32");
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        model_path: model_path.expect("--model required"),
        voice_path,
        output_path: output_path.expect("--output required"),
        temperature,
    }
}

// ---------------------------------------------------------------------------
// RNG using rand + rand_distr (already in tts-core deps)
// ---------------------------------------------------------------------------

struct SimpleRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl SimpleRng {
    fn new(temperature: f32) -> Self {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self { inner: rng, distr }
    }
}

impl Rng for SimpleRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

// ---------------------------------------------------------------------------
// WAV writer (Float32 PCM, format type 3, mono, 24000 Hz)
// ---------------------------------------------------------------------------

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    let num_samples = samples.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size = num_samples * 4; // float32 = 4 bytes each
    let chunk_size = 36 + data_size;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk (18 bytes for float32 PCM: standard 16 + 2-byte cbSize = 0)
    f.write_all(b"fmt ")?;
    f.write_all(&18u32.to_le_bytes())?;    // chunk size (18 for float32 format)
    f.write_all(&3u16.to_le_bytes())?;     // audio format: IEEE_FLOAT = 3
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?;     // cbSize = 0

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor statistics helper for logging
// ---------------------------------------------------------------------------

fn log_tensor_stats(label: &str, data: &[f32]) {
    if data.is_empty() {
        eprintln!("[{label}] empty");
        return;
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    eprintln!(
        "[{label}] len={} min={:.4} max={:.4} mean={:.4} nan={nan_count} inf={inf_count}",
        data.len(), min, max, mean
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> CResult<()> {
    let args = parse_args();

    eprintln!("=== TTS Generate ===");
    eprintln!("model: {}", args.model_path);
    eprintln!("voice: {:?}", args.voice_path);
    eprintln!("output: {}", args.output_path);
    eprintln!("temperature: {}", args.temperature);

    // --- Load model ---
    eprintln!("\n[1] Loading model...");
    let model_bytes = std::fs::read(&args.model_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read model: {e}")))?;
    eprintln!("  read {} MB", model_bytes.len() / (1024 * 1024));

    let mut gguf = mimi_rs::gguf_loader::GgufTensors::from_bytes(&model_bytes, &Device::Cpu)?;
    let cfg = tts_core::config::TTSConfig::v202601(args.temperature);
    let model = tts_core::tts_model::TTSModel::load_gguf(&mut gguf, &cfg)?;

    let sample_rate = model.sample_rate() as u32;
    eprintln!("  model loaded OK (sample_rate={})", sample_rate);
    eprintln!("  ldim={} dim={}", cfg.flow_lm.ldim, cfg.flow_lm.d_model);

    // --- Load voice (optional) ---
    let voice_state = if let Some(ref voice_path) = args.voice_path {
        eprintln!("\n[2] Loading voice from {}...", voice_path);
        let voice_bytes = std::fs::read(voice_path)
            .map_err(|e| candle_core::Error::Msg(format!("failed to read voice: {e}")))?;
        eprintln!("  read {} KB", voice_bytes.len() / 1024);

        let tensors = candle_core::safetensors::load_buffer(&voice_bytes, &Device::Cpu)?;
        eprintln!("  loaded {} tensors from voice file", tensors.len());

        // Check format: KV cache (per-layer) or audio_prompt (single tensor)
        if tensors.contains_key("audio_prompt") {
            // audio_prompt format: feed through backbone to build KV cache
            let audio_prompt = tensors.get("audio_prompt").unwrap();
            eprintln!("  audio_prompt shape: {:?}", audio_prompt.shape());
            let mut state = model.init_flow_lm_state();
            model.prompt_text(&mut state, &[])?; // no text tokens, but we need to init
            // Feed audio_prompt as backbone input (it's already in embedding space)
            // For now, use prompt_text with empty tokens + backbone directly
            eprintln!("  WARN: audio_prompt voice format not yet supported for backbone injection");
            eprintln!("  Using empty state instead");
            model.init_flow_lm_state()
        } else {
            // KV cache format
            let num_layers = 6usize;
            let mut layer_states = Vec::with_capacity(num_layers);

            for i in 0..num_layers {
                let cache_name = format!("transformer.layers.{i}.self_attn/cache");
                let cache = tensors
                    .get(&cache_name)
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {cache_name}")))?;

                let k = cache.narrow(0, 0, 1)?.squeeze(0)?;
                let v = cache.narrow(0, 1, 1)?.squeeze(0)?;
                let seq_len = k.dim(1)?;

                if i == 0 {
                    eprintln!("  voice seq_len from layer 0: {seq_len}");
                    eprintln!("  k shape: {:?}", k.shape());
                }

                layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState::with_kv(
                    k.contiguous()?,
                    v.contiguous()?,
                    seq_len,
                )));
            }

            TTSState {
                flow_lm_state: FlowLMState {
                    transformer_state: StreamingTransformerState { layer_states },
                },
            }
        }
    } else {
        eprintln!("\n[2] No voice file, using empty state");
        model.init_flow_lm_state()
    };
    eprintln!("  voice state ready");

    // --- Prepare text and token IDs ---
    eprintln!("\n[3] Preparing text...");
    let raw_text = "Hello, this is a test of the text to speech system.";
    let (prepared_text, frames_after_eos) = prepare_text_prompt(raw_text);
    eprintln!("  raw: {raw_text:?}");
    eprintln!("  prepared: {prepared_text:?}");
    eprintln!("  frames_after_eos: {frames_after_eos}");

    // Real token IDs from sentencepiece encoding of the prepared text
    // Obtained by running: python3 -c "import sentencepiece as spm; sp=spm.SentencePieceProcessor(); sp.Load('tokenizer.model'); print(sp.EncodeAsIds('Hello, this is a test of the text to speech system.'))"
    let token_ids: Vec<u32> = vec![2994, 262, 285, 277, 267, 1115, 272, 265, 2009, 266, 260, 3476, 260, 848, 263];
    eprintln!("  token_ids ({} tokens): {:?}", token_ids.len(), token_ids);

    // --- Run prompt_text ---
    eprintln!("\n[4] Running prompt_text...");
    let mut tts_state = voice_state.clone();
    model.prompt_text(&mut tts_state, &token_ids)?;
    let seq_len = tts_state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  prompt_text done, seq_len={seq_len}");

    // --- Init mimi state and RNG ---
    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu)?;
    let mut rng = SimpleRng::new(args.temperature);

    // --- BOS latent: NaN tensor [1, 1, ldim] ---
    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let mut prev_latent = Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu)?;

    // --- Generation loop ---
    // Use same max_frames formula as wasm binding
    let max_frames = ((token_ids.len() as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    eprintln!("\n[5] Generating audio ({max_frames} max frames, frames_after_eos={frames_after_eos})...");

    let mut audio_chunks: Vec<f32> = Vec::new();
    let mut eos_countdown: Option<usize> = None;
    let mut total_steps = 0usize;

    for step in 0..max_frames {
        let (next_latent, is_eos) =
            model.generate_step(&mut tts_state, &prev_latent, &mut rng)?;

        let audio_chunk = model.decode_latent(&next_latent, &mut mimi_state)?;
        let pcm = audio_chunk.flatten_all()?.to_vec1::<f32>()?;

        let pcm_min = pcm.iter().cloned().fold(f32::INFINITY, f32::min);
        let pcm_max = pcm.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pcm_mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
        let nan_count = pcm.iter().filter(|x| x.is_nan()).count();
        let inf_count = pcm.iter().filter(|x| x.is_infinite()).count();

        eprintln!(
            "step {:3}: pcm_len={} is_eos={} min={:.4} max={:.4} mean={:.4} nan={nan_count} inf={inf_count}",
            step, pcm.len(), is_eos, pcm_min, pcm_max, pcm_mean
        );

        audio_chunks.extend_from_slice(&pcm);
        total_steps = step + 1;

        if is_eos && eos_countdown.is_none() {
            eprintln!("  EOS detected at step {step}, starting countdown ({frames_after_eos} frames)");
            eos_countdown = Some(frames_after_eos);
        }

        if let Some(ref mut c) = eos_countdown {
            if *c == 0 {
                eprintln!("  EOS countdown reached 0 at step {step}, stopping");
                break;
            }
            *c -= 1;
        }

        prev_latent = next_latent;
    }

    let total_seconds = audio_chunks.len() as f64 / sample_rate as f64;
    eprintln!("\n[6] Generation complete:");
    eprintln!("  total_steps: {total_steps}");
    eprintln!("  total_samples: {}", audio_chunks.len());
    eprintln!("  total_duration: {:.2}s", total_seconds);
    log_tensor_stats("final_audio", &audio_chunks);

    // --- Write WAV ---
    eprintln!("\n[7] Writing WAV to {}...", args.output_path);
    write_wav(&args.output_path, &audio_chunks, sample_rate)
        .map_err(|e| candle_core::Error::Msg(format!("failed to write WAV: {e}")))?;
    eprintln!("  wrote {} samples ({:.2}s) at {}Hz", audio_chunks.len(), total_seconds, sample_rate);
    eprintln!("\nDone! Audio saved to {}", args.output_path);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
