/// End-to-end TADA TTS generation using Burn/wgpu LLM + candle VibeVoice/decoder.
///
/// Usage:
///   cargo run --example tada_generate_burn -p tada-wasm --release -- \
///     --model /path/to/tada-1b-q4_0.gguf \
///     --tokenizer tokenizer.json \
///     --output /tmp/test_tada_burn.wav \
///     [--temperature 0.9] \
///     [--noise-temp 0.6] \
///     [--flow-steps 10] \
///     [--text "Hello world"] \
///     [--voice /path/to/voice.safetensors] \
///     [--seed 42] \
///     [--transition-steps 5]
///
/// Mirrors the candle-only `tada_generate` example but runs the LLM backbone
/// on GPU via Burn/wgpu and VibeVoice + decoder on CPU via candle.

use std::io::Cursor;
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;

use tada_core::config::TadaConfig;
use tada_core::tada_model::{Rng, TadaModel, VoicePrompt};
use tada_wasm::gguf;
use tada_wasm::model;

// ---------------------------------------------------------------------------
// CLI arg parsing
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    tokenizer_path: String,
    output_path: String,
    temperature: f32,
    noise_temp: f32,
    flow_steps: usize,
    max_gen: usize,
    text: String,
    voice_path: Option<String>,
    seed: u64,
    transition_steps: usize,
    cfg_scale: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = None;
    let mut tokenizer_path = None;
    let mut output_path = None;
    let mut temperature = 0.9f32;
    let mut noise_temp = 0.6f32;
    let mut flow_steps = 10usize;
    let mut max_gen = 128usize;
    let mut text = String::from("The quick brown fox jumps over the lazy dog.");
    let mut voice_path: Option<String> = None;
    let mut seed = 42u64;
    let mut transition_steps = 5usize;
    let mut cfg_scale = 1.0f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = Some(args[i].clone()); }
            "--tokenizer" => { i += 1; tokenizer_path = Some(args[i].clone()); }
            "--output" => { i += 1; output_path = Some(args[i].clone()); }
            "--temperature" => { i += 1; temperature = args[i].parse().expect("temperature must be f32"); }
            "--noise-temp" => { i += 1; noise_temp = args[i].parse().expect("noise-temp must be f32"); }
            "--flow-steps" => { i += 1; flow_steps = args[i].parse().expect("flow-steps must be usize"); }
            "--max-gen" => { i += 1; max_gen = args[i].parse().expect("max-gen must be usize"); }
            "--text" => { i += 1; text = args[i].clone(); }
            "--voice" => { i += 1; voice_path = Some(args[i].clone()); }
            "--seed" => { i += 1; seed = args[i].parse().expect("seed must be u64"); }
            "--transition-steps" => { i += 1; transition_steps = args[i].parse().expect("transition-steps must be usize"); }
            "--cfg-scale" => { i += 1; cfg_scale = args[i].parse().expect("cfg-scale must be f32"); }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        model_path: model_path.unwrap_or_else(|| "/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf".to_string()),
        tokenizer_path: tokenizer_path.unwrap_or_else(|| "tokenizer.json".to_string()),
        output_path: output_path.unwrap_or_else(|| "/tmp/test_tada_burn.wav".to_string()),
        temperature,
        noise_temp,
        flow_steps,
        max_gen,
        text,
        voice_path,
        seed,
        transition_steps,
        cfg_scale,
    }
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

struct SimpleRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, 1.0).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { inner: rng, distr }
    }
}

impl Rng for SimpleRng {
    fn sample_normal(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

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
// Tokenize with Llama 3 chat template
// ---------------------------------------------------------------------------

/// Build the token sequence matching Python's generate() logic.
///
/// With voice prompt:
///   [BOS, <start_header>system<end_header><eot>, <start_header>assistant<end_header>,
///    prompt_text_tokens..., target_text_tokens..., <eot>, <eot> * shift_acoustic]
///
/// Without voice prompt (zero-shot):
///   [BOS, <start_header>system<end_header><eot>, <start_header>assistant<end_header>,
///    target_text_tokens...]
///
/// Returns (token_ids, prefix_len) where prefix_len is the number of tokens
/// before the prompt text tokens start (used for voice feature alignment).
fn tokenize(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    prompt_text: Option<&str>,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let enc = |s: &str| -> anyhow::Result<Vec<u32>> {
        tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("{e}"))
    };

    if let Some(prompt_txt) = prompt_text {
        let prompt_toks = enc(prompt_txt)?;
        let target_toks = enc(text)?;

        let mut ids = vec![128000u32]; // BOS
        let prefix = enc("<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>")?;
        ids.extend(&prefix);
        let prefix_len = 1 + prefix.len(); // BOS + prefix tokens

        ids.extend(&prompt_toks);
        ids.extend(&target_toks);
        ids.push(128009); // <|eot_id|>
        // Trailing EOT tokens (shift_acoustic = 5)
        for _ in 0..5 {
            ids.push(128009);
        }

        Ok((ids, prefix_len))
    } else {
        let text_tokens = enc(text)?;
        let mut ids = vec![128000u32]; // BOS
        let prefix = enc("<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>")?;
        ids.extend(&prefix);
        let prefix_len = ids.len(); // BOS + prefix
        ids.extend(&text_tokens);
        // No trailing eot_id in zero-shot mode

        Ok((ids, prefix_len))
    }
}

// ---------------------------------------------------------------------------
// Token sampling (Gumbel-max)
// ---------------------------------------------------------------------------

fn sample_token(logits: &[f32], temperature: f32, rng: &mut SimpleRng) -> (u32, bool) {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &l) in logits.iter().enumerate() {
        let scaled = if (temperature - 1.0).abs() > 1e-6 { l / temperature } else { l };

        let u: f32 = loop {
            let n = rng.sample_normal();
            let u = 1.0 / (1.0 + (-n).exp());
            if u > 0.0 && u < 1.0 { break u; }
        };
        let gumbel = -((-u.ln()).ln());
        let val = scaled + gumbel;
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }

    let is_eos = best_idx == 128001 || best_idx == 128009;
    (best_idx, is_eos)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> anyhow::Result<()> {
    let args = parse_args();
    let cfg = TadaConfig::tada_1b();
    let acoustic_dim = cfg.acoustic_dim;
    let shift_acoustic = cfg.shift_acoustic;
    let sample_rate = 24000u32;

    eprintln!("=== TADA Generate (Burn/wgpu LLM + candle VibeVoice/Decoder) ===");
    eprintln!("model:       {}", args.model_path);
    eprintln!("tokenizer:   {}", args.tokenizer_path);
    eprintln!("output:      {}", args.output_path);
    eprintln!("text:        {:?}", args.text);
    eprintln!("temperature: {}", args.temperature);
    eprintln!("noise_temp:  {}", args.noise_temp);
    eprintln!("flow_steps:  {}", args.flow_steps);
    eprintln!("seed:        {}", args.seed);
    if let Some(ref vp) = args.voice_path {
        eprintln!("voice:       {}", vp);
        eprintln!("transition_steps: {}", args.transition_steps);
    }

    // --- Read GGUF file ---
    eprintln!("\n[1] Reading GGUF file...");
    let t_read_start = Instant::now();
    let gguf_bytes = std::fs::read(&args.model_path)?;
    let t_read = t_read_start.elapsed();
    eprintln!("  read {} MB in {:.1}s", gguf_bytes.len() / (1024 * 1024), t_read.as_secs_f64());

    // --- Load Burn LLM (GPU) ---
    eprintln!("\n[2] Loading Burn LLM on GPU...");
    let t_burn_start = Instant::now();
    let device = WgpuDevice::default();
    let cursor = Cursor::new(&gguf_bytes);
    let mut reader = gguf::GgufReader::open(cursor)?;
    let burn_model = model::load_tada_llama_gguf(&mut reader, &device)?;
    drop(reader);
    let t_burn_load = t_burn_start.elapsed();
    eprintln!("  Burn LLM loaded in {:.1}s", t_burn_load.as_secs_f64());

    // --- Load candle VibeVoice + decoder (CPU) ---
    eprintln!("\n[3] Loading candle VibeVoice + decoder on CPU...");
    let t_candle_start = Instant::now();
    let mut candle_model = TadaModel::load_gguf(&gguf_bytes, &cfg, &candle_core::Device::Cpu)?;
    let t_candle_load = t_candle_start.elapsed();
    eprintln!("  candle model loaded in {:.1}s", t_candle_load.as_secs_f64());

    let t_load = t_read.as_secs_f64() + t_burn_load.as_secs_f64() + t_candle_load.as_secs_f64();
    eprintln!("  total load: {:.1}s (read={:.1}s, burn={:.1}s, candle={:.1}s)",
        t_load, t_read.as_secs_f64(), t_burn_load.as_secs_f64(), t_candle_load.as_secs_f64());

    // --- Load tokenizer ---
    eprintln!("\n[4] Loading tokenizer...");
    let tokenizer_bytes = std::fs::read(&args.tokenizer_path)?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // --- Load optional voice prompt ---
    let (voice_prompt, voice_text): (Option<VoicePrompt>, Option<String>) =
        if let Some(ref vp_path) = args.voice_path {
            eprintln!("\n[4b] Loading voice prompt from {vp_path}...");
            let vp_bytes = std::fs::read(vp_path)
                .map_err(|e| anyhow::anyhow!("failed to read voice prompt: {e}"))?;
            let vp = VoicePrompt::load(&vp_bytes, cfg.acoustic_dim, cfg.num_time_classes as u32)
                .map_err(|e| anyhow::anyhow!("failed to load voice prompt: {e}"))?;
            let meta_path = vp_path.replace(".safetensors", ".json");
            let vt = if let Ok(meta_bytes) = std::fs::read(&meta_path) {
                // Parse "text" field from JSON without serde_json dependency.
                // Looks for: "text": "..." or "text":"..."
                let meta_str = String::from_utf8_lossy(&meta_bytes);
                meta_str
                    .find("\"text\"")
                    .and_then(|pos| meta_str[pos + 6..].find('"').map(|q| pos + 6 + q + 1))
                    .and_then(|start| {
                        let rest = &meta_str[start..];
                        rest.find('"').map(|end| rest[..end].to_string())
                    })
            } else {
                None
            };
            eprintln!("  voice prompt loaded: {} tokens, text={:?}", vp.len(), vt);
            (Some(vp), vt)
        } else {
            (None, None)
        };

    // --- Tokenize ---
    let (token_ids, prefix_len) = tokenize(&tokenizer, &args.text, voice_text.as_deref())?;
    let prompt_len = token_ids.len();
    let has_voice = voice_prompt.is_some();

    // In voice-prompted mode, run for exactly prompt_len steps (matches Python).
    // In zero-shot mode, add autoregressive steps.
    let total_tokens = if has_voice {
        prompt_len
    } else {
        prompt_len + args.max_gen.max(50)
    };
    eprintln!("  {} tokens (prefix_len={}): {:?}", token_ids.len(), prefix_len, token_ids);

    // --- Voice prompt alignment parameters (mirrors tada_generate.rs) ---
    let num_transition_steps: usize = if has_voice { args.transition_steps } else { 0 };
    let prefix_len_py = prefix_len.saturating_sub(1); // Python's prefix_len excludes BOS
    let effective_voice_len = voice_prompt
        .as_ref()
        .map(|vp| vp.len().saturating_sub(num_transition_steps))
        .unwrap_or(0);
    let prompt_phase_len = prefix_len_py + effective_voice_len;

    // --- Generation loop ---
    eprintln!("\n[5] Running generation (prompt_len={}, total_tokens={}, shift_acoustic={})...", prompt_len, total_tokens, shift_acoustic);
    eprintln!("  voice alignment: prefix_len_py={prefix_len_py}, effective_voice_len={effective_voice_len}, prompt_phase_len={prompt_phase_len}");

    candle_model.clear_state();
    let mut cache = burn_model.create_cache(total_tokens + 64);
    let mut rng = SimpleRng::new(args.seed);

    let mut acoustics: Vec<Vec<f32>> = Vec::new();
    let mut times_before: Vec<u32> = Vec::new();
    let mut times_after: Vec<u32> = Vec::new();
    let mut next_token: u32 = token_ids[0];
    let mut acoustic = vec![0.0f32; acoustic_dim];
    let mut acoustic_mask: u32 = 0;
    let mut time_before: u32 = 0;
    let mut time_after: u32 = 0;
    let mut eos_countdown: Option<usize> = None;

    let mut total_llm_ms: f64 = 0.0;
    let mut total_vibe_ms: f64 = 0.0;

    let t_gen_start = Instant::now();
    for step in 0..total_tokens {
        let current_token = if step < prompt_len { token_ids[step] } else { next_token };

        // --- Burn LLM forward (GPU) ---
        let t_step = Instant::now();
        let hidden_burn = burn_model.forward_step(
            current_token,
            &acoustic,
            acoustic_mask,
            time_before,
            time_after,
            &mut cache,
        );

        // Read hidden state back to CPU for candle
        let hidden_data = hidden_burn.clone().into_data();
        let hidden_vec: Vec<f32> = hidden_data.to_vec().unwrap();
        let llm_ms = t_step.elapsed().as_secs_f64() * 1000.0;
        total_llm_ms += llm_ms;

        let hidden_candle = candle_core::Tensor::from_vec(
            hidden_vec,
            (1, 1, cfg.llama.hidden_size),
            &candle_core::Device::Cpu,
        )?;

        // --- VibeVoice flow matching (CPU) ---
        // Skip during EOS countdown — post-EOS hidden states decode to noise.
        if step >= shift_acoustic && eos_countdown.is_none() {
            let t_vibe = Instant::now();
            let (acou, tb, ta) = candle_model.generate_acoustic(
                &hidden_candle,
                None,  // neg_hidden: TODO compute for proper CFG
                args.noise_temp,
                &mut rng,
                args.flow_steps,
                args.cfg_scale,
            )?;
            let vibe_ms = t_vibe.elapsed().as_secs_f64() * 1000.0;
            total_vibe_ms += vibe_ms;

            let acou_vec = acou.squeeze(0)?.to_vec1::<f32>()?;
            acoustics.push(acou_vec);
            times_before.push(tb);
            times_after.push(ta);
        }

        // --- Sample next token ---
        let mut is_eos = false;
        let mut sampled_id = current_token;
        if step >= prompt_len - 1 {
            let logits_burn = burn_model.lm_head(hidden_burn);
            let logits_data = logits_burn.into_data();
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();

            let (token_id, eos) = sample_token(&logits_vec, args.temperature, &mut rng);
            sampled_id = token_id;
            is_eos = eos;
            next_token = sampled_id;
        }

        // Update acoustic/time for the next step input.
        //
        // Mirrors tada_generate.rs logic:
        //   - During prompt phase (voice-prompted mode): feed zeros or voice features
        //   - After prompt phase: autoregressive (use model's own last prediction)
        //
        // IMPORTANT: the update prepares INPUT for step+1, so we check whether
        // *next* step's prompt_idx is still within the prompt phase.
        if step >= shift_acoustic {
            let next_prompt_idx = (step + 1) - shift_acoustic;

            if has_voice && next_prompt_idx < prompt_phase_len {
                if next_prompt_idx >= prefix_len_py
                    && next_prompt_idx < prefix_len_py + effective_voice_len
                {
                    // Next step is in the voice-feature region.
                    if let Some(ref vp) = voice_prompt {
                        if let Some((vp_acoustic, vp_mask, vp_tb, vp_ta)) =
                            vp.get_step(step + 1, shift_acoustic + prefix_len_py)
                        {
                            acoustic = vp_acoustic.clone();
                            acoustic_mask = vp_mask;
                            time_before = vp_tb;
                            time_after = vp_ta;
                        }
                    }
                } else {
                    // Next step is in the padding region: feed zeros, mask=0.
                    acoustic = vec![0.0; acoustic_dim];
                    acoustic_mask = 0;
                    time_before = 0;
                    time_after = 0;
                }
            } else if let Some(ac) = acoustics.last() {
                // Next step is autoregressive: use the model's own last prediction.
                acoustic = ac.clone();
                acoustic_mask = 1;
                time_before = *times_before.last().unwrap_or(&0);
                time_after = *times_after.last().unwrap_or(&0);
            }
        }

        // Log progress every 10 steps
        if step % 10 == 0 || is_eos {
            eprintln!(
                "  step {:3}/{}: token={:6} eos={} frames={} llm={:.0}ms vibe={:.0}ms",
                step, total_tokens, sampled_id, is_eos, acoustics.len(),
                llm_ms, if step >= shift_acoustic { total_vibe_ms / (acoustics.len() as f64).max(1.0) } else { 0.0 },
            );
        }

        // EOS countdown (only in zero-shot mode; voice mode runs for exactly prompt_len steps)
        if !has_voice && is_eos && eos_countdown.is_none() {
            eprintln!("  >>> EOS at step {step}, {shift_acoustic} more steps");
            eos_countdown = Some(shift_acoustic);
        }
        if let Some(ref mut countdown) = eos_countdown {
            if *countdown == 0 {
                eprintln!("  >>> stopping at step {step}");
                break;
            }
            *countdown -= 1;
        }
    }

    let t_gen = t_gen_start.elapsed();
    let num_gen_steps = acoustics.len();
    eprintln!(
        "\n  Generation: {} frames in {:.1}s (llm={:.1}s, vibe={:.1}s)",
        num_gen_steps,
        t_gen.as_secs_f64(),
        total_llm_ms / 1000.0,
        total_vibe_ms / 1000.0,
    );
    if num_gen_steps > 0 {
        eprintln!(
            "  Per-step avg: llm={:.0}ms, vibe={:.0}ms",
            total_llm_ms / num_gen_steps as f64,
            total_vibe_ms / num_gen_steps as f64,
        );
    }

    // --- Strip leading frames (voice-prompted mode) ---
    // Python: encoded = acoustic_features[..., num_prompt_tokens + num_transition_steps - 1:, :]
    // where num_prompt_tokens = prompt_phase_len
    let strip_frames = if prompt_phase_len + num_transition_steps >= 1 {
        prompt_phase_len + num_transition_steps - 1
    } else {
        0
    };
    if strip_frames > 0 && strip_frames < acoustics.len() {
        eprintln!(
            "\n  Stripping first {} acoustic frames (prompt_phase={}, transition={})",
            strip_frames, prompt_phase_len, num_transition_steps
        );
        acoustics.drain(..strip_frames);
        times_before.drain(..strip_frames);
        times_after.drain(..strip_frames);
    }

    // In voice-prompted mode, trim the last acoustic frame.
    if has_voice && !acoustics.is_empty() {
        acoustics.pop();
        times_before.pop();
        times_after.pop();
        eprintln!("  Trimmed 1 trailing meaningless acoustic frame (voice-prompted mode)");
    }

    // Trailing time entry (must be 1, not the first frame's duration)
    times_before.push(1);

    eprintln!("  times_before ({} entries): {:?}", times_before.len(), &times_before);

    // --- Decode audio ---
    eprintln!("\n[6] Decoding audio...");
    let t_decode_start = Instant::now();
    let samples = candle_model.decode_audio(&acoustics, &times_before)?;
    let t_decode = t_decode_start.elapsed();

    let duration_secs = samples.len() as f64 / sample_rate as f64;
    eprintln!("  {} samples, {:.2}s audio, decoded in {:.1}s", samples.len(), duration_secs, t_decode.as_secs_f64());

    // --- Write WAV ---
    eprintln!("\n[7] Writing WAV to {}...", args.output_path);
    write_wav(&args.output_path, &samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("WAV write: {e}"))?;

    // --- Summary ---
    let rtf = t_gen.as_secs_f64() / duration_secs;
    eprintln!("\n=== Timing Summary ===");
    eprintln!("  load:       {:.1}s (read={:.1}s burn={:.1}s candle={:.1}s)",
        t_load, t_read.as_secs_f64(), t_burn_load.as_secs_f64(), t_candle_load.as_secs_f64());
    eprintln!("  generation: {:.1}s (llm={:.1}s vibe={:.1}s)",
        t_gen.as_secs_f64(), total_llm_ms / 1000.0, total_vibe_ms / 1000.0);
    eprintln!("  decode:     {:.1}s", t_decode.as_secs_f64());
    eprintln!("  audio:      {:.2}s", duration_secs);
    eprintln!("  RTF (gen):  {:.2}x", rtf);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
