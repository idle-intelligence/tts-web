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
///     [--text "Hello world"]
///
/// Mirrors the candle-only `tada_generate` example but runs the LLM backbone
/// on GPU via Burn/wgpu and VibeVoice + decoder on CPU via candle.

use std::io::Cursor;
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;

use tada_core::config::TadaConfig;
use tada_core::tada_model::{Rng, TadaModel};
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
    fn new() -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, 1.0).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
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
// Tokenize with Llama 3 chat template (zero-shot)
// ---------------------------------------------------------------------------

fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    let enc = |s: &str| -> Vec<u32> {
        tokenizer.encode(s, false).unwrap().get_ids().to_vec()
    };

    let mut ids = vec![128000u32]; // BOS
    let prefix = enc("<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>");
    ids.extend(&prefix);
    ids.extend(enc(text));
    // No trailing eot_id in zero-shot mode
    ids
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

    // --- Tokenize ---
    let token_ids = tokenize(&tokenizer, &args.text);
    let prompt_len = token_ids.len();
    let total_tokens = prompt_len + args.max_gen.max(50);
    eprintln!("  {} tokens: {:?}", token_ids.len(), token_ids);

    // --- Generation loop ---
    eprintln!("\n[5] Running generation (prompt_len={}, total_tokens={})...", prompt_len, total_tokens);

    candle_model.clear_state();
    let mut cache = burn_model.create_cache(total_tokens + 64);
    let mut rng = SimpleRng::new();

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
        if step >= shift_acoustic {
            let t_vibe = Instant::now();
            let (acou, tb, ta) = candle_model.generate_acoustic(
                &hidden_candle,
                args.noise_temp,
                &mut rng,
                args.flow_steps,
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

        // Update acoustic for next step
        if step >= shift_acoustic {
            if let Some(ac) = acoustics.last() {
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

        // EOS countdown
        if is_eos && eos_countdown.is_none() {
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

    // Trailing time entry
    let trailing_time = *times_before.first().unwrap_or(&0);
    times_before.push(trailing_time);

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
