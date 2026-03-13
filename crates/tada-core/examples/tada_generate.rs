/// End-to-end TADA TTS generation example.
///
/// Usage:
///   cargo run --example tada_generate -p tada-core -- \
///     --model /path/to/tada-1b-q4_0.gguf \
///     --tokenizer tokenizer.json \
///     --output /tmp/test_tada.wav \
///     [--temperature 0.9] \
///     [--noise-temp 0.9] \
///     [--flow-steps 10] \
///     [--text "Hello world"] \
///     [--voice /path/to/voice.safetensors]
///
/// Default model path: /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf
/// Default tokenizer:  tokenizer.json (repo root)
///
/// When --voice is provided the generation loop uses the pre-computed acoustic
/// features from the voice prompt file for each step that falls within the
/// prompt range, then continues autoregressively.
///
/// Writes a Float32 PCM WAV at 24000 Hz (mono).

use candle_core::{Device, Result as CResult, Tensor};
use tada_core::audio_check::{check_generation, AudioStats};
use tada_core::config::TadaConfig;
use tada_core::tada_model::{Rng, TadaModel, VoicePrompt};

// ---------------------------------------------------------------------------
// CLI arg parsing (manual, no external deps)
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
    cpu: bool,
    /// Optional path to a `.safetensors` voice-prompt file.
    voice_path: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = None;
    let mut tokenizer_path = None;
    let mut output_path = None;
    let mut temperature = 0.9f32;
    let mut noise_temp = 0.9f32;
    let mut flow_steps = 10usize;
    let mut max_gen = 128usize;
    let mut text = String::from("Hello world");
    let mut cpu = false;
    let mut voice_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args[i].clone());
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                output_path = Some(args[i].clone());
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("temperature must be f32");
            }
            "--noise-temp" => {
                i += 1;
                noise_temp = args[i].parse().expect("noise-temp must be f32");
            }
            "--flow-steps" => {
                i += 1;
                flow_steps = args[i].parse().expect("flow-steps must be usize");
            }
            "--text" => {
                i += 1;
                text = args[i].clone();
            }
            "--max-gen" => {
                i += 1;
                max_gen = args[i].parse().expect("max-gen must be usize");
            }
            "--cpu" => {
                cpu = true;
            }
            "--voice" => {
                i += 1;
                voice_path = Some(args[i].clone());
            }
            other => {
                eprintln!("Unknown arg: {other}");
                eprintln!("Usage: tada_generate --model <path.gguf> --tokenizer <path.json> --output <path.wav> [--cpu] [--temperature 0.9] [--noise-temp 0.9] [--flow-steps 10] [--max-gen 128] [--text \"Hello world\"] [--voice <path.safetensors>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        model_path: model_path.expect("--model required"),
        tokenizer_path: tokenizer_path.expect("--tokenizer required"),
        output_path: output_path.expect("--output required"),
        temperature,
        noise_temp,
        flow_steps,
        max_gen,
        text,
        cpu,
        voice_path,
    }
}

// ---------------------------------------------------------------------------
// RNG using rand + rand_distr
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
    let data_size = num_samples * 4;
    let chunk_size = 36 + data_size;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk (18 bytes for float32 PCM)
    f.write_all(b"fmt ")?;
    f.write_all(&18u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?; // IEEE_FLOAT
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?; // cbSize = 0

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

fn log_stats(label: &str, data: &[f32]) {
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
        data.len(),
        min,
        max,
        mean
    );
}

// ---------------------------------------------------------------------------
// Tokenize with Llama 3 chat template
// ---------------------------------------------------------------------------

/// Build the token sequence matching Python's generate() logic.
///
/// With voice prompt:
///   [BOS, <start_header>system<end_header><eot>, <start_header>assistant<end_header>,
///    prompt_text_tokens..., target_text_tokens..., <eot>, <eos> * shift_acoustic]
///
/// Without voice prompt (zero-shot):
///   [BOS, <start_header>assistant<end_header>, \n\n, target_text_tokens..., <eot>]
///   (no trailing EOS — Python strips them when num_extra_steps > 0)
///
/// Returns (token_ids, prefix_len) where prefix_len is the number of tokens
/// before the prompt text tokens start (so we know where to align voice features).
fn tokenize(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    prompt_text: Option<&str>,
) -> CResult<(Vec<u32>, usize)> {
    let enc = |s: &str| -> CResult<Vec<u32>> {
        tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))
    };

    if let Some(prompt_txt) = prompt_text {
        // Voice-prompted mode — match Python's generate() exactly.
        //
        // Python builds: _add_bos_eos(prompt_text_tokens + target_text_tokens)
        // Then inserts prefix_text after BOS:
        //   prefix_text = "<|start_header_id|>system<|end_header_id|><|eot_id|>
        //                   <|start_header_id|>assistant<|end_header_id|>"

        // Encode prompt text + target text (no special tokens)
        let prompt_toks = enc(prompt_txt)?;
        let target_toks = enc(text)?;

        // _add_bos_eos: [prompt..., target..., eos*5]
        // Then BOS is prepended implicitly by the model? Actually, looking at the code:
        // text_tokens = encode(prompt_text, False) + encode(text, False)
        // input_ids = _add_bos_eos(tensor(text_tokens)) which pads with eos_id
        // but BOS comes from... the original encode() in line 1199 uses default (adds BOS)
        // Actually _add_bos_eos just pads EOS at end and adjusts lengths.
        // BOS comes from the tokenizer's encode() at line 1199 which uses add_special_tokens=True

        // Build raw sequence: BOS + text_tokens + eos*5
        let mut ids = vec![128000u32]; // BOS
        let bos_plus_prompt_start = 1; // BOS is at index 0

        // Now insert prefix: system header + assistant header
        // <|start_header_id|>system<|end_header_id|><|eot_id|>
        // <|start_header_id|>assistant<|end_header_id|>
        let prefix = enc("<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>")?;
        ids.extend(&prefix);
        let prefix_len = 1 + prefix.len(); // BOS + prefix tokens

        // Prompt text tokens
        ids.extend(&prompt_toks);
        // Target text tokens
        ids.extend(&target_toks);
        // <|eot_id|>
        ids.push(128009);
        // Trailing EOS tokens (shift_acoustic = 5)
        // Llama 3's eos_token_id is 128009 (<|eot_id|>), not 128001 (<|end_of_text|>)
        for _ in 0..5 {
            ids.push(128009);
        }

        Ok((ids, prefix_len))
    } else {
        // Zero-shot mode — match Python's generate() exactly.
        //
        // Python builds: _add_bos_eos(text_tokens) → [BOS, text_tokens, eos*5]
        // Then strips trailing EOS: input_ids[:, :-num_eos_tokens] → [BOS, text_tokens]
        // Then inserts prefix: [BOS, prefix_tokens, text_tokens]
        //
        // prefix_text = "<|start_header_id|>system<|end_header_id|><|eot_id|>
        //                 <|start_header_id|>assistant<|end_header_id|>"
        //
        // NO trailing <|eot_id|> in zero-shot mode!
        let text_tokens = enc(text)?;
        let mut ids = vec![128000u32]; // BOS
        // Encode prefix with special tokens
        let prefix = enc("<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>")?;
        ids.extend(&prefix);
        let prefix_len = ids.len(); // BOS + prefix
        ids.extend(&text_tokens);
        // No trailing eot_id — Python strips it in zero-shot mode

        Ok((ids, prefix_len))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> CResult<()> {
    let args = parse_args();

    // --- Select device ---
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0).unwrap_or_else(|_| {
            eprintln!("Metal not available, falling back to CPU");
            Device::Cpu
        })
    };

    eprintln!("=== TADA Generate ===");
    eprintln!("device:      {:?}", device);
    eprintln!("model:       {}", args.model_path);
    eprintln!("tokenizer:   {}", args.tokenizer_path);
    eprintln!("output:      {}", args.output_path);
    eprintln!("text:        {:?}", args.text);
    eprintln!("temperature: {}", args.temperature);
    eprintln!("noise_temp:  {}", args.noise_temp);
    eprintln!("flow_steps:  {}", args.flow_steps);
    eprintln!("max_gen:     {}", args.max_gen);
    if let Some(ref vp) = args.voice_path {
        eprintln!("voice:       {}", vp);
    }

    // --- Load model ---
    eprintln!("\n[1] Loading model...");
    let t_load_start = std::time::Instant::now();
    let model_bytes = std::fs::read(&args.model_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read model: {e}")))?;
    eprintln!("  read {} MB", model_bytes.len() / (1024 * 1024));

    let cfg = TadaConfig::tada_1b();
    let mut model = TadaModel::load_gguf(&model_bytes, &cfg, &device)?;
    let sample_rate = TadaModel::sample_rate() as u32;
    let t_load = t_load_start.elapsed();
    eprintln!("  model loaded OK (sample_rate={sample_rate}) in {:.1}s", t_load.as_secs_f64());

    // --- Load tokenizer ---
    eprintln!("\n[2] Loading tokenizer...");
    let tokenizer_bytes = std::fs::read(&args.tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read tokenizer: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;
    eprintln!("  tokenizer loaded OK");

    // --- Load optional voice prompt (before tokenization — we need the text) ---
    let (voice_prompt, voice_text): (Option<VoicePrompt>, Option<String>) =
        if let Some(ref vp_path) = args.voice_path {
            eprintln!("\n[3] Loading voice prompt from {vp_path}...");
            let vp_bytes = std::fs::read(vp_path)
                .map_err(|e| candle_core::Error::Msg(format!("failed to read voice prompt: {e}")))?;
            let vp = VoicePrompt::load(&vp_bytes, cfg.acoustic_dim, cfg.num_time_classes as u32)
                .map_err(|e| candle_core::Error::Msg(format!("failed to load voice prompt: {e}")))?;
            // Read companion .json for the voice prompt text
            let meta_path = vp_path.replace(".safetensors", ".json");
            let vt = if let Ok(meta_bytes) = std::fs::read(&meta_path) {
                let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
                    .map_err(|e| candle_core::Error::Msg(format!("voice meta json: {e}")))?;
                meta.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())
            } else {
                None
            };
            eprintln!("  voice prompt loaded: {} tokens, text={:?}", vp.len(), vt);
            (Some(vp), vt)
        } else {
            (None, None)
        };

    // --- Tokenize ---
    eprintln!("\n[4] Tokenizing...");
    let (token_ids, prefix_len) = tokenize(
        &tokenizer,
        &args.text,
        voice_text.as_deref(),
    )?;
    eprintln!("  {} tokens (prefix_len={}): {:?}", token_ids.len(), prefix_len, token_ids);

    // --- Generation setup ---
    let prompt_len = token_ids.len();
    let acoustic_dim = cfg.acoustic_dim;
    let shift_acoustic = cfg.shift_acoustic;
    let has_voice = voice_prompt.is_some();
    // In voice-prompted mode, Python runs for exactly prompt_len steps.
    // In zero-shot mode, Python uses num_extra_steps=50.
    let total_tokens = if has_voice {
        prompt_len
    } else {
        prompt_len + args.max_gen.max(50)
    };

    model.clear_state();
    let mut rng = SimpleRng::new();

    // State for generation loop
    let mut acoustics: Vec<Vec<f32>> = Vec::new();
    let mut times_before: Vec<u32> = Vec::new();
    let mut times_after: Vec<u32> = Vec::new();
    let mut next_token: u32 = token_ids[0];
    let mut acoustic = vec![0.0f32; acoustic_dim];
    let mut acoustic_mask: u32 = 0;
    let mut time_before: u32 = 0;
    let mut time_after: u32 = 0;
    let mut eos_countdown: Option<usize> = None;

    // ---------- Voice prompt alignment (matching Python's generate()) ----------
    //
    // Python does:
    //   1. Pad prompt_acoustic_features with prefix_len zeros at front
    //   2. Trim last num_transition_steps entries
    //   3. Index at step - shift_acoustic during generation
    //
    // With T voice tokens and transition_steps trimming:
    //   effective_voice_len = max(0, T - transition_steps)
    //   prompt_phase_len = prefix_len_py + effective_voice_len
    //
    // During prompt phase (step - shift_acoustic < prompt_phase_len):
    //   - First prefix_len_py steps: feed zeros, mask=0
    //   - Next effective_voice_len steps: feed voice features, mask=1
    // After prompt phase: autoregressive (feed predictions, mask=1)
    let num_transition_steps: usize = if has_voice { 5 } else { 0 };
    let prefix_len_py = prefix_len.saturating_sub(1); // Python's prefix_len excludes BOS
    let effective_voice_len = voice_prompt
        .as_ref()
        .map(|vp| vp.len().saturating_sub(num_transition_steps))
        .unwrap_or(0);
    let prompt_phase_len = prefix_len_py + effective_voice_len;

    eprintln!(
        "\n[5] Running generation (prompt_len={prompt_len}, total_tokens={total_tokens}, shift_acoustic={shift_acoustic})..."
    );
    eprintln!(
        "  voice alignment: prefix_len_py={prefix_len_py}, effective_voice_len={effective_voice_len}, prompt_phase_len={prompt_phase_len}"
    );
    if let Some(ref vp) = voice_prompt {
        if effective_voice_len > 0 {
            let voice_start = shift_acoustic + prefix_len_py;
            eprintln!("  voice prompt: {} effective tokens will steer steps {}..{}",
                effective_voice_len, voice_start, voice_start + effective_voice_len - 1);
        } else {
            eprintln!("  voice prompt: {} tokens, ALL trimmed by transition_steps={} (text-only conditioning)",
                vp.len(), num_transition_steps);
        }
    }

    let t_gen_start = std::time::Instant::now();
    for step in 0..total_tokens {
        // Get current token
        let current_token = if step < prompt_len {
            token_ids[step]
        } else {
            next_token
        };

        // Build input embeddings
        let token_tensor = Tensor::from_vec(vec![current_token], (1, 1), &device)?;
        let acoustic_tensor =
            Tensor::from_vec(acoustic.clone(), (1, 1, acoustic_dim), &device)?;
        let mask_tensor = Tensor::from_vec(vec![acoustic_mask], (1, 1), &device)?;
        let time_before_tensor = Tensor::from_vec(vec![time_before], (1, 1), &device)?;
        let time_after_tensor = Tensor::from_vec(vec![time_after], (1, 1), &device)?;

        let input_embeds = model.build_input_embeds(
            &token_tensor,
            &acoustic_tensor,
            &mask_tensor,
            &time_before_tensor,
            &time_after_tensor,
        )?;

        // Forward step
        let hidden = model.forward_step(&input_embeds)?;

        // Generate acoustics after shift_acoustic steps
        if step >= shift_acoustic {
            let (acou, tb, ta) = model.generate_acoustic(
                &hidden,
                args.noise_temp,
                &mut rng,
                args.flow_steps,
            )?;

            let acou_vec = acou.squeeze(0)?.to_vec1::<f32>()?;
            acoustics.push(acou_vec);
            times_before.push(tb);
            times_after.push(ta);
        }

        // Sample next token after prompt
        let mut is_eos = false;
        let mut sampled_id = current_token;
        if step >= prompt_len - 1 {
            let (token_id, eos) = model.sample_next_token(&hidden, args.temperature, &mut rng)?;
            sampled_id = token_id;
            is_eos = eos;
            next_token = sampled_id;
        }

        // Update acoustic/time for next step input.
        //
        // Match Python's generate() logic:
        //   - During prompt phase (first prompt_phase_len steps after shift_acoustic):
        //     feed zeros or voice features depending on position
        //   - After prompt phase: autoregressive (feed model's own predictions)
        if step >= shift_acoustic {
            let prompt_idx = step - shift_acoustic;

            if has_voice && prompt_idx < prompt_phase_len {
                // Still in prompt phase — check if we're in the voice region
                if prompt_idx >= prefix_len_py && prompt_idx < prefix_len_py + effective_voice_len {
                    // Feed voice features
                    if let Some(vp) = voice_prompt.as_ref() {
                        // Use get_step with offset = shift_acoustic + prefix_len_py
                        if let Some((vp_acoustic, vp_mask, vp_tb, vp_ta)) =
                            vp.get_step(step, shift_acoustic + prefix_len_py)
                        {
                            acoustic = vp_acoustic.clone();
                            acoustic_mask = vp_mask;
                            time_before = vp_tb;
                            time_after = vp_ta;
                        }
                    }
                } else {
                    // Padding region: feed zeros, mask=0
                    acoustic = vec![0.0; acoustic_dim];
                    acoustic_mask = 0;
                    time_before = 0;
                    time_after = 0;
                }
            } else if let Some(ac) = acoustics.last() {
                // Autoregressive: use the model's own last prediction.
                acoustic = ac.clone();
                acoustic_mask = 1;
                time_before = *times_before.last().unwrap_or(&0);
                time_after = *times_after.last().unwrap_or(&0);
            }
        }

        // Debug: log top-5 logits and hidden stats at key steps
        if step >= prompt_len - 1 && step < prompt_len + 3 {
            let logits_dbg = model.lm_head_logits(&hidden).unwrap();
            let logits_vec: Vec<f32> = logits_dbg.squeeze(0).unwrap().squeeze(0).unwrap().to_vec1().unwrap();
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("  [DEBUG step {step}] top-5 logits: {:?}", &indexed[..5]);
            eprintln!("  [DEBUG step {step}] EOS(128001) logit: {:.4}, EOT(128009) logit: {:.4}",
                logits_vec[128001], logits_vec[128009]);
            let h_vec: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();
            let h_mean = h_vec.iter().sum::<f32>() / h_vec.len() as f32;
            let h_max = h_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let h_min = h_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!("  [DEBUG step {step}] hidden: mean={h_mean:.4} min={h_min:.4} max={h_max:.4}");
        }

        // Log progress
        let voice_tag = if step >= shift_acoustic && has_voice {
            let pidx = step - shift_acoustic;
            if pidx < prompt_phase_len {
                if pidx >= prefix_len_py && pidx < prefix_len_py + effective_voice_len {
                    " [voice]"
                } else {
                    " [zeros]"
                }
            } else {
                ""
            }
        } else {
            ""
        };
        eprintln!(
            "  step {:3}/{}: token_id={:6} is_eos={} acoustic_frames={}{}{}",
            step,
            total_tokens,
            sampled_id,
            is_eos,
            acoustics.len(),
            if step < shift_acoustic { " (pre-shift)" } else { "" },
            voice_tag,
        );

        // Handle EOS countdown (only in zero-shot mode; voice mode runs
        // for exactly prompt_len steps like Python)
        if !has_voice && is_eos && eos_countdown.is_none() {
            eprintln!("  >>> EOS detected at step {step}, starting countdown ({shift_acoustic} more steps)");
            eos_countdown = Some(shift_acoustic);
        }

        if let Some(ref mut countdown) = eos_countdown {
            if *countdown == 0 {
                eprintln!("  >>> EOS countdown reached 0 at step {step}, stopping");
                break;
            }
            *countdown -= 1;
        }
    }

    let t_gen = t_gen_start.elapsed();
    let hit_eos = eos_countdown.is_some();
    eprintln!(
        "\n  Generation complete: {} acoustic frames, {} time values, eos={}, generation={:.1}s",
        acoustics.len(),
        times_before.len(),
        hit_eos,
        t_gen.as_secs_f64(),
    );

    // --- Strip leading frames ---
    // Python: encoded = acoustic_features[..., num_prompt_tokens + num_transition_steps - 1:, :]
    // where num_prompt_tokens = prompt_acoustic_features.shape[1] = prompt_phase_len
    // strip = prompt_phase_len + num_transition_steps - 1
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

    // Python also adds one extra time_before at the end (from the last step)
    // and removes leading silence: wav[..., int(24000 * time_before[0] / 50):]
    // For now, add trailing time entry (use first times_before or 0).
    let trailing_time = *times_before.first().unwrap_or(&0);
    times_before.push(trailing_time);

    // --- Validate generation ---
    let num_text_tokens = prompt_len - shift_acoustic; // just the actual text tokens
    let gen_failures = check_generation(
        num_text_tokens,
        acoustics.len(),
        &times_before,
        hit_eos,
    );
    if !gen_failures.is_empty() {
        eprintln!("\n  ⚠ Generation sanity checks FAILED:");
        for f in &gen_failures {
            eprintln!("    - {f}");
        }
    }

    // --- Debug acoustic stats ---
    {
        let all_flat: Vec<f32> = acoustics.iter().flat_map(|v| v.iter().copied()).collect();
        let a_mean = all_flat.iter().sum::<f32>() / all_flat.len() as f32;
        let a_min = all_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let a_max = all_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let a_std = (all_flat.iter().map(|x| (x - a_mean).powi(2)).sum::<f32>() / all_flat.len() as f32).sqrt();
        eprintln!("  [acoustic features] mean={a_mean:.4} std={a_std:.4} min={a_min:.4} max={a_max:.4}");
        eprintln!("  [times_before] {:?}", &times_before);
        eprintln!("  [times_after]  {:?}", &times_after);
    }

    // --- Decode audio ---
    eprintln!("\n[6] Decoding audio...");
    let t_decode_start = std::time::Instant::now();
    let samples = model.decode_audio(&acoustics, &times_before)?;
    let t_decode = t_decode_start.elapsed();
    let stats = AudioStats::from_samples(&samples, sample_rate as usize);
    eprintln!("  {stats}");

    let audio_failures = stats.check(num_text_tokens);
    if !audio_failures.is_empty() {
        eprintln!("\n  ⚠ Audio sanity checks FAILED:");
        for f in &audio_failures {
            eprintln!("    - {f}");
        }
    }

    let all_failures: Vec<String> = gen_failures.into_iter().chain(audio_failures).collect();
    if !all_failures.is_empty() {
        eprintln!("\n  ❌ {} sanity check(s) failed — output is likely garbage", all_failures.len());
    } else {
        eprintln!("\n  ✓ All sanity checks passed");
    }

    // --- Write WAV ---
    eprintln!("\n[7] Writing WAV to {}...", args.output_path);
    write_wav(&args.output_path, &samples, sample_rate)
        .map_err(|e| candle_core::Error::Msg(format!("failed to write WAV: {e}")))?;
    eprintln!("  wrote {} samples ({:.2}s)", samples.len(), stats.duration_secs);
    eprintln!("\n=== Timing ===");
    eprintln!("  load:       {:.1}s", t_load.as_secs_f64());
    eprintln!("  generation: {:.1}s", t_gen.as_secs_f64());
    eprintln!("  decode:     {:.1}s", t_decode.as_secs_f64());
    eprintln!("  audio:      {:.1}s", stats.duration_secs);
    let rtf = t_gen.as_secs_f64() / stats.duration_secs;
    eprintln!("  RTF (gen):  {:.2}x", rtf);
    eprintln!("\nDone! Audio saved to {}", args.output_path);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
