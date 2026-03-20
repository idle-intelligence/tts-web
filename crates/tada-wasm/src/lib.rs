//! TADA TTS — browser-native text-to-speech.
//!
//! Hybrid architecture:
//! - Llama 3.2 1B backbone runs on GPU via Burn/wgpu (Q4_0 quantized)
//! - VibeVoice diffusion head + DAC decoder run on CPU via candle
//!
//! The WASM API exposes step-by-step generation so the JS worker can
//! stream partial audio to the main thread.

pub mod gguf;
pub mod model;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use model::vibevoice::BurnVibeVoice;
use model::{LayerCaches, TadaLlama};
use tada_core::config::TadaConfig;
use tada_core::tada_model::{self, BOS_TOKEN_ID, EOS_TOKEN_ID, EOT_TOKEN_ID, VoicePrompt};

// ---------------------------------------------------------------------------
// WasmRng — WASM-compatible RNG
// ---------------------------------------------------------------------------

struct WasmRng {
    inner: Box<rand::rngs::StdRng>,
    distr: rand_distr::Normal<f32>,
}

impl WasmRng {
    fn new() -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, 1.0).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self {
            inner: Box::new(rng),
            distr,
        }
    }
}

impl tada_model::Rng for WasmRng {
    fn sample_normal(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

impl model::vibevoice::Rng for WasmRng {
    fn sample_normal(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

// ---------------------------------------------------------------------------
// Generation state
// ---------------------------------------------------------------------------

struct GenState {
    acoustics: Vec<Vec<f32>>,
    times_before: Vec<u32>,
    times_after: Vec<u32>,
    token_ids: Vec<u32>,
    next_token: u32,
    acoustic: Vec<f32>,
    acoustic_mask: u32,
    time_before: u32,
    time_after: u32,
    step: usize,
    total_tokens: usize,
    shift_acoustic: usize,
    rng: WasmRng,
    temperature: f32,
    noise_temp: f32,
    num_flow_steps: usize,
    is_done: bool,
    eos_countdown: Option<usize>,
    cache: LayerCaches,
    // Voice prompt alignment parameters (all zero/None in zero-shot mode).
    voice_prompt: Option<VoicePrompt>,
    /// Number of transition steps (Python default: 5, zero-shot: 0).
    num_transition_steps: usize,
    /// prefix_len_py = number of text/prefix tokens before voice acoustic data starts
    /// (excludes BOS — matches Python's prefix_len).
    prefix_len_py: usize,
    /// effective_voice_len = voice_prompt.len() - num_transition_steps
    effective_voice_len: usize,
    /// prompt_phase_len = prefix_len_py + effective_voice_len
    prompt_phase_len: usize,
    /// CFG scale for acoustic features (1.0 = off, 1.6 = Python default)
    cfg_scale: f32,
}

// ---------------------------------------------------------------------------
// HybridTadaModel — Burn LLM + candle VibeVoice/decoder
// ---------------------------------------------------------------------------

/// The hybrid TADA model combining GPU LLM and CPU diffusion/decoder.
pub struct HybridTadaModel {
    /// Llama backbone on GPU (Burn/wgpu)
    llama: TadaLlama,
    /// VibeVoice diffusion head on GPU (Burn/wgpu) — optional.
    /// When present, flow matching runs entirely on GPU and only
    /// 512 acoustic floats + 2 time values are read back per step.
    burn_vv: Option<BurnVibeVoice>,
    /// VibeVoice + decoder on CPU (candle) — always present as fallback.
    candle_model: tada_core::tada_model::TadaModel,
    /// Config
    cfg: TadaConfig,
    /// Generation state
    gen_state: Option<GenState>,
    /// Tokenizer
    tokenizer: tokenizers::Tokenizer,
}

impl HybridTadaModel {
    /// Load the hybrid model from GGUF bytes.
    ///
    /// The same GGUF file contains both the Llama weights (loaded into Burn/GPU)
    /// and the VibeVoice/decoder weights (loaded into candle/CPU).
    pub fn load(
        gguf_bytes: Vec<Vec<u8>>,
        tokenizer_bytes: &[u8],
        device: &WgpuDevice,
    ) -> anyhow::Result<Self> {
        let cfg = TadaConfig::tada_1b();

        // Single buffer: move the first shard if only one, else flatten.
        // Both Burn and candle borrow from this — zero extra copies.
        let buf = if gguf_bytes.len() == 1 {
            gguf_bytes.into_iter().next().unwrap() // move, no copy
        } else {
            let total_len: usize = gguf_bytes.iter().map(|s| s.len()).sum();
            let mut flat = Vec::with_capacity(total_len);
            for shard in gguf_bytes {
                flat.extend_from_slice(&shard);
            }
            flat
        };

        // Load Llama backbone into Burn/GPU (borrows buf)
        let cursor = std::io::Cursor::new(&buf);
        let mut reader = gguf::GgufReader::open(cursor)?;
        let llama = model::load_tada_llama_gguf(&mut reader, device)?;

        // Try to load VibeVoice into Burn/GPU as well.
        // This is optional — if it fails (e.g. weights not present or dtype unsupported),
        // we fall back to the candle CPU path without hard-failing the load.
        // Try loading BurnVibeVoice with Q8_0 weights on GPU.
        // F32 was too slow (5s/step), but Q8_0 shader should be ~280ms/step.
        let burn_vv = match model::load_burn_vibevoice(&mut reader, device) {
            Ok(vv) => {
                eprintln!("[tada] BurnVibeVoice loaded (Q8_0 GPU)");
                Some(vv)
            }
            Err(e) => {
                eprintln!("[tada] BurnVibeVoice skipped (candle CPU fallback): {e}");
                None
            }
        };
        drop(reader);

        // Load decoder + adapters into candle/CPU (skip LLM, skip VV if Burn handles it)
        let candle_model = if burn_vv.is_some() {
            tada_core::tada_model::TadaModel::load_gguf_no_llm_no_vv(&buf, &cfg, &candle_core::Device::Cpu)?
        } else {
            tada_core::tada_model::TadaModel::load_gguf_no_llm(&buf, &cfg, &candle_core::Device::Cpu)?
        };

        // Free the raw GGUF bytes — all tensors are now in GPU/CPU memory
        drop(buf);

        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        Ok(Self {
            llama,
            burn_vv,
            candle_model,
            cfg,
            gen_state: None,
            tokenizer,
        })
    }

    /// Tokenize text with Llama 3 chat template.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let (ids, _) = self.tokenize_with_voice(text, None)?;
        Ok(ids)
    }

    /// Tokenize text with optional voice prompt text prepended.
    ///
    /// Returns `(token_ids, prefix_len)` where `prefix_len` is the number of
    /// tokens before the voice/target text region starts (BOS + header tokens).
    /// In zero-shot mode (`voice_text = None`) `prefix_len` is unused.
    pub fn tokenize_with_voice(
        &self,
        text: &str,
        voice_text: Option<&str>,
    ) -> anyhow::Result<(Vec<u32>, usize)> {
        let enc = |s: &str| -> anyhow::Result<Vec<u32>> {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!("{e}"))
        };

        let mut ids = vec![BOS_TOKEN_ID]; // <|begin_of_text|>
        ids.push(128006); // <|start_header_id|>
        ids.extend(enc("system")?);
        ids.push(128007); // <|end_header_id|>
        ids.push(EOT_TOKEN_ID); // <|eot_id|>
        ids.push(128006); // <|start_header_id|>
        ids.extend(enc("assistant")?);
        ids.push(128007); // <|end_header_id|>

        // prefix_len = everything up to (but not including) voice/text content
        let prefix_len = ids.len();

        if let Some(vt) = voice_text {
            ids.extend(enc(vt)?);
        }

        ids.extend(enc(text)?);
        ids.push(EOT_TOKEN_ID);

        for _ in 0..self.cfg.shift_acoustic {
            ids.push(EOT_TOKEN_ID);
        }

        Ok((ids, prefix_len))
    }

    /// Start a new generation run.
    ///
    /// `voice_prompt` — optional pre-computed voice prompt (pass `None` for
    ///   zero-shot mode).
    /// `prefix_len`   — number of tokens before the prompt text starts
    ///   (BOS + header prefix).  Used to align acoustic features during the
    ///   voice prompt phase.  Pass 0 for zero-shot mode.
    /// `transition_steps` — number of acoustic frames to withhold from the
    ///   prompt tail and skip at the start of generated output (Python default
    ///   is 5 for voice-prompted mode, 0 for zero-shot).
    pub fn start_generation(
        &mut self,
        token_ids: &[u32],
        temperature: f32,
        noise_temp: f32,
        num_flow_steps: u32,
        voice_prompt: Option<VoicePrompt>,
        prefix_len: usize,
        transition_steps: usize,
        cfg_scale: f32,
    ) {
        let acoustic_dim = self.cfg.acoustic_dim;
        let shift_acoustic = self.cfg.shift_acoustic;
        let has_voice = voice_prompt.is_some();

        // In voice-prompted mode Python runs for exactly prompt_len steps.
        // In zero-shot mode we add extra autoregressive steps.
        let prompt_len = token_ids.len();
        let max_gen = 128;
        let total_tokens = if has_voice {
            prompt_len
        } else {
            prompt_len + max_gen
        };
        let max_cache_len = total_tokens + 64;

        // Compute voice prompt alignment parameters (mirror of tada_generate.rs).
        //
        // Python pads prompt_acoustic_features with `prefix_len` zeros at the front,
        // then truncates the last `num_transition_steps` entries:
        //   padded length = prefix_len + T
        //   after truncation = prefix_len + T - num_transition_steps
        //
        // During generation, step `s` reads voice index `s - shift_acoustic - prefix_len_py`.
        // prefix_len_py = prefix_len (excluding BOS from our count, matching Python).
        let num_transition_steps = if has_voice { transition_steps } else { 0 };
        // Python's prefix_len excludes BOS (prefix_len_py = prefix_len - 1 in tada_generate.rs).
        let prefix_len_py = prefix_len.saturating_sub(1);
        let effective_voice_len = voice_prompt
            .as_ref()
            .map(|vp| vp.len().saturating_sub(num_transition_steps))
            .unwrap_or(0);
        let prompt_phase_len = prefix_len_py + effective_voice_len;

        // Clear candle model state
        self.candle_model.clear_state();

        let cache = self.llama.create_cache(max_cache_len);

        self.gen_state = Some(GenState {
            acoustics: Vec::new(),
            times_before: Vec::new(),
            times_after: Vec::new(),
            token_ids: token_ids.to_vec(),
            next_token: token_ids[0],
            acoustic: vec![0.0; acoustic_dim],
            acoustic_mask: 0,
            time_before: 0,
            time_after: 0,
            step: 0,
            total_tokens,
            shift_acoustic,
            rng: WasmRng::new(),
            temperature,
            noise_temp,
            num_flow_steps: num_flow_steps as usize,
            is_done: false,
            eos_countdown: None,
            cache,
            voice_prompt,
            num_transition_steps,
            prefix_len_py,
            effective_voice_len,
            prompt_phase_len,
            cfg_scale,
        });
    }

    /// Run one generation step.
    ///
    /// Returns JSON progress string, or None if generation is complete.
    pub async fn generation_step(&mut self) -> anyhow::Result<Option<String>> {
        let state = match self.gen_state.as_mut() {
            Some(s) => s,
            None => return Ok(None),
        };

        if state.is_done {
            return Ok(None);
        }

        let step = state.step;
        let prompt_len = state.token_ids.len();

        // Get current token
        let current_token = if step < prompt_len {
            state.token_ids[step]
        } else {
            state.next_token
        };

        // --- GPU forward: Burn LLM ---
        let hidden_burn: Tensor<Wgpu, 3> = self.llama.forward_step(
            current_token,
            &state.acoustic,
            state.acoustic_mask,
            state.time_before,
            state.time_after,
            &mut state.cache,
        );

        // Determine if we need VibeVoice this step
        let need_vibevoice = step >= state.shift_acoustic
            && state.eos_countdown.is_none()
            && !(state.voice_prompt.is_some()
                && (step - state.shift_acoustic) < state.prompt_phase_len);

        // Only run VibeVoice when needed.
        // If BurnVibeVoice is loaded, flow matching runs entirely on GPU —
        // the hidden state never leaves the GPU until the final 512-float readback.
        // Otherwise, fall back to the candle CPU path (full ~2KB hidden readback).
        if need_vibevoice {
            let (acoustic_vec, time_before, time_after) = if let Some(ref burn_vv) = self.burn_vv {
                // GPU path: hidden stays on GPU, only result comes back
                model::vibevoice::solve_flow_matching_burn(
                    burn_vv,
                    hidden_burn.clone(),
                    state.noise_temp,
                    state.num_flow_steps,
                    state.cfg_scale,
                    &mut state.rng,
                )
                .await?
            } else {
                // CPU fallback: read hidden back, run candle VV
                let hidden_data = hidden_burn.clone().into_data_async().await
                    .map_err(|e| anyhow::anyhow!("GPU readback failed: {e:?}"))?;
                let hidden_vec: Vec<f32> = hidden_data.to_vec().unwrap();
                let hidden_size = self.cfg.llama.hidden_size;

                let hidden_candle = candle_core::Tensor::from_vec(
                    hidden_vec,
                    (1, 1, hidden_size),
                    &candle_core::Device::Cpu,
                )?;

                let (acoustic, time_before, time_after) = self.candle_model.generate_acoustic(
                    &hidden_candle,
                    state.noise_temp,
                    &mut state.rng,
                    state.num_flow_steps,
                    state.cfg_scale,
                )?;

                let av = acoustic.squeeze(0)?.to_vec1::<f32>()?;
                (av, time_before, time_after)
            };

            state.acoustics.push(acoustic_vec);
            state.times_before.push(time_before);
            state.times_after.push(time_after);
        } else if step >= state.shift_acoustic {
            // Prompt phase or EOS: push dummy values (will be stripped)
            state.acoustics.push(vec![0.0; self.cfg.acoustic_dim]);
            state.times_before.push(0);
            state.times_after.push(0);
        }

        // Sample next token after prompt
        let mut is_eos = false;
        let mut sampled_id = current_token;
        if step >= prompt_len - 1 {
            // Use Burn LLM's lm_head on GPU for logits
            let logits_burn = self.llama.lm_head(hidden_burn);
            let logits_data = logits_burn.into_data_async().await
                .map_err(|e| anyhow::anyhow!("GPU readback failed: {e:?}"))?;
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();

            // Sample with temperature + Gumbel-max
            let (token_id, eos) =
                sample_token(&logits_vec, state.temperature, &mut state.rng);
            sampled_id = token_id;
            is_eos = eos;
            state.next_token = sampled_id;
        }

        // Update acoustic/time for the next step input.
        //
        // Mirrors tada_generate.rs logic:
        //   - During prompt phase (voice-prompted mode): feed zeros or voice features
        //   - After prompt phase: autoregressive (use model's own last prediction)
        //
        // IMPORTANT: the update prepares the INPUT for step+1, so we check
        // whether *next* step's prompt_idx is still within the prompt phase.
        // Using the current step's prompt_idx caused the first AR step to
        // receive zeros/voice-features instead of the model's own prediction.
        if step >= state.shift_acoustic {
            let next_prompt_idx = (step + 1) - state.shift_acoustic;
            let has_voice = state.voice_prompt.is_some();

            if has_voice && next_prompt_idx < state.prompt_phase_len {
                // Next step is still in the prompt phase — prepare its input.
                if next_prompt_idx >= state.prefix_len_py
                    && next_prompt_idx < state.prefix_len_py + state.effective_voice_len
                {
                    // Next step is in the voice-feature region: look up step+1's features.
                    if let Some(vp) = state.voice_prompt.as_ref() {
                        let voice_step_offset = state.shift_acoustic + state.prefix_len_py;
                        if let Some((vp_acoustic, vp_mask, vp_tb, vp_ta)) =
                            vp.get_step(step + 1, voice_step_offset)
                        {
                            state.acoustic = vp_acoustic.clone();
                            state.acoustic_mask = vp_mask;
                            state.time_before = vp_tb;
                            state.time_after = vp_ta;
                        }
                    }
                } else {
                    // Next step is in the padding region: feed zeros, mask=0.
                    let acoustic_dim = self.cfg.acoustic_dim;
                    state.acoustic = vec![0.0; acoustic_dim];
                    state.acoustic_mask = 0;
                    state.time_before = 0;
                    state.time_after = 0;
                }
            } else if let Some(ac) = state.acoustics.last() {
                // Next step is autoregressive: use the model's own last prediction.
                state.acoustic = ac.clone();
                state.acoustic_mask = 1;
                state.time_before = *state.times_before.last().unwrap_or(&0);
                state.time_after = *state.times_after.last().unwrap_or(&0);
            }
        }

        // EOS countdown (only in zero-shot mode; voice mode runs for exactly prompt_len steps)
        if state.voice_prompt.is_none() && is_eos && state.eos_countdown.is_none() {
            state.eos_countdown = Some(state.shift_acoustic);
        }
        if let Some(ref mut countdown) = state.eos_countdown {
            if *countdown == 0 {
                state.is_done = true;
            } else {
                *countdown -= 1;
            }
        }

        state.step += 1;
        if state.step >= state.total_tokens {
            state.is_done = true;
        }

        let progress = format!(
            r#"{{"step":{},"total_tokens":{},"is_eos":{},"token_id":{}}}"#,
            step, state.total_tokens, is_eos, sampled_id,
        );

        Ok(Some(progress))
    }

    /// Decode accumulated acoustics to PCM audio (candle/CPU).
    ///
    /// When a voice prompt was used, strips the leading `prompt_phase_len +
    /// num_transition_steps - 1` acoustic frames before decoding — matching
    /// Python's:
    ///   `encoded = acoustic_features[..., num_prompt_tokens + num_transition_steps - 1:, :]`
    /// where `num_prompt_tokens = prompt_phase_len`.
    pub fn decode_audio(&self) -> anyhow::Result<Vec<f32>> {
        let state = self
            .gen_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no generation state"))?;

        // Strip leading prompt + transition frames (voice-prompted mode only).
        // Python: strip index = num_prompt_tokens + num_transition_steps - 1
        //       = prompt_phase_len + num_transition_steps - 1
        let strip_frames = if state.voice_prompt.is_some()
            && state.prompt_phase_len + state.num_transition_steps >= 1
        {
            state.prompt_phase_len + state.num_transition_steps - 1
        } else {
            0
        };

        // Build owned, mutable working copies so we can apply further trimming.
        let mut acoustics: Vec<Vec<f32>>;
        let mut times_before: Vec<u32>;

        if strip_frames > 0 && strip_frames < state.acoustics.len() {
            acoustics = state.acoustics[strip_frames..].to_vec();
            times_before = state.times_before[strip_frames..].to_vec();
        } else {
            acoustics = state.acoustics.clone();
            times_before = state.times_before.clone();
        }

        // In voice-prompted mode, trim the last acoustic frame.
        //
        // With 1 EOT + shift_acoustic trailing EOT tokens = shift_acoustic+1 trailing
        // token steps. Due to the shift_acoustic=5 offset, only the LAST trailing
        // step's acoustic is truly meaningless — it encodes the EOT token itself rather
        // than a shifted text token. Popping it prevents a junk frame from being decoded.
        if state.voice_prompt.is_some() && !acoustics.is_empty() {
            acoustics.pop();
            times_before.pop();
            eprintln!("[tada] Trimmed 1 trailing meaningless acoustic frame (voice-prompted mode)");
        }

        // Append a small sentinel trailing time so expand_durations has a boundary
        // for the last real speech frame. Use 1 rather than duplicating the last
        // frame's duration, to avoid adding unnecessary silence.
        times_before.push(1);

        // Clamp anomalous time_before values to prevent trailing noise.
        // Values like 59 or 196 cause expand_durations to insert many zero
        // frames that decode to noise ("dzouib"). Normal range is ~1-37.
        const MAX_TIME_BEFORE: u32 = 40;
        let clamped_times: Vec<u32> = times_before
            .iter()
            .enumerate()
            .map(|(i, &tb)| {
                if tb > MAX_TIME_BEFORE {
                    eprintln!("[tada] Clamped time_before[{i}] from {tb} to {MAX_TIME_BEFORE}");
                    MAX_TIME_BEFORE
                } else {
                    tb
                }
            })
            .collect();
        let times_before = &clamped_times[..];

        let samples = self
            .candle_model
            .decode_audio(&acoustics, times_before)?;

        Ok(samples)
    }

    pub fn cancel_generation(&mut self) {
        self.gen_state = None;
    }

    pub fn sample_rate(&self) -> usize {
        24000
    }
}

// ---------------------------------------------------------------------------
// Token sampling
// ---------------------------------------------------------------------------

/// Sample a token from logits using temperature + Gumbel-max trick.
fn sample_token(logits: &[f32], temperature: f32, rng: &mut WasmRng) -> (u32, bool) {
    use tada_model::Rng;

    let mut gumbel_logits = Vec::with_capacity(logits.len());
    for &l in logits {
        let scaled = if (temperature - 1.0).abs() > 1e-6 {
            l / temperature
        } else {
            l
        };

        let u: f32 = loop {
            let n = rng.sample_normal();
            let u = 1.0 / (1.0 + (-n).exp());
            if u > 0.0 && u < 1.0 {
                break u;
            }
        };
        let gumbel = -((-u.ln()).ln());
        gumbel_logits.push(scaled + gumbel);
    }

    let token_id = gumbel_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);

    let is_eos = token_id == EOS_TOKEN_ID || token_id == EOT_TOKEN_ID;

    (token_id, is_eos)
}

// ===========================================================================
// WASM bindings (feature = "wasm")
// ===========================================================================

#[cfg(feature = "wasm")]
pub mod web {
    use super::*;
    use std::sync::OnceLock;
    use wasm_bindgen::prelude::*;

    /// Device initialized by `initWgpuDevice()`.
    static WGPU_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

    fn wasm_log(msg: &str) {
        #[cfg(target_family = "wasm")]
        web_sys::console::log_1(&msg.into());
        #[cfg(not(target_family = "wasm"))]
        let _ = msg;
    }

    /// Initialize panic hook for better error messages in browser console.
    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
    }

    /// Initialize the WebGPU device asynchronously.
    ///
    /// **Must** be called (and awaited) before creating `TadaEngine`.
    #[wasm_bindgen(js_name = initWgpuDevice)]
    pub async fn init_wgpu_device() {
        use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuSetup};

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("No WebGPU adapter found");

        let info = adapter.get_info();
        let adapter_limits = adapter.limits();
        wasm_log(&format!(
            "[wgpu] Adapter: {} ({:?}), backend: {:?}",
            info.name, info.device_type, info.backend
        ));

        let features = adapter.features() - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tada-wgpu"),
                required_features: features,
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create WebGPU device");

        wasm_log(&format!(
            "[wgpu] Device created: max_compute_invocations={}",
            device.limits().max_compute_invocations_per_workgroup,
        ));

        let setup = WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: info.backend,
        };

        // Increase task batching: default is 32, but our LLM has ~50 dispatches per step.
        // Batching all ops into fewer GPU submissions reduces dispatch overhead significantly.
        let options = RuntimeOptions {
            tasks_max: 512,
            ..RuntimeOptions::default()
        };
        let wgpu_device = init_device(setup, options);
        WGPU_DEVICE.set(wgpu_device).ok();
    }

    // -----------------------------------------------------------------------
    // TadaEngine — WASM-facing wrapper around HybridTadaModel
    // -----------------------------------------------------------------------

    /// Browser-facing TADA TTS engine.
    ///
    /// Wraps the hybrid Burn+candle model with JS-friendly APIs.
    #[wasm_bindgen]
    pub struct TadaEngine {
        inner: Option<HybridTadaModel>,
        shard_bufs: Vec<Vec<u8>>,
    }

    #[wasm_bindgen]
    impl TadaEngine {
        /// Create a new empty engine. Call `appendModelShard` then `loadModel`.
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            console_error_panic_hook::set_once();
            Self {
                inner: None,
                shard_bufs: Vec::new(),
            }
        }

        /// Append a model weight shard (for multi-shard GGUF loading).
        #[wasm_bindgen(js_name = appendModelShard)]
        pub fn append_model_shard(&mut self, shard: &[u8]) {
            self.shard_bufs.push(shard.to_vec());
            wasm_log(&format!(
                "[tada] Shard appended ({} bytes, {} total shards)",
                shard.len(),
                self.shard_bufs.len()
            ));
        }

        /// Load the model from previously appended shards + tokenizer bytes.
        #[wasm_bindgen(js_name = loadModel)]
        pub fn load_model(&mut self, tokenizer_bytes: &[u8]) -> Result<(), JsError> {
            if self.shard_bufs.is_empty() {
                return Err(JsError::new("No shards appended. Call appendModelShard first."));
            }

            let device = WGPU_DEVICE
                .get()
                .cloned()
                .unwrap_or_else(WgpuDevice::default);

            wasm_log("[tada] Loading hybrid model...");

            let shards = std::mem::take(&mut self.shard_bufs);
            let model = HybridTadaModel::load(shards, tokenizer_bytes, &device)
                .map_err(|e| JsError::new(&format!("Model load failed: {e}")))?;

            self.inner = Some(model);
            wasm_log("[tada] Model loaded successfully");
            Ok(())
        }

        /// Run warmup passes to pre-compile GPU shader pipelines.
        /// Call after loadModel to avoid ~3s warmup on first generation.
        #[wasm_bindgen]
        pub async fn warmup(&mut self) -> Result<(), JsError> {
            let model = self.model_mut()?;
            wasm_log("[tada] GPU warmup: running dummy forward passes...");

            let acoustic = vec![0.0f32; model.cfg.acoustic_dim];
            let mut cache = model.llama.create_cache(16);

            // Run 5 forward passes to compile all WGSL shader variants
            for i in 0..5u32 {
                let hidden = model.llama.forward_step(
                    128000 + i, // dummy token
                    &acoustic,
                    0,
                    0,
                    0,
                    &mut cache,
                );
                // Force GPU execution by reading back (async)
                let _ = hidden.into_data_async().await;
            }

            // Reset cache but keep GPU buffer allocations
            cache.reset_keep_buffers();

            // Warm up VV Q8_0 shaders too (if BurnVibeVoice is loaded)
            if let Some(ref burn_vv) = model.burn_vv {
                wasm_log("[tada] GPU warmup: VV Q8_0 shaders...");
                let hidden_size = model.cfg.llama.hidden_size;
                let device = WGPU_DEVICE.get().cloned().unwrap_or_else(WgpuDevice::default);
                let dummy_hidden = burn::tensor::Tensor::<Wgpu, 3>::zeros(
                    [1, 1, hidden_size],
                    &device,
                );
                // Run one VV forward (2 ODE steps) to compile all Q8 matmul pipelines
                let result = model::vibevoice::solve_flow_matching_burn(
                    burn_vv,
                    dummy_hidden,
                    0.9,  // noise_temp
                    2,    // just 2 ODE steps for warmup (compiles same shaders as 10)
                    1.0,  // no CFG
                    &mut crate::WasmRng::new(),
                ).await;
                let _ = result; // ignore result, just wanted shader compilation
            }

            wasm_log("[tada] GPU warmup complete");
            Ok(())
        }

        fn model(&self) -> Result<&HybridTadaModel, JsError> {
            self.inner.as_ref().ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))
        }

        fn model_mut(&mut self) -> Result<&mut HybridTadaModel, JsError> {
            self.inner.as_mut().ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))
        }

        /// Tokenize text with Llama 3 chat template.
        /// Returns token IDs as Uint32Array.
        pub fn tokenize(&self, text: &str) -> Result<js_sys::Uint32Array, JsError> {
            let ids = self.model()?.tokenize(text)
                .map_err(|e| JsError::new(&e.to_string()))?;
            Ok(js_sys::Uint32Array::from(ids.as_slice()))
        }

        /// Tokenize text with an optional voice prompt text prepended.
        ///
        /// Returns a JS object `{ tokenIds: Uint32Array, prefixLen: number }`.
        /// `voiceText` should be the transcript of the voice prompt audio.
        /// Pass `null` or `undefined` for zero-shot mode (equivalent to `tokenize`).
        #[wasm_bindgen(js_name = tokenizeWithVoice)]
        pub fn tokenize_with_voice(
            &self,
            text: &str,
            voice_text: Option<String>,
        ) -> Result<JsValue, JsError> {
            let (ids, prefix_len) = self
                .model()?
                .tokenize_with_voice(text, voice_text.as_deref())
                .map_err(|e| JsError::new(&e.to_string()))?;

            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &"tokenIds".into(),
                &js_sys::Uint32Array::from(ids.as_slice()),
            )
            .unwrap();
            js_sys::Reflect::set(&obj, &"prefixLen".into(), &(prefix_len as u32).into()).unwrap();
            Ok(obj.into())
        }

        /// Start generation from token IDs.
        ///
        /// `voice_prompt_bytes` — optional raw safetensors bytes for the voice
        ///   prompt.  Pass an empty `Uint8Array` (or omit) for zero-shot mode.
        /// `prefix_len` — number of tokens before prompt text starts (BOS +
        ///   header prefix).  Pass 0 for zero-shot mode.
        /// `transition_steps` — transition gap size (Python default: 5 for
        ///   voice-prompted, 0 for zero-shot).
        #[wasm_bindgen(js_name = startGeneration)]
        pub fn start_generation(
            &mut self,
            token_ids: &[u32],
            temperature: f32,
            noise_temp: f32,
            num_flow_steps: u32,
            cfg_scale: f32,
            voice_prompt_bytes: Option<Vec<u8>>,
            prefix_len: u32,
            transition_steps: u32,
        ) -> Result<(), JsError> {
            let model = self.model_mut()?;
            let cfg = model.cfg.clone();

            // Load voice prompt if bytes were provided
            let voice_prompt = match voice_prompt_bytes {
                Some(bytes) if !bytes.is_empty() => {
                    let vp = VoicePrompt::load(
                        &bytes,
                        cfg.acoustic_dim,
                        cfg.num_time_classes as u32,
                    )
                    .map_err(|e| JsError::new(&format!("Failed to load voice prompt: {e}")))?;
                    Some(vp)
                }
                _ => None,
            };

            model.start_generation(
                token_ids,
                temperature,
                noise_temp,
                num_flow_steps,
                voice_prompt,
                prefix_len as usize,
                transition_steps as usize,
                cfg_scale,
            );
            Ok(())
        }

        /// Run one generation step.
        ///
        /// Returns JSON progress string or null if done:
        /// `{ "step": N, "total_tokens": M, "is_eos": bool, "token_id": N }`
        ///
        /// Returns a Promise (async for WebGPU mapAsync compatibility).
        #[wasm_bindgen(js_name = generationStep)]
        pub async fn generation_step(&mut self) -> Result<JsValue, JsError> {
            match self
                .model_mut()?
                .generation_step()
                .await
                .map_err(|e| JsError::new(&e.to_string()))?
            {
                Some(json) => Ok(JsValue::from_str(&json)),
                None => Ok(JsValue::NULL),
            }
        }

        /// Decode all accumulated acoustics to PCM audio.
        /// Returns Float32Array of 24kHz PCM samples.
        #[wasm_bindgen(js_name = decodeAudio)]
        pub fn decode_audio(&self) -> Result<js_sys::Float32Array, JsError> {
            let samples = self.model()?.decode_audio()
                .map_err(|e| JsError::new(&e.to_string()))?;
            Ok(js_sys::Float32Array::from(samples.as_slice()))
        }

        /// Get the sample rate (24000 Hz).
        #[wasm_bindgen(js_name = sampleRate)]
        pub fn sample_rate(&self) -> usize {
            24000
        }

        /// Cancel the current generation.
        #[wasm_bindgen(js_name = cancelGeneration)]
        pub fn cancel_generation(&mut self) {
            if let Some(ref mut model) = self.inner {
                model.cancel_generation();
            }
        }

        /// Check if the model is loaded and ready.
        #[wasm_bindgen(js_name = isReady)]
        pub fn is_ready(&self) -> bool {
            self.inner.is_some()
        }
    }
}
