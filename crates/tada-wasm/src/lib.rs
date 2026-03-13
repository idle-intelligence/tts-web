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

use model::{LayerCaches, TadaLlama};
use tada_core::config::TadaConfig;
use tada_core::tada_model::{self, BOS_TOKEN_ID, EOS_TOKEN_ID, EOT_TOKEN_ID};

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
}

// ---------------------------------------------------------------------------
// HybridTadaModel — Burn LLM + candle VibeVoice/decoder
// ---------------------------------------------------------------------------

/// The hybrid TADA model combining GPU LLM and CPU diffusion/decoder.
pub struct HybridTadaModel {
    /// Llama backbone on GPU (Burn/wgpu)
    llama: TadaLlama,
    /// VibeVoice + decoder on CPU (candle)
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

        // Load Llama backbone into Burn/GPU
        let cursor = gguf::ShardedCursor::new(gguf_bytes.clone());
        let mut reader = gguf::GgufReader::open(cursor)?;
        let llama = model::load_tada_llama_gguf(&mut reader, device)?;
        drop(reader);

        // Load VibeVoice + decoder into candle/CPU
        // Flatten shards for candle's GGUF loader
        let total_len: usize = gguf_bytes.iter().map(|s| s.len()).sum();
        let mut flat = Vec::with_capacity(total_len);
        for shard in &gguf_bytes {
            flat.extend_from_slice(shard);
        }
        let candle_model =
            tada_core::tada_model::TadaModel::load_gguf(&flat, &cfg, &candle_core::Device::Cpu)?;

        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        Ok(Self {
            llama,
            candle_model,
            cfg,
            gen_state: None,
            tokenizer,
        })
    }

    /// Tokenize text with Llama 3 chat template.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let mut ids = vec![BOS_TOKEN_ID]; // <|begin_of_text|>
        ids.push(128006); // <|start_header_id|>

        let assistant_enc = self
            .tokenizer
            .encode("assistant", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ids.extend(assistant_enc.get_ids());

        ids.push(128007); // <|end_header_id|>

        let text_enc = self
            .tokenizer
            .encode(format!("\n\n{text}"), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ids.extend(text_enc.get_ids());

        ids.push(EOT_TOKEN_ID); // <|eot_id|>

        // Add shift_acoustic EOS tokens
        for _ in 0..self.cfg.shift_acoustic {
            ids.push(EOS_TOKEN_ID);
        }

        Ok(ids)
    }

    /// Start a new generation run.
    pub fn start_generation(
        &mut self,
        token_ids: &[u32],
        temperature: f32,
        noise_temp: f32,
        num_flow_steps: u32,
    ) {
        let acoustic_dim = self.cfg.acoustic_dim;
        let max_gen = 128;
        let total_tokens = token_ids.len() + max_gen;
        let max_cache_len = total_tokens + 64;

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
            shift_acoustic: self.cfg.shift_acoustic,
            rng: WasmRng::new(),
            temperature,
            noise_temp,
            num_flow_steps: num_flow_steps as usize,
            is_done: false,
            eos_countdown: None,
            cache,
        });
    }

    /// Run one generation step.
    ///
    /// Returns JSON progress string, or None if generation is complete.
    pub fn generation_step(&mut self) -> anyhow::Result<Option<String>> {
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

        // Read hidden state back to CPU for candle
        let hidden_data = hidden_burn.clone().into_data();
        let hidden_vec: Vec<f32> = hidden_data.to_vec().unwrap();
        let hidden_size = self.cfg.llama.hidden_size;

        let hidden_candle = candle_core::Tensor::from_vec(
            hidden_vec,
            (1, 1, hidden_size),
            &candle_core::Device::Cpu,
        )?;

        // If past the acoustic shift, run flow matching (candle/CPU)
        if step >= state.shift_acoustic {
            let (acoustic, time_before, time_after) = self.candle_model.generate_acoustic(
                &hidden_candle,
                state.noise_temp,
                &mut state.rng,
                state.num_flow_steps,
            )?;

            state
                .acoustics
                .push(acoustic.squeeze(0)?.to_vec1::<f32>()?);
            state.times_before.push(time_before);
            state.times_after.push(time_after);
        }

        // Sample next token
        let mut is_eos = false;
        let mut sampled_id = current_token;
        if step >= prompt_len - 1 {
            // Use Burn LLM's lm_head on GPU for logits
            let logits_burn = self.llama.lm_head(hidden_burn);
            let logits_data = logits_burn.into_data();
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();

            // Sample with temperature + Gumbel-max
            let (token_id, eos) =
                sample_token(&logits_vec, state.temperature, &mut state.rng);
            sampled_id = token_id;
            is_eos = eos;
            state.next_token = sampled_id;

            // Update acoustic/time for next step
            if step >= state.shift_acoustic {
                if let Some(ac) = state.acoustics.last() {
                    state.acoustic = ac.clone();
                    state.acoustic_mask = 1;
                    state.time_before = *state.times_before.last().unwrap_or(&0);
                    state.time_after = *state.times_after.last().unwrap_or(&0);
                }
            }
        }

        // EOS countdown
        if is_eos && state.eos_countdown.is_none() {
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
    pub fn decode_audio(&self) -> anyhow::Result<Vec<f32>> {
        let state = self
            .gen_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no generation state"))?;

        let samples = self
            .candle_model
            .decode_audio(&state.acoustics, &state.times_before)?;

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

        let wgpu_device = init_device(setup, RuntimeOptions::default());
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

        /// Start generation from token IDs.
        #[wasm_bindgen(js_name = startGeneration)]
        pub fn start_generation(
            &mut self,
            token_ids: &[u32],
            temperature: f32,
            noise_temp: f32,
            num_flow_steps: u32,
        ) -> Result<(), JsError> {
            self.model_mut()?
                .start_generation(token_ids, temperature, noise_temp, num_flow_steps);
            Ok(())
        }

        /// Run one generation step.
        ///
        /// Returns JSON progress string or null if done:
        /// `{ "step": N, "total_tokens": M, "is_eos": bool, "token_id": N }`
        #[wasm_bindgen(js_name = generationStep)]
        pub fn generation_step(&mut self) -> Result<JsValue, JsError> {
            match self
                .model_mut()?
                .generation_step()
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
