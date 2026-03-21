//! TADA TTS orchestrator — ties together the Llama backbone, VibeVoice
//! diffusion head, and codec decoder into an end-to-end speech synthesizer.

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

use crate::config::TadaConfig;
use crate::decoder::{expand_durations, Decoder};
use crate::flow_matching::{decode_gray_code_to_time, solve_flow_matching};
use crate::llama::LlamaModel;
use crate::vibevoice::VibeVoiceDiffusionHead;
use mimi_rs::gguf_loader::GgufTensors;
use mimi_rs::qlinear::QLinear;

// ---------------------------------------------------------------------------
// Debug dump
// ---------------------------------------------------------------------------

/// Per-step intermediates captured during `generate_acoustic`.
pub struct AcousticDebugInfo {
    /// LLM hidden state (condition vector) passed to flow matching — [hidden_size].
    pub hidden_state: Vec<f32>,
    /// Initial noise sample passed into the flow matching ODE — [total_latent_dim].
    pub flow_input: Vec<f32>,
    /// Full flow matching output (acoustic + time bits) — [total_latent_dim].
    pub flow_output: Vec<f32>,
    /// Acoustic features extracted from flow_output — [acoustic_dim].
    pub acoustic: Vec<f32>,
    /// Raw time bit floats before thresholding — [time_dim].
    pub time_bits: Vec<f32>,
    /// Decoded time_before integer.
    pub time_before: u32,
    /// Decoded time_after integer.
    pub time_after: u32,
    /// Token ID that was the input at this step.
    pub token_id: u32,
    /// time_before value that was fed as input to this step.
    pub time_before_input: u32,
    /// time_after value that was fed as input to this step.
    pub time_after_input: u32,
}

// ---------------------------------------------------------------------------
// VoicePrompt
// ---------------------------------------------------------------------------

/// Pre-computed voice prompt loaded from a `.safetensors` file.
///
/// Contains per-token acoustic features and frame positions produced by the
/// TADA encoder for a reference audio clip.  During generation the loop feeds
/// these values into the model instead of its own predictions, steering the
/// synthesised voice towards the reference speaker.
///
/// ## File layout
/// The `.safetensors` file must contain three tensors:
/// - `token_values`    — `[T, acoustic_dim]` f32, normalised acoustic features
/// - `token_positions` — `[T]` i64, frame-centre positions (50 fps)
/// - `token_masks`     — `[T]` f32, 1.0 where the token has acoustic data
pub struct VoicePrompt {
    /// Per-token acoustic features, `[T][acoustic_dim]`.
    pub token_values: Vec<Vec<f32>>,
    /// Frame-centre positions for each token (50 fps clock).
    pub token_positions: Vec<i64>,
    /// Per-token mask (0.0 = no data, 1.0 = has data).
    pub token_masks: Vec<f32>,
    /// `time_len_before[i]` — gap in frames **before** token `i`.
    /// Derived from consecutive differences of `token_positions`.
    /// Length = T.
    pub time_len_before: Vec<u32>,
    /// `time_len_after[i]` — gap in frames **after** token `i`.
    /// Equals `time_gaps[i + 1]` (the gap that precedes token `i+1`).
    /// Length = T.
    pub time_len_after: Vec<u32>,
}

impl VoicePrompt {
    /// Load a voice prompt from raw safetensors bytes.
    ///
    /// `acoustic_dim` must match the model config (typically 512).
    /// `num_time_classes` is the upper bound for clamping time gaps
    /// (typically 1024, from `TadaConfig`).
    pub fn load(data: &[u8], acoustic_dim: usize, num_time_classes: u32) -> candle_core::Result<Self> {
        let tensors = candle_core::safetensors::load_buffer(data, &Device::Cpu)?;

        // ---- token_values [T, acoustic_dim] --------------------------------
        let tv = tensors
            .get("token_values")
            .ok_or_else(|| candle_core::Error::Msg("voice prompt missing 'token_values'".into()))?;
        let tv_shape = tv.dims().to_vec();
        if tv_shape.len() != 2 || tv_shape[1] != acoustic_dim {
            return Err(candle_core::Error::Msg(format!(
                "token_values shape {:?} does not match acoustic_dim={}",
                tv_shape, acoustic_dim
            )));
        }
        let t = tv_shape[0];
        let tv_flat = tv.flatten_all()?.to_vec1::<f32>()?;
        let token_values: Vec<Vec<f32>> = tv_flat
            .chunks(acoustic_dim)
            .map(|c| c.to_vec())
            .collect();

        // ---- token_positions [T] -------------------------------------------
        let tp = tensors
            .get("token_positions")
            .ok_or_else(|| candle_core::Error::Msg("voice prompt missing 'token_positions'".into()))?;
        // Accept both i64 and i32 to be robust.
        let token_positions: Vec<i64> = match tp.dtype() {
            candle_core::DType::I64 => tp.to_vec1::<i64>()?,
            candle_core::DType::I32 => tp.to_vec1::<i32>()?.into_iter().map(|x| x as i64).collect(),
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "token_positions has unexpected dtype {:?}",
                    other
                )))
            }
        };
        if token_positions.len() != t {
            return Err(candle_core::Error::Msg(format!(
                "token_positions length {} != token_values length {}",
                token_positions.len(),
                t
            )));
        }

        // ---- token_masks [T] -----------------------------------------------
        // The encoder's token_masks may be frame-level (length != T).
        // For voice prompts all T tokens have valid acoustic data, so if
        // the saved mask doesn't match T we default to all-ones.
        let token_masks: Vec<f32> = match tensors.get("token_masks") {
            Some(tm) => {
                let flat = match tm.dtype() {
                    candle_core::DType::F32 => tm.flatten_all()?.to_vec1::<f32>()?,
                    candle_core::DType::I64 => tm.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|x| x as f32).collect(),
                    _ => vec![1.0; t],
                };
                if flat.len() == t { flat } else { vec![1.0; t] }
            }
            None => vec![1.0; t],
        };

        // ---- Derive time gaps from positions --------------------------------
        //
        // Following the Python reference (`generate()` in tada.py):
        //
        //   selected_positions_with_ending = token_positions  (simplified)
        //   time_gaps = (pos[i] - pos[i-1]).clamp(0, num_time_classes-1)
        //   time_gaps = [0, time_gaps...]          (prepend a zero)
        //   time_len_before = time_gaps[:-1]       (length T)
        //   time_len_after  = time_gaps[1:]        (length T)
        //
        // We need length T+1 for both so that step `s` can look up index
        // `s - shift + 1`.  We store the full `time_gaps` vector (length T+1)
        // and expose slices as `time_len_before` / `time_len_after`.
        let max_tc = (num_time_classes as i64) - 1;
        let mut time_gaps: Vec<u32> = Vec::with_capacity(t + 1);
        time_gaps.push(0); // leading zero
        let mut prev = 1i64; // matches Python's `pad(…, value=1)` before diff
        for &pos in &token_positions {
            let gap = (pos - prev).clamp(0, max_tc) as u32;
            time_gaps.push(gap);
            prev = pos;
        }
        // time_len_before[i] = time_gaps[i]   (indices 0..T)
        // time_len_after[i]  = time_gaps[i+1] (indices 0..T)
        let time_len_before = time_gaps[..t].to_vec();  // length T
        let time_len_after  = time_gaps[1..].to_vec();  // length T

        Ok(Self {
            token_values,
            token_positions,
            token_masks,
            time_len_before,
            time_len_after,
        })
    }

    /// Number of voice-prompt tokens.
    pub fn len(&self) -> usize {
        self.token_values.len()
    }

    /// True if the voice prompt has no tokens.
    pub fn is_empty(&self) -> bool {
        self.token_values.is_empty()
    }

    /// Return the acoustic features, mask, and time values to feed at a given
    /// generation step, or `None` if this step is beyond the voice prompt.
    ///
    /// `step` is the global step index (same counter used in the generation
    /// loop).  `shift_acoustic` is the model's `shift_acoustic` offset
    /// (typically 5).
    ///
    /// Returns `(acoustic_vec, acoustic_mask, time_before, time_after)`.
    pub fn get_step(
        &self,
        step: usize,
        shift_acoustic: usize,
    ) -> Option<(&Vec<f32>, u32, u32, u32)> {
        if step < shift_acoustic {
            return None;
        }
        let idx = step - shift_acoustic; // index into token_values / token_masks
        if idx >= self.len() {
            return None;
        }
        let acoustic = &self.token_values[idx];
        let mask = self.token_masks[idx] as u32; // 0 or 1
        // time indices: before uses idx, after uses idx (== time_gaps[idx+1])
        // Both slices have length T so guard against out-of-bounds.
        let tb = self.time_len_before.get(idx).copied().unwrap_or(0);
        let ta = self.time_len_after.get(idx).copied().unwrap_or(0);
        Some((acoustic, mask, tb, ta))
    }
}

// ---------------------------------------------------------------------------
// Special token IDs
// ---------------------------------------------------------------------------

pub const BOS_TOKEN_ID: u32 = 128000;
pub const EOS_TOKEN_ID: u32 = 128001;
pub const EOT_TOKEN_ID: u32 = 128009;

// ---------------------------------------------------------------------------
// Rng trait (WASM-compatible)
// ---------------------------------------------------------------------------

/// Minimal RNG trait for generating standard normal samples.
///
/// Decoupled from `rand` so that WASM callers can provide their own
/// source of randomness (e.g. `Math.random()` via JS).
pub trait Rng {
    fn sample_normal(&mut self) -> f32;
}

// ---------------------------------------------------------------------------
// TadaModel
// ---------------------------------------------------------------------------

/// End-to-end TADA TTS model.
///
/// Combines:
/// - A Llama 3.2 1B backbone (autoregressive text + acoustic token generation)
/// - A VibeVoice diffusion head (flow-matching acoustic prediction)
/// - A DAC-style codec decoder (acoustic latents → 24 kHz PCM)
///
/// Plus small adapter layers that project acoustic features and time encodings
/// into the LLM's hidden space.
pub struct TadaModel {
    llama: LlamaModel,
    prediction_head: VibeVoiceDiffusionHead,
    decoder: Decoder,
    acoustic_proj: QLinear,        // [acoustic_dim → hidden_size] with bias
    acoustic_mask_emb: Embedding,  // 2 embeddings of hidden_size
    time_start_embed: Embedding,   // num_time_classes embeddings
    time_end_embed: Embedding,     // num_time_classes embeddings
    cfg: TadaConfig,
    device: Device,
    position: usize, // current position in sequence (for KV cache)
}

impl TadaModel {
    /// Load a complete TADA model from GGUF bytes.
    pub fn load_gguf(data: &[u8], cfg: &TadaConfig, device: &Device) -> Result<Self> {
        let mut gguf = GgufTensors::from_bytes(data, device)?;

        let llama = LlamaModel::load_gguf(&mut gguf, &cfg.llama)?;
        let prediction_head = VibeVoiceDiffusionHead::load_gguf(&mut gguf, cfg)?;
        let decoder = Decoder::load_gguf(&mut gguf, "_decoder", &cfg.decoder)?;

        // TADA-specific adapter layers
        let acoustic_proj = gguf.qlinear("acoustic_proj")?;

        let acoustic_mask_w = gguf.tensor("acoustic_mask_emb.weight")?;
        let acoustic_mask_emb = Embedding::new(acoustic_mask_w, cfg.llama.hidden_size);

        let time_start_w = gguf.tensor("time_start_embed.weight")?;
        let time_start_embed = Embedding::new(time_start_w, cfg.llama.hidden_size);

        let time_end_w = gguf.tensor("time_end_embed.weight")?;
        let time_end_embed = Embedding::new(time_end_w, cfg.llama.hidden_size);

        Ok(Self {
            llama,
            prediction_head,
            decoder,
            acoustic_proj,
            acoustic_mask_emb,
            time_start_embed,
            time_end_embed,
            cfg: cfg.clone(),
            device: device.clone(),
            position: 0,
        })
    }

    /// Load only VibeVoice + decoder + adapters from GGUF (skip LLM).
    /// Used in hybrid mode where Burn handles the LLM on GPU.
    pub fn load_gguf_no_llm(data: &[u8], cfg: &TadaConfig, device: &Device) -> Result<Self> {
        Self::load_gguf_no_llm_inner(data, cfg, device, true)
    }

    /// Load without LLM and without VV (decoder + adapters only).
    /// Used when Burn handles both LLM and VV on GPU — saves ~400MB WASM memory.
    pub fn load_gguf_no_llm_no_vv(data: &[u8], cfg: &TadaConfig, device: &Device) -> Result<Self> {
        Self::load_gguf_no_llm_inner(data, cfg, device, false)
    }

    fn load_gguf_no_llm_inner(data: &[u8], cfg: &TadaConfig, device: &Device, load_vv: bool) -> Result<Self> {
        let mut gguf = GgufTensors::from_bytes(data, device)?;

        // Only load VV if not handled by Burn GPU
        let prediction_head = if load_vv {
            VibeVoiceDiffusionHead::load_gguf(&mut gguf, cfg)?
        } else {
            // Dummy — Burn handles VV on GPU. We still need the struct but it won't be called.
            // Load with zero-size layers to avoid allocating real weights.
            VibeVoiceDiffusionHead::load_gguf(&mut gguf, &TadaConfig { head_layers: 0, ..cfg.clone() })?
        };
        let decoder = Decoder::load_gguf(&mut gguf, "_decoder", &cfg.decoder)?;

        let acoustic_proj = gguf.qlinear("acoustic_proj")?;
        let acoustic_mask_w = gguf.tensor("acoustic_mask_emb.weight")?;
        let acoustic_mask_emb = Embedding::new(acoustic_mask_w, cfg.llama.hidden_size);
        let time_start_w = gguf.tensor("time_start_embed.weight")?;
        let time_start_embed = Embedding::new(time_start_w, cfg.llama.hidden_size);
        let time_end_w = gguf.tensor("time_end_embed.weight")?;
        let time_end_embed = Embedding::new(time_end_w, cfg.llama.hidden_size);

        // Skip LlamaModel — create a minimal placeholder.
        // In hybrid mode, Burn handles the LLM. The llama field is only used
        // for embed_tokens/lm_head/forward_step which are handled by Burn.
        // We still need the struct to satisfy the type, but we skip the expensive
        // loading of 16 transformer layers + KV caches.
        //
        // NOTE: Calling forward_step/embed_tokens on this dummy will panic.
        // Only generate_acoustic/decode_audio are safe to call.
        let llama = LlamaModel::load_gguf_adapters_only(&mut gguf, &cfg.llama)?;

        Ok(Self {
            llama,
            prediction_head,
            decoder,
            acoustic_proj,
            acoustic_mask_emb,
            time_start_embed,
            time_end_embed,
            cfg: cfg.clone(),
            device: device.clone(),
            position: 0,
        })
    }

    /// Output sample rate in Hz.
    pub fn sample_rate() -> usize {
        24000
    }

    /// The device tensors live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reset KV cache and position counter (call between sequences).
    pub fn clear_state(&mut self) {
        self.llama.clear_kv_cache();
        self.position = 0;
    }

    /// Build the combined input embedding for one autoregressive step.
    ///
    /// All inputs are `[1, 1, ...]` (batch=1, seq_len=1) for step-by-step
    /// generation:
    ///
    /// - `token_ids`: `[1, 1]` u32 — the text token
    /// - `acoustic`: `[1, 1, acoustic_dim]` — acoustic features (or zeros)
    /// - `acoustic_mask`: `[1, 1]` u32 — 0 if no acoustic, 1 if yes
    /// - `time_before`: `[1, 1]` u32 — duration from previous step (or 0)
    /// - `time_after`: `[1, 1]` u32 — duration from previous step (or 0)
    ///
    /// Returns `[1, 1, hidden_size]`.
    pub fn build_input_embeds(
        &self,
        token_ids: &Tensor,
        acoustic: &Tensor,
        acoustic_mask: &Tensor,
        time_before: &Tensor,
        time_after: &Tensor,
    ) -> Result<Tensor> {
        // Token embedding: [1, 1, hidden_size]
        let tok_emb = self.llama.embed_tokens(token_ids)?;

        // Acoustic projection: [1, 1, acoustic_dim] → [1, 1, hidden_size]
        let acou_emb = self.acoustic_proj.forward(acoustic)?;

        // Acoustic mask embedding: [1, 1] u32 → [1, 1, hidden_size]
        let mask_emb = self.acoustic_mask_emb.forward(acoustic_mask)?;

        // Time embeddings: [1, 1] u32 → [1, 1, hidden_size]
        let time_start_emb = self.time_start_embed.forward(time_before)?;
        let time_end_emb = self.time_end_embed.forward(time_after)?;

        // Sum all embeddings
        let embed = (tok_emb + acou_emb)?;
        let embed = (embed + mask_emb)?;
        let embed = (embed + time_start_emb)?;
        let embed = (embed + time_end_emb)?;

        Ok(embed)
    }

    /// Run one autoregressive step through the Llama backbone.
    ///
    /// - `input_embeds`: `[1, 1, hidden_size]` from `build_input_embeds`
    ///
    /// Returns hidden states `[1, 1, hidden_size]`.
    pub fn forward_step(&mut self, input_embeds: &Tensor) -> Result<Tensor> {
        let hidden = self.llama.forward(input_embeds, self.position)?;
        let (_b, seq_len, _h) = input_embeds.dims3()?;
        self.position += seq_len;
        Ok(hidden)
    }

    /// Run one step with layer-0 attention debug dumps.
    ///
    /// Same as `forward_step` but prints intermediate tensors for layer 0's attention.
    /// Use on step 2 to compare with Burn's `forward_with_cache_debug`.
    pub fn forward_step_debug_layer0(&mut self, input_embeds: &Tensor) -> Result<Tensor> {
        let hidden = self.llama.forward_debug_layer0(input_embeds, self.position)?;
        let (_b, seq_len, _h) = input_embeds.dims3()?;
        self.position += seq_len;
        Ok(hidden)
    }

    /// Run only the first `num_layers` transformer layers for per-layer debug comparison.
    ///
    /// Does NOT increment the position counter — this is for debug comparison only.
    /// Call `clear_state()` between runs to reset caches.
    pub fn forward_step_n_layers(&mut self, input_embeds: &Tensor, num_layers: usize) -> Result<Tensor> {
        self.llama.forward_n_layers(input_embeds, self.position, num_layers)
    }

    /// Undo the last forward_step: decrement position and pop last KV cache entry.
    /// Used for CFG negative condition: run LLM with zero acoustics, capture hidden,
    /// then undo so the real forward step's KV cache is correct.
    pub fn undo_last_step(&mut self) {
        self.position = self.position.saturating_sub(1);
        self.llama.pop_kv_cache();
    }

    /// Compute logits from hidden states (for debugging).
    pub fn lm_head_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.llama.lm_head(hidden)
    }

    /// Generate acoustic features and time values from the LLM hidden state
    /// using flow-matching diffusion.
    ///
    /// - `hidden`: `[1, 1, hidden_size]` from `forward_step`
    /// - `noise_temp`: noise temperature for the initial sample
    /// - `rng`: WASM-compatible RNG source
    /// - `num_steps`: number of Euler ODE steps (e.g. 32)
    /// - `acoustic_cfg_scale`: CFG scale for acoustic features (1.0 = disabled)
    ///
    /// Returns `(acoustic, time_before, time_after)` where:
    /// - `acoustic`: `[1, acoustic_dim]`
    /// - `time_before`, `time_after`: scalar u32 duration values
    pub fn generate_acoustic(
        &self,
        hidden: &Tensor,
        neg_hidden: Option<&Tensor>,
        noise_temp: f32,
        rng: &mut dyn Rng,
        num_steps: usize,
        acoustic_cfg_scale: f32,
    ) -> Result<(Tensor, u32, u32)> {
        let (acoustic, time_before, time_after, _debug) =
            self.generate_acoustic_debug(hidden, neg_hidden, noise_temp, rng, num_steps, acoustic_cfg_scale, false)?;
        Ok((acoustic, time_before, time_after))
    }

    /// Like `generate_acoustic` but optionally captures intermediate tensors
    /// for debugging.  When `capture_debug` is true the returned
    /// `Option<AcousticDebugInfo>` is `Some`; otherwise it is `None`.
    ///
    /// The extra scalar inputs `token_id`, `time_before_input`, and
    /// `time_after_input` are not computed here — the caller must fill them in
    /// after the call (they are initialised to 0 inside the returned struct).
    pub fn generate_acoustic_debug(
        &self,
        hidden: &Tensor,
        neg_hidden: Option<&Tensor>,
        noise_temp: f32,
        rng: &mut dyn Rng,
        num_steps: usize,
        acoustic_cfg_scale: f32,
        capture_debug: bool,
    ) -> Result<(Tensor, u32, u32, Option<AcousticDebugInfo>)> {
        // Squeeze hidden from [1, 1, hidden_size] → [1, hidden_size]
        let cond = hidden.squeeze(1)?;
        let neg_cond = neg_hidden.map(|h| h.squeeze(1)).transpose()?;

        // Sample noise: [1, total_latent_dim]
        let total_dim = self.cfg.total_latent_dim();
        let noise_data: Vec<f32> = (0..total_dim)
            .map(|_| rng.sample_normal() * noise_temp)
            .collect();
        let noise = Tensor::from_vec(noise_data.clone(), (1, total_dim), hidden.device())?;

        // Flow matching ODE solve
        let result = solve_flow_matching(
            &noise,
            &cond,
            neg_cond.as_ref(),
            &self.prediction_head,
            num_steps,
            "logsnr",
            acoustic_cfg_scale,
            self.cfg.acoustic_dim,
        )?;

        // Split result into acoustic [1, acoustic_dim] and time gray bits [1, time_dim]
        let acoustic_dim = self.cfg.acoustic_dim;
        let time_dim = self.cfg.time_dim();
        let num_bits = self.cfg.num_time_bits();

        let acoustic = result.narrow(1, 0, acoustic_dim)?;
        let time_bits = result.narrow(1, acoustic_dim, time_dim)?;

        // Time bits are interleaved: [time_before_bits..., time_after_bits...]
        // Each half has num_bits elements
        let time_before_bits = time_bits.narrow(1, 0, num_bits)?;
        let time_after_bits = time_bits.narrow(1, num_bits, num_bits)?;

        // Decode gray code bits → integer time values
        let time_before_t = decode_gray_code_to_time(&time_before_bits, num_bits)?;
        let time_after_t = decode_gray_code_to_time(&time_after_bits, num_bits)?;

        let time_before = time_before_t.to_vec1::<u32>()?[0];
        let time_after = time_after_t.to_vec1::<u32>()?[0];

        let debug_info = if capture_debug {
            let hidden_state = cond.flatten_all()?.to_vec1::<f32>()?;
            let flow_input = noise_data; // already Vec<f32>
            let flow_output = result.flatten_all()?.to_vec1::<f32>()?;
            let acoustic_vec = acoustic.flatten_all()?.to_vec1::<f32>()?;
            let time_bits_vec = time_bits.flatten_all()?.to_vec1::<f32>()?;
            Some(AcousticDebugInfo {
                hidden_state,
                flow_input,
                flow_output,
                acoustic: acoustic_vec,
                time_bits: time_bits_vec,
                time_before,
                time_after,
                // Caller fills these in after the call
                token_id: 0,
                time_before_input: 0,
                time_after_input: 0,
            })
        } else {
            None
        };

        Ok((acoustic, time_before, time_after, debug_info))
    }

    /// Sample the next text token from the LLM hidden state.
    ///
    /// - `hidden`: `[1, 1, hidden_size]` from `forward_step`
    /// - `temperature`: sampling temperature (1.0 = no scaling)
    /// - `rng`: RNG for stochastic sampling (Gumbel-max trick)
    ///
    /// Returns `(token_id, is_eos)` where `is_eos` is true if the token
    /// is EOS (128001) or EOT (128009).
    pub fn sample_next_token(
        &self,
        hidden: &Tensor,
        temperature: f32,
        rng: &mut dyn Rng,
    ) -> Result<(u32, bool)> {
        // Compute logits: [1, 1, vocab_size]
        let logits = self.llama.lm_head(hidden)?;

        // Take last position: [1, vocab_size]
        let logits = logits.i((.., logits.dim(1)? - 1, ..))?;

        // Temperature scaling
        let logits = if (temperature - 1.0).abs() > 1e-6 {
            (logits / temperature as f64)?
        } else {
            logits
        };

        // Gumbel-max trick: argmax(logits + gumbel_noise) = multinomial sample
        // gumbel_noise = -log(-log(uniform(0,1)))
        // We use the normal RNG to generate uniform samples via the CDF
        let logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;
        let vocab_size = logits_vec.len();
        let mut gumbel_logits = Vec::with_capacity(vocab_size);
        for &l in &logits_vec {
            // Convert normal sample to uniform via probit function approximation
            let u: f32 = loop {
                // Use normal sample → clamp to (0,1) range via sigmoid
                let n = rng.sample_normal();
                let u = 1.0 / (1.0 + (-n).exp());
                if u > 0.0 && u < 1.0 {
                    break u;
                }
            };
            let gumbel = -((-u.ln()).ln());
            gumbel_logits.push(l + gumbel);
        }

        let token_id = gumbel_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        let is_eos = token_id == EOS_TOKEN_ID || token_id == EOT_TOKEN_ID;

        Ok((token_id, is_eos))
    }

    /// Decode collected acoustic features and durations into PCM audio.
    ///
    /// - `acoustics`: slice of acoustic vectors, each of length `acoustic_dim`
    /// - `times_before`: slice of duration values. Length can be `N` (same as
    ///   acoustics) or `N+1` (trailing element adds trailing zeros, matching
    ///   Python's `_decode_wav`).
    ///
    /// Returns flattened PCM audio samples at 24 kHz.
    pub fn decode_audio(
        &self,
        acoustics: &[Vec<f32>],
        times_before: &[u32],
    ) -> Result<Vec<f32>> {
        let n = acoustics.len();
        let acoustic_dim = self.cfg.acoustic_dim;
        let device = &self.device;

        // Build acoustic tensor [N, acoustic_dim]
        let flat: Vec<f32> = acoustics.iter().flat_map(|v| v.iter().copied()).collect();
        let acoustic_tensor = Tensor::from_vec(flat, (n, acoustic_dim), device)?;

        // Denormalize: acoustic = acoustic * std + mean
        let acoustic_tensor = (acoustic_tensor * self.cfg.acoustic_std)?;
        let acoustic_tensor = (acoustic_tensor + self.cfg.acoustic_mean)?;

        // Expand durations → [1, total_frames, embed_dim] + [1, total_frames]
        let (expanded, token_masks) =
            expand_durations(&acoustic_tensor, times_before, device)?;

        // Decode to PCM: [1, 1, samples]
        let pcm = self.decoder.forward(&expanded, &token_masks)?;

        // Flatten to Vec<f32>
        let mut samples = pcm.flatten_all()?.to_vec1::<f32>()?;

        // Remove leading silence (matching Python):
        //   wav = wav[..., int(24000 * time_before[0] / 50) :]
        let leading_silence = if !times_before.is_empty() {
            (24000.0 * times_before[0] as f64 / 50.0) as usize
        } else {
            0
        };
        if leading_silence > 0 && leading_silence < samples.len() {
            samples.drain(..leading_silence);
        }

        Ok(samples)
    }
}
