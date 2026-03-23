use candle_core::{Result, Tensor, D};
use candle_nn::{Module, RmsNorm};
use mimi_rs::gguf_loader::GgufTensors;
use mimi_rs::qlinear::QLinear;

use crate::config::TadaConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// AdaLN modulation: x * (1 + scale) + shift.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one_plus_scale = (scale + 1.0f64)?;
    x.broadcast_mul(&one_plus_scale)?.broadcast_add(shift)
}

/// RMSNorm without learnable affine parameters.
/// Normalizes by sqrt(mean(x^2) + eps).
fn rms_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(variance + eps)?.sqrt()?)
}

// ---------------------------------------------------------------------------
// TimestepEmbedder
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding followed by a 2-layer MLP (no bias).
///
/// `t` (scalar per sample) → sinusoidal frequencies → Linear → SiLU → Linear.
pub struct TimestepEmbedder {
    mlp_0: QLinear,
    mlp_2: QLinear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        frequency_embedding_size: usize,
    ) -> Result<Self> {
        let mlp_0 = gguf.qlinear(&format!("{prefix}.mlp.0"))?;
        let mlp_2 = gguf.qlinear(&format!("{prefix}.mlp.2"))?;
        Ok(Self { mlp_0, mlp_2, frequency_embedding_size })
    }

    /// Compute sinusoidal positional embedding for timesteps.
    ///
    /// `t` has shape `[B]` or `[B, 1]` — scalar timestep per sample.
    /// Returns shape `[B, frequency_embedding_size]`.
    fn timestep_embedding(&self, t: &Tensor) -> Result<Tensor> {
        // Flatten to [B] if [B, 1]
        let t = if t.rank() == 2 { t.squeeze(D::Minus1)? } else { t.clone() };

        let half = self.frequency_embedding_size / 2;
        let device = t.device();
        let dtype = t.dtype();

        // freqs = exp(-ln(10000) * arange(0, half) / half)
        let log_10000 = (10000.0f64).ln();
        let scale = (-log_10000 / half as f64) as f32;
        let arange: Vec<f32> = (0..half).map(|i| i as f32 * scale).collect();
        let freqs = Tensor::from_vec(arange, (half,), device)?.exp()?;
        let freqs = freqs.to_dtype(dtype)?;

        // args = t[:, None] * freqs[None, :]  →  [B, half]
        let t_unsqueezed = t.unsqueeze(D::Minus1)?;
        let args = t_unsqueezed.broadcast_mul(&freqs.unsqueeze(0)?)?;

        // embedding = cat([cos(args), sin(args)], dim=-1)  →  [B, frequency_embedding_size]
        Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_freq = self.timestep_embedding(t)?;
        let x = self.mlp_0.forward(&t_freq)?;
        let x = candle_nn::ops::silu(&x)?;
        self.mlp_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// FeedForwardNetwork (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward network: SiLU(gate_proj(x)) * up_proj(x) → down_proj.
pub struct FeedForwardNetwork {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl FeedForwardNetwork {
    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str) -> Result<Self> {
        let gate_proj = gguf.qlinear(&format!("{prefix}.gate_proj"))?;
        let up_proj = gguf.qlinear(&format!("{prefix}.up_proj"))?;
        let down_proj = gguf.qlinear(&format!("{prefix}.down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// HeadLayer
// ---------------------------------------------------------------------------

/// One diffusion transformer block with AdaLN modulation.
///
/// The AdaLN modulation produces 3 × hidden_size outputs (shift, scale, gate)
/// that condition the FFN on the combined timestep + condition embedding.
pub struct HeadLayer {
    ffn: FeedForwardNetwork,
    norm: RmsNorm,
    ada_ln_modulation: QLinear,
}

impl HeadLayer {
    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, rms_norm_eps: f64) -> Result<Self> {
        let ffn = FeedForwardNetwork::load_gguf(gguf, &format!("{prefix}.ffn"))?;
        let norm_weight = gguf.tensor(&format!("{prefix}.norm.weight"))?;
        let norm = RmsNorm::new(norm_weight, rms_norm_eps);
        let ada_ln_modulation = gguf.qlinear(&format!("{prefix}.adaLN_modulation.1"))?;
        Ok(Self { ffn, norm, ada_ln_modulation })
    }

    /// Forward pass for one transformer block.
    ///
    /// `c_silu` is `silu(condition + timestep_embedding)`, pre-computed once
    /// and shared across all layers.
    pub fn forward(&self, x: &Tensor, c_silu: &Tensor) -> Result<Tensor> {
        let hidden_size = x.dim(D::Minus1)?;
        let ada = self.ada_ln_modulation.forward(c_silu)?;

        let shift = ada.narrow(D::Minus1, 0, hidden_size)?.contiguous()?;
        let scale = ada.narrow(D::Minus1, hidden_size, hidden_size)?.contiguous()?;
        let gate = ada.narrow(D::Minus1, 2 * hidden_size, hidden_size)?.contiguous()?;

        // x = x + gate * ffn(modulate(norm(x), shift, scale))
        let normed = self.norm.forward(x)?;
        let modulated = modulate(&normed, &shift, &scale)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        x.broadcast_add(&gate.broadcast_mul(&ffn_out)?)
    }
}

// ---------------------------------------------------------------------------
// FinalLayer
// ---------------------------------------------------------------------------

/// Final output layer with AdaLN (no elementwise_affine on the norm).
///
/// Uses an affine-free RMSNorm, then modulates with (shift, scale) from the
/// combined condition, then projects to the output latent dimension.
pub struct FinalLayer {
    norm_eps: f64,
    linear: QLinear,
    ada_ln_modulation: QLinear,
}

impl FinalLayer {
    pub fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let linear = gguf.qlinear(&format!("{prefix}.linear"))?;
        let ada_ln_modulation = gguf.qlinear(&format!("{prefix}.adaLN_modulation.1"))?;
        Ok(Self { norm_eps: rms_norm_eps, linear, ada_ln_modulation })
    }

    /// Forward pass: affine-free RMSNorm → AdaLN modulate → linear projection.
    pub fn forward(&self, x: &Tensor, c_silu: &Tensor) -> Result<Tensor> {
        let hidden_size = x.dim(D::Minus1)?;
        let ada = self.ada_ln_modulation.forward(c_silu)?;

        let shift = ada.narrow(D::Minus1, 0, hidden_size)?.contiguous()?;
        let scale = ada.narrow(D::Minus1, hidden_size, hidden_size)?.contiguous()?;

        let x = rms_norm_no_affine(x, self.norm_eps)?;
        let x = modulate(&x, &shift, &scale)?;
        self.linear.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// VibeVoiceDiffusionHead
// ---------------------------------------------------------------------------

/// VibeVoice diffusion prediction head for TADA.
///
/// A transformer-like architecture that predicts velocity for flow matching.
/// Takes noisy acoustic latents, scalar timesteps, and LLM conditioning as
/// input and outputs predicted velocity in acoustic latent space.
pub struct VibeVoiceDiffusionHead {
    noisy_images_proj: QLinear,
    cond_proj: QLinear,
    t_embedder: TimestepEmbedder,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
}

impl VibeVoiceDiffusionHead {
    /// Load from a GGUF tensor store.
    ///
    /// Expects tensors prefixed with `prediction_head.`.
    pub fn load_gguf(gguf: &mut GgufTensors, cfg: &TadaConfig) -> Result<Self> {
        let prefix = "prediction_head";
        let rms_norm_eps = cfg.llama.rms_norm_eps;

        let noisy_images_proj = gguf.qlinear(&format!("{prefix}.noisy_images_proj"))?;
        let cond_proj = gguf.qlinear(&format!("{prefix}.cond_proj"))?;
        let t_embedder = TimestepEmbedder::load_gguf(
            gguf,
            &format!("{prefix}.t_embedder"),
            256,
        )?;

        let mut layers = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            layers.push(HeadLayer::load_gguf(
                gguf,
                &format!("{prefix}.layers.{i}"),
                rms_norm_eps,
            )?);
        }

        let final_layer = FinalLayer::load_gguf(
            gguf,
            &format!("{prefix}.final_layer"),
            rms_norm_eps,
        )?;

        Ok(Self { noisy_images_proj, cond_proj, t_embedder, layers, final_layer })
    }

    /// Predict velocity for flow matching.
    ///
    /// # Arguments
    /// - `noisy_images` — noisy acoustic latent, shape `[B, latent_size]` (528)
    /// - `timesteps` — scalar timesteps, shape `[B]` or `[B, 1]`
    /// - `condition` — conditioning from LLM backbone, shape `[B, hidden_size]` (2048)
    ///
    /// # Returns
    /// Predicted velocity, shape `[B, latent_size]` (528).
    pub fn forward(
        &self,
        noisy_images: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        // 1. Project noisy latent to hidden dimension
        let mut x = self.noisy_images_proj.forward(noisy_images)?;

        // 2. Timestep embedding
        let t = self.t_embedder.forward(timesteps)?;

        // 3. Project condition
        let condition = self.cond_proj.forward(condition)?;

        // 4. Combine condition + timestep
        let c = (&condition + &t)?;
        let c_silu = candle_nn::ops::silu(&c)?;

        // 5. Transformer layers with AdaLN
        for layer in &self.layers {
            x = layer.forward(&x, &c_silu)?;
        }

        // 6. Final layer
        self.final_layer.forward(&x, &c_silu)
    }
}
