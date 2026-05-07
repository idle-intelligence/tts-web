//! Burn/wgpu port of the VibeVoice diffusion head + flow matching ODE solver.
//!
//! Mirrors the candle implementation in `tada-core/src/vibevoice.rs` and
//! `tada-core/src/flow_matching.rs`, but keeps all tensors on GPU throughout
//! the ODE loop. Only 512 acoustic floats + 2 u32 time values are read back
//! to CPU after the final step.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::silu;
use burn::tensor::Tensor;

use super::F32Linear;
use crate::gguf::Q8Linear;

// ---------------------------------------------------------------------------
// VVLinear — wraps either Q8Linear or F32Linear for VibeVoice layers
// ---------------------------------------------------------------------------

pub enum VVLinear {
    Q8(Q8Linear),
    F32(F32Linear),
}

impl VVLinear {
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        match self {
            Self::Q8(l) => l.forward(x),
            Self::F32(l) => l.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// AdaLN modulation: x * (1 + scale) + shift.
fn modulate(
    x: Tensor<Wgpu, 3>,
    shift: Tensor<Wgpu, 3>,
    scale: Tensor<Wgpu, 3>,
) -> Tensor<Wgpu, 3> {
    x * (scale + 1.0) + shift
}

/// RMSNorm without learnable affine parameters.
fn rms_norm_no_affine(x: Tensor<Wgpu, 3>, eps: f32) -> Tensor<Wgpu, 3> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(2); // [B, S, 1]
    let rms = (variance + eps).sqrt();
    x / rms
}

// ---------------------------------------------------------------------------
// BurnTimestepEmbedder
// ---------------------------------------------------------------------------

pub struct BurnTimestepEmbedder {
    mlp_0: F32Linear,
    mlp_2: F32Linear,
    frequency_embedding_size: usize,
}

impl BurnTimestepEmbedder {
    pub fn new(mlp_0: F32Linear, mlp_2: F32Linear, frequency_embedding_size: usize) -> Self {
        Self { mlp_0, mlp_2, frequency_embedding_size }
    }

    /// Sinusoidal timestep embedding.
    ///
    /// `t` — scalar value (single timestep).
    /// Returns `[1, 1, frequency_embedding_size]` on GPU.
    fn timestep_embedding(&self, t: f32, device: &WgpuDevice) -> Tensor<Wgpu, 3> {
        let half = self.frequency_embedding_size / 2;
        let log_10000 = (10000.0f64).ln();
        let scale = (-log_10000 / half as f64) as f32;

        let mut freq_data = Vec::with_capacity(self.frequency_embedding_size);
        for i in 0..half {
            let freq = (i as f32 * scale).exp();
            freq_data.push((t * freq).cos());
        }
        for i in 0..half {
            let freq = (i as f32 * scale).exp();
            freq_data.push((t * freq).sin());
        }

        Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(freq_data, [1, 1, self.frequency_embedding_size]),
            device,
        )
    }

    pub fn forward(&self, t: f32, device: &WgpuDevice) -> Tensor<Wgpu, 3> {
        let t_freq = self.timestep_embedding(t, device);
        let x = self.mlp_0.forward(t_freq);
        let x = silu(x);
        self.mlp_2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// BurnFeedForwardNetwork (SwiGLU)
// ---------------------------------------------------------------------------

pub struct BurnFeedForwardNetwork {
    gate_proj: VVLinear,
    up_proj: VVLinear,
    down_proj: VVLinear,
}

impl BurnFeedForwardNetwork {
    pub fn new(gate_proj: VVLinear, up_proj: VVLinear, down_proj: VVLinear) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ---------------------------------------------------------------------------
// BurnHeadLayer
// ---------------------------------------------------------------------------

pub struct BurnHeadLayer {
    ffn: BurnFeedForwardNetwork,
    /// RMSNorm weight [head_dim] stored as a 1D f32 vec on CPU for manual application.
    /// We apply it manually to avoid the Burn RmsNorm gamma broadcasting issue with 3D input.
    norm_weight: Vec<f32>,
    norm_eps: f32,
    ada_ln_modulation: VVLinear,
    head_dim: usize,
    device: WgpuDevice,
}

impl BurnHeadLayer {
    pub fn new(
        ffn: BurnFeedForwardNetwork,
        norm_weight: Vec<f32>,
        norm_eps: f32,
        ada_ln_modulation: VVLinear,
        head_dim: usize,
        device: WgpuDevice,
    ) -> Self {
        Self { ffn, norm_weight, norm_eps, ada_ln_modulation, head_dim, device }
    }

    fn apply_rms_norm(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let normed = rms_norm_no_affine(x, self.norm_eps);
        // Multiply by gamma weight [1, 1, head_dim]
        let gamma = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(self.norm_weight.clone(), [1, 1, self.head_dim]),
            &self.device,
        );
        normed * gamma
    }

    /// `x` shape: `[1, 1, head_dim]`
    /// `c_silu` shape: `[1, 1, head_dim]` — pre-computed silu(cond + t_embed)
    pub fn forward(&self, x: Tensor<Wgpu, 3>, c_silu: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let ada = self.ada_ln_modulation.forward(c_silu);
        // ada: [1, 1, 3*head_dim] — split into shift, scale, gate
        let shift = ada.clone().slice([0..1, 0..1, 0..self.head_dim]);
        let scale = ada.clone().slice([0..1, 0..1, self.head_dim..2 * self.head_dim]);
        let gate = ada.slice([0..1, 0..1, 2 * self.head_dim..3 * self.head_dim]);

        let normed = self.apply_rms_norm(x.clone());
        let modulated = modulate(normed, shift, scale);
        let ffn_out = self.ffn.forward(modulated);
        x + gate * ffn_out
    }
}

// ---------------------------------------------------------------------------
// BurnFinalLayer
// ---------------------------------------------------------------------------

pub struct BurnFinalLayer {
    linear: VVLinear,
    ada_ln_modulation: VVLinear,
    norm_eps: f32,
    head_dim: usize,
    total_latent_dim: usize,
    device: WgpuDevice,
}

impl BurnFinalLayer {
    pub fn new(
        linear: VVLinear,
        ada_ln_modulation: VVLinear,
        norm_eps: f32,
        head_dim: usize,
        total_latent_dim: usize,
        device: WgpuDevice,
    ) -> Self {
        Self { linear, ada_ln_modulation, norm_eps, head_dim, total_latent_dim, device }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>, c_silu: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let ada = self.ada_ln_modulation.forward(c_silu);
        let shift = ada.clone().slice([0..1, 0..1, 0..self.head_dim]);
        let scale = ada.slice([0..1, 0..1, self.head_dim..2 * self.head_dim]);

        let normed = rms_norm_no_affine(x, self.norm_eps);
        let modulated = modulate(normed, shift, scale);
        self.linear.forward(modulated)
    }
}

// ---------------------------------------------------------------------------
// BurnVibeVoice
// ---------------------------------------------------------------------------

pub struct BurnVibeVoice {
    noisy_images_proj: VVLinear,
    cond_proj: VVLinear,
    t_embedder: BurnTimestepEmbedder,
    layers: Vec<BurnHeadLayer>,
    final_layer: BurnFinalLayer,
    head_dim: usize,
    total_latent_dim: usize,
    acoustic_dim: usize,
    device: WgpuDevice,
}

impl BurnVibeVoice {
    pub fn new(
        noisy_images_proj: VVLinear,
        cond_proj: VVLinear,
        t_embedder: BurnTimestepEmbedder,
        layers: Vec<BurnHeadLayer>,
        final_layer: BurnFinalLayer,
        head_dim: usize,
        total_latent_dim: usize,
        acoustic_dim: usize,
        device: WgpuDevice,
    ) -> Self {
        Self {
            noisy_images_proj,
            cond_proj,
            t_embedder,
            layers,
            final_layer,
            head_dim,
            total_latent_dim,
            acoustic_dim,
            device,
        }
    }

    /// Predict velocity for one flow matching step.
    ///
    /// - `noisy_images`: `[1, 1, total_latent_dim]` (528 dims)
    /// - `t`: scalar timestep value
    /// - `condition`: `[1, 1, hidden_size]` (2048 dims) — LLM hidden state
    ///
    /// Returns predicted velocity `[1, 1, total_latent_dim]`.
    pub fn forward(
        &self,
        noisy_images: Tensor<Wgpu, 3>,
        t: f32,
        condition: Tensor<Wgpu, 3>,
    ) -> Tensor<Wgpu, 3> {
        // 1. Project noisy latent to head_dim
        let mut x = self.noisy_images_proj.forward(noisy_images);

        // 2. Timestep embedding [1, 1, head_dim]
        let t_emb = self.t_embedder.forward(t, &self.device);

        // 3. Project condition [1, 1, head_dim]
        let cond = self.cond_proj.forward(condition);

        // 4. Combine condition + timestep, apply SiLU
        let c = cond + t_emb;
        let c_silu = silu(c);

        // 5. Transformer layers with AdaLN
        for layer in &self.layers {
            x = layer.forward(x, c_silu.clone());
        }

        // 6. Final projection to total_latent_dim
        self.final_layer.forward(x, c_silu)
    }

    pub fn acoustic_dim(&self) -> usize {
        self.acoustic_dim
    }

    pub fn total_latent_dim(&self) -> usize {
        self.total_latent_dim
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// Gray code decode (CPU, post-readback)
// ---------------------------------------------------------------------------

fn gray_to_int(mut gray: u32) -> u32 {
    let mut mask = gray >> 1;
    while mask != 0 {
        gray ^= mask;
        mask >>= 1;
    }
    gray
}

/// Decode 8 float bits ({-1, 1} encoded) from a slice into a u32 time value.
pub fn decode_gray_bits(bits: &[f32], num_bits: usize) -> u32 {
    let mut gray: u32 = 0;
    for (i, &bit) in bits.iter().take(num_bits).enumerate() {
        if bit > 0.0 {
            gray |= 1 << (num_bits - 1 - i);
        }
    }
    gray_to_int(gray)
}

// ---------------------------------------------------------------------------
// logsnr time schedule (mirrors flow_matching.rs)
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn build_logsnr_schedule(num_steps: usize) -> Vec<f32> {
    let n = num_steps + 1;
    let mut schedule = Vec::with_capacity(n);
    for i in 0..n {
        let logsnr = 5.0 - 10.0 * (i as f64 / num_steps as f64);
        let t = sigmoid(-logsnr / 2.0);
        schedule.push(t as f32);
    }
    schedule[0] = 0.0;
    schedule[n - 1] = 1.0;
    schedule
}

fn scheduled_cfg(base_scale: f32, t: f32) -> f32 {
    1.0 + (base_scale - 1.0) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
}

// ---------------------------------------------------------------------------
// Burn flow matching ODE solver
// ---------------------------------------------------------------------------

/// Solve flow matching ODE on GPU.
///
/// - `condition`: `[1, 1, hidden_size]` LLM hidden state (stays on GPU)
/// - `noise_temp`: initial noise scale
/// - `num_steps`: Euler steps
/// - `cfg_scale`: CFG scale (1.0 = off)
///
/// Returns `(acoustic_f32, time_before, time_after)`.
/// Only 528 floats are read back to CPU at the end.
pub async fn solve_flow_matching_burn(
    vv: &BurnVibeVoice,
    condition: Tensor<Wgpu, 3>,
    noise_temp: f32,
    num_steps: usize,
    cfg_scale: f32,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<f32>, u32, u32)> {
    let total_dim = vv.total_latent_dim;
    let acoustic_dim = vv.acoustic_dim;
    let num_time_bits = 8usize;
    let device = vv.device();

    // Build initial noise on CPU, upload to GPU
    let mut noise = vec![0.0f32; total_dim];
    for v in noise.iter_mut() {
        *v = rng.sample_normal() * noise_temp;
    }
    let mut x: Tensor<Wgpu, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(noise, [1, 1, total_dim]),
        device,
    );

    let schedule = build_logsnr_schedule(num_steps);
    let use_cfg = (cfg_scale - 1.0).abs() > 1e-6;

    // Pre-build zero condition for CFG negative pass
    let cond_neg = if use_cfg {
        let zeros = vec![0.0f32; condition.dims()[2]];
        let hidden_size = condition.dims()[2];
        Some(Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(zeros, [1, 1, hidden_size]),
            device,
        ))
    } else {
        None
    };

    for i in 0..num_steps {
        let t_val = schedule[i];
        let t_next = schedule[i + 1];
        let dt = t_next - t_val;

        let velocity = if use_cfg {
            let vel_pos = vv.forward(x.clone(), t_val, condition.clone());
            let vel_neg = vv.forward(x.clone(), t_val, cond_neg.as_ref().unwrap().clone());

            let cfg = scheduled_cfg(cfg_scale, t_val);

            // Acoustic dims: neg + cfg * (pos - neg)
            let vel_pos_ac = vel_pos.clone().slice([0..1, 0..1, 0..acoustic_dim]);
            let vel_neg_ac = vel_neg.clone().slice([0..1, 0..1, 0..acoustic_dim]);
            let vel_ac = vel_neg_ac.clone() + (vel_pos_ac - vel_neg_ac) * cfg;

            // Duration dims: use positive velocity unchanged
            let duration_dim = total_dim - acoustic_dim;
            if duration_dim > 0 {
                let vel_dur = vel_pos.slice([0..1, 0..1, acoustic_dim..total_dim]);
                Tensor::cat(vec![vel_ac, vel_dur], 2)
            } else {
                vel_ac
            }
        } else {
            vv.forward(x.clone(), t_val, condition.clone())
        };

        x = x + velocity * dt;
    }

    // Read back the final state — only 528 floats = 2KB
    let x_data = x
        .into_data_async()
        .await
        .map_err(|e| anyhow::anyhow!("GPU readback failed: {e:?}"))?;
    let x_vec: Vec<f32> = x_data.to_vec().unwrap();

    // Split acoustic [0..acoustic_dim] and time bits [acoustic_dim..]
    let acoustic = x_vec[..acoustic_dim].to_vec();

    // Decode time_before (bits [acoustic_dim .. acoustic_dim+num_time_bits])
    let tb_bits = &x_vec[acoustic_dim..acoustic_dim + num_time_bits];
    let time_before = decode_gray_bits(tb_bits, num_time_bits);

    // Decode time_after (bits [acoustic_dim+num_time_bits .. acoustic_dim+2*num_time_bits])
    let ta_bits = &x_vec[acoustic_dim + num_time_bits..acoustic_dim + 2 * num_time_bits];
    let time_after = decode_gray_bits(ta_bits, num_time_bits);

    Ok((acoustic, time_before, time_after))
}

// ---------------------------------------------------------------------------
// Rng trait (mirrors tada_model::Rng)
// ---------------------------------------------------------------------------

pub trait Rng {
    fn sample_normal(&mut self) -> f32;
}
