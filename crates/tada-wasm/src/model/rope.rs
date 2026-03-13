//! Rotary Position Embeddings (RoPE) with Llama 3-style frequency scaling.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

/// Rotary Position Embeddings with precomputed cos/sin tables.
pub struct RoPE {
    cos: Tensor<Wgpu, 2>, // [max_pos, half_dim]
    sin: Tensor<Wgpu, 2>, // [max_pos, half_dim]
}

impl RoPE {
    /// Create RoPE with Llama 3-style frequency scaling.
    ///
    /// `rope_scaling`: optional `(factor, high_freq_factor, low_freq_factor, original_max_pos)`
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        rope_scaling: Option<(f64, f64, f64, usize)>,
        device: &WgpuDevice,
    ) -> Self {
        let half_dim = head_dim / 2;

        // Base inverse frequencies: 1 / (theta^(2i/d))
        let mut inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        // Apply Llama 3 RoPE scaling if configured
        if let Some((factor, high_freq_factor, low_freq_factor, orig_max_pos)) = rope_scaling {
            let old_ctx = orig_max_pos as f64;
            let low_freq_wavelen = old_ctx / low_freq_factor;
            let high_freq_wavelen = old_ctx / high_freq_factor;

            for freq in inv_freq.iter_mut() {
                let wavelen = 2.0 * std::f64::consts::PI / *freq;
                if wavelen < high_freq_wavelen {
                    // High frequency: keep unchanged
                } else if wavelen > low_freq_wavelen {
                    // Low frequency: divide by factor
                    *freq /= factor;
                } else {
                    // Smooth interpolation
                    let smooth = (old_ctx / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    *freq = (1.0 - smooth) * (*freq / factor) + smooth * *freq;
                }
            }
        }

        // Compute outer product: freqs[pos, i] = pos * inv_freq[i]
        let mut freqs = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                freqs[pos * half_dim + i] = (pos as f64 * inv_freq[i]) as f32;
            }
        }

        let freqs = Tensor::<Wgpu, 1>::from_floats(freqs.as_slice(), device)
            .reshape([max_seq_len, half_dim]);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        RoPE { cos, sin }
    }

    /// Apply rotary embeddings to Q and K tensors.
    ///
    /// q, k shape: [batch, seq, heads, head_dim]
    pub fn apply(
        &self,
        q: Tensor<Wgpu, 4>,
        k: Tensor<Wgpu, 4>,
        offset: usize,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let seq_len = q.dims()[1];
        let [_max_len, half_dim] = self.cos.dims();
        let cos = self
            .cos
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]);
        let sin = self
            .sin
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]);

        let q_rot = self.apply_rotation(q, cos.clone(), sin.clone());
        let k_rot = self.apply_rotation(k, cos, sin);
        (q_rot, k_rot)
    }

    fn apply_rotation(
        &self,
        x: Tensor<Wgpu, 4>,
        cos: Tensor<Wgpu, 2>,
        sin: Tensor<Wgpu, 2>,
    ) -> Tensor<Wgpu, 4> {
        let [batch, seq, heads, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        let x_pairs = x.reshape([batch, seq, heads, half_dim, 2]);

        let x_r: Tensor<Wgpu, 4> = x_pairs
            .clone()
            .slice([0..batch, 0..seq, 0..heads, 0..half_dim, 0..1])
            .reshape([batch, seq, heads, half_dim]);
        let x_i: Tensor<Wgpu, 4> = x_pairs
            .slice([0..batch, 0..seq, 0..heads, 0..half_dim, 1..2])
            .reshape([batch, seq, heads, half_dim]);

        // Broadcast cos/sin: [seq, half_dim] -> [1, seq, 1, half_dim]
        let cos: Tensor<Wgpu, 4> = cos.unsqueeze_dim::<3>(0).unsqueeze_dim(2);
        let sin: Tensor<Wgpu, 4> = sin.unsqueeze_dim::<3>(0).unsqueeze_dim(2);

        let out_r = x_r.clone() * cos.clone() - x_i.clone() * sin.clone();
        let out_i = x_r * sin + x_i * cos;

        let out_r: Tensor<Wgpu, 5> = out_r.unsqueeze_dim(4);
        let out_i: Tensor<Wgpu, 5> = out_i.unsqueeze_dim(4);
        let out = Tensor::cat(vec![out_r, out_i], 4);
        out.reshape([batch, seq, heads, head_dim])
    }
}
