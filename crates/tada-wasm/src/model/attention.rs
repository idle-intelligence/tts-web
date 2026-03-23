//! Q4 Attention and SwiGLU MLP for Llama 3.2.

use burn::backend::wgpu::Wgpu;
use burn::nn::RmsNorm;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::Tensor;

use crate::gguf::Q4Linear;
use super::kv_cache::KVCache;
use super::rope::RoPE;

// ---------------------------------------------------------------------------
// Causal mask
// ---------------------------------------------------------------------------

/// Apply causal mask for prefill (seq_len > 1). Single-step decoding skips this.
fn apply_causal_mask(
    scores: Tensor<Wgpu, 4>,
    q_len: usize,
    kv_len: usize,
    offset: usize,
) -> Tensor<Wgpu, 4> {
    if q_len == 1 {
        return scores;
    }
    let device = scores.device();
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let actual_pos = offset + i;
        for j in 0..kv_len {
            if j > actual_pos {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<Wgpu, 2> = mask.reshape([q_len, kv_len]);
    let mask: Tensor<Wgpu, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

// ---------------------------------------------------------------------------
// Q4Attention — Llama-style GQA with separate Q/K/V/O projections
// ---------------------------------------------------------------------------

/// Grouped-query attention with Q4-quantized weights.
///
/// Llama 3.2 1B: 32 query heads, 8 KV heads, head_dim=64.
/// Separate Q/K/V/O projections (unlike stt-web's combined in_proj).
pub struct Q4Attention {
    q_proj: Q4Linear,
    k_proj: Q4Linear,
    v_proj: Q4Linear,
    o_proj: Q4Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Q4Attention {
    pub fn new(
        q_proj: Q4Linear,
        k_proj: Q4Linear,
        v_proj: Q4Linear,
        o_proj: Q4Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }

    /// Forward pass with KV cache.
    ///
    /// x shape: [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.offset();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);
        let total_seq_len = cache.seq_len();

        // GQA: expand KV heads to match Q heads
        let (k, v) = self.expand_kv(k, v);

        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        let scores = apply_causal_mask(scores, seq_len, total_seq_len, offset);

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.o_proj.forward(out)
    }

    fn expand_kv(
        &self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        if self.n_heads == self.n_kv_heads {
            return (k, v);
        }
        let repeat_factor = self.n_heads / self.n_kv_heads;
        let [batch, n_kv_heads, seq, head_dim] = k.dims();

        let k = k
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        let v = v
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        (k, v)
    }
}

// ---------------------------------------------------------------------------
// Q4FeedForward — SwiGLU MLP with separate gate/up/down projections
// ---------------------------------------------------------------------------

/// SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
///
/// Llama 3.2 1B: hidden=2048, intermediate=8192.
pub struct Q4FeedForward {
    gate_proj: Q4Linear,
    up_proj: Q4Linear,
    down_proj: Q4Linear,
}

impl Q4FeedForward {
    pub fn new(gate_proj: Q4Linear, up_proj: Q4Linear, down_proj: Q4Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ---------------------------------------------------------------------------
// Q4TransformerBlock
// ---------------------------------------------------------------------------

/// Pre-LN transformer block: RMSNorm → attention → residual → RMSNorm → MLP → residual
pub struct Q4TransformerBlock {
    input_layernorm: RmsNorm<Wgpu>,
    attention: Q4Attention,
    post_attention_layernorm: RmsNorm<Wgpu>,
    ffn: Q4FeedForward,
}

impl Q4TransformerBlock {
    pub fn new(
        input_layernorm: RmsNorm<Wgpu>,
        attention: Q4Attention,
        post_attention_layernorm: RmsNorm<Wgpu>,
        ffn: Q4FeedForward,
    ) -> Self {
        Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            ffn,
        }
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache);
        let x = x + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}
