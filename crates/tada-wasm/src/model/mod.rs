//! Burn+wgpu Llama 3.2 1B model for TADA TTS.
//!
//! This is the GPU-accelerated LLM backbone. It takes pre-computed input
//! embeddings (token + acoustic + mask + time) and produces hidden states
//! for the VibeVoice diffusion head.

pub mod attention;
pub mod kv_cache;
pub mod rope;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use burn::nn::RmsNorm;

use crate::gguf::{self, EmbeddingStore};
pub use attention::{Q4Attention, Q4FeedForward, Q4TransformerBlock};
pub use kv_cache::{KVCache, LayerCaches};
use rope::RoPE;

// ---------------------------------------------------------------------------
// F32Linear — simple f32 linear layer on GPU
// ---------------------------------------------------------------------------

/// A linear layer with f32 weights on GPU.
///
/// For small adapter layers (acoustic_proj) that aren't Q4-quantized.
pub struct F32Linear {
    weight: Tensor<Wgpu, 2>, // [out_features, in_features]
    bias: Option<Tensor<Wgpu, 1>>,
}

impl F32Linear {
    pub fn new(weight: Tensor<Wgpu, 2>, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weight, bias }
    }

    /// Forward: x @ weight^T + bias
    ///
    /// x shape: [B, M, K], returns [B, M, N]
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        // weight: [N, K], transpose to [K, N]
        let wt = self.weight.clone().transpose();
        let wt: Tensor<Wgpu, 3> = wt.unsqueeze_dim(0); // [1, K, N]
        let out = x.matmul(wt);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze_dim::<2>(0).unsqueeze_dim(0),
            None => out,
        }
    }
}

// ---------------------------------------------------------------------------
// F32Embedding — f32 embedding table on GPU
// ---------------------------------------------------------------------------

/// An f32 embedding table on GPU, for small embedding tables.
///
/// For adapter embeddings (acoustic_mask, time_start, time_end) that
/// aren't Q4-quantized.
pub struct F32Embedding {
    weight: Tensor<Wgpu, 2>, // [vocab_size, dim]
    dim: usize,
}

impl F32Embedding {
    pub fn new(weight: Tensor<Wgpu, 2>, dim: usize) -> Self {
        Self { weight, dim }
    }

    /// Look up embedding for id. Adds the result to `out_buf` on CPU.
    pub fn embed_id_add_cpu(&self, id: u32, out_buf: &mut [f32]) {
        let selected = self.weight.clone().slice([id as usize..id as usize + 1, 0..self.dim]);
        let data = selected.into_data();
        let vec: Vec<f32> = data.to_vec().unwrap();
        for (i, &v) in vec.iter().enumerate() {
            out_buf[i] += v;
        }
    }
}

// ---------------------------------------------------------------------------
// TadaLlama — the full model
// ---------------------------------------------------------------------------

/// Llama 3.2 1B backbone for TADA TTS, running on WebGPU via Burn.
///
/// Holds:
/// - Token embedding (CPU Q4, dequanted per step)
/// - Acoustic projection (F32Linear, [512 → 2048])
/// - Acoustic mask embedding (2 entries of 2048)
/// - Time start/end embeddings (num_time_classes entries of 2048)
/// - 16 transformer layers (Q4 attention + SwiGLU MLP)
/// - Final RMSNorm
/// - embed_tokens_gpu: full f32 embedding table on GPU for tied lm_head
pub struct TadaLlama {
    embed_tokens: EmbeddingStore,
    embed_tokens_gpu: Tensor<Wgpu, 2>, // [vocab_size, hidden_size] for tied lm_head
    acoustic_proj: F32Linear,
    acoustic_mask_emb: F32Embedding,
    time_start_embed: F32Embedding,
    time_end_embed: F32Embedding,
    layers: Vec<Q4TransformerBlock>,
    final_norm: RmsNorm<Wgpu>,
    rope: RoPE,
    hidden_size: usize,
    device: WgpuDevice,
}

impl TadaLlama {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed_tokens: EmbeddingStore,
        embed_tokens_gpu: Tensor<Wgpu, 2>,
        acoustic_proj: F32Linear,
        acoustic_mask_emb: F32Embedding,
        time_start_embed: F32Embedding,
        time_end_embed: F32Embedding,
        layers: Vec<Q4TransformerBlock>,
        final_norm: RmsNorm<Wgpu>,
        rope: RoPE,
        hidden_size: usize,
        device: WgpuDevice,
    ) -> Self {
        Self {
            embed_tokens,
            embed_tokens_gpu,
            acoustic_proj,
            acoustic_mask_emb,
            time_start_embed,
            time_end_embed,
            layers,
            final_norm,
            rope,
            hidden_size,
            device,
        }
    }

    /// Single-step forward pass for autoregressive generation.
    ///
    /// Builds the combined input embedding from all components, runs through
    /// the transformer layers, and returns the hidden state.
    ///
    /// - `token_id`: text token ID
    /// - `acoustic`: acoustic feature vector [acoustic_dim] (512 floats)
    /// - `acoustic_mask`: 0 (no acoustic data) or 1 (has acoustic data)
    /// - `time_before`: duration class index for time-before embedding
    /// - `time_after`: duration class index for time-after embedding
    /// - `cache`: mutable layer KV caches
    ///
    /// Returns hidden state `[1, 1, hidden_size]`.
    pub fn forward_step(
        &self,
        token_id: u32,
        acoustic: &[f32],
        acoustic_mask: u32,
        time_before: u32,
        time_after: u32,
        cache: &mut LayerCaches,
    ) -> Tensor<Wgpu, 3> {
        let dim = self.hidden_size;

        // Build combined embedding on CPU, upload once
        // Token embedding is Q4 on CPU; adapter embeddings are F32 on GPU
        // but small enough that CPU readback is fine for accumulation.
        let mut sum = vec![0.0f32; dim];

        // Token embedding (Q4 CPU dequant)
        self.embed_tokens.embed_id_add_cpu(token_id, &mut sum);

        // Acoustic mask embedding (F32 GPU → CPU readback)
        self.acoustic_mask_emb.embed_id_add_cpu(acoustic_mask, &mut sum);

        // Time embeddings (F32 GPU → CPU readback)
        self.time_start_embed.embed_id_add_cpu(time_before, &mut sum);
        self.time_end_embed.embed_id_add_cpu(time_after, &mut sum);

        // Acoustic projection on GPU: [1, 1, acoustic_dim] → [1, 1, hidden_size]
        let acoustic_tensor = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(acoustic.to_vec(), [1, 1, acoustic.len()]),
            &self.device,
        );
        let acoustic_proj = self.acoustic_proj.forward(acoustic_tensor);

        // Upload CPU embeddings sum
        let cpu_embed = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(sum, [1, 1, dim]),
            &self.device,
        );

        // Combine: token + acoustic_proj + mask + time_start + time_end
        let input = cpu_embed + acoustic_proj;

        // Run through transformer layers
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
        }

        // Final norm
        self.final_norm.forward(x)
    }

    /// Single-step forward pass running only the first `num_layers` transformer layers.
    ///
    /// Identical to `forward_step` but stops after `num_layers` layers and skips
    /// the final RMSNorm when `num_layers < total layers`. Useful for per-layer
    /// hidden state comparison during debugging.
    ///
    /// Returns hidden state `[1, 1, hidden_size]`.
    pub fn forward_step_layers(
        &self,
        token_id: u32,
        acoustic: &[f32],
        acoustic_mask: u32,
        time_before: u32,
        time_after: u32,
        cache: &mut LayerCaches,
        num_layers: usize,
    ) -> Tensor<Wgpu, 3> {
        let dim = self.hidden_size;

        let mut sum = vec![0.0f32; dim];
        self.embed_tokens.embed_id_add_cpu(token_id, &mut sum);
        self.acoustic_mask_emb.embed_id_add_cpu(acoustic_mask, &mut sum);
        self.time_start_embed.embed_id_add_cpu(time_before, &mut sum);
        self.time_end_embed.embed_id_add_cpu(time_after, &mut sum);

        let acoustic_tensor = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(acoustic.to_vec(), [1, 1, acoustic.len()]),
            &self.device,
        );
        let acoustic_proj = self.acoustic_proj.forward(acoustic_tensor);

        let cpu_embed = Tensor::<Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(sum, [1, 1, dim]),
            &self.device,
        );
        let input = cpu_embed + acoustic_proj;

        let total_layers = self.layers.len();
        let run_layers = num_layers.min(total_layers);

        let mut x = input;
        for i in 0..run_layers {
            let layer = &self.layers[i];
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
        }

        if run_layers >= total_layers {
            self.final_norm.forward(x)
        } else {
            x
        }
    }

    /// Compute tied lm_head logits: hidden @ embed_tokens^T.
    ///
    /// For TADA, the lm_head shares weights with embed_tokens (tied).
    /// `embed_tokens_gpu` is the pre-dequantized f32 embedding table on GPU.
    ///
    /// `hidden`: [1, 1, hidden_size]
    /// Returns: [1, 1, vocab_size]
    pub fn lm_head(&self, hidden: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        // embed_tokens_gpu: [vocab_size, hidden_size]
        // hidden: [1, 1, hidden_size]
        // Want: hidden @ embed^T → [1, 1, vocab_size]
        let embed_t = self.embed_tokens_gpu.clone().transpose(); // [hidden_size, vocab_size]
        let embed_t: Tensor<Wgpu, 3> = embed_t.unsqueeze_dim(0); // [1, hidden_size, vocab_size]
        hidden.matmul(embed_t)
    }

    /// Create a new LayerCaches for this model.
    ///
    /// TADA has no sliding window, so we use a large max cache length.
    pub fn create_cache(&self, max_seq_len: usize) -> LayerCaches {
        LayerCaches::new(
            self.layers.len(),
            max_seq_len,
            8, // n_kv_heads for Llama 3.2 1B
            self.hidden_size / 32, // head_dim = hidden_size / n_heads
            &self.device,
        )
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

use std::io::{Read, Seek};

/// Load a TadaLlama from a GGUF file.
///
/// Reads the Llama backbone weights + TADA adapter weights from GGUF.
/// The VibeVoice head and decoder are NOT loaded here (they stay on candle/CPU).
pub fn load_tada_llama_gguf<R: Read + Seek>(
    reader: &mut gguf::GgufReader<R>,
    device: &WgpuDevice,
) -> anyhow::Result<TadaLlama> {
    use tada_core::config::TadaConfig;

    let cfg = TadaConfig::tada_1b();
    let llama_cfg = &cfg.llama;
    let hidden_size = llama_cfg.hidden_size;
    let n_heads = llama_cfg.num_attention_heads;
    let n_kv_heads = llama_cfg.num_key_value_heads;
    let head_dim = llama_cfg.head_dim;
    let eps = llama_cfg.rms_norm_eps;

    // --- Transformer layers ---
    let mut layers = Vec::with_capacity(llama_cfg.num_hidden_layers);
    for i in 0..llama_cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");

        let q_proj = gguf::load_q4_linear(reader, &format!("{prefix}.self_attn.q_proj.weight"), device)?;
        let k_proj = gguf::load_q4_linear(reader, &format!("{prefix}.self_attn.k_proj.weight"), device)?;
        let v_proj = gguf::load_q4_linear(reader, &format!("{prefix}.self_attn.v_proj.weight"), device)?;
        let o_proj = gguf::load_q4_linear(reader, &format!("{prefix}.self_attn.o_proj.weight"), device)?;

        let attention = Q4Attention::new(q_proj, k_proj, v_proj, o_proj, n_heads, n_kv_heads, head_dim);

        let gate_proj = gguf::load_q4_linear(reader, &format!("{prefix}.mlp.gate_proj.weight"), device)?;
        let up_proj = gguf::load_q4_linear(reader, &format!("{prefix}.mlp.up_proj.weight"), device)?;
        let down_proj = gguf::load_q4_linear(reader, &format!("{prefix}.mlp.down_proj.weight"), device)?;

        let ffn = Q4FeedForward::new(gate_proj, up_proj, down_proj);

        let input_ln = gguf::load_rms_norm(reader, &format!("{prefix}.input_layernorm.weight"), eps, device)?;
        let post_attn_ln = gguf::load_rms_norm(reader, &format!("{prefix}.post_attention_layernorm.weight"), eps, device)?;

        layers.push(Q4TransformerBlock::new(input_ln, attention, post_attn_ln, ffn));
    }

    // --- Final norm ---
    let final_norm = gguf::load_rms_norm(reader, "model.norm.weight", eps, device)?;

    // --- RoPE ---
    let rope_scaling = llama_cfg.rope_scaling.as_ref().map(|s| {
        (s.factor, s.high_freq_factor, s.low_freq_factor, s.original_max_position_embeddings)
    });
    // Use a reasonable max_seq_len for the RoPE table (not the full 131072)
    let max_rope_len = 4096;
    let rope = RoPE::new(head_dim, max_rope_len, llama_cfg.rope_theta, rope_scaling, device);

    // --- Embeddings (CPU Q4) ---
    let embed_info = reader.tensor_info("model.embed_tokens.weight")
        .ok_or_else(|| anyhow::anyhow!("model.embed_tokens.weight not found"))?
        .clone();
    let embed_shape = gguf::reverse_gguf_dims(embed_info.shape());
    let embed_bytes = reader.tensor_data("model.embed_tokens.weight")?;
    let embed_tokens = EmbeddingStore::new(embed_bytes, embed_shape[0], embed_shape[1]);

    // Dequant full embedding table to f32 on GPU for tied lm_head
    let vocab_size = embed_shape[0];
    let mut embed_f32 = vec![0.0f32; vocab_size * hidden_size];
    for id in 0..vocab_size {
        let offset = id * hidden_size;
        embed_tokens.embed_id_add_cpu(id as u32, &mut embed_f32[offset..offset + hidden_size]);
    }
    let embed_tokens_gpu = Tensor::<Wgpu, 2>::from_data(
        burn::tensor::TensorData::new(embed_f32, [vocab_size, hidden_size]),
        device,
    );

    // --- Acoustic projection (F32 linear) ---
    let acoustic_proj = load_f32_linear(reader, "acoustic_proj", device)?;

    // --- Adapter embeddings (F32) ---
    let acoustic_mask_emb = load_f32_embedding(reader, "acoustic_mask_emb.weight", device)?;
    let time_start_embed = load_f32_embedding(reader, "time_start_embed.weight", device)?;
    let time_end_embed = load_f32_embedding(reader, "time_end_embed.weight", device)?;

    Ok(TadaLlama::new(
        embed_tokens,
        embed_tokens_gpu,
        acoustic_proj,
        acoustic_mask_emb,
        time_start_embed,
        time_end_embed,
        layers,
        final_norm,
        rope,
        hidden_size,
        device.clone(),
    ))
}

/// Load an F32 linear layer from GGUF (weight + optional bias).
fn load_f32_linear<R: Read + Seek>(
    reader: &mut gguf::GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> anyhow::Result<F32Linear> {
    let weight_name = format!("{name}.weight");
    let weight_data = gguf::load_f32_tensor(reader, &weight_name)?;
    let info = reader.tensor_info(&weight_name)
        .ok_or_else(|| anyhow::anyhow!("{weight_name} not found"))?
        .clone();
    let shape = gguf::reverse_gguf_dims(info.shape());
    let weight = Tensor::<Wgpu, 2>::from_data(
        burn::tensor::TensorData::new(weight_data, [shape[0], shape[1]]),
        device,
    );

    let bias_name = format!("{name}.bias");
    let bias = if reader.tensor_info(&bias_name).is_some() {
        let bias_data = gguf::load_f32_tensor(reader, &bias_name)?;
        let len = bias_data.len();
        Some(Tensor::<Wgpu, 1>::from_data(
            burn::tensor::TensorData::new(bias_data, [len]),
            device,
        ))
    } else {
        None
    };

    Ok(F32Linear::new(weight, bias))
}

/// Load an F32 embedding table from GGUF.
fn load_f32_embedding<R: Read + Seek>(
    reader: &mut gguf::GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> anyhow::Result<F32Embedding> {
    let data = gguf::load_f32_tensor(reader, name)?;
    let info = reader.tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("{name} not found"))?
        .clone();
    let shape = gguf::reverse_gguf_dims(info.shape());
    let dim = shape[1];
    let weight = Tensor::<Wgpu, 2>::from_data(
        burn::tensor::TensorData::new(data, [shape[0], shape[1]]),
        device,
    );
    Ok(F32Embedding::new(weight, dim))
}
