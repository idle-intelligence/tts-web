//! Burn+wgpu Llama 3.2 1B model for TADA TTS.
//!
//! This is the GPU-accelerated LLM backbone. It takes pre-computed input
//! embeddings (token + acoustic + mask + time) and produces hidden states
//! for the VibeVoice diffusion head.

pub mod attention;
pub mod kv_cache;
pub mod rope;
pub mod vibevoice;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use burn::nn::RmsNorm;

use burn::tensor::activation::{silu, softmax};
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

/// An f32 embedding table on CPU, for small adapter embeddings.
///
/// Stores data as a flat Vec<f32> on CPU to avoid GPU readback issues on WASM
/// (into_data() is sync but WebGPU mapAsync is async).
pub struct F32Embedding {
    data: Vec<f32>, // [vocab_size * dim] flattened
    dim: usize,
}

impl F32Embedding {
    pub fn new(weight: Tensor<Wgpu, 2>, dim: usize) -> Self {
        // Read from GPU to CPU once at load time.
        // This works during model loading (sync context) but not during
        // generation on WASM. Pre-extracting avoids the async issue.
        let data: Vec<f32> = weight.into_data().to_vec().unwrap();
        Self { data, dim }
    }

    /// Create from CPU f32 data directly (avoids GPU readback).
    pub fn from_cpu(data: Vec<f32>, dim: usize) -> Self {
        Self { data, dim }
    }

    /// Look up embedding for id. Adds the result to `out_buf` on CPU.
    pub fn embed_id_add_cpu(&self, id: u32, out_buf: &mut [f32]) {
        let offset = id as usize * self.dim;
        for i in 0..self.dim {
            out_buf[i] += self.data[offset + i];
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

/// Load a TadaLlama from a GGUF file with ALL transformer weights dequantized to F32.
///
/// Supports both Q4_0 GGUF (dequantizes to F32 at load time) and F32 GGUF (reads directly).
/// Uses F32Linear for all projections instead of Q4Linear — bypasses the custom Q4 WGSL shader
/// and uses Burn's built-in matmul. For debugging only.
pub fn load_tada_llama_gguf_f32<R: Read + Seek>(
    reader: &mut gguf::GgufReader<R>,
    device: &WgpuDevice,
) -> anyhow::Result<TadaLlamaF32> {
    use tada_core::config::TadaConfig;
    use burn::tensor::TensorData;

    let cfg = TadaConfig::tada_1b();
    let llama_cfg = &cfg.llama;
    let hidden_size = llama_cfg.hidden_size;
    let n_heads = llama_cfg.num_attention_heads;
    let n_kv_heads = llama_cfg.num_key_value_heads;
    let head_dim = llama_cfg.head_dim;
    let eps = llama_cfg.rms_norm_eps;

    let load_proj = |reader: &mut gguf::GgufReader<R>, name: &str| -> anyhow::Result<F32Linear> {
        let (data, shape) = gguf::load_f32_weight_any(reader, name)?;
        let weight = burn::tensor::Tensor::<burn::backend::wgpu::Wgpu, 2>::from_data(
            TensorData::new(data, [shape[0], shape[1]]),
            device,
        );
        Ok(F32Linear::new(weight, None))
    };

    // --- Transformer layers ---
    let mut layers = Vec::with_capacity(llama_cfg.num_hidden_layers);
    for i in 0..llama_cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let q_proj = load_proj(reader, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let k_proj = load_proj(reader, &format!("{prefix}.self_attn.k_proj.weight"))?;
        let v_proj = load_proj(reader, &format!("{prefix}.self_attn.v_proj.weight"))?;
        let o_proj = load_proj(reader, &format!("{prefix}.self_attn.o_proj.weight"))?;
        let attention = F32Attention::new(q_proj, k_proj, v_proj, o_proj, n_heads, n_kv_heads, head_dim);

        let gate_proj = load_proj(reader, &format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_proj = load_proj(reader, &format!("{prefix}.mlp.up_proj.weight"))?;
        let down_proj = load_proj(reader, &format!("{prefix}.mlp.down_proj.weight"))?;
        let ffn = F32FeedForward::new(gate_proj, up_proj, down_proj);

        let input_ln = gguf::load_rms_norm(reader, &format!("{prefix}.input_layernorm.weight"), eps, device)?;
        let post_attn_ln = gguf::load_rms_norm(reader, &format!("{prefix}.post_attention_layernorm.weight"), eps, device)?;

        layers.push(F32TransformerBlock::new(input_ln, attention, post_attn_ln, ffn));
    }

    // --- Final norm ---
    let final_norm = gguf::load_rms_norm(reader, "model.norm.weight", eps, device)?;

    // --- RoPE ---
    let rope_scaling = llama_cfg.rope_scaling.as_ref().map(|s| {
        (s.factor, s.high_freq_factor, s.low_freq_factor, s.original_max_position_embeddings)
    });
    let rope = RoPE::new(head_dim, 4096, llama_cfg.rope_theta, rope_scaling, device);

    // --- Embeddings ---
    let embed_info = reader.tensor_info("model.embed_tokens.weight")
        .ok_or_else(|| anyhow::anyhow!("model.embed_tokens.weight not found"))?
        .clone();
    let embed_shape = gguf::reverse_gguf_dims(embed_info.shape());
    let vocab_size = embed_shape[0];

    let embed_bytes = reader.tensor_data("model.embed_tokens.weight")?;
    let embed_tokens = EmbeddingStore::new(embed_bytes.clone(), vocab_size, hidden_size);
    let (embed_data, _) = gguf::load_f32_weight_any(reader, "model.embed_tokens.weight")?;
    let embed_tokens_gpu = burn::tensor::Tensor::<burn::backend::wgpu::Wgpu, 2>::from_data(
        TensorData::new(embed_data, [vocab_size, hidden_size]),
        device,
    );

    // --- Acoustic projection (F32 linear) ---
    let acoustic_proj = load_f32_linear(reader, "acoustic_proj", device)?;

    // --- Adapter embeddings (F32) ---
    let acoustic_mask_emb = load_f32_embedding(reader, "acoustic_mask_emb.weight", device)?;
    let time_start_embed = load_f32_embedding(reader, "time_start_embed.weight", device)?;
    let time_end_embed = load_f32_embedding(reader, "time_end_embed.weight", device)?;

    Ok(TadaLlamaF32 {
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
        device: device.clone(),
    })
}

/// TadaLlama with all transformer weights as F32. For debugging only.
pub struct TadaLlamaF32 {
    embed_tokens: EmbeddingStore,
    embed_tokens_gpu: burn::tensor::Tensor<burn::backend::wgpu::Wgpu, 2>,
    acoustic_proj: F32Linear,
    acoustic_mask_emb: F32Embedding,
    time_start_embed: F32Embedding,
    time_end_embed: F32Embedding,
    layers: Vec<F32TransformerBlock>,
    final_norm: burn::nn::RmsNorm<burn::backend::wgpu::Wgpu>,
    rope: RoPE,
    hidden_size: usize,
    device: burn::backend::wgpu::WgpuDevice,
}

impl TadaLlamaF32 {
    pub fn forward_step(
        &self,
        token_id: u32,
        acoustic: &[f32],
        acoustic_mask: u32,
        time_before: u32,
        time_after: u32,
        cache: &mut LayerCaches,
    ) -> burn::tensor::Tensor<burn::backend::wgpu::Wgpu, 3> {
        let dim = self.hidden_size;

        // Build combined embedding on CPU (no GPU readback needed)
        let mut sum = vec![0.0f32; dim];
        self.embed_tokens.embed_id_add_cpu(token_id, &mut sum);
        self.acoustic_mask_emb.embed_id_add_cpu(acoustic_mask, &mut sum);
        self.time_start_embed.embed_id_add_cpu(time_before, &mut sum);
        self.time_end_embed.embed_id_add_cpu(time_after, &mut sum);

        let acoustic_tensor = burn::tensor::Tensor::<burn::backend::wgpu::Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(acoustic.to_vec(), [1, 1, acoustic.len()]),
            &self.device,
        );
        let acoustic_proj = self.acoustic_proj.forward(acoustic_tensor);

        let cpu_embed = burn::tensor::Tensor::<burn::backend::wgpu::Wgpu, 3>::from_data(
            burn::tensor::TensorData::new(sum, [1, 1, dim]),
            &self.device,
        );
        let mut x = cpu_embed + acoustic_proj;

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
        }
        self.final_norm.forward(x)
    }

    pub fn create_cache(&self, max_seq_len: usize) -> LayerCaches {
        LayerCaches::new(
            self.layers.len(),
            max_seq_len,
            8,
            self.hidden_size / 32,
            &self.device,
        )
    }

    /// Forward step with layer-0 attention debug dumps.
    ///
    /// Runs through all layers but prints intermediate tensors for layer 0's attention.
    /// Use on step 2 to compare with candle's `forward_step_debug_layer0`.
    pub fn forward_step_debug_layer0(
        &self,
        token_id: u32,
        acoustic: &[f32],
        acoustic_mask: u32,
        time_before: u32,
        time_after: u32,
        cache: &mut LayerCaches,
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
        let mut x = cpu_embed + acoustic_proj;

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                if i == 0 {
                    // Layer 0: run with debug
                    println!("[BURN] === Layer 0 debug ===");
                    let residual = x.clone();
                    let normed = layer.input_layernorm.forward(x.clone());
                    {
                        let dims = normed.dims();
                        let flat: Tensor<Wgpu, 1> = normed.clone().flatten(0, 2);
                        let data: Vec<f32> = flat.into_data().to_vec().unwrap();
                        println!("  [BURN] after_input_layernorm: shape={dims:?} first4={:?}", &data[..data.len().min(4)]);
                    }
                    let attn_out = layer.attention.forward_with_cache_debug(normed, &self.rope, c);
                    {
                        let dims = attn_out.dims();
                        let flat: Tensor<Wgpu, 1> = attn_out.clone().flatten(0, 2);
                        let data: Vec<f32> = flat.into_data().to_vec().unwrap();
                        println!("  [BURN] after_attn: shape={dims:?} first4={:?}", &data[..data.len().min(4)]);
                    }
                    let after_residual = attn_out + residual;
                    {
                        let dims = after_residual.dims();
                        let flat: Tensor<Wgpu, 1> = after_residual.clone().flatten(0, 2);
                        let data: Vec<f32> = flat.into_data().to_vec().unwrap();
                        println!("[BURN] after layer 0 residual: shape={dims:?} first4={:?}", &data[..data.len().min(4)]);
                    }
                    // Complete rest of layer 0 (MLP)
                    let residual2 = after_residual.clone();
                    let normed2 = layer.post_attention_layernorm.forward(after_residual);
                    let mlp_out = layer.ffn.forward(normed2);
                    x = mlp_out + residual2;
                } else {
                    x = layer.forward_with_cache(x, &self.rope, c);
                }
            }
        }
        self.final_norm.forward(x)
    }
}

// ---------------------------------------------------------------------------
// F32Attention, F32FeedForward, F32TransformerBlock — F32 versions for debugging
// ---------------------------------------------------------------------------

pub struct F32Attention {
    q_proj: F32Linear,
    k_proj: F32Linear,
    v_proj: F32Linear,
    o_proj: F32Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl F32Attention {
    pub fn new(
        q_proj: F32Linear,
        k_proj: F32Linear,
        v_proj: F32Linear,
        o_proj: F32Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self { q_proj, k_proj, v_proj, o_proj, n_heads, n_kv_heads, head_dim, scale: (head_dim as f32).powf(-0.5) }
    }

    /// Debug forward: identical computation to forward_with_cache but prints
    /// the first 4 values and shape of each intermediate tensor.
    /// Only call for a single layer on a single step.
    pub fn forward_with_cache_debug(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        fn peek(label: &str, t: &Tensor<Wgpu, 4>) {
            let dims = t.dims();
            let flat: Tensor<Wgpu, 1> = t.clone().flatten(0, 3);
            let data: Vec<f32> = flat.into_data().to_vec().unwrap();
            let n = data.len().min(4);
            println!("  [BURN] {label}: shape={dims:?} first4={:?}", &data[..n]);
        }
        fn peek3(label: &str, t: &Tensor<Wgpu, 3>) {
            let dims = t.dims();
            let flat: Tensor<Wgpu, 1> = t.clone().flatten(0, 2);
            let data: Vec<f32> = flat.into_data().to_vec().unwrap();
            let n = data.len().min(4);
            println!("  [BURN] {label}: shape={dims:?} first4={:?}", &data[..n]);
        }

        let [batch, seq_len, _] = x.dims();
        let offset = cache.offset();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Print Q,K,V before RoPE (reshape to 4D for peek; dims are [b,seq,heads,hd])
        peek("q_before_rope [b,seq,nh,hd]", &q);
        peek("k_before_rope [b,seq,nkv,hd]", &k);
        peek("v_before_rope [b,seq,nkv,hd]", &v);

        let (q, k) = rope.apply(q, k, offset);

        peek("q_after_rope  [b,seq,nh,hd]", &q);
        peek("k_after_rope  [b,seq,nkv,hd]", &k);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        peek("k_pre_cache   [b,nkv,seq,hd]", &k);
        peek("v_pre_cache   [b,nkv,seq,hd]", &v);

        let (k, v) = cache.update(k, v);

        peek("k_from_cache  [b,nkv,total,hd]", &k);
        peek("v_from_cache  [b,nkv,total,hd]", &v);

        let (k, v) = if self.n_heads != self.n_kv_heads {
            let repeat_factor = self.n_heads / self.n_kv_heads;
            let [b, nkv, s, hd] = k.dims();
            let k2 = k.unsqueeze_dim::<5>(2).repeat_dim(2, repeat_factor).reshape([b, nkv * repeat_factor, s, hd]);
            let v2 = v.unsqueeze_dim::<5>(2).repeat_dim(2, repeat_factor).reshape([b, nkv * repeat_factor, s, hd]);
            (k2, v2)
        } else {
            (k, v)
        };

        peek("k_expand_kv   [b,nh,total,hd]", &k);
        peek("v_expand_kv   [b,nh,total,hd]", &v);

        let k_t = k.swap_dims(2, 3);
        peek("k_t           [b,nh,hd,total]", &k_t);

        let scores = q.matmul(k_t) * self.scale;
        peek("scores        [b,nh,seq,total]", &scores);

        let total_seq_len = cache.seq_len();
        let scores = if seq_len > 1 {
            let device = scores.device();
            let mut mask_data = vec![0.0f32; seq_len * total_seq_len];
            for i in 0..seq_len {
                let actual_pos = offset + i;
                for j in 0..total_seq_len {
                    if j > actual_pos {
                        mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
            let mask: Tensor<Wgpu, 4> = mask.reshape([seq_len, total_seq_len]).unsqueeze_dim::<3>(0).unsqueeze_dim(0);
            scores + mask
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        peek("attn_weights  [b,nh,seq,total]", &attn);

        let out = attn.matmul(v);
        peek("attn_out      [b,nh,seq,hd]", &out);

        let out = out.swap_dims(1, 2).reshape([batch, seq_len, self.n_heads * self.head_dim]);
        peek3("attn_out_flat [b,seq,hidden]", &out);

        self.o_proj.forward(out)
    }

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

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);

        let (k, v) = if self.n_heads != self.n_kv_heads {
            let repeat_factor = self.n_heads / self.n_kv_heads;
            let [b, nkv, s, hd] = k.dims();
            let k2 = k.unsqueeze_dim::<5>(2).repeat_dim(2, repeat_factor).reshape([b, nkv * repeat_factor, s, hd]);
            let v2 = v.unsqueeze_dim::<5>(2).repeat_dim(2, repeat_factor).reshape([b, nkv * repeat_factor, s, hd]);
            (k2, v2)
        } else {
            (k, v)
        };

        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        let total_seq_len = cache.seq_len();
        let scores = if seq_len > 1 {
            let device = scores.device();
            let mut mask_data = vec![0.0f32; seq_len * total_seq_len];
            for i in 0..seq_len {
                let actual_pos = offset + i;
                for j in 0..total_seq_len {
                    if j > actual_pos {
                        mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
            let mask: Tensor<Wgpu, 4> = mask.reshape([seq_len, total_seq_len]).unsqueeze_dim::<3>(0).unsqueeze_dim(0);
            scores + mask
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.o_proj.forward(out)
    }
}

pub struct F32FeedForward {
    gate_proj: F32Linear,
    up_proj: F32Linear,
    down_proj: F32Linear,
}

impl F32FeedForward {
    pub fn new(gate_proj: F32Linear, up_proj: F32Linear, down_proj: F32Linear) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

pub struct F32TransformerBlock {
    pub input_layernorm: RmsNorm<Wgpu>,
    pub attention: F32Attention,
    pub post_attention_layernorm: RmsNorm<Wgpu>,
    pub ffn: F32FeedForward,
}

impl F32TransformerBlock {
    pub fn new(
        input_layernorm: RmsNorm<Wgpu>,
        attention: F32Attention,
        post_attention_layernorm: RmsNorm<Wgpu>,
        ffn: F32FeedForward,
    ) -> Self {
        Self { input_layernorm, attention, post_attention_layernorm, ffn }
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

/// Load an F32 embedding table from GGUF (stays on CPU, no GPU upload).
fn load_f32_embedding<R: Read + Seek>(
    reader: &mut gguf::GgufReader<R>,
    name: &str,
    _device: &WgpuDevice,
) -> anyhow::Result<F32Embedding> {
    let data = gguf::load_f32_tensor(reader, name)?;
    let info = reader.tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("{name} not found"))?
        .clone();
    let shape = gguf::reverse_gguf_dims(info.shape());
    let dim = shape[1];
    Ok(F32Embedding::from_cpu(data, dim))
}

/// Load the VibeVoice diffusion head weights from GGUF into Burn/GPU.
///
/// Reads `prediction_head.*` tensors. Handles Q8_0, Q4_0, F16, and F32 —
/// all dequantized to F32 at load time via `load_f32_weight_any`.
///
/// This is intentionally separate from `load_tada_llama_gguf` so it can be
/// called optionally (the candle path is still available as a fallback).
pub fn load_burn_vibevoice<R: Read + Seek>(
    reader: &mut gguf::GgufReader<R>,
    device: &WgpuDevice,
) -> anyhow::Result<vibevoice::BurnVibeVoice> {
    use burn::tensor::TensorData;
    use tada_core::config::TadaConfig;
    use vibevoice::{
        BurnFeedForwardNetwork, BurnFinalLayer, BurnHeadLayer, BurnTimestepEmbedder, BurnVibeVoice,
        VVLinear,
    };

    let cfg = TadaConfig::tada_1b();
    let head_dim = 2048usize; // prediction_head hidden dim (same as LLM hidden_size)
    let _hidden_size = cfg.llama.hidden_size; // 2048 — kept for documentation
    let total_latent_dim = cfg.total_latent_dim(); // 528
    let acoustic_dim = cfg.acoustic_dim; // 512
    let num_head_layers = cfg.head_layers; // 6
    let frequency_embedding_size = 256usize;
    let eps = cfg.llama.rms_norm_eps as f32;

    let prefix = "prediction_head";

    // Helper: load Q8Linear if Q8_0, else F32Linear (for small t_embedder layers)
    let load_vv = |reader: &mut gguf::GgufReader<R>, name: &str| -> anyhow::Result<VVLinear> {
        gguf::load_q8_or_f32_linear(reader, name, device)
    };

    // Helper: load small F32Linear (stays F32 regardless — for t_embedder)
    let load_f32 = |reader: &mut gguf::GgufReader<R>, name: &str| -> anyhow::Result<F32Linear> {
        let (data, shape) = gguf::load_f32_weight_any(reader, name)?;
        let w = Tensor::<Wgpu, 2>::from_data(TensorData::new(data, [shape[0], shape[1]]), device);
        Ok(F32Linear::new(w, None))
    };

    // noisy_images_proj: [head_dim, total_latent_dim]
    let noisy_images_proj = load_vv(reader, &format!("{prefix}.noisy_images_proj.weight"))?;

    // cond_proj: [head_dim, hidden_size]
    let cond_proj = load_vv(reader, &format!("{prefix}.cond_proj.weight"))?;

    // t_embedder MLP — small, stays F32
    let t_mlp_0 = load_f32(reader, &format!("{prefix}.t_embedder.mlp.0.weight"))?;
    let t_mlp_2 = load_f32(reader, &format!("{prefix}.t_embedder.mlp.2.weight"))?;
    let t_embedder = BurnTimestepEmbedder::new(t_mlp_0, t_mlp_2, frequency_embedding_size);

    // Transformer layers
    let mut layers = Vec::with_capacity(num_head_layers);
    for i in 0..num_head_layers {
        let lp = format!("{prefix}.layers.{i}");

        let gate_proj = load_vv(reader, &format!("{lp}.ffn.gate_proj.weight"))?;
        let up_proj = load_vv(reader, &format!("{lp}.ffn.up_proj.weight"))?;
        let down_proj = load_vv(reader, &format!("{lp}.ffn.down_proj.weight"))?;
        let ffn = BurnFeedForwardNetwork::new(gate_proj, up_proj, down_proj);

        let norm_weight = gguf::load_f32_tensor(reader, &format!("{lp}.norm.weight"))?;

        let ada = load_vv(reader, &format!("{lp}.adaLN_modulation.1.weight"))?;

        layers.push(BurnHeadLayer::new(ffn, norm_weight, eps, ada, head_dim, device.clone()));
    }

    // Final layer
    let fl_prefix = format!("{prefix}.final_layer");
    let final_linear = load_vv(reader, &format!("{fl_prefix}.linear.weight"))?;
    let final_ada = load_vv(reader, &format!("{fl_prefix}.adaLN_modulation.1.weight"))?;
    let final_layer = BurnFinalLayer::new(
        final_linear,
        final_ada,
        eps,
        head_dim,
        total_latent_dim,
        device.clone(),
    );

    Ok(BurnVibeVoice::new(
        noisy_images_proj,
        cond_proj,
        t_embedder,
        layers,
        final_layer,
        head_dim,
        total_latent_dim,
        acoustic_dim,
        device.clone(),
    ))
}
