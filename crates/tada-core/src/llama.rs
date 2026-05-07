//! Llama 3.2 1B backbone for TADA TTS.
//!
//! This is a minimal, WASM-friendly Llama implementation:
//! - No flash-attn, no tracing spans
//! - Quantized weights via QMatMul (loaded from GGUF)
//! - Returns hidden states for the diffusion head, and optionally logits via tied lm_head
//! - GQA with key/value head repetition
//! - Llama3-style RoPE scaling

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, RmsNorm};

use crate::config::LlamaConfig;
use mimi_rs::gguf_loader::GgufTensors;
use mimi_rs::qlinear::QLinear;

// ---------------------------------------------------------------------------
// RoPE cache
// ---------------------------------------------------------------------------

/// Pre-computed cos/sin tables for rotary position embeddings.
pub struct RopeCache {
    cos: Tensor, // [max_pos, head_dim/2]
    sin: Tensor, // [max_pos, head_dim/2]
}

impl RopeCache {
    /// Build a RoPE cache with Llama3-style frequency scaling.
    pub fn new(cfg: &LlamaConfig, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let half = head_dim / 2;

        // Base inverse frequencies: theta^(-2i/d) for i in 0..half
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / head_dim as f64);
            inv_freq.push(freq);
        }

        // Apply Llama3 RoPE scaling if configured.
        if let Some(ref s) = cfg.rope_scaling {
            let factor = s.factor;
            let low_freq_factor = s.low_freq_factor;
            let high_freq_factor = s.high_freq_factor;
            let old_ctx = s.original_max_position_embeddings as f64;

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
                    // Smooth interpolation between the two regimes
                    let smooth = (old_ctx / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    *freq = (1.0 - smooth) * (*freq / factor) + smooth * *freq;
                }
            }
        }

        let inv_freq_f32: Vec<f32> = inv_freq.iter().map(|&f| f as f32).collect();
        let inv_freq_tensor = Tensor::from_vec(inv_freq_f32, (1, half), device)?;

        // Position indices [0, 1, ..., max_pos-1]
        let max_pos = cfg.max_position_embeddings;
        let positions = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;

        // Outer product: [max_pos, half]
        let freqs = positions.matmul(&inv_freq_tensor)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    /// Slice cos/sin for positions [pos .. pos + seq_len].
    fn get(&self, pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.i(pos..pos + seq_len)?;
        let sin = self.sin.i(pos..pos + seq_len)?;
        Ok((cos, sin))
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// Per-layer key/value cache for autoregressive decoding.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Append new key/value and return the full accumulated tensors.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k_full, v_full) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k_full = Tensor::cat(&[prev_k, k], 2)?;
                let v_full = Tensor::cat(&[prev_v, v], 2)?;
                (k_full, v_full)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k_full.clone());
        self.v = Some(v_full.clone());
        Ok((k_full, v_full))
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Grouped-query attention (GQA) with RoPE.
pub struct CausalSelfAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// Repeat KV heads to match the number of query heads (for GQA).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, hd))?
        .reshape((b, n_kv * n_rep, seq, hd))
}

fn peek_candle(label: &str, t: &Tensor) {
    let shape: Vec<usize> = t.dims().to_vec();
    let flat = t.flatten_all().and_then(|f| f.to_vec1::<f32>()).unwrap_or_default();
    let n = flat.len().min(4);
    println!("  [CANDLE] {label}: shape={shape:?} first4={:?}", &flat[..n]);
}

impl CausalSelfAttention {
    fn load(gguf: &mut GgufTensors, prefix: &str, cfg: &LlamaConfig) -> Result<Self> {
        let q_proj = gguf.qlinear(&format!("{prefix}.q_proj"))?;
        let k_proj = gguf.qlinear(&format!("{prefix}.k_proj"))?;
        let v_proj = gguf.qlinear(&format!("{prefix}.v_proj"))?;
        let o_proj = gguf.qlinear(&format!("{prefix}.o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [b, heads, seq, head_dim]
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;

        // KV cache: append and get full K, V
        let (k, v) = kv_cache.append(&k, &v)?;

        // GQA: repeat KV heads to match Q heads
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        // Causal mask (only needed for prefill, seq_len > 1)
        let total_seq = k.dim(2)?;
        let attn_weights = if seq_len > 1 {
            let mask = create_causal_mask(seq_len, total_seq, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to [b, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    /// Debug forward: identical computation to forward but prints intermediate tensors.
    pub fn forward_debug(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Note: candle transposes first (to [b,heads,seq,hd]) then applies RoPE
        // Burn applies RoPE before swap_dims (on [b,seq,heads,hd]) — same math, different layout
        peek_candle("q_before_rope [b,nh,seq,hd]", &q);
        peek_candle("k_before_rope [b,nkv,seq,hd]", &k);
        peek_candle("v_before_rope [b,nkv,seq,hd]", &v);

        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;

        peek_candle("q_after_rope  [b,nh,seq,hd]", &q);
        peek_candle("k_after_rope  [b,nkv,seq,hd]", &k);

        peek_candle("k_pre_cache   [b,nkv,seq,hd]", &k);
        peek_candle("v_pre_cache   [b,nkv,seq,hd]", &v);

        let (k, v) = kv_cache.append(&k, &v)?;

        peek_candle("k_from_cache  [b,nkv,total,hd]", &k);
        peek_candle("v_from_cache  [b,nkv,total,hd]", &v);

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        peek_candle("k_expand_kv   [b,nh,total,hd]", &k);
        peek_candle("v_expand_kv   [b,nh,total,hd]", &v);

        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        peek_candle("k_t           [b,nh,hd,total]", &k_t);

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_t)? * scale)?;
        peek_candle("scores        [b,nh,seq,total]", &attn_weights);

        let total_seq = k_t.dim(3)?;
        let attn_weights = if seq_len > 1 {
            let mask = create_causal_mask(seq_len, total_seq, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        peek_candle("attn_weights  [b,nh,seq,total]", &attn_weights);

        let attn_output = attn_weights.matmul(&v)?;
        peek_candle("attn_out      [b,nh,seq,hd]", &attn_output);

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;
        peek_candle("attn_out_flat [b,seq,hidden]", &attn_output);

        self.o_proj.forward(&attn_output)
    }
}

/// Create a causal attention mask for prefill.
/// Returns a tensor of shape [1, 1, seq_len, total_seq] with 0 for allowed
/// positions and -inf for masked positions.
fn create_causal_mask(seq_len: usize, total_seq: usize, device: &Device) -> Result<Tensor> {
    // During prefill with KV cache, the new tokens occupy positions
    // [total_seq - seq_len .. total_seq]. Token at row i can attend to
    // columns 0..=(total_seq - seq_len + i).
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            let allowed = total_seq - seq_len + i + 1;
            (0..total_seq).map(move |j| {
                if j < allowed {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, total_seq), device)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl Mlp {
    fn load(gguf: &mut GgufTensors, prefix: &str) -> Result<Self> {
        let gate_proj = gguf.qlinear(&format!("{prefix}.gate_proj"))?;
        let up_proj = gguf.qlinear(&format!("{prefix}.up_proj"))?;
        let down_proj = gguf.qlinear(&format!("{prefix}.down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

/// One transformer layer: pre-norm attention + post-norm MLP.
pub struct Block {
    attn: CausalSelfAttention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attn_layernorm: RmsNorm,
}

impl Block {
    fn forward_debug_layer0(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        println!("  [CANDLE] after_input_layernorm first4={:?}", {
            let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
            v[..v.len().min(4)].to_vec()
        });
        let x = self.attn.forward_debug(&x, cos, sin, kv_cache)?;
        println!("  [CANDLE] after_attn first4={:?}", {
            let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
            v[..v.len().min(4)].to_vec()
        });
        residual + x
    }

    fn load(gguf: &mut GgufTensors, prefix: &str, cfg: &LlamaConfig) -> Result<Self> {
        let attn = CausalSelfAttention::load(gguf, &format!("{prefix}.self_attn"), cfg)?;
        let mlp = Mlp::load(gguf, &format!("{prefix}.mlp"))?;
        let input_ln_w = gguf.tensor(&format!("{prefix}.input_layernorm.weight"))?;
        let post_attn_ln_w = gguf.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let eps = cfg.rms_norm_eps;
        Ok(Self {
            attn,
            mlp,
            input_layernorm: RmsNorm::new(input_ln_w, eps),
            post_attn_layernorm: RmsNorm::new(post_attn_ln_w, eps),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        // Pre-norm attention with residual
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.attn.forward(&x, cos, sin, kv_cache)?;
        let x = (residual + x)?;

        // Post-norm MLP with residual
        let residual = &x;
        let h = self.post_attn_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// Llama 3.2 1B model used as the backbone of TADA TTS.
///
/// The model takes pre-computed input embeddings (token + acoustic + time)
/// and returns hidden states for the diffusion prediction head. It also
/// provides a tied lm_head for text token logits.
pub struct LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    rope: RopeCache,
    kv_caches: Vec<KvCache>,
    /// Separate KV cache for the CFG negative (zero-acoustic) path.
    neg_kv_caches: Vec<KvCache>,
}

impl LlamaModel {
    /// Load the Llama model from GGUF tensors.
    pub fn load_gguf(gguf: &mut GgufTensors, cfg: &LlamaConfig) -> Result<Self> {
        let embed_w = gguf.tensor("model.embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_w, cfg.hidden_size);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(Block::load(gguf, &prefix, cfg)?);
        }

        let norm_w = gguf.tensor("model.norm.weight")?;
        let norm = RmsNorm::new(norm_w, cfg.rms_norm_eps);

        let rope = RopeCache::new(cfg, &gguf.device)?;

        let kv_caches = (0..cfg.num_hidden_layers).map(|_| KvCache::new()).collect();
        let neg_kv_caches = (0..cfg.num_hidden_layers).map(|_| KvCache::new()).collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            kv_caches,
            neg_kv_caches,
        })
    }

    /// Create a minimal placeholder LLM (no weights loaded).
    /// Used in hybrid mode where Burn handles the entire LLM on GPU.
    /// Only safe to call generate_acoustic/decode_audio on the parent TadaModel.
    pub fn load_gguf_adapters_only(gguf: &mut GgufTensors, cfg: &LlamaConfig) -> Result<Self> {
        // Create tiny dummy tensors — these methods will never be called in hybrid mode
        let dummy = Tensor::zeros((1, cfg.hidden_size), DType::F32, &gguf.device)?;
        let embed_tokens = Embedding::new(dummy.clone(), cfg.hidden_size);
        let norm = RmsNorm::new(dummy.squeeze(0)?, cfg.rms_norm_eps);
        let rope = RopeCache::new(cfg, &gguf.device)?;

        Ok(Self {
            embed_tokens,
            layers: Vec::new(),
            norm,
            rope,
            kv_caches: Vec::new(),
            neg_kv_caches: Vec::new(),
        })
    }

    /// Run the transformer on pre-computed input embeddings.
    ///
    /// - `input_embeds`: [batch, seq_len, hidden_size] — combined token + acoustic + time embeddings
    /// - `index_pos`: starting position in the sequence (for RoPE and KV cache)
    ///
    /// Returns hidden states of shape [batch, seq_len, hidden_size].
    pub fn forward(&mut self, input_embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len, _hidden) = input_embeds.dims3()?;
        let (cos, sin) = self.rope.get(index_pos, seq_len)?;

        let mut x = input_embeds.clone();
        for (layer, kv_cache) in self.layers.iter().zip(self.kv_caches.iter_mut()) {
            x = layer.forward(&x, &cos, &sin, kv_cache)?;
        }

        self.norm.forward(&x)
    }

    /// Run the transformer on pre-computed input embeddings for only `num_layers` layers.
    ///
    /// Same as `forward` but stops after `num_layers` transformer layers and skips
    /// the final RMSNorm when `num_layers < total layers`. Useful for per-layer
    /// debugging comparisons.
    pub fn forward_n_layers(&mut self, input_embeds: &Tensor, index_pos: usize, num_layers: usize) -> Result<Tensor> {
        let (_b, seq_len, _hidden) = input_embeds.dims3()?;
        let (cos, sin) = self.rope.get(index_pos, seq_len)?;

        let total_layers = self.layers.len();
        let run_layers = num_layers.min(total_layers);

        let mut x = input_embeds.clone();
        for i in 0..run_layers {
            x = self.layers[i].forward(&x, &cos, &sin, &mut self.kv_caches[i])?;
        }

        if run_layers >= total_layers {
            self.norm.forward(&x)
        } else {
            Ok(x)
        }
    }

    /// Debug forward: runs layer 0 with intermediate tensor dumps, then continues normally.
    ///
    /// Call this on step 2 to pinpoint where Burn diverges from candle at the attention level.
    pub fn forward_debug_layer0(&mut self, input_embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len, _hidden) = input_embeds.dims3()?;
        let (cos, sin) = self.rope.get(index_pos, seq_len)?;

        let mut x = input_embeds.clone();

        // Layer 0: run with debug
        println!("[CANDLE] === Layer 0 debug ===");
        {
            let cache = &mut self.kv_caches[0];
            x = self.layers[0].forward_debug_layer0(&x, &cos, &sin, cache)?;
        }
        println!("[CANDLE] after layer 0 residual first4={:?}", {
            let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
            v[..v.len().min(4)].to_vec()
        });

        // Remaining layers: normal forward
        for i in 1..self.layers.len() {
            x = self.layers[i].forward(&x, &cos, &sin, &mut self.kv_caches[i])?;
        }

        self.norm.forward(&x)
    }

    /// Look up token embeddings by ID.
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    /// Tied lm_head: hidden_states @ embed_tokens.weight^T → logits.
    ///
    /// - `hidden`: [batch, seq_len, hidden_size]
    ///
    /// Returns logits of shape [batch, seq_len, vocab_size].
    pub fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let embed_w = self.embed_tokens.embeddings(); // [vocab_size, hidden_size]
        hidden.broadcast_matmul(&embed_w.t()?)
    }

    /// Run the transformer using the negative (zero-acoustic) KV cache.
    ///
    /// Uses the same weights as `forward` but writes to `neg_kv_caches` instead
    /// of `kv_caches`. This allows CFG to maintain independent KV histories for
    /// the positive (real acoustic) and negative (zero acoustic) paths.
    ///
    /// - `input_embeds`: [batch, seq_len, hidden_size]
    /// - `index_pos`: starting position in the sequence (same as the pos path)
    ///
    /// Returns hidden states of shape [batch, seq_len, hidden_size].
    pub fn forward_neg(&mut self, input_embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len, _hidden) = input_embeds.dims3()?;
        let (cos, sin) = self.rope.get(index_pos, seq_len)?;

        let mut x = input_embeds.clone();
        for (layer, kv_cache) in self.layers.iter().zip(self.neg_kv_caches.iter_mut()) {
            x = layer.forward(&x, &cos, &sin, kv_cache)?;
        }

        self.norm.forward(&x)
    }

    /// Clear all KV caches (call between sequences).
    pub fn clear_kv_cache(&mut self) {
        for cache in self.kv_caches.iter_mut() {
            cache.clear();
        }
        for cache in self.neg_kv_caches.iter_mut() {
            cache.clear();
        }
    }
}
