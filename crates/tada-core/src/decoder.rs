use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Module};
use mimi_rs::gguf_loader::GgufTensors;
use mimi_rs::qlinear::QLinear;
use mimi_rs::rope::RotaryEmbedding;

use crate::config::DecoderConfig;

// ---------------------------------------------------------------------------
// Snake1d activation: x + (1/alpha) * sin²(alpha * x)
// ---------------------------------------------------------------------------

struct Snake1d {
    alpha: Tensor, // [1, C, 1]
}

impl Snake1d {
    fn load_gguf(gguf: &mut GgufTensors, name: &str) -> Result<Self> {
        let alpha = gguf.tensor(name)?;
        Ok(Self { alpha })
    }

    /// x: [B, C, T]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // sin²(alpha * x) / alpha
        let ax = x.broadcast_mul(&self.alpha)?;
        let sin_ax = ax.sin()?;
        let sin2 = sin_ax.sqr()?;
        let alpha_recip = self.alpha.recip()?;
        let term = sin2.broadcast_mul(&alpha_recip)?;
        x + &term
    }
}

// ---------------------------------------------------------------------------
// ResidualUnit: Snake1d → Conv1d(k=7, dilation) → Snake1d → Conv1d(k=1) + skip
// ---------------------------------------------------------------------------

struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d, // k=7, dilated
    snake2: Snake1d,
    conv2: Conv1d, // k=1
}

impl ResidualUnit {
    fn load_gguf(gguf: &mut GgufTensors, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        let snake1 = Snake1d::load_gguf(gguf, &format!("{prefix}.block.0.alpha"))?;
        let snake2 = Snake1d::load_gguf(gguf, &format!("{prefix}.block.2.alpha"))?;

        // Dilated conv: k=7, padding = ((7-1)*dilation)/2
        let pad = ((7 - 1) * dilation) / 2;
        let w1 = gguf.tensor(&format!("{prefix}.block.1.weight"))?;
        let b1 = gguf.tensor(&format!("{prefix}.block.1.bias"))?;
        let conv1 = Conv1d::new(
            w1,
            Some(b1),
            Conv1dConfig { padding: pad, stride: 1, dilation, groups: 1, ..Default::default() },
        );

        // Pointwise conv: k=1
        let w2 = gguf.tensor(&format!("{prefix}.block.3.weight"))?;
        let b2 = gguf.tensor(&format!("{prefix}.block.3.bias"))?;
        let conv2 = Conv1d::new(
            w2,
            Some(b2),
            Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, ..Default::default() },
        );

        let _ = channels; // shapes baked into weights
        Ok(Self { snake1, conv1, snake2, conv2 })
    }

    /// x: [B, C, T]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.snake2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        x + &h
    }
}

// ---------------------------------------------------------------------------
// DecoderBlock: Snake1d → ConvTranspose1d(stride) → 3×ResidualUnit
// ---------------------------------------------------------------------------

struct DecoderBlock {
    snake: Snake1d,
    conv_tr: ConvTranspose1d,
    res_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let snake = Snake1d::load_gguf(gguf, &format!("{prefix}.block.0.alpha"))?;

        // ConvTranspose1d: kernel = 2*stride, padding to trim output
        let w = gguf.tensor(&format!("{prefix}.block.1.weight"))?;
        let b = gguf.tensor(&format!("{prefix}.block.1.bias"))?;
        // DAC uses padding = ceil(stride/2) on each side, which for ConvTranspose1d
        // means padding = stride/2 (integer div) and output_padding to fix length.
        // Actually: ConvTranspose1d with kernel=2*stride, stride=stride, padding=stride//2
        // gives output length = (L-1)*stride + 2*stride - 2*(stride//2)
        // For even stride: padding = stride/2, output_padding = 0
        // This gives output = (L-1)*stride + 2*stride - stride = L*stride ✓
        let padding = (stride + 1) / 2; // ceil(stride / 2)
        let conv_tr = ConvTranspose1d::new(
            w,
            Some(b),
            ConvTranspose1dConfig {
                padding,
                stride,
                output_padding: 0,
                dilation: 1,
                groups: 1,
            },
        );

        // 3 residual units with dilations [1, 3, 9]
        let dilations = [1, 3, 9];
        let mut res_units = Vec::with_capacity(3);
        for (j, &d) in dilations.iter().enumerate() {
            // block indices: 2, 3, 4 (after snake=0, convtr=1)
            let ru = ResidualUnit::load_gguf(
                gguf,
                &format!("{prefix}.block.{}", j + 2),
                out_channels,
                d,
            )?;
            res_units.push(ru);
        }

        let _ = (in_channels, out_channels); // shapes baked into weights
        Ok(Self { snake, conv_tr, res_units })
    }

    /// x: [B, C_in, T] → [B, C_out, T*stride]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let h = self.conv_tr.forward(&h)?;
        let mut h = h;
        for ru in &self.res_units {
            h = ru.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// DACDecoder: Conv1d → 4×DecoderBlock → Snake1d → Conv1d → Tanh
// ---------------------------------------------------------------------------

struct DACDecoder {
    initial_conv: Conv1d,
    blocks: Vec<DecoderBlock>,
    final_snake: Snake1d,
    final_conv: Conv1d,
}

impl DACDecoder {
    fn load_gguf(gguf: &mut GgufTensors, prefix: &str, cfg: &DecoderConfig) -> Result<Self> {
        // Initial conv: Conv1d(hidden_dim, channels, k=7, pad=3)
        let w0 = gguf.tensor(&format!("{prefix}.model.0.weight"))?;
        let b0 = gguf.tensor(&format!("{prefix}.model.0.bias"))?;
        let initial_conv = Conv1d::new(
            w0,
            Some(b0),
            Conv1dConfig { padding: 3, stride: 1, dilation: 1, groups: 1, ..Default::default() },
        );

        // Decoder blocks with channel progression: channels → channels/2 → channels/4 → ...
        let mut blocks = Vec::with_capacity(cfg.strides.len());
        let mut c = cfg.wav_decoder_channels; // 1536
        for (i, &stride) in cfg.strides.iter().enumerate() {
            let c_out = c / 2;
            let block = DecoderBlock::load_gguf(
                gguf,
                &format!("{prefix}.model.{}", i + 1),
                c,
                c_out,
                stride,
            )?;
            blocks.push(block);
            c = c_out;
        }
        // c is now 96 (1536 / 2^4)

        // Final snake + conv
        let final_snake = Snake1d::load_gguf(gguf, &format!("{prefix}.model.5.alpha"))?;

        let w_final = gguf.tensor(&format!("{prefix}.model.6.weight"))?;
        let b_final = gguf.tensor(&format!("{prefix}.model.6.bias"))?;
        let final_conv = Conv1d::new(
            w_final,
            Some(b_final),
            Conv1dConfig { padding: 3, stride: 1, dilation: 1, groups: 1, ..Default::default() },
        );

        Ok(Self { initial_conv, blocks, final_snake, final_conv })
    }

    /// x: [B, hidden_dim, T] → [B, 1, T*480]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.initial_conv.forward(x)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        h = self.final_snake.forward(&h)?;
        h = self.final_conv.forward(&h)?;
        h.tanh()
    }
}

// ---------------------------------------------------------------------------
// LocalSelfAttention: QKV → split heads → RoPE → SDPA → out_proj
// ---------------------------------------------------------------------------

struct LocalSelfAttention {
    qkv: QLinear,
    out_proj: QLinear,
    layer_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl LocalSelfAttention {
    fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let qkv = gguf.qlinear(&format!("{prefix}.qkv"))?;
        let out_proj = gguf.qlinear(&format!("{prefix}.out_proj"))?;

        let ln_w = gguf.tensor(&format!("{prefix}.layer_norm.weight"))?;
        let ln_b = gguf.tensor(&format!("{prefix}.layer_norm.bias"))?;
        let layer_norm = LayerNorm::new(ln_w, ln_b, 1e-5);

        let head_dim = hidden_dim / num_heads;
        Ok(Self { qkv, out_proj, layer_norm, num_heads, head_dim })
    }

    /// x: [B, T, D], mask: [T, T] (additive, 0 = attend, -inf = block)
    /// rope: shared RoPE embeddings
    fn forward(&self, x: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let residual = x;

        // No pre-norm: compute QKV on raw input (post-norm architecture)
        let (b, t, _d) = x.dims3()?;

        // QKV projection → [B, T, 3*D]
        let qkv = self.qkv.forward(&x)?;

        // Split into Q, K, V each [B, T, D]
        let d = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, d)?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?;

        // Reshape to [B, T, H, head_dim]
        let q = q.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?;

        // Apply RoPE (expects [B, T, H, D])
        let (q, k) = rope.forward(&q, &k, 0)?;

        // Transpose to [B, H, T, head_dim] for attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // Apply additive mask: [T, T] → broadcast to [B, H, T, T]
        let attn = attn.broadcast_add(mask)?;

        // Softmax over last dim
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // Weighted sum: [B, H, T, head_dim]
        let out = attn.matmul(&v)?;

        // Transpose back to [B, T, H, head_dim] → reshape [B, T, D]
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;

        // Output projection
        let out = self.out_proj.forward(&out)?;

        // Post-norm: LayerNorm(residual + out)
        self.layer_norm.forward(&(residual + &out)?)
    }
}

// ---------------------------------------------------------------------------
// LocalAttentionEncoderLayer: self_attn + FFN with residual + LayerNorm
// ---------------------------------------------------------------------------

struct LocalAttentionEncoderLayer {
    self_attn: LocalSelfAttention,
    ffn_linear1: QLinear,
    ffn_linear2: QLinear,
    norm: LayerNorm,
}

impl LocalAttentionEncoderLayer {
    fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let self_attn = LocalSelfAttention::load_gguf(
            gguf,
            &format!("{prefix}.self_attn"),
            hidden_dim,
            num_heads,
        )?;

        let ffn_linear1 = gguf.qlinear(&format!("{prefix}.ffn.0"))?;
        let ffn_linear2 = gguf.qlinear(&format!("{prefix}.ffn.3"))?;

        let norm_w = gguf.tensor(&format!("{prefix}.norm.weight"))?;
        let norm_b = gguf.tensor(&format!("{prefix}.norm.bias"))?;
        let norm = LayerNorm::new(norm_w, norm_b, 1e-5);

        Ok(Self { self_attn, ffn_linear1, ffn_linear2, norm })
    }

    /// x: [B, T, D], mask: [T, T]
    fn forward(&self, x: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        // Self-attention with post-norm (handled inside LocalSelfAttention)
        let x = self.self_attn.forward(&x, mask, rope)?;

        // FFN with post-norm: LayerNorm(x + FFN(x))
        let ffn_out = self.ffn_linear1.forward(&x)?;
        let ffn_out = ffn_out.gelu()?;
        let ffn_out = self.ffn_linear2.forward(&ffn_out)?;
        self.norm.forward(&(&x + &ffn_out)?)
    }
}

// ---------------------------------------------------------------------------
// LocalAttentionEncoder: N layers + final LayerNorm
// ---------------------------------------------------------------------------

struct LocalAttentionEncoder {
    layers: Vec<LocalAttentionEncoderLayer>,
    final_norm: LayerNorm,
    rope: RotaryEmbedding,
}

impl LocalAttentionEncoder {
    fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        cfg: &DecoderConfig,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_attn_layers);
        for i in 0..cfg.num_attn_layers {
            let layer = LocalAttentionEncoderLayer::load_gguf(
                gguf,
                &format!("{prefix}.layers.{i}"),
                cfg.hidden_dim,
                cfg.num_attn_heads,
            )?;
            layers.push(layer);
        }

        let fn_w = gguf.tensor(&format!("{prefix}.final_norm.weight"))?;
        let fn_b = gguf.tensor(&format!("{prefix}.final_norm.bias"))?;
        let final_norm = LayerNorm::new(fn_w, fn_b, 1e-5);

        let head_dim = cfg.hidden_dim / cfg.num_attn_heads; // 1024/8 = 128
        // Pre-compute RoPE table for a reasonable max length; will fall back to
        // on-the-fly computation for longer sequences.
        let rope = RotaryEmbedding::new(head_dim, 8192, 10000.0, &gguf.device)?;

        Ok(Self { layers, final_norm, rope })
    }

    /// x: [B, T, D], mask: [T, T]
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, mask, &self.rope)?;
        }
        self.final_norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Segment attention mask v2
// ---------------------------------------------------------------------------

/// Create a segment-based attention mask (v2) from token_masks.
///
/// token_masks: [B, T] — 1 where a real acoustic token is, 0 for padding.
/// In v2, block_ids = cumsum(token_masks). Position i can attend to position j if:
/// - Same block: block_ids[j] == block_ids[i]
/// - Previous block: block_ids[j] == block_ids[i] - 1
/// - Causal: j <= i
/// - Special rule for marked positions (token_masks[i] == 1): they start a new
///   block, so they can also attend to the entire previous block.
///
/// Returns additive mask [T, T]: 0.0 = attend, -inf = block.
fn create_segment_attention_mask_v2(
    token_masks: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    // Work on CPU for indexing, move result to target device at the end.
    let masks_vec = token_masks.squeeze(0)?.to_vec1::<f32>()?;
    let t = masks_vec.len();

    // block_ids = cumsum(token_masks) - token_masks
    // This ensures marked positions (mask=1) belong to the same block as
    // the preceding padding positions, so blocks end at marked positions.
    let mut block_ids = Vec::with_capacity(t);
    let mut cumsum = 0i64;
    for &m in &masks_vec {
        cumsum += if m > 0.5 { 1 } else { 0 };
        let subtract = if m > 0.5 { 1 } else { 0 };
        block_ids.push(cumsum - subtract);
    }

    // Build mask: [T, T] — NOT causal.
    // Position i can attend to position j (forward or backward) if
    // block_ids[j] == block_ids[i] (same block) or
    // block_ids[j] == block_ids[i] - 1 (previous block).
    let mut mask_data = vec![f32::NEG_INFINITY; t * t];
    for i in 0..t {
        let bid_i = block_ids[i];
        for j in 0..t {
            let bid_j = block_ids[j];
            if bid_j == bid_i || bid_j == bid_i - 1 {
                mask_data[i * t + j] = 0.0;
            }
        }
    }

    Tensor::from_vec(mask_data, (t, t), device)
}

// ---------------------------------------------------------------------------
// Full Decoder: decoder_proj → local_attention_decoder → wav_decoder
// ---------------------------------------------------------------------------

/// TADA codec decoder: converts acoustic features to 24kHz PCM audio.
///
/// Architecture:
/// 1. Linear projection from embed_dim (512) to hidden_dim (1024)
/// 2. Local self-attention encoder (6 layers, 8 heads, RoPE)
/// 3. DAC-style CNN decoder with Snake activations (strides [4,4,5,6] → 480x upsample)
pub struct Decoder {
    decoder_proj: QLinear,
    local_attention_decoder: LocalAttentionEncoder,
    wav_decoder: DACDecoder,
}

impl Decoder {
    /// Load the decoder from GGUF tensors.
    ///
    /// `prefix` should be `"_decoder"` to match the tensor names like
    /// `_decoder.decoder_proj.weight`, etc.
    pub fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        cfg: &DecoderConfig,
    ) -> Result<Self> {
        let decoder_proj = gguf.qlinear(&format!("{prefix}.decoder_proj"))?;
        let local_attention_decoder = LocalAttentionEncoder::load_gguf(
            gguf,
            &format!("{prefix}.local_attention_decoder"),
            cfg,
        )?;
        let wav_decoder = DACDecoder::load_gguf(
            gguf,
            &format!("{prefix}.wav_decoder"),
            cfg,
        )?;

        Ok(Self { decoder_proj, local_attention_decoder, wav_decoder })
    }

    /// Decode acoustic features to PCM audio.
    ///
    /// # Arguments
    /// - `encoded_expanded` — duration-expanded acoustic features, [B, T, embed_dim]
    ///   where T includes zero-padded frames for duration expansion.
    /// - `token_masks` — [B, T], 1.0 where real acoustic token, 0.0 for padding.
    ///
    /// # Returns
    /// PCM audio tensor [B, 1, samples] where samples = T * 480.
    pub fn forward(&self, encoded_expanded: &Tensor, token_masks: &Tensor) -> Result<Tensor> {
        // 1. Project to hidden dim: [B, T, embed_dim] → [B, T, hidden_dim]
        let decoder_input = self.decoder_proj.forward(encoded_expanded)?;

        // 2. Create segment attention mask from token_masks (v2)
        let device = decoder_input.device();
        let mask = create_segment_attention_mask_v2(token_masks, device)?;

        // 3. Local attention decoder: [B, T, hidden_dim] → [B, T, hidden_dim]
        let decoded = self.local_attention_decoder.forward(&decoder_input, &mask)?;

        // 4. Transpose for 1D convolutions: [B, T, hidden_dim] → [B, hidden_dim, T]
        let x = decoded.transpose(1, 2)?;

        // 5. WAV decoder (CNN upsample): [B, hidden_dim, T] → [B, 1, T*480]
        self.wav_decoder.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Duration expansion utility
// ---------------------------------------------------------------------------

/// Expand acoustic features according to predicted durations.
///
/// For each acoustic vector at position i with `time_before[i] = T`, insert
/// `T - 1` zero frames before it (so the real frame sits at the end of a
/// T-frame window). The last `time_before` value adds trailing zeros.
///
/// # Arguments
/// - `acoustics` — [N, embed_dim] acoustic vectors (no batch dim)
/// - `time_before` — [N] integer durations (how many frames each token spans)
///
/// # Returns
/// `(expanded, token_masks)`:
/// - `expanded` — [1, total_frames, embed_dim]
/// - `token_masks` — [1, total_frames] with 1.0 at real acoustic positions, 0.0 elsewhere
/// Expand acoustic features according to predicted durations.
///
/// Matches Python's `_decode_wav`:
///
///   For each acoustic frame `i` (0..N):
///     - Insert `max(time_before[i] - 1, 0)` zero frames
///     - Insert the acoustic frame itself
///   After all frames:
///     - Insert `time_before[N]` trailing zero frames (optional extra element)
///
/// # Arguments
/// - `acoustics` — `[N, embed_dim]` acoustic vectors
/// - `time_before` — `[N]` or `[N+1]` integer durations. If `N+1`, the
///   last element adds trailing zero frames (matching Python).
///
/// # Returns
/// `(expanded, token_masks)`:
/// - `expanded` — `[1, total_frames, embed_dim]`
/// - `token_masks` — `[1, total_frames]` with 1.0 at non-zero positions
pub fn expand_durations(
    acoustics: &Tensor,
    time_before: &[u32],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let n_acoustics = acoustics.dim(0)?;
    let embed_dim = acoustics.dim(D::Minus1)?;
    let acoustics_data = acoustics.to_vec2::<f32>()?;

    // Build expanded sequence matching Python's _decode_wav exactly:
    //   for pos in range(N):
    //       zeros(max(time_before[pos] - 1, 0), dim)
    //       encoded[pos]
    //   zeros(time_before[N], dim)  # trailing
    let mut frames: Vec<f32> = Vec::new();
    let mut masks: Vec<f32> = Vec::new();

    for i in 0..n_acoustics {
        let tb = time_before.get(i).copied().unwrap_or(1) as usize;
        // Insert (tb - 1) zero frames before the acoustic frame
        let n_zeros = if tb > 1 { tb - 1 } else { 0 };
        for _ in 0..n_zeros {
            frames.extend(std::iter::repeat(0.0f32).take(embed_dim));
            masks.push(0.0);
        }
        // Insert the acoustic frame
        frames.extend_from_slice(&acoustics_data[i]);
        masks.push(1.0);
    }

    // Trailing zero frames (from extra time_before element, if present)
    if time_before.len() > n_acoustics {
        let trailing = time_before[n_acoustics] as usize;
        for _ in 0..trailing {
            frames.extend(std::iter::repeat(0.0f32).take(embed_dim));
            masks.push(0.0);
        }
    }

    let total = masks.len();
    let expanded = Tensor::from_vec(frames, (1, total, embed_dim), device)?;
    let masks = Tensor::from_vec(masks, (1, total), device)?;

    // Override masks: Python uses norm-based detection
    // token_masks = (norm(expanded, dim=-1) != 0).long()
    // Our manual mask is equivalent since we only place non-zero data at acoustic positions.

    Ok((expanded, masks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_snake1d_forward() -> Result<()> {
        let device = Device::Cpu;
        let alpha = Tensor::ones((1, 4, 1), DType::F32, &device)?;
        let snake = Snake1d { alpha };
        let x = Tensor::randn(0f32, 1.0, (1, 4, 16), &device)?;
        let y = snake.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4, 16]);
        // Snake output should be >= x (since sin²/alpha >= 0)
        let diff: f32 = (&y - &x)?.min(D::Minus1)?.min(D::Minus1)?.min(D::Minus1)?.to_scalar()?;
        assert!(diff >= -1e-6, "Snake output should be >= x, got min diff {diff}");
        Ok(())
    }

    #[test]
    fn test_segment_mask_v2_simple() -> Result<()> {
        let device = Device::Cpu;
        // 5 frames: real at 0, pad, pad, real at 3, pad
        // token_masks = [1, 0, 0, 1, 0]
        // block_ids =   [1, 1, 1, 2, 2]
        let masks = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 0.0], (1, 5), &device)?;
        let mask = create_segment_attention_mask_v2(&masks, &device)?;
        let m = mask.to_vec2::<f32>()?;

        // Position 0 (block 1): can attend to itself (block 1) and block 0 (=0, nonexistent)
        assert_eq!(m[0][0], 0.0); // self
        assert!(m[0][1].is_infinite()); // future = blocked by causal

        // Position 3 (block 2): can attend to block 2 and block 1
        assert_eq!(m[3][0], 0.0); // block 1
        assert_eq!(m[3][1], 0.0); // block 1
        assert_eq!(m[3][2], 0.0); // block 1
        assert_eq!(m[3][3], 0.0); // block 2 (self)

        // Position 4 (block 2): can attend to block 2 and block 1
        assert_eq!(m[4][0], 0.0); // block 1
        assert_eq!(m[4][3], 0.0); // block 2
        assert_eq!(m[4][4], 0.0); // block 2 (self)

        Ok(())
    }

    #[test]
    fn test_expand_durations() -> Result<()> {
        let device = Device::Cpu;
        // 3 acoustic vectors with embed_dim=2
        let acoustics = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3, 2),
            &device,
        )?;
        let time_before = &[2u32, 1, 3]; // total = 6 frames

        let (expanded, masks) = expand_durations(&acoustics, time_before, &device)?;

        assert_eq!(expanded.dims(), &[1, 6, 2]);
        assert_eq!(masks.dims(), &[1, 6]);

        let e = expanded.squeeze(0)?.to_vec2::<f32>()?;
        let m = masks.squeeze(0)?.to_vec1::<f32>()?;

        // Frame 0: zero (padding for first token's duration=2)
        assert_eq!(e[0], &[0.0, 0.0]);
        assert_eq!(m[0], 0.0);

        // Frame 1: first acoustic [1, 2] (real pos = 0 + 2 - 1 = 1)
        assert_eq!(e[1], &[1.0, 2.0]);
        assert_eq!(m[1], 1.0);

        // Frame 2: second acoustic [3, 4] (duration=1, real pos = 2 + 1 - 1 = 2)
        assert_eq!(e[2], &[3.0, 4.0]);
        assert_eq!(m[2], 1.0);

        // Frame 3-4: zero padding for third token's duration=3
        assert_eq!(e[3], &[0.0, 0.0]);
        assert_eq!(m[3], 0.0);
        assert_eq!(e[4], &[0.0, 0.0]);
        assert_eq!(m[4], 0.0);

        // Frame 5: third acoustic [5, 6] (real pos = 3 + 3 - 1 = 5)
        assert_eq!(e[5], &[5.0, 6.0]);
        assert_eq!(m[5], 1.0);

        Ok(())
    }

    #[test]
    fn test_segment_mask_all_real() -> Result<()> {
        let device = Device::Cpu;
        // All positions are real tokens — each is its own block
        let masks = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (1, 3), &device)?;
        let mask = create_segment_attention_mask_v2(&masks, &device)?;
        let m = mask.to_vec2::<f32>()?;

        // block_ids = [1, 2, 3]
        // Pos 0 (block 1): attends to block 1 and block 0 (none) → only self
        assert_eq!(m[0][0], 0.0);

        // Pos 1 (block 2): attends to block 2 and block 1 → pos 0 and 1
        assert_eq!(m[1][0], 0.0);
        assert_eq!(m[1][1], 0.0);

        // Pos 2 (block 3): attends to block 3 and block 2 → pos 1 and 2
        assert!(m[2][0].is_infinite()); // block 1, not adjacent
        assert_eq!(m[2][1], 0.0);
        assert_eq!(m[2][2], 0.0);

        Ok(())
    }
}
