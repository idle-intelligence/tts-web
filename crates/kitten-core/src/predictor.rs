use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

fn depthwise_conv_transpose1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    padding: usize,
    stride: usize,
    output_padding: usize,
) -> Result<Tensor> {
    let channels = x.dim(1)?;
    let mut outputs = Vec::with_capacity(channels);
    for c in 0..channels {
        let xc = x.i((.., c..c + 1, ..))?.contiguous()?;
        let wc = weight.i(c..c + 1)?.contiguous()?; // [1, 1, k]
        let out_c = xc.conv_transpose1d(&wc, padding, output_padding, stride, 1, 1)?;
        outputs.push(out_c);
    }
    let out = Tensor::cat(&outputs, 1)?;
    if let Some(b) = bias {
        Ok(out.broadcast_add(&b.reshape((1, channels, 1))?)?)
    } else {
        Ok(out)
    }
}

use crate::config::KittenConfig;

// ---------------------------------------------------------------------------
// BiLSTM (same structure as in text_encoder — duplicated until we refactor)
// ---------------------------------------------------------------------------

struct BiLstm {
    // Forward direction: W [4*H, I], R [4*H, H], B [8*H]
    w_fwd: Tensor,
    r_fwd: Tensor,
    b_fwd: Tensor,
    // Backward direction
    w_bwd: Tensor,
    r_bwd: Tensor,
    b_bwd: Tensor,
    hidden_size: usize,
}

impl BiLstm {
    fn load(vb: VarBuilder, input_size: usize, hidden_size: usize) -> Result<Self> {
        // ONNX packs both directions: W [2, 4H, I], R [2, 4H, H], B [2, 8H]
        let w = vb.get((2, 4 * hidden_size, input_size), "W")?;
        let r = vb.get((2, 4 * hidden_size, hidden_size), "R")?;
        let b = vb.get((2, 8 * hidden_size), "B")?;

        let w_fwd = w.i(0)?.contiguous()?; // [4H, I]
        let w_bwd = w.i(1)?.contiguous()?;
        let r_fwd = r.i(0)?.contiguous()?; // [4H, H]
        let r_bwd = r.i(1)?.contiguous()?;
        let b_fwd = b.i(0)?.contiguous()?; // [8H]
        let b_bwd = b.i(1)?.contiguous()?;

        Ok(Self { w_fwd, r_fwd, b_fwd, w_bwd, r_bwd, b_bwd, hidden_size })
    }

    // x: [batch, seq, input_size] → [batch, seq, 2*hidden_size]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fwd = self.run_direction(x, &self.w_fwd, &self.r_fwd, &self.b_fwd, false)?;
        let bwd = self.run_direction(x, &self.w_bwd, &self.r_bwd, &self.b_bwd, true)?;
        Ok(Tensor::cat(&[fwd, bwd], 2)?)
    }

    fn run_direction(
        &self,
        x: &Tensor,
        w: &Tensor, // [4H, I]
        r: &Tensor, // [4H, H]
        b: &Tensor, // [8H]
        reverse: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let h = self.hidden_size;
        let dev = x.device();
        let dtype = x.dtype();

        // Split biases: wb [4H], rb [4H]
        let wb = b.i(..4 * h)?;
        let rb = b.i(4 * h..)?;

        let mut hidden = Tensor::zeros((batch, h), dtype, dev)?;
        let mut cell = Tensor::zeros((batch, h), dtype, dev)?;

        let mut outputs = Vec::with_capacity(seq_len);

        for step in 0..seq_len {
            let t = if reverse { seq_len - 1 - step } else { step };
            // x_t: [batch, input_size]
            let x_t = x.i((.., t, ..))?.contiguous()?;

            // gates = x_t @ W^T + wb + hidden @ R^T + rb
            let gates = (x_t.matmul(&w.t()?)?.broadcast_add(&wb)?
                + hidden.matmul(&r.t()?)?.broadcast_add(&rb)?)?;

            // ONNX gate order: i, o, f, c
            let i_gate = gates.i((.., 0..h))?;
            let o_gate = gates.i((.., h..2 * h))?;
            let f_gate = gates.i((.., 2 * h..3 * h))?;
            let c_gate = gates.i((.., 3 * h..4 * h))?;

            let i = sigmoid(&i_gate)?;
            let o = sigmoid(&o_gate)?;
            let f = sigmoid(&f_gate)?;
            let g = c_gate.tanh()?;

            cell = (f.mul(&cell)? + i.mul(&g)?)?;
            hidden = o.mul(&cell.tanh()?)?;

            outputs.push(hidden.clone());
        }

        if reverse {
            outputs.reverse();
        }

        // Stack [batch, h] → [batch, seq, h]
        let stacked = Tensor::stack(&outputs, 1)?;
        Ok(stacked)
    }
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::sigmoid(x)?)
}

// ---------------------------------------------------------------------------
// InstanceNorm1d (per-channel, over the time dim)
// Expects x: [batch, channels, time]
// ---------------------------------------------------------------------------
fn instance_norm(x: &Tensor) -> Result<Tensor> {
    let (_b, _c, t) = x.dims3()?;
    if t == 0 {
        return Ok(x.clone());
    }
    let mean = x.mean_keepdim(2)?;
    let var = (x.broadcast_sub(&mean)?.powf(2.0)?).mean_keepdim(2)?;
    let eps = 1e-5_f64;
    Ok(x.broadcast_sub(&mean)?.broadcast_div(&(var + eps)?.sqrt()?)?)
}

fn leaky_relu_02(x: &Tensor) -> Result<Tensor> {
    let neg = (x * 0.2_f64)?;
    Ok(x.maximum(&neg)?)
}

// ---------------------------------------------------------------------------
// AdaIN: instance_norm → gamma*x + beta, where [gamma, beta] = fc(style).split()
// norm_weight/norm_bias are Optional (affine=True/False instance norm).
// ---------------------------------------------------------------------------
struct AdaIn {
    fc: Linear,
    norm_weight: Option<Tensor>,
    norm_bias: Option<Tensor>,
}

impl AdaIn {
    // Auto-detect affine: try loading norm.weight + norm.bias, skip if missing.
    fn load(vb: VarBuilder, style_in: usize, channels: usize) -> Result<Self> {
        let fc = linear(style_in, channels * 2, vb.pp("fc"))?;
        let norm_weight = vb.pp("norm").get(channels, "weight").ok();
        let norm_bias = vb.pp("norm").get(channels, "bias").ok();
        Ok(Self { fc, norm_weight, norm_bias })
    }

    // x: [batch, channels, T], style: [batch, style_in] → [batch, channels, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        // instance norm (per-channel, over time)
        let normed = if let (Some(w), Some(b)) = (&self.norm_weight, &self.norm_bias) {
            let n = instance_norm(x)?;
            // w, b: [C] → reshape to [1, C, 1] for broadcast
            let w = w.reshape((1, x.dim(1)?, 1))?;
            let b = b.reshape((1, x.dim(1)?, 1))?;
            n.broadcast_mul(&w)?.broadcast_add(&b)?
        } else {
            instance_norm(x)?
        };

        // AdaIN modulation: gamma = fc[:C] + 1, beta = fc[C:]
        let proj = self.fc.forward(style)?; // [batch, 2*C]
        let c = x.dim(1)?;
        let gamma_raw = proj.i((.., ..c))?.unsqueeze(2)?; // [batch, C, 1]
        let beta = proj.i((.., c..))?.unsqueeze(2)?;
        let gamma = (gamma_raw + 1.0_f64)?;
        Ok(normed.broadcast_mul(&gamma)?.broadcast_add(&beta)?)
    }
}

// ---------------------------------------------------------------------------
// Block 0: 128ch → 128ch (identity residual)
// norm1 has affine, norm2 does NOT have affine
// ---------------------------------------------------------------------------
struct Block0 {
    norm1: AdaIn,
    conv1: Conv1d,
    norm2: AdaIn,
    conv2: Conv1d,
}

impl Block0 {
    fn load(vb: VarBuilder, style_half: usize, ch: usize) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            norm1: AdaIn::load(vb.pp("norm1"), style_half, ch)?,
            conv1: conv1d(ch, ch, 3, cfg, vb.pp("conv1"))?,
            norm2: AdaIn::load(vb.pp("norm2"), style_half, ch)?,
            conv2: conv1d(ch, ch, 3, cfg, vb.pp("conv2"))?,
        })
    }

    // x: [batch, ch, T] → [batch, ch, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv2.forward(&h)?;
        // residual + /2
        Ok(((x + h)? / 2.0)?)
    }
}

// ---------------------------------------------------------------------------
// Block 1: 128ch → 64ch (conv1x1 residual skip, has pool conv)
// norm1 has NO affine (norm1.norm.* absent), norm2 HAS affine
// ---------------------------------------------------------------------------
struct Block1 {
    conv1x1: Conv1d,
    norm1: AdaIn,
    conv1: Conv1d,
    norm2: AdaIn,
    conv2: Conv1d,
    pool_weight: Tensor,
    pool_bias: Tensor,
}

impl Block1 {
    fn load(vb: VarBuilder, style_half: usize, in_ch: usize, out_ch: usize) -> Result<Self> {
        let cfg3 = Conv1dConfig { padding: 1, ..Default::default() };
        let cfg1 = Conv1dConfig { ..Default::default() };
        // conv1x1: residual skip (no bias — load weight manually)
        let conv1x1_w = vb.get((out_ch, in_ch, 1), "conv1x1.weight")?;
        let conv1x1 = Conv1d::new(conv1x1_w, None, cfg1);
        // pool: depthwise conv, weight [in_ch, 1, 3] — stored as tensors, applied via depthwise_conv1d
        let pool_weight = vb.get((in_ch, 1, 3), "pool.weight")?;
        let pool_bias = vb.get(in_ch, "pool.bias")?;
        Ok(Self {
            conv1x1,
            norm1: AdaIn::load(vb.pp("norm1"), style_half, in_ch)?,
            conv1: conv1d(in_ch, out_ch, 3, cfg3, vb.pp("conv1"))?,
            norm2: AdaIn::load(vb.pp("norm2"), style_half, out_ch)?,
            conv2: conv1d(out_ch, out_ch, 3, cfg3, vb.pp("conv2"))?,
            pool_weight,
            pool_bias,
        })
    }

    // x: [batch, in_ch, T] → [batch, out_ch, 2T]
    // The pool is a depthwise ConvTranspose1d (stride=2) that upsamples T→2T
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Main path with channel reduction
        let skip = self.conv1x1.forward(x)?; // [batch, out_ch, T]
        let h = self.norm1.forward(x, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv1.forward(&h)?;     // in_ch→out_ch
        let h = self.norm2.forward(&h, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv2.forward(&h)?;     // out_ch→out_ch
        let out = (skip + h)?;               // [batch, out_ch, T]

        // Pool: depthwise ConvTranspose1d (stride=2, pad=1, output_pad=1) upsamples T→2T
        // Weight: [in_ch, 1, 3]. Applied to the original in_ch input, then take first out_ch.
        let pooled = depthwise_conv_transpose1d(x, &self.pool_weight, Some(&self.pool_bias), 1, 2, 1)?;
        // pooled: [batch, in_ch, 2T]
        let out_ch = out.dim(1)?;
        let pooled = pooled.i((.., ..out_ch, ..))?.contiguous()?; // [batch, out_ch, 2T]

        // Upsample `out` to 2T via nearest neighbor to match pooled
        let t2 = pooled.dim(2)?;
        let out_up = out.upsample_nearest1d(t2)?; // [batch, out_ch, 2T]

        Ok(((out_up + pooled)? / 2.0)?)
    }
}

// ---------------------------------------------------------------------------
// Block 2: 64ch → 64ch (identity residual)
// NEITHER norm1 nor norm2 has affine params
// ---------------------------------------------------------------------------
struct Block2 {
    norm1: AdaIn,
    conv1: Conv1d,
    norm2: AdaIn,
    conv2: Conv1d,
}

impl Block2 {
    fn load(vb: VarBuilder, style_half: usize, ch: usize) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            norm1: AdaIn::load(vb.pp("norm1"), style_half, ch)?,
            conv1: conv1d(ch, ch, 3, cfg, vb.pp("conv1"))?,
            norm2: AdaIn::load(vb.pp("norm2"), style_half, ch)?,
            conv2: conv1d(ch, ch, 3, cfg, vb.pp("conv2"))?,
        })
    }

    // x: [batch, ch, T] → [batch, ch, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv2.forward(&h)?;
        Ok((x + h)?)
    }
}

// ---------------------------------------------------------------------------
// F0 or N predictor branch
// ---------------------------------------------------------------------------
struct PredictorBranch {
    block0: Block0,
    block1: Block1,
    block2: Block2,
    proj: Conv1d,
}

impl PredictorBranch {
    fn load(vb: VarBuilder, name: &str, style_half: usize, lstm_out: usize) -> Result<Self> {
        let half = lstm_out / 2;
        let block0 = Block0::load(vb.pp(name).pp("0"), style_half, lstm_out)?;
        let block1 = Block1::load(vb.pp(name).pp("1"), style_half, lstm_out, half)?;
        let block2 = Block2::load(vb.pp(name).pp("2"), style_half, half)?;
        let proj_cfg = Conv1dConfig { ..Default::default() };
        let proj = conv1d(half, 1, 1, proj_cfg, vb.pp(format!("{}_proj", name)))?;
        Ok(Self { block0, block1, block2, proj })
    }

    // x: [batch, lstm_out, T], style: [batch, style_half] → [batch, 1, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x, style)?;
        let x = self.block1.forward(&x, style)?;
        let x = self.block2.forward(&x, style)?;
        Ok(self.proj.forward(&x)?)
    }
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------
pub struct Predictor {
    // Duration sub-network
    dur_lstm: BiLstm,
    duration_proj: Linear, // 128 → max_duration (50)

    // Shared LSTM for F0 + N
    shared_lstm: BiLstm,

    // F0 branch
    f0_branch: PredictorBranch,

    // N branch
    n_branch: PredictorBranch,

    max_duration: usize,
    style_half: usize, // style_dim / 2 = 128
}

impl Predictor {
    pub fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        // text encoder output is 256 (lstm_out 128 + style_half 128), lstm_hidden = 64 → BiLSTM out = 128
        let input_size = cfg.lstm_hidden * 2 + cfg.style_dim / 2; // 128 + 128 = 256
        let hidden_size = cfg.lstm_hidden; // 64
        let lstm_out = hidden_size * 2; // 128
        let style_half = cfg.style_dim / 2; // 128

        let dur_lstm = BiLstm::load(vb.pp("predictor").pp("lstm"), input_size, hidden_size)?;
        let duration_proj = linear(lstm_out, cfg.max_duration, vb.pp("predictor").pp("duration_proj").pp("linear_layer"))?;

        let shared_lstm = BiLstm::load(vb.pp("shared").pp("lstm"), input_size, hidden_size)?;

        let f0_branch = PredictorBranch::load(vb.pp("predictor"), "F0", style_half, lstm_out)?;
        let n_branch = PredictorBranch::load(vb.pp("predictor"), "N", style_half, lstm_out)?;

        Ok(Self {
            dur_lstm,
            duration_proj,
            shared_lstm,
            f0_branch,
            n_branch,
            max_duration: cfg.max_duration,
            style_half,
        })
    }

    /// text_features: [batch, seq, hidden_dim]
    /// style: [batch, style_dim] — uses style[:, 128:] for AdaIN
    /// Returns (durations, expanded_features, shared_lstm_out, f0, n_amp)
    ///   durations: [batch, seq] i64
    ///   expanded_features: [batch, hidden_dim, T]
    ///   shared_lstm_out: [batch, 128, T] — shared LSTM output (needed by decoder)
    ///   f0: [batch, 1, T]
    ///   n_amp: [batch, 1, T]
    pub fn forward(
        &self,
        text_features: &Tensor,
        style: &Tensor,
        speed: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let style_half = style.i((.., self.style_half..))?; // [batch, 128]

        // --- Duration predictor ---
        // dur_lstm: [batch, seq, 128]
        let dur_hidden = self.dur_lstm.forward(text_features)?;
        // duration_proj: [batch, seq, max_duration]
        let logits = self.duration_proj.forward(&dur_hidden)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        // sum over last dim: [batch, seq]
        let dur_raw = probs.sum(2)?;
        // divide by speed, round, clip to [1, max_duration]
        let dur_scaled = (dur_raw / speed as f64)?;
        let dur_rounded = round_tensor(&dur_scaled)?;
        let durations = clamp_tensor(&dur_rounded, 1.0, self.max_duration as f64)?
            .to_dtype(DType::I64)?;

        // --- Length regulator ---
        let expanded = length_regulator(text_features, &durations)?;
        // expanded: [batch, T, hidden_dim] → transpose → [batch, hidden_dim, T]
        let expanded_t = expanded.transpose(1, 2)?.contiguous()?;

        // --- Shared LSTM for F0 + N ---
        // shared_lstm input: [batch, T, hidden_dim] (transpose back)
        let expanded_seq = expanded_t.transpose(1, 2)?.contiguous()?;
        let shared_hidden = self.shared_lstm.forward(&expanded_seq)?; // [batch, T, 128]
        let shared_t = shared_hidden.transpose(1, 2)?.contiguous()?; // [batch, 128, T]

        // --- F0 and N ---
        let f0 = self.f0_branch.forward(&shared_t, &style_half)?; // [batch, 1, T]
        let n_amp = self.n_branch.forward(&shared_t, &style_half)?; // [batch, 1, T]

        Ok((durations, expanded_t, shared_t, f0, n_amp))
    }
}

// ---------------------------------------------------------------------------
// Length regulator: expand text_features by integer durations
// text_features: [batch, seq, C]
// durations: [batch, seq] i64
// Returns: [batch, T, C] where T = sum(durations[b]) for each b
// ---------------------------------------------------------------------------
fn length_regulator(text_features: &Tensor, durations: &Tensor) -> Result<Tensor> {
    let (batch, _seq, c) = text_features.dims3()?;
    let dev = text_features.device();
    let dtype = text_features.dtype();

    let dur_vec = durations.to_vec2::<i64>()?;

    let mut batch_outputs = Vec::with_capacity(batch);
    for b in 0..batch {
        let feats = text_features.i(b)?; // [seq, C]
        let durs = &dur_vec[b];
        let total: i64 = durs.iter().sum();
        let total = total as usize;

        let mut frames = Vec::with_capacity(total);
        for (s, &d) in durs.iter().enumerate() {
            let feat = feats.i(s)?; // [C]
            for _ in 0..d as usize {
                frames.push(feat.clone());
            }
        }
        // stack frames: [total, C]
        let expanded = if frames.is_empty() {
            Tensor::zeros((0, c), dtype, dev)?
        } else {
            Tensor::stack(&frames, 0)?
        };
        batch_outputs.push(expanded);
    }

    // Pad to max total length across batch
    let max_t = batch_outputs.iter().map(|t: &Tensor| t.dim(0).unwrap_or(0)).max().unwrap_or(0);
    let mut padded = Vec::with_capacity(batch);
    for out in batch_outputs {
        let t: usize = out.dim(0)?;
        let p = if t < max_t {
            let pad = Tensor::zeros((max_t - t, c), dtype, dev)?;
            Tensor::cat(&[out, pad], 0)?
        } else {
            out
        };
        padded.push(p.unsqueeze(0)?); // [1, max_t, C]
    }

    Ok(Tensor::cat(&padded, 0)?) // [batch, max_t, C]
}

// ---------------------------------------------------------------------------
// Helpers: round and clamp (element-wise, no native ops in candle-core)
// ---------------------------------------------------------------------------
fn round_tensor(x: &Tensor) -> Result<Tensor> {
    // floor(x + 0.5)
    Ok((x + 0.5_f64)?.floor()?)
}

fn clamp_tensor(x: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    Ok(x.clamp(min, max)?)
}
