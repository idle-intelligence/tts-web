use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{conv1d, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

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

// ---------------------------------------------------------------------------
// AdaIN block (128 channels)
// ---------------------------------------------------------------------------
struct AdaInBlock {
    norm1_fc: Linear, // 128 → 256 (gamma+beta for norm1)
    norm2_fc: Linear, // 128 → 256 (gamma+beta for norm2)
    conv1: Conv1d,    // 128 → 128, k=3, pad=1
    conv2: Conv1d,    // 128 → 128, k=3, pad=1
}

impl AdaInBlock {
    fn load(vb: VarBuilder, style_half: usize, channels: usize) -> Result<Self> {
        let norm1_fc = linear(style_half, channels * 2, vb.pp("norm1").pp("fc"))?;
        let norm2_fc = linear(style_half, channels * 2, vb.pp("norm2").pp("fc"))?;
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = conv1d(channels, channels, 3, cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(channels, channels, 3, cfg, vb.pp("conv2"))?;
        Ok(Self { norm1_fc, norm2_fc, conv1, conv2 })
    }

    // x: [batch, channels, T], style_half: [batch, style_half]
    fn forward(&self, x: &Tensor, style_half: &Tensor) -> Result<Tensor> {
        // First sub-block
        let x = adain_apply(&self.norm1_fc, x, style_half)?;
        let x = leaky_relu_02(&x)?;
        let x = self.conv1.forward(&x)?;
        // Second sub-block
        let x2 = adain_apply(&self.norm2_fc, &x, style_half)?;
        let x2 = leaky_relu_02(&x2)?;
        let x2 = self.conv2.forward(&x2)?;
        // Residual + divide by 2
        Ok(((x + x2)? / 2.0)?)
    }
}

fn adain_apply(fc: &Linear, x: &Tensor, style_half: &Tensor) -> Result<Tensor> {
    let proj = fc.forward(style_half)?; // [batch, 2*C]
    let c = x.dim(1)?;
    // gamma: [batch, C], beta: [batch, C]
    let gamma = proj.i((.., ..c))?;
    let beta = proj.i((.., c..))?;
    // reshape for broadcast: [batch, C, 1]
    let gamma = gamma.unsqueeze(2)?;
    let beta = beta.unsqueeze(2)?;
    let normed = instance_norm(x)?;
    Ok(normed.broadcast_mul(&gamma)?.broadcast_add(&beta)?)
}

fn leaky_relu_02(x: &Tensor) -> Result<Tensor> {
    // LeakyReLU(0.2): max(0.2*x, x)
    let neg = (x * 0.2_f64)?;
    Ok(x.maximum(&neg)?)
}

// ---------------------------------------------------------------------------
// DownsampleConvBlock  (F0.1, F0.2, N.1, N.2)
//
// F0.1: downsample 128→64, then residual conv block
// F0.2: no downsample (64→64), just residual conv block
// ---------------------------------------------------------------------------
struct ConvResBlock {
    downsample: Option<Conv1d>, // Some for F0.1 (128→64), None for F0.2
    conv0: Conv1d,              // (in_ch→mid_ch, k=3)
    norm0: LayerNorm,
    conv3: Conv1d,              // (mid_ch→out_ch, k=3)
    norm3: LayerNorm,
    #[allow(dead_code)]
    out_ch: usize,
}

impl ConvResBlock {
    // has_downsample: true for block 1 (128→64), false for block 2 (64→64)
    fn load(vb: VarBuilder, has_downsample: bool, in_ch: usize, mid_ch: usize, out_ch: usize) -> Result<Self> {
        let downsample = if has_downsample {
            let cfg = Conv1dConfig { ..Default::default() };
            Some(conv1d(in_ch, out_ch, 1, cfg, vb.pp("downsample"))?)
        } else {
            None
        };
        let cfg3 = Conv1dConfig { padding: 1, ..Default::default() };
        let conv0 = conv1d(in_ch, mid_ch, 3, cfg3, vb.pp("conv_block").pp("0"))?;
        let norm0 = layer_norm(mid_ch, LayerNormConfig::default(), vb.pp("conv_block").pp("1"))?;
        let conv3 = conv1d(mid_ch, out_ch, 3, cfg3, vb.pp("conv_block").pp("3"))?;
        let norm3 = layer_norm(out_ch, LayerNormConfig::default(), vb.pp("conv_block").pp("4"))?;
        Ok(Self { downsample, conv0, norm0, conv3, norm3, out_ch })
    }

    // x: [batch, in_ch, T] → [batch, out_ch, T]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some(ds) = &self.downsample {
            ds.forward(x)?
        } else {
            x.clone()
        };

        // conv_block input is the downsampled or original
        let x_in = residual.clone();

        // Conv(in/out_ch → mid_ch) → LayerNorm → LeakyReLU
        let h = self.conv0.forward(&x_in)?;
        // LayerNorm expects [..., C] — transpose to [batch, T, mid_ch], norm, transpose back
        let h = layer_norm_on_channels(h, &self.norm0)?;
        let h = leaky_relu_02(&h)?;

        // Conv(mid_ch → out_ch) → LayerNorm → LeakyReLU
        let h = self.conv3.forward(&h)?;
        let h = layer_norm_on_channels(h, &self.norm3)?;
        let h = leaky_relu_02(&h)?;

        Ok((residual + h)?)
    }
}

// LayerNorm is applied over the last dimension.
// Our tensors are [batch, channels, T] — we need to norm over channels for each position.
// Transpose to [batch, T, channels], norm, transpose back.
fn layer_norm_on_channels(x: Tensor, ln: &LayerNorm) -> Result<Tensor> {
    let x_t = x.transpose(1, 2)?.contiguous()?; // [batch, T, C]
    let normed = ln.forward(&x_t)?;
    Ok(normed.transpose(1, 2)?.contiguous()?)
}

// ---------------------------------------------------------------------------
// F0 or N predictor branch (shared LSTM → AdaIN block → 2x ConvResBlock → proj)
// ---------------------------------------------------------------------------
struct PredictorBranch {
    adain: AdaInBlock,
    block1: ConvResBlock,
    block2: ConvResBlock,
    proj: Conv1d,
}

impl PredictorBranch {
    fn load(vb: VarBuilder, name: &str, style_half: usize, lstm_out: usize) -> Result<Self> {
        // Block 0: AdaIN at lstm_out channels (128)
        let adain = AdaInBlock::load(vb.pp(name).pp("0"), style_half, lstm_out)?;
        // Block 1: downsample 128→64
        let block1 = ConvResBlock::load(
            vb.pp(name).pp("1"),
            true,
            lstm_out,
            lstm_out,
            lstm_out / 2,
        )?;
        // Block 2: 64→64
        let block2 = ConvResBlock::load(
            vb.pp(name).pp("2"),
            false,
            lstm_out / 2,
            lstm_out / 2,
            lstm_out / 2,
        )?;
        let proj_cfg = Conv1dConfig { ..Default::default() };
        let proj = conv1d(lstm_out / 2, 1, 1, proj_cfg, vb.pp(format!("{}_proj", name)))?;
        Ok(Self { adain, block1, block2, proj })
    }

    // x: [batch, lstm_out, T], style_half: [batch, style_half]
    fn forward(&self, x: &Tensor, style_half: &Tensor) -> Result<Tensor> {
        let x = self.adain.forward(x, style_half)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
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
        // hidden_dim = 256 (text encoder output), lstm_hidden = 64 → BiLSTM out = 128
        let input_size = cfg.hidden_dim; // 256
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
    /// Returns (durations, expanded_features, f0, n_amp)
    ///   durations: [batch, seq] i64
    ///   expanded_features: [batch, hidden_dim, T]
    ///   f0: [batch, 1, T]
    ///   n_amp: [batch, 1, T]
    pub fn forward(
        &self,
        text_features: &Tensor,
        style: &Tensor,
        speed: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
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

        Ok((durations, expanded_t, f0, n_amp))
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
