use candle_core::{IndexOp, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use anyhow::Result;

use crate::config::KittenConfig;

// ── InstanceNorm1d ────────────────────────────────────────────────────────────
// Input: [batch, channels, length]
// Normalizes over the `length` dimension for each (batch, channel).
struct InstanceNorm1d {
    gamma: Tensor, // [channels]
    beta: Tensor,  // [channels]
    eps: f64,
}

impl InstanceNorm1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(channels, "gamma")?;
        let beta = vb.get(channels, "beta")?;
        Ok(Self { gamma, beta, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, channels, length]
        let (batch, channels, length) = x.dims3()?;
        // mean/var over length dim
        let mean = x.mean_keepdim(2)?; // [batch, channels, 1]
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(2)?; // [batch, channels, 1]
        let std = (var + self.eps)?.sqrt()?;
        let normed = diff.broadcast_div(&std)?; // [batch, channels, length]

        // gamma/beta: [channels] → [1, channels, 1]
        let gamma = self.gamma.reshape((1, channels, 1))?.broadcast_as((batch, channels, length))?;
        let beta = self.beta.reshape((1, channels, 1))?.broadcast_as((batch, channels, length))?;
        Ok((normed * gamma)?.add(&beta)?)
    }
}

// ── Conv1d + InstanceNorm block ───────────────────────────────────────────────
struct CnnBlock {
    conv: candle_nn::Conv1d,
    norm: InstanceNorm1d,
}

impl CnnBlock {
    fn load(in_c: usize, out_c: usize, kernel: usize, pad: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv1dConfig { padding: pad, ..Default::default() };
        let conv = candle_nn::conv1d(in_c, out_c, kernel, cfg, vb.pp("0"))?;
        let norm = InstanceNorm1d::load(out_c, vb.pp("1"))?;
        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, channels, length]
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        // LeakyReLU(0.2)
        let neg = (x.clone() * 0.2)?;
        let mask = x.ge(0f32)?;
        Ok(mask.where_cond(&x, &neg)?)
    }
}

// ── BiLSTM ────────────────────────────────────────────────────────────────────
// Weights stored in ONNX convention:
//   W: [num_directions, 4*hidden, input_size]  — gate order: i, o, f, c
//   R: [num_directions, 4*hidden, hidden_size]
//   B: [num_directions, 8*hidden]              — first 4*h input bias, next 4*h recurrent bias
struct BiLstm {
    // direction 0 = forward, direction 1 = backward
    w_fwd: Tensor, // [4*h, input]
    r_fwd: Tensor, // [4*h, hidden]
    b_fwd: Tensor, // [4*h]  (input_bias + recurrent_bias combined)
    w_bwd: Tensor,
    r_bwd: Tensor,
    b_bwd: Tensor,
    hidden_size: usize,
}

impl BiLstm {
    fn load(hidden_size: usize, input_size: usize, vb_w: &str, vb_r: &str, vb_b: &str, vb: &VarBuilder) -> Result<Self> {
        let h4 = 4 * hidden_size;
        // W: [2, 4*h, input]
        let w = vb.get((2, h4, input_size), vb_w)?;
        // R: [2, 4*h, hidden]
        let r = vb.get((2, h4, hidden_size), vb_r)?;
        // B: [2, 8*h]
        let b_full = vb.get((2, 8 * hidden_size), vb_b)?;

        // Split directions
        let w_fwd = w.i(0)?.contiguous()?; // [4*h, input]
        let w_bwd = w.i(1)?.contiguous()?;
        let r_fwd = r.i(0)?.contiguous()?; // [4*h, hidden]
        let r_bwd = r.i(1)?.contiguous()?;

        // Combine input bias + recurrent bias: both are [4*h]
        let b0_fwd = b_full.i((0, ..h4))?.contiguous()?;
        let b1_fwd = b_full.i((0, h4..))?.contiguous()?;
        let b_fwd = b0_fwd.add(&b1_fwd)?;

        let b0_bwd = b_full.i((1, ..h4))?.contiguous()?;
        let b1_bwd = b_full.i((1, h4..))?.contiguous()?;
        let b_bwd = b0_bwd.add(&b1_bwd)?;

        Ok(Self { w_fwd, r_fwd, b_fwd, w_bwd, r_bwd, b_bwd, hidden_size })
    }

    // Run one direction of LSTM.
    // xs: Vec of [batch, input] tensors (one per time step)
    // Returns Vec of [batch, hidden] outputs
    fn run_direction(
        xs: &[Tensor],
        w: &Tensor,
        r: &Tensor,
        b: &Tensor,
        hidden_size: usize,
    ) -> Result<Vec<Tensor>> {
        let batch = xs[0].dim(0)?;
        let dev = xs[0].device();
        let dtype = xs[0].dtype();

        let mut h = Tensor::zeros((batch, hidden_size), dtype, dev)?;
        let mut c = Tensor::zeros((batch, hidden_size), dtype, dev)?;

        // Precompute W^T and b once
        // w: [4*h, input] → w^T: [input, 4*h]
        let wt = w.t()?.contiguous()?; // [input, 4*h]
        let rt = r.t()?.contiguous()?; // [hidden, 4*h]
        // b: [4*h] → [1, 4*h] for broadcast
        let b = b.unsqueeze(0)?;

        let mut outputs = Vec::with_capacity(xs.len());
        for x in xs {
            // [batch, 4*h]
            let gates = (x.matmul(&wt)?.add(&h.matmul(&rt)?)?.broadcast_add(&b))?;

            let h4 = 4 * hidden_size;
            let i_gate = candle_nn::ops::sigmoid(&gates.i((.., ..hidden_size))?)?;
            let o_gate = candle_nn::ops::sigmoid(&gates.i((.., hidden_size..2 * hidden_size))?)?;
            let f_gate = candle_nn::ops::sigmoid(&gates.i((.., 2 * hidden_size..3 * hidden_size))?)?;
            let g_gate = gates.i((.., 3 * hidden_size..h4))?.tanh()?;

            c = (f_gate.mul(&c)?.add(&i_gate.mul(&g_gate)?))?;
            h = o_gate.mul(&c.tanh()?)?;
            outputs.push(h.clone());
        }
        Ok(outputs)
    }

    // Input: [seq, batch, input_size]
    // Output: [seq, batch, 2*hidden_size]  (fwd || bwd concatenated)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, _batch, _) = x.dims3()?;

        // Slice into per-step tensors: each [batch, input]
        let steps: Vec<Tensor> = (0..seq)
            .map(|t| x.i((t, .., ..))?.contiguous())
            .collect::<Result<Vec<_>, _>>()?;

        let fwd_outs = Self::run_direction(&steps, &self.w_fwd, &self.r_fwd, &self.b_fwd, self.hidden_size)?;
        let rev_steps: Vec<Tensor> = steps.iter().rev().cloned().collect();
        let bwd_outs_rev = Self::run_direction(&rev_steps, &self.w_bwd, &self.r_bwd, &self.b_bwd, self.hidden_size)?;
        let bwd_outs: Vec<Tensor> = bwd_outs_rev.into_iter().rev().collect();

        // Concat [batch, hidden] || [batch, hidden] → [batch, 2*hidden], then stack → [seq, batch, 2*hidden]
        let combined: Vec<Tensor> = fwd_outs.into_iter().zip(bwd_outs)
            .map(|(f, b)| Tensor::cat(&[f, b], 1))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Tensor::stack(&combined, 0)?)
    }
}

// ── AdaIN block ───────────────────────────────────────────────────────────────
// fc: Linear(style_in → 2*feat_dim), splits into scale + bias
// Applied to: [seq, batch, feat_dim] using style [batch, style_in]
struct AdaIn {
    fc: Linear,
    feat_dim: usize,
}

impl AdaIn {
    fn load(style_in: usize, feat_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc = linear(style_in, 2 * feat_dim, vb.pp("fc"))?;
        Ok(Self { fc, feat_dim })
    }

    // x: [seq, batch, feat_dim]
    // style: [batch, style_in]
    // returns: [seq, batch, feat_dim]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (seq, batch, feat) = x.dims3()?;

        // ONNX uses LayerNorm over the feature dimension (axis=-1, i.e. last dim = feat)
        // Input to LN is treated as [batch, seq, feat], so x: [seq, batch, feat] → [batch, seq, feat]
        let x_bsf = x.transpose(0, 1)?.contiguous()?; // [batch, seq, feat]
        let mean = x_bsf.mean_keepdim(2)?;             // [batch, seq, 1]
        let diff = x_bsf.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(2)?;
        let std = (var + 1e-5_f64)?.sqrt()?;
        let normed_bsf = diff.broadcast_div(&std)?;    // [batch, seq, feat]
        // back to [seq, batch, feat]
        let normed = normed_bsf.transpose(0, 1)?.contiguous()?;

        // Project style; ONNX AdaIN: scale = fc[:C] + 1, bias = fc[C:]
        let proj = self.fc.forward(style)?; // [batch, 2*feat]
        let scale_raw = proj.i((.., ..self.feat_dim))?.contiguous()?; // [batch, feat]
        let bias = proj.i((.., self.feat_dim..))?.contiguous()?;      // [batch, feat]
        let scale = (scale_raw + 1.0_f64)?;

        // Broadcast to [seq, batch, feat]
        let scale = scale.unsqueeze(0)?.broadcast_as((seq, batch, feat))?;
        let bias = bias.unsqueeze(0)?.broadcast_as((seq, batch, feat))?;

        Ok(normed.mul(&scale)?.add(&bias)?)
    }
}

// ── TextEncoder ───────────────────────────────────────────────────────────────

pub struct TextEncoder {
    // CNN path (for decoder)
    embedding: Embedding, // text_encoder.embedding.weight [178, 128]
    cnn0: CnnBlock,
    cnn1: CnnBlock,
    cnn_lstm: BiLstm,

    // LSTM chain path (for predictor)
    lstm0: BiLstm,
    adain1: AdaIn,
    lstm2: BiLstm,
    adain3: AdaIn,

    style_half: usize, // 128 (style_dim / 2, used for lstm chain)
}

impl TextEncoder {
    pub fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let hidden_dim = cfg.hidden_dim;       // 128
        let lstm_hidden = cfg.lstm_hidden;     // 64
        let style_dim = cfg.style_dim;         // 256
        let style_half = style_dim / 2;        // 128

        // CNN path weights are under text_encoder.*
        let vb_te = vb.pp("text_encoder");
        let embedding = embedding(cfg.n_token - 1, hidden_dim, vb_te.pp("embedding"))?;
        let cnn0 = CnnBlock::load(hidden_dim, hidden_dim, 5, 2, vb_te.pp("cnn").pp("0"))?;
        let cnn1 = CnnBlock::load(hidden_dim, hidden_dim, 5, 2, vb_te.pp("cnn").pp("1"))?;

        // CNN LSTM and LSTM chain weights are under predictor.text_encoder.*
        let vb_pte = vb.pp("predictor").pp("text_encoder");

        // CNN path LSTM: input_size=128, hidden=64, bidirectional → output 128
        let cnn_lstm = BiLstm::load(lstm_hidden, hidden_dim, "lstm.W", "lstm.R", "lstm.B", &vb_pte)?;

        // LSTM chain: lstms.0 input_size=256 (bert 128 + style_half 128)
        let lstm0 = BiLstm::load(lstm_hidden, hidden_dim + style_half, "lstms.0.W", "lstms.0.R", "lstms.0.B", &vb_pte)?;
        // AdaIN after lstms.0: style_in=style_half=128, feat_dim=lstm_out=128
        let adain1 = AdaIn::load(style_half, hidden_dim, vb_pte.pp("lstms.1"))?;
        // lstms.2: input_size=256 (lstm0_out 128 + style_half 128)
        let lstm2 = BiLstm::load(lstm_hidden, hidden_dim + style_half, "lstms.2.W", "lstms.2.R", "lstms.2.B", &vb_pte)?;
        let adain3 = AdaIn::load(style_half, hidden_dim, vb_pte.pp("lstms.3"))?;

        Ok(Self {
            embedding,
            cnn0, cnn1, cnn_lstm,
            lstm0, adain1, lstm2, adain3,
            style_half,
        })
    }

    /// Returns `(lstm_features, cnn_features)`:
    /// - `lstm_features`: `[batch, seq, 256]` — LSTM chain output for predictor
    /// - `cnn_features`:  `[batch, 128, seq]` — CNN path output for decoder
    pub fn forward(&self, bert_output: &Tensor, input_ids: &Tensor, style: &Tensor) -> Result<(Tensor, Tensor)> {
        // bert_output: [batch, seq, 128]
        // input_ids: [batch, seq] u32
        // style: [batch, 256]
        let (batch, seq, _) = bert_output.dims3()?;

        // Split style into first half and second half
        // The LSTM chain uses style[:, 128:] = last 128 dims
        let style_for_lstm = style.i((.., self.style_half..))?.contiguous()?; // [batch, 128]

        // ── CNN path ──────────────────────────────────────────────────────────
        // Embed input_ids using text_encoder's own embedding (not BERT output)
        let cnn_embed = self.embedding.forward(input_ids)?; // [batch, seq, 128]
        let cnn_in = cnn_embed.transpose(1, 2)?.contiguous()?;
        let cnn_out = self.cnn0.forward(&cnn_in)?;
        let cnn_out = self.cnn1.forward(&cnn_out)?;
        // [batch, 128, seq] → [seq, batch, 128]
        let cnn_out = cnn_out.transpose(1, 2)?.transpose(0, 1)?.contiguous()?;
        let cnn_lstm_out = self.cnn_lstm.forward(&cnn_out)?; // [seq, batch, 128]
        // [seq, batch, 128] → [batch, 128, seq]
        let cnn_features = cnn_lstm_out.transpose(0, 1)?.transpose(1, 2)?.contiguous()?;

        // ── LSTM chain path ───────────────────────────────────────────────────
        // Transpose bert: [batch, seq, 128] → [seq, batch, 128]
        let bert_t = bert_output.transpose(0, 1)?.contiguous()?; // [seq, batch, 128]

        // Broadcast style_for_lstm [batch, 128] → [seq, batch, 128]
        let style_broadcast = style_for_lstm.unsqueeze(0)?.broadcast_as((seq, batch, self.style_half))?;
        // Concat on last dim → [seq, batch, 256]
        let lstm0_in = Tensor::cat(&[bert_t, style_broadcast], 2)?;
        let lstm0_out = self.lstm0.forward(&lstm0_in)?; // [seq, batch, 128]

        // AdaIN with style_for_lstm
        let lstm0_normed = self.adain1.forward(&lstm0_out, &style_for_lstm)?; // [seq, batch, 128]

        // Concat with style again → [seq, batch, 256]
        let style_broadcast2 = style_for_lstm.unsqueeze(0)?.broadcast_as((seq, batch, self.style_half))?;
        let lstm2_in = Tensor::cat(&[lstm0_normed, style_broadcast2], 2)?;
        let lstm2_out = self.lstm2.forward(&lstm2_in)?; // [seq, batch, 128]

        // AdaIN
        let lstm2_normed = self.adain3.forward(&lstm2_out, &style_for_lstm)?; // [seq, batch, 128]

        // Concat LSTM output with style → [seq, batch, 256]
        let style_broadcast3 = style_for_lstm.unsqueeze(0)?.broadcast_as((seq, batch, self.style_half))?;
        let lstm_features_seq = Tensor::cat(&[lstm2_normed, style_broadcast3], 2)?; // [seq, batch, 256]

        // [seq, batch, 256] → [batch, seq, 256]
        let lstm_features = lstm_features_seq.transpose(0, 1)?.contiguous()?;

        Ok((lstm_features, cnn_features))
    }
}
