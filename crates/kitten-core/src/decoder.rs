use candle_core::{IndexOp, Tensor};
use candle_nn::{linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use anyhow::Result;

use crate::config::KittenConfig;

// ── InstanceNorm1d ────────────────────────────────────────────────────────────
// [batch, channels, length] — normalizes over length per (batch, channel)
struct InstanceNorm1d {
    weight: Tensor, // gamma [channels]
    bias: Tensor,   // beta  [channels]
    eps: f64,
}

impl InstanceNorm1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(channels, "weight")?;
        let bias = vb.get(channels, "bias")?;
        Ok(Self { weight, bias, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, length) = x.dims3()?;
        let mean = x.mean_keepdim(2)?;
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(2)?;
        let std = (var + self.eps)?.sqrt()?;
        let normed = diff.broadcast_div(&std)?;
        let w = self.weight.reshape((1, channels, 1))?.broadcast_as((batch, channels, length))?;
        let b = self.bias.reshape((1, channels, 1))?.broadcast_as((batch, channels, length))?;
        Ok((normed * w)?.add(&b)?)
    }
}

// ── AdaIN (style-conditional InstanceNorm) ────────────────────────────────────
// norm: learned InstanceNorm parameters (separate from style projection)
// fc:   Linear(style_in → 2*channels) — predicts scale+bias from style
// Input: [batch, channels, length], style: [batch, style_in]
struct AdaInConv {
    norm: InstanceNorm1d,
    fc: Linear,
    channels: usize,
}

impl AdaInConv {
    fn load(channels: usize, style_in: usize, vb: VarBuilder) -> Result<Self> {
        let norm = InstanceNorm1d::load(channels, vb.pp("norm"))?;
        let fc = linear(style_in, 2 * channels, vb.pp("fc"))?;
        Ok(Self { norm, fc, channels })
    }

    // x: [batch, channels, length], style: [batch, style_in]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (batch, _channels, length) = x.dims3()?;
        let normed = self.norm.forward(x)?;
        let proj = self.fc.forward(style)?; // [batch, 2*channels]
        let scale = proj.i((.., ..self.channels))?.contiguous()?; // [batch, channels]
        let shift = proj.i((.., self.channels..))?.contiguous()?;
        let scale = scale.reshape((batch, self.channels, 1))?.broadcast_as((batch, self.channels, length))?;
        let shift = shift.reshape((batch, self.channels, 1))?.broadcast_as((batch, self.channels, length))?;
        Ok(normed.mul(&scale)?.add(&shift)?)
    }
}

// ── BiLSTM ────────────────────────────────────────────────────────────────────
// Weights in ONNX convention:
//   W: [2, 4*h, input]  gate order: i,o,f,c
//   R: [2, 4*h, hidden]
//   B: [2, 8*h]
struct BiLstm {
    w_fwd: Tensor, // [4*h, input]
    r_fwd: Tensor, // [4*h, hidden]
    b_fwd: Tensor, // [4*h]
    w_bwd: Tensor,
    r_bwd: Tensor,
    b_bwd: Tensor,
    hidden_size: usize,
}

impl BiLstm {
    fn load(hidden_size: usize, input_size: usize, vb: &VarBuilder) -> Result<Self> {
        let h4 = 4 * hidden_size;
        let w = vb.get((2, h4, input_size), "W")?;
        let r = vb.get((2, h4, hidden_size), "R")?;
        let b_full = vb.get((2, 8 * hidden_size), "B")?;

        let w_fwd = w.i(0)?.contiguous()?;
        let w_bwd = w.i(1)?.contiguous()?;
        let r_fwd = r.i(0)?.contiguous()?;
        let r_bwd = r.i(1)?.contiguous()?;

        let b0_fwd = b_full.i((0, ..h4))?.contiguous()?;
        let b1_fwd = b_full.i((0, h4..))?.contiguous()?;
        let b_fwd = b0_fwd.add(&b1_fwd)?;
        let b0_bwd = b_full.i((1, ..h4))?.contiguous()?;
        let b1_bwd = b_full.i((1, h4..))?.contiguous()?;
        let b_bwd = b0_bwd.add(&b1_bwd)?;

        Ok(Self { w_fwd, r_fwd, b_fwd, w_bwd, r_bwd, b_bwd, hidden_size })
    }

    fn run_direction(xs: &[Tensor], w: &Tensor, r: &Tensor, b: &Tensor, hidden_size: usize) -> Result<Vec<Tensor>> {
        let batch = xs[0].dim(0)?;
        let dev = xs[0].device();
        let dtype = xs[0].dtype();
        let mut h = Tensor::zeros((batch, hidden_size), dtype, dev)?;
        let mut c = Tensor::zeros((batch, hidden_size), dtype, dev)?;
        let wt = w.t()?.contiguous()?;
        let rt = r.t()?.contiguous()?;
        let b = b.unsqueeze(0)?;
        let mut outputs = Vec::with_capacity(xs.len());
        for x in xs {
            let gates = x.matmul(&wt)?.add(&h.matmul(&rt)?)?.broadcast_add(&b)?;
            let i_gate = candle_nn::ops::sigmoid(&gates.i((.., ..hidden_size))?)?;
            let o_gate = candle_nn::ops::sigmoid(&gates.i((.., hidden_size..2 * hidden_size))?)?;
            let f_gate = candle_nn::ops::sigmoid(&gates.i((.., 2 * hidden_size..3 * hidden_size))?)?;
            let g_gate = gates.i((.., 3 * hidden_size..))?.tanh()?;
            c = f_gate.mul(&c)?.add(&i_gate.mul(&g_gate)?)?;
            h = o_gate.mul(&c.tanh()?)?;
            outputs.push(h.clone());
        }
        Ok(outputs)
    }

    // x: [batch, channels, seq] — NCL format
    // Returns: [batch, 2*hidden, seq]
    fn forward_ncl(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, ch, seq] → permute to [seq, batch, ch]
        let x = x.transpose(1, 2)?.transpose(0, 1)?.contiguous()?; // [seq, batch, ch]
        let (seq, _batch, _) = x.dims3()?;
        let steps: Vec<Tensor> = (0..seq)
            .map(|t| x.i((t, .., ..))?.contiguous())
            .collect::<Result<Vec<_>, _>>()?;
        let fwd = Self::run_direction(&steps, &self.w_fwd, &self.r_fwd, &self.b_fwd, self.hidden_size)?;
        let rev_steps: Vec<Tensor> = steps.iter().rev().cloned().collect();
        let bwd_rev = Self::run_direction(&rev_steps, &self.w_bwd, &self.r_bwd, &self.b_bwd, self.hidden_size)?;
        let bwd: Vec<Tensor> = bwd_rev.into_iter().rev().collect();
        let combined: Vec<Tensor> = fwd.into_iter().zip(bwd)
            .map(|(f, b)| Tensor::cat(&[f, b], 1))
            .collect::<Result<Vec<_>, _>>()?;
        // stack → [seq, batch, 2*h] → [batch, 2*h, seq]
        let out = Tensor::stack(&combined, 0)?; // [seq, batch, 2*h]
        Ok(out.transpose(0, 1)?.transpose(1, 2)?.contiguous()?) // [batch, 2*h, seq]
    }
}

fn leaky_relu(x: &Tensor) -> Result<Tensor> {
    let neg = (x.clone() * 0.2)?;
    let mask = x.ge(0f32)?;
    Ok(mask.where_cond(x, &neg)?)
}

// ── EncodeBlock ───────────────────────────────────────────────────────────────
// Input: [batch, 130, T] (128 LSTM + 1 F0 + 1 N)
// norm1: AdaIN over 130ch (pre-conv1)
// conv1: Conv1d(130→256, k=3, pad=1)
// norm2: AdaIN over 256ch
// conv2: Conv1d(256→256, k=3, pad=1)
// conv1x1: Conv1d(130→256, k=1) residual projection
struct EncodeBlock {
    norm1: AdaInConv,
    conv1: Conv1d,
    norm2: AdaInConv,
    conv2: Conv1d,
    conv1x1: Tensor, // [256, 130, 1] — loaded manually (no bias)
}

impl EncodeBlock {
    fn load(style_in: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = AdaInConv::load(130, style_in, vb.pp("norm1"))?;
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = candle_nn::conv1d(130, 256, 3, cfg, vb.pp("conv1"))?;
        let norm2 = AdaInConv::load(256, style_in, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv1d(256, 256, 3, cfg, vb.pp("conv2"))?;
        let conv1x1 = vb.get((256, 130, 1), "conv1x1.weight")?;
        Ok(Self { norm1, conv1, norm2, conv2, conv1x1 })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Residual skip: conv1x1 on original x
        let residual = x.conv1d(&self.conv1x1, 0, 1, 1, 1)?; // [batch, 256, T]

        let x = self.norm1.forward(x, style)?;
        let x = leaky_relu(&x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x, style)?;
        let x = leaky_relu(&x)?;
        let x = self.conv2.forward(&x)?;
        Ok(x.add(&residual)?)
    }
}

// ── DecodeBlock ───────────────────────────────────────────────────────────────
// Input: [batch, 322, T] (256 features + 64 asr_res + 1 F0 + 1 N)
// norm1: AdaIN over 322ch
// conv1: Conv1d(322→256, k=3, pad=1)
// norm2: AdaIN over 256ch
// conv2: Conv1d(256→256, k=3, pad=1)
// conv1x1: Conv1d(322→256, k=1) residual
struct DecodeBlock {
    norm1: AdaInConv,
    conv1: Conv1d,
    norm2: AdaInConv,
    conv2: Conv1d,
    conv1x1: Tensor, // [256, 322, 1] — no bias
}

impl DecodeBlock {
    fn load(style_in: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = AdaInConv::load(322, style_in, vb.pp("norm1"))?;
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = candle_nn::conv1d(322, 256, 3, cfg, vb.pp("conv1"))?;
        let norm2 = AdaInConv::load(256, style_in, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv1d(256, 256, 3, cfg, vb.pp("conv2"))?;
        let conv1x1 = vb.get((256, 322, 1), "conv1x1.weight")?;
        Ok(Self { norm1, conv1, norm2, conv2, conv1x1 })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let residual = x.conv1d(&self.conv1x1, 0, 1, 1, 1)?; // [batch, 256, T]

        let x = self.norm1.forward(x, style)?;
        let x = leaky_relu(&x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x, style)?;
        let x = leaky_relu(&x)?;
        let x = self.conv2.forward(&x)?;
        Ok(x.add(&residual)?)
    }
}

// ── HiFi-GAN AdaIN ResBlock ───────────────────────────────────────────────────
// Each resblock has 3 dilated convs in convs1, 3 in convs2, with AdaIN between.
// Plus alpha (learned residual scaling) and adain1/adain2.
struct HifiResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaInConv>, // 3 AdaIN for convs1
    adain2: Vec<AdaInConv>, // 3 AdaIN for convs2
    alpha1: Vec<Tensor>,    // [1, channels, 1] per conv
    alpha2: Vec<Tensor>,
}

impl HifiResBlock {
    fn load(channels: usize, kernel: usize, style_in: usize, vb: VarBuilder) -> Result<Self> {
        let dilations = [1usize, 3, 5];
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();

        for (j, &dil) in dilations.iter().enumerate() {
            let pad = dil * (kernel - 1) / 2;
            let cfg1 = Conv1dConfig { padding: pad, dilation: dil, ..Default::default() };
            convs1.push(candle_nn::conv1d(channels, channels, kernel, cfg1, vb.pp(format!("convs1.{j}")))?);
            let cfg2 = Conv1dConfig { padding: pad, dilation: dil, ..Default::default() };
            convs2.push(candle_nn::conv1d(channels, channels, kernel, cfg2, vb.pp(format!("convs2.{j}")))?);
            adain1.push(AdaInConv::load(channels, style_in, vb.pp(format!("adain1.{j}")))?);
            adain2.push(AdaInConv::load(channels, style_in, vb.pp(format!("adain2.{j}")))?);
            alpha1.push(vb.get((1, channels, 1), format!("alpha1.{j}").as_str())?);
            alpha2.push(vb.get((1, channels, 1), format!("alpha2.{j}").as_str())?);
        }

        Ok(Self { convs1, convs2, adain1, adain2, alpha1, alpha2 })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for j in 0..3 {
            // First sub-block
            let normed = self.adain1[j].forward(&out, style)?;
            let activated = leaky_relu(&normed)?;
            let conv_out = self.convs1[j].forward(&activated)?;
            // Alpha-weighted residual: x * alpha + conv_out
            let alpha = self.alpha1[j].broadcast_as(out.shape())?;
            out = (out.mul(&alpha)?.add(&conv_out))?;

            // Second sub-block
            let normed = self.adain2[j].forward(&out, style)?;
            let activated = leaky_relu(&normed)?;
            let conv_out = self.convs2[j].forward(&activated)?;
            let alpha = self.alpha2[j].broadcast_as(out.shape())?;
            out = (out.mul(&alpha)?.add(&conv_out))?;
        }
        Ok(out)
    }
}

// ── ConvTranspose1d helper ─────────────────────────────────────────────────────
// candle stores ConvTranspose1d weight as [in_ch, out_ch, kernel].
// We use Tensor::conv_transpose1d directly.
fn conv_transpose1d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Tensor> {
    // x: [batch, in_ch, T], weight: [in_ch, out_ch, kernel]
    let y = x.conv_transpose1d(weight, padding, 0, stride, 1, 1)?;
    if let Some(b) = bias {
        let out_ch = b.dim(0)?;
        let b = b.reshape((1, out_ch, 1))?;
        Ok(y.broadcast_add(&b)?)
    } else {
        Ok(y)
    }
}

// ── Harmonic Source (m_source) ────────────────────────────────────────────────
// Takes F0 [batch, 1, T] and generates a multi-harmonic excitation.
// l_linear: weight stored as [9, 1] (9 harmonics → 1 output), bias [1]
// We interpret it as: for each time step, apply linear(harmonic_features → 1)
struct HarmonicSource {
    l_linear_w: Tensor, // [9, 1] — we'll use matmul with this directly
    l_linear_b: Tensor, // [1]
    n_harmonics: usize,
    sample_rate: usize,
}

impl HarmonicSource {
    fn load(n_harmonics: usize, sample_rate: usize, vb: VarBuilder) -> Result<Self> {
        let l_linear_w = vb.get((9, 1), "l_linear.weight")?;
        let l_linear_b = vb.get(1, "l_linear.bias")?;
        Ok(Self { l_linear_w, l_linear_b, n_harmonics, sample_rate })
    }

    // f0: [batch, 1, T] — fundamental frequency in Hz
    // Returns: [batch, 1, T] — harmonic excitation signal
    fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        let (batch, _, t) = f0.dims3()?;
        let dev = f0.device();
        let dtype = f0.dtype();

        // Generate harmonics: h_k(t) = sin(2*pi*k*f0*t/sr) for k=1..n_harmonics
        // Simplified: generate from f0 phase accumulation
        // For each harmonic k, the phase increment per frame is 2*pi*k*f0/sr
        // We accumulate phase across time steps
        // f0: [batch, 1, T] → broadcast to [batch, n_harmonics, T]

        // Build harmonic multipliers: [1, n_harmonics, 1]
        let harmonics: Vec<f32> = (1..=self.n_harmonics).map(|k| k as f32).collect();
        let h_mult = Tensor::new(harmonics.as_slice(), dev)?
            .to_dtype(dtype)?
            .reshape((1, self.n_harmonics, 1))?; // [1, n_harmonics, 1]

        // f0_expanded: [batch, n_harmonics, T]
        let f0_exp = f0.broadcast_as((batch, self.n_harmonics, t))?;
        let h_mult_exp = h_mult.broadcast_as((batch, self.n_harmonics, t))?;

        // Phase increment per step: 2*pi*k*f0 / sr
        let pi2 = (2.0 * std::f64::consts::PI) as f32;
        let scale = Tensor::new(pi2 / self.sample_rate as f32, f0.device())?.to_dtype(dtype)?;
        let phase_inc = f0_exp.mul(&h_mult_exp)?.broadcast_mul(&scale)?; // [batch, n_h, T]

        // Cumulative sum over T → phase at each step
        // candle doesn't have cumsum directly, so we compute it step by step
        // For efficiency, we stack slices
        let phase_slices: Vec<Tensor> = (0..t)
            .map(|i| phase_inc.i((.., .., ..=i))?.sum_keepdim(2))
            .collect::<Result<Vec<_>, _>>()?;
        let phase = Tensor::cat(&phase_slices, 2)?; // [batch, n_h, T]

        // sin of phase: [batch, n_h, T]
        let harmonics_sin = phase.sin()?;

        // l_linear: [n_h, T] × w[n_h, 1] + b[1] — but we apply per-timestep
        // Reshape: [batch, T, n_h] @ [n_h, 1] → [batch, T, 1] → [batch, 1, T]
        let hs_t = harmonics_sin.transpose(1, 2)?.contiguous()?; // [batch, T, n_h]
        let w = self.l_linear_w.contiguous()?; // [n_h, 1]
        let out = hs_t.matmul(&w)?; // [batch, T, 1]
        let b = self.l_linear_b.reshape((1, 1, 1))?;
        let out = out.broadcast_add(&b)?; // [batch, T, 1]
        let out = out.transpose(1, 2)?.contiguous()?; // [batch, 1, T]

        Ok(out)
    }
}

// ── STFT-based harmonic decoder ───────────────────────────────────────────────
// conv_post output (22ch) → split into amp (11) and phase (11)
// → harmonic synthesis via STFT inverse weights
struct StftSynth {
    #[allow(dead_code)]
    weight_fwd_real: Tensor, // [11, 1, 20] — loaded but used only for analysis pass
    #[allow(dead_code)]
    weight_fwd_imag: Tensor,
    weight_bwd_real: Tensor, // [11, 1, 20]
    weight_bwd_imag: Tensor,
}

impl StftSynth {
    fn load(vb: VarBuilder) -> Result<Self> {
        let weight_fwd_real = vb.get((11, 1, 20), "weight_forward_real")?;
        let weight_fwd_imag = vb.get((11, 1, 20), "weight_forward_imag")?;
        let weight_bwd_real = vb.get((11, 1, 20), "weight_backward_real")?;
        let weight_bwd_imag = vb.get((11, 1, 20), "weight_backward_imag")?;
        Ok(Self { weight_fwd_real, weight_fwd_imag, weight_bwd_real, weight_bwd_imag })
    }

    // x: [batch, 22, T] — 11 log-amps + 11 phases
    // Returns: [batch, 1, T*stride] waveform
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n_harm = 11usize;
        let (batch, _, t) = x.dims3()?;

        // Split: log_amp [batch, 11, T], phase [batch, 11, T]
        let log_amp = x.i((.., ..n_harm, ..))?.contiguous()?;
        let phase = x.i((.., n_harm.., ..))?.contiguous()?;

        // Amplitude
        let amp = log_amp.exp()?; // [batch, 11, T]

        // Real and imaginary parts
        let sin_phase = phase.sin()?;
        let cos_phase = phase.cos()?;
        let real = amp.mul(&cos_phase)?; // [batch, 11, T]
        let imag = amp.mul(&sin_phase)?; // [batch, 11, T]

        // STFT inverse: each harmonic channel [batch, 1, T] convolved with [1, 1, 20] backward weights
        // Then sum across harmonics
        // weight_bwd_real/imag: [11, 1, 20] — treat as per-harmonic 1D filter
        // We do: out = sum_k ( real_k * bwd_real_k + imag_k * bwd_imag_k )
        // Implement via grouped convolution: [batch, 11, T] with [11, 1, 20] groups=11

        // Pad for "same" output: kernel=20, stride=1, pad=19//2=9
        let cfg = Conv1dConfig { padding: 9, groups: n_harm, ..Default::default() };

        // Reshape weights: [11, 1, 20] is already correct for groups=11
        let real_out = real.conv1d(&self.weight_bwd_real, 9, 1, 1, n_harm)?; // [batch, 11, T]
        let imag_out = imag.conv1d(&self.weight_bwd_imag, 9, 1, 1, n_harm)?;

        let _ = cfg; // suppress unused warning

        // Sum across harmonic channels → [batch, 1, T]
        let waveform = real_out.add(&imag_out)?.sum_keepdim(1)?;
        let waveform = waveform.reshape((batch, 1, t))?;
        Ok(waveform)
    }
}

// ── HiFi-GAN Generator ────────────────────────────────────────────────────────
struct Generator {
    // Upsampling transposed convs
    ups_weight: Vec<Tensor>, // [in_ch, out_ch, kernel]
    ups_bias: Vec<Tensor>,
    // AdaIN ResBlocks after each upsampling
    resblocks: Vec<HifiResBlock>,
    // Noise injection
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<HifiResBlock>,
    // Post conv: [22, 64, 7]
    conv_post: Conv1d,
    // STFT synthesis
    stft: StftSynth,
    // Harmonic source
    m_source: HarmonicSource,

    upsample_rates: Vec<usize>,
    #[allow(dead_code)]
    upsample_kernels: Vec<usize>,
    #[allow(dead_code)]
    generator_channels: Vec<usize>, // [256, 128, 64]
    #[allow(dead_code)]
    style_in: usize,
}

impl Generator {
    fn load(style_in: usize, cfg: &KittenConfig, vb: VarBuilder) -> Result<Self> {
        let rates = &cfg.generator_upsample_rates;   // [10, 6]
        let kernels = &cfg.generator_upsample_kernels; // [20, 12]
        let channels = &cfg.generator_channels;       // [256, 128, 64]

        let mut ups_weight = Vec::new();
        let mut ups_bias = Vec::new();
        let mut resblocks = Vec::new();
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();

        // n_ups = 2, n_resblocks = 4 (2 per upsampling stage)
        for i in 0..rates.len() {
            let in_ch = channels[i];
            let out_ch = channels[i + 1];
            // ConvTranspose1d weight: [in_ch, out_ch, kernel]
            let w = vb.get((in_ch, out_ch, kernels[i]), format!("ups.{i}.weight").as_str())?;
            let b = vb.get(out_ch, format!("ups.{i}.bias").as_str())?;
            ups_weight.push(w);
            ups_bias.push(b);

            // 2 resblocks per upsample stage
            for j in 0..2 {
                let rb_idx = i * 2 + j;
                let rb_kernel = if out_ch == 128 { 3 } else { 3 };
                resblocks.push(HifiResBlock::load(out_ch, rb_kernel, style_in, vb.pp(format!("resblocks.{rb_idx}")))?);
            }

            // Noise conv: takes 22-ch source and projects to out_ch
            let noise_kernel = if i == 0 { kernels[0] } else { 1 };
            let noise_stride = if i == 0 { rates[0] } else { 1 };
            let noise_pad = if i == 0 { (noise_kernel - noise_stride) / 2 } else { 0 };
            let noise_cfg = Conv1dConfig { padding: noise_pad, stride: noise_stride, ..Default::default() };
            noise_convs.push(candle_nn::conv1d(22, out_ch, noise_kernel, noise_cfg, vb.pp(format!("noise_convs.{i}")))?);
            noise_res.push(HifiResBlock::load(out_ch, if out_ch == 128 { 7 } else { 11 }, style_in, vb.pp(format!("noise_res.{i}")))?);
        }

        let post_cfg = Conv1dConfig { padding: 3, ..Default::default() };
        let conv_post = candle_nn::conv1d(channels[channels.len() - 1], cfg.post_conv_channels, 7, post_cfg, vb.pp("conv_post"))?;

        let stft = StftSynth::load(vb.pp("stft"))?;
        let m_source = HarmonicSource::load(cfg.n_harmonics - 2, cfg.sample_rate, vb.pp("m_source"))?;

        Ok(Self {
            ups_weight,
            ups_bias,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            stft,
            m_source,
            upsample_rates: rates.clone(),
            upsample_kernels: kernels.clone(),
            generator_channels: channels.clone(),
            style_in,
        })
    }

    // x: [batch, 256, T], f0: [batch, 1, T], style: [batch, style_in]
    // Returns: [batch, 1, T*60]  (total upsampling = 10*6 = 60)
    fn forward(&self, x: &Tensor, f0: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Generate harmonic source signal at the target sample rate
        // f0 is at frame rate; upsample to audio rate
        let total_upsample: usize = self.upsample_rates.iter().product();
        let (_batch, _, t) = x.dims3()?;
        let audio_t = t * total_upsample;

        // Generate harmonic excitation at audio rate by upsampling f0
        // Simple nearest-neighbor upsample of f0 to audio_t
        let f0_audio = f0.upsample_nearest1d(audio_t)?; // [batch, 1, audio_t]
        let harmonic_src = self.m_source.forward(&f0_audio)?; // [batch, 1, audio_t]

        let mut h = x.clone();
        let n_ups = self.upsample_rates.len();

        for i in 0..n_ups {
            h = leaky_relu(&h)?;
            let stride = self.upsample_rates[i];
            let kernel = self.upsample_kernels[i];
            let padding = (kernel - stride) / 2;
            h = conv_transpose1d(&h, &self.ups_weight[i], Some(&self.ups_bias[i]), stride, padding)?;

            // Compute the length of harmonic source for this stage
            let ups_so_far: usize = self.upsample_rates[..=i].iter().product();
            let stage_t = t * ups_so_far;

            // Noise injection: downsample harmonic src to this stage's time resolution
            let src_at_stage = harmonic_src.upsample_nearest1d(stage_t)?; // approximate; could also slice
            let noise = self.noise_convs[i].forward(&src_at_stage)?; // [batch, out_ch, stage_t]
            let noise = self.noise_res[i].forward(&noise, style)?;
            h = h.add(&noise)?;

            // AdaIN ResBlocks
            let rb0 = i * 2;
            h = self.resblocks[rb0].forward(&h, style)?;
            h = self.resblocks[rb0 + 1].forward(&h, style)?;
        }

        h = leaky_relu(&h)?;
        h = self.conv_post.forward(&h)?; // [batch, 22, T*60]

        // STFT-based harmonic synthesis
        let waveform = self.stft.forward(&h)?; // [batch, 1, T*60]
        Ok(waveform)
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────────

pub struct Decoder {
    // LSTM: input=256, hidden=64, bidirectional → output 128
    lstm: BiLstm,
    // F0 downsampling conv
    f0_conv: Conv1d,
    // N downsampling conv
    n_conv: Conv1d,
    // asr_res projection: Conv1d(128→64, k=1)
    asr_res: Conv1d,
    // Encode block
    encode: EncodeBlock,
    // 4 decode blocks
    decode: Vec<DecodeBlock>,
    // pool conv (decode.3): [322, 1, 3] — generates features from noise
    pool_weight: Tensor,
    pool_bias: Tensor,
    // Generator
    generator: Generator,

    #[allow(dead_code)]
    style_dim: usize, // 256 (full style)
}

impl Decoder {
    pub fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let vb_d = vb.pp("decoder");
        let style_dim = cfg.style_dim; // 256

        // BiLSTM: input=256 (decoder_dim), hidden=64
        let lstm = BiLstm::load(cfg.lstm_hidden, cfg.decoder_dim, &vb_d.pp("lstm"))?;

        // F0/N downsampling: Conv1d(1, 1, k=3, stride=2, pad=1)
        let f0_cfg = Conv1dConfig { padding: 1, stride: 2, ..Default::default() };
        let f0_conv = candle_nn::conv1d(1, 1, 3, f0_cfg, vb_d.pp("F0_conv"))?;
        let n_conv = candle_nn::conv1d(1, 1, 3, f0_cfg, vb_d.pp("N_conv"))?;

        // asr_res: Conv1d(128→64, k=1)
        let asr_res = candle_nn::conv1d(128, 64, 1, Default::default(), vb_d.pp("asr_res").pp("0"))?;

        // Encode block: style_in = style_dim
        let encode = EncodeBlock::load(style_dim, vb_d.pp("encode"))?;

        // 4 decode blocks
        let mut decode = Vec::new();
        for i in 0..4 {
            decode.push(DecodeBlock::load(style_dim, vb_d.pp(format!("decode.{i}")))?);
        }

        // decode.3.pool: Conv1d(1, 322, k=3, pad=1)
        let pool_weight = vb_d.get((322, 1, 3), "decode.3.pool.weight")?;
        let pool_bias = vb_d.get(322, "decode.3.pool.bias")?;

        let generator = Generator::load(style_dim, cfg, vb_d.pp("generator"))?;

        Ok(Self {
            lstm, f0_conv, n_conv, asr_res, encode, decode,
            pool_weight, pool_bias, generator, style_dim,
        })
    }

    /// Returns waveform [batch, 1, num_samples]
    ///
    /// - `expanded_features`: [batch, 256, T] — duration-expanded acoustic features
    /// - `asr_features`:      [batch, 128, T] — CNN text encoder output, already duration-expanded
    /// - `f0`:                [batch, 1, T] — fundamental frequency
    /// - `n_amp`:             [batch, 1, T] — noise amplitude
    /// - `style`:             [batch, 256] — speaker style vector
    pub fn forward(
        &self,
        expanded_features: &Tensor,
        asr_features: &Tensor,
        f0: &Tensor,
        n_amp: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        // 1. BiLSTM on expanded features: [batch, 256, T] → [batch, 128, T]
        let lstm_out = self.lstm.forward_ncl(expanded_features)?; // [batch, 128, T]

        // 2. F0 and N downsampling: [batch, 1, T] → [batch, 1, T/2]
        let f0_down = self.f0_conv.forward(f0)?; // [batch, 1, T/2]
        let n_down = self.n_conv.forward(n_amp)?; // [batch, 1, T/2]

        // 3. asr_res projection: [batch, 128, T] → [batch, 64, T]
        let asr_proj = self.asr_res.forward(asr_features)?; // [batch, 64, T]

        // 4. Downsample LSTM output to T/2 to match F0/N
        let (_batch, _, t) = lstm_out.dims3()?;
        let t_half = f0_down.dim(2)?;
        let lstm_down = if t != t_half {
            lstm_out.upsample_nearest1d(t_half)?
        } else {
            lstm_out
        };

        // 5. Downsample asr_proj to T/2 as well
        let asr_half = asr_proj.upsample_nearest1d(t_half)?; // [batch, 64, T/2]

        // 6. Encode block: concat [lstm_down(128), f0_down(1), n_down(1)] = 130ch
        let encode_in = Tensor::cat(&[&lstm_down, &f0_down, &n_down], 1)?; // [batch, 130, T/2]
        let encode_out = self.encode.forward(&encode_in, style)?; // [batch, 256, T/2]

        // 7. Four decode blocks: each takes concat[prev_256, asr_64, f0_1, n_1] = 322ch
        let mut h = encode_out;
        for (i, block) in self.decode.iter().enumerate() {
            let decode_in = Tensor::cat(&[&h, &asr_half, &f0_down, &n_down], 1)?; // [batch, 322, T/2]
            if i == 3 {
                // decode.3 has an extra pool conv that processes noise and adds to input
                // pool: Conv1d(1, 322, k=3, pad=1) applied to noise (use n_down as proxy)
                let pool_out = {
                    let y = n_down.conv1d(&self.pool_weight, 1, 1, 1, 1)?; // [batch, 322, T/2]
                    let b = self.pool_bias.reshape((1, 322, 1))?;
                    y.broadcast_add(&b)?
                };
                let decode_in = decode_in.add(&pool_out)?;
                h = block.forward(&decode_in, style)?;
            } else {
                h = block.forward(&decode_in, style)?;
            }
        }
        // h: [batch, 256, T/2]

        // 8. Upsample back to T for generator
        let (_, _, t_half2) = h.dims3()?;
        let t_full = t_half2 * 2;
        let h_up = h.upsample_nearest1d(t_full)?; // [batch, 256, T]

        // Upsample f0 back to T for generator
        let f0_t = f0.upsample_nearest1d(t_full)?;

        // 9. Generator: produces waveform
        let waveform = self.generator.forward(&h_up, &f0_t, style)?;
        Ok(waveform)
    }
}
