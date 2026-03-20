use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

use crate::config::KittenConfig;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn leaky_relu_02(x: &Tensor) -> Result<Tensor> {
    let neg = (x * 0.2_f64)?;
    Ok(x.maximum(&neg)?)
}

/// InstanceNorm1d: per-channel normalisation over the time dimension.
/// x: [batch, channels, time] → [batch, channels, time]
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

fn debug_stats(name: &str, t: &Tensor) {
    let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let nans = data.iter().filter(|x| x.is_nan()).count();
    let first5: Vec<f32> = data.iter().copied().take(5).collect();
    eprintln!("[GEN] {name}: shape={:?} min={min:.6} max={max:.6} nans={nans} first5={first5:.6?}", t.shape());
}

// ── AdaIN ─────────────────────────────────────────────────────────────────────
// Reuses the same pattern as predictor.rs: optional affine instance norm.
// norm.weight/norm.bias may or may not be present; fc always present.
struct AdaIn {
    fc: Linear,
    norm_weight: Option<Tensor>,
    norm_bias: Option<Tensor>,
}

impl AdaIn {
    fn load(vb: VarBuilder, style_in: usize, channels: usize) -> Result<Self> {
        let fc = linear(style_in, channels * 2, vb.pp("fc"))?;
        let norm_weight = vb.pp("norm").get(channels, "weight").ok();
        let norm_bias = vb.pp("norm").get(channels, "bias").ok();
        Ok(Self { fc, norm_weight, norm_bias })
    }

    // x: [batch, channels, T], style: [batch, style_in] → [batch, channels, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let normed = if let (Some(w), Some(b)) = (&self.norm_weight, &self.norm_bias) {
            let n = instance_norm(x)?;
            let w = w.reshape((1, x.dim(1)?, 1))?;
            let b = b.reshape((1, x.dim(1)?, 1))?;
            n.broadcast_mul(&w)?.broadcast_add(&b)?
        } else {
            instance_norm(x)?
        };

        let proj = self.fc.forward(style)?; // [batch, 2*C]
        let c = x.dim(1)?;
        let gamma = (proj.i((.., ..c))?.unsqueeze(2)? + 1.0_f64)?; // [batch, C, 1], +1 residual
        let beta = proj.i((.., c..))?.unsqueeze(2)?;
        Ok(normed.broadcast_mul(&gamma)?.broadcast_add(&beta)?)
    }
}

// ── EncodeBlock ────────────────────────────────────────────────────────────────
// Prefix: `decoder.encode`
// Input: concat[shared_lstm(128ch), f0_down(1ch), n_down(1ch)] = 130ch
//
// skip = conv1x1(x)          130→256, k=1, no bias
// h = adain_norm1(x, style)  130ch, style=128→260=2*130
// h = leaky_relu(h, 0.2)
// h = conv1(h)               130→256, k=3, pad=1
// h = adain_norm2(h, style)  256ch, style=128→512=2*256
// h = leaky_relu(h, 0.2)
// h = conv2(h)               256→256, k=3, pad=1
// output = skip + h
struct EncodeBlock {
    conv1x1_w: Tensor, // [256, 130, 1] — no bias
    norm1: AdaIn,
    conv1: Conv1d,
    norm2: AdaIn,
    conv2: Conv1d,
}

impl EncodeBlock {
    fn load(vb: VarBuilder, style_in: usize) -> Result<Self> {
        let conv1x1_w = vb.get((256, 130, 1), "conv1x1.weight")?;
        let norm1 = AdaIn::load(vb.pp("norm1"), style_in, 130)?;
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = conv1d(130, 256, 3, cfg, vb.pp("conv1"))?;
        let norm2 = AdaIn::load(vb.pp("norm2"), style_in, 256)?;
        let conv2 = conv1d(256, 256, 3, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1x1_w, norm1, conv1, norm2, conv2 })
    }

    // x: [batch, 130, T] → [batch, 256, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let skip = x.conv1d(&self.conv1x1_w, 0, 1, 1, 1)?; // [batch, 256, T]
        let h = self.norm1.forward(x, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h, style)?;
        let h = leaky_relu_02(&h)?;
        let h = self.conv2.forward(&h)?;
        // ONNX applies scalar 1/√2 to the residual sum output
        Ok(((skip + h)? * (1.0_f64 / std::f64::consts::SQRT_2))?)
    }
}

// ── DecodeBlock ────────────────────────────────────────────────────────────────
// Prefix: `decoder.decode.{i}`, i=0..2 (standard), i=3 (upsampling variant)
// Input: concat[prev_256ch, asr_64ch, f0_1ch, n_1ch] = 322ch
//
// Standard (i=0,1,2):
//   skip = conv1x1(x)           322→256, k=1, no bias
//   h = adain_norm1(x, style)   322ch
//   h = leaky_relu(h, 0.2)
//   h = conv1(h)                322→256, k=3, pad=1
//   h = adain_norm2(h, style)   256ch
//   h = leaky_relu(h, 0.2)
//   h = conv2(h)                256→256, k=3, pad=1
//   output = skip + h
//
// Upsampling (i=3):
//   x_up  = nearest×2(x)        322ch, T→2T
//   skip  = conv1x1(x_up)       322→256
//   h_up  = depthwise_conv_transpose(x) via pool weights, T→2T
//   h = adain_norm1(h_up, style)
//   ... (same as above)
//   output = skip + h
struct DecodeBlock {
    conv1x1_w: Tensor, // [256, 322, 1] — no bias
    norm1: AdaIn,
    conv1: Conv1d,
    norm2: AdaIn,
    conv2: Conv1d,
    /// Only present for decode.3: depthwise ConvTranspose1d, weight [322, 1, 3]
    pool_weight: Option<Tensor>,
    pool_bias: Option<Tensor>,
}

impl DecodeBlock {
    fn load(vb: VarBuilder, style_in: usize, is_upsample: bool) -> Result<Self> {
        let conv1x1_w = vb.get((256, 322, 1), "conv1x1.weight")?;
        let norm1 = AdaIn::load(vb.pp("norm1"), style_in, 322)?;
        let cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = conv1d(322, 256, 3, cfg, vb.pp("conv1"))?;
        let norm2 = AdaIn::load(vb.pp("norm2"), style_in, 256)?;
        let conv2 = conv1d(256, 256, 3, cfg, vb.pp("conv2"))?;
        let (pool_weight, pool_bias) = if is_upsample {
            let pw = vb.get((322, 1, 3), "pool.weight")?;
            let pb = vb.get(322, "pool.bias")?;
            (Some(pw), Some(pb))
        } else {
            (None, None)
        };
        Ok(Self { conv1x1_w, norm1, conv1, norm2, conv2, pool_weight, pool_bias })
    }

    // x: [batch, 322, T] → [batch, 256, T] (or [batch, 256, 2T] for upsample block)
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        if self.pool_weight.is_some() {
            // Upsampling decode block (decode.3)
            let t = x.dim(2)?;
            let t2 = t * 2;

            // Skip path: nearest upsample then conv1x1
            let x_up = x.upsample_nearest1d(t2)?; // [batch, 322, 2T]
            let skip = x_up.conv1d(&self.conv1x1_w, 0, 1, 1, 1)?; // [batch, 256, 2T]

            // Main path: depthwise ConvTranspose1d (stride=2)
            let h_up = depthwise_conv_transpose1d(
                x,
                self.pool_weight.as_ref().unwrap(),
                self.pool_bias.as_ref().unwrap(),
                2, // stride
                1, // padding
                1, // output_padding
            )?; // [batch, 322, 2T]

            let h = self.norm1.forward(&h_up, style)?;
            let h = leaky_relu_02(&h)?;
            let h = self.conv1.forward(&h)?;
            let h = self.norm2.forward(&h, style)?;
            let h = leaky_relu_02(&h)?;
            let h = self.conv2.forward(&h)?;
            // ONNX applies scalar 1/√2 to the residual sum output
            Ok(((skip + h)? * (1.0_f64 / std::f64::consts::SQRT_2))?)
        } else {
            // Standard decode block (decode.0,1,2)
            let skip = x.conv1d(&self.conv1x1_w, 0, 1, 1, 1)?; // [batch, 256, T]
            let h = self.norm1.forward(x, style)?;
            let h = leaky_relu_02(&h)?;
            let h = self.conv1.forward(&h)?;
            let h = self.norm2.forward(&h, style)?;
            let h = leaky_relu_02(&h)?;
            let h = self.conv2.forward(&h)?;
            // ONNX applies scalar 1/√2 to the residual sum output
            Ok(((skip + h)? * (1.0_f64 / std::f64::consts::SQRT_2))?)
        }
    }
}

/// Depthwise ConvTranspose1d: each channel uses its own 1×kernel filter.
/// weight: [C, 1, kernel], bias: [C]
/// Implements groups=C by looping over channels.
fn depthwise_conv_transpose1d(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Tensor> {
    let (batch, channels, _t) = x.dims3()?;
    let bias_vec = bias.to_vec1::<f32>()?;
    let mut out_channels = Vec::with_capacity(channels);
    for c in 0..channels {
        // x_c: [batch, 1, T]
        let x_c = x.i((.., c..c + 1, ..))?.contiguous()?;
        // w_c: [1, 1, kernel]
        let w_c = weight.i((c..c + 1, .., ..))?.contiguous()?;
        // conv_transpose1d: [batch, 1, T_out]
        let y_c = x_c.conv_transpose1d(&w_c, padding, output_padding, stride, 1, 1)?;
        // add per-channel bias scalar
        let b_c = Tensor::new(bias_vec[c], x.device())?.to_dtype(x.dtype())?;
        let y_c = y_c.broadcast_add(&b_c.reshape((1, 1, 1))?)?;
        out_channels.push(y_c);
    }
    // cat along channel dim → [batch, C, T_out]
    let result = Tensor::cat(&out_channels, 1)?;
    // ensure batch dim is correct
    let _ = batch;
    Ok(result)
}

// ── AdaIN ResBlock (HiFi-GAN style) ───────────────────────────────────────────
// 3 dilated layers each with adain+snake activation on both sides.
//
// snake activation: x + sin²(α·x) / α
struct AdaInResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn>,
    adain2: Vec<AdaIn>,
    alpha1: Vec<Tensor>, // [1, ch, 1]
    alpha2: Vec<Tensor>,
}

impl AdaInResBlock {
    fn load(channels: usize, kernel: usize, style_in: usize, vb: VarBuilder) -> Result<Self> {
        let dilations = [1usize, 3, 5];
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        for (j, &dil) in dilations.iter().enumerate() {
            let pad1 = dil * (kernel - 1) / 2;
            let cfg1 = Conv1dConfig { padding: pad1, dilation: dil, ..Default::default() };
            convs1.push(conv1d(channels, channels, kernel, cfg1, vb.pp(format!("convs1.{j}")))?);
            let cfg2 = Conv1dConfig { padding: (kernel - 1) / 2, ..Default::default() };
            convs2.push(conv1d(channels, channels, kernel, cfg2, vb.pp(format!("convs2.{j}")))?);
            adain1.push(AdaIn::load(vb.pp(format!("adain1.{j}")), style_in, channels)?);
            adain2.push(AdaIn::load(vb.pp(format!("adain2.{j}")), style_in, channels)?);
            alpha1.push(vb.get((1, channels, 1), &format!("alpha1.{j}"))?);
            alpha2.push(vb.get((1, channels, 1), &format!("alpha2.{j}"))?);
        }
        Ok(Self { convs1, convs2, adain1, adain2, alpha1, alpha2 })
    }

    // x: [batch, channels, T] → [batch, channels, T]
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for j in 0..3 {
            let h = self.adain1[j].forward(&out, style)?;
            let h = snake(&h, &self.alpha1[j])?;
            let h = self.convs1[j].forward(&h)?;
            let h = self.adain2[j].forward(&h, style)?;
            let h = snake(&h, &self.alpha2[j])?;
            let h = self.convs2[j].forward(&h)?;
            out = (out + h)?;
        }
        Ok(out)
    }
}

/// Snake activation: x + sin²(α·x) / α
/// alpha: [1, channels, 1] (will be broadcast to x shape)
fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    // sin²(αx)/α = (1 - cos(2αx)) / (2α)
    // Equivalent: x + sin(αx)² / α
    let ax = x.broadcast_mul(alpha)?;
    let sin_ax = ax.sin()?;
    let sin2 = sin_ax.sqr()?;
    // divide by alpha, broadcast
    let result = (x + sin2.broadcast_div(alpha)?)?;
    Ok(result)
}

/// Element-wise atan2(y, x) computed on CPU via f32 stdlib.
fn tensor_atan2(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    let shape = y.shape().clone();
    let yd = y.flatten_all()?.to_vec1::<f32>()?;
    let xd = x.flatten_all()?.to_vec1::<f32>()?;
    let out: Vec<f32> = yd.iter().zip(xd.iter()).map(|(&yv, &xv)| yv.atan2(xv)).collect();
    Ok(Tensor::from_vec(out, shape, y.device())?.to_dtype(y.dtype())?)
}

// ── Harmonic Source ────────────────────────────────────────────────────────────
// Generates multi-harmonic excitation from F0.
// l_linear: [1, n_harmonics] weight + [1] bias (combines harmonics → 1 signal)
struct HarmonicSource {
    l_linear_w: Tensor, // [1, n_harmonics] stored as [n_harmonics, 1] in ONNX
    l_linear_b: Tensor, // [1]
    n_harmonics: usize,
    sample_rate: usize,
}

impl HarmonicSource {
    fn load(n_harmonics: usize, sample_rate: usize, vb: VarBuilder) -> Result<Self> {
        // ONNX weight shape is [1, n_harmonics] (out_features=1, in_features=n_harmonics)
        let l_linear_w = vb.get((1, n_harmonics), "l_linear.weight")?;
        let l_linear_b = vb.get(1, "l_linear.bias")?;
        Ok(Self { l_linear_w, l_linear_b, n_harmonics, sample_rate })
    }

    // f0: [batch, 1, T_audio] — in Hz at audio sample rate
    // Returns: [batch, 1, T_audio]
    fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        let (batch, _, t) = f0.dims3()?;
        let dev = f0.device();
        let dtype = f0.dtype();

        // harmonic multipliers [1, n_harmonics, 1]
        let harmonics: Vec<f32> = (1..=self.n_harmonics).map(|k| k as f32).collect();
        let h_mult = Tensor::new(harmonics.as_slice(), dev)?
            .to_dtype(dtype)?
            .reshape((1, self.n_harmonics, 1))?;

        // f0_exp: [batch, n_harmonics, T]
        let f0_exp = f0.broadcast_as((batch, self.n_harmonics, t))?;
        let h_mult_exp = h_mult.broadcast_as((batch, self.n_harmonics, t))?;

        // phase increment per sample: 2π·k·f0/sr
        let two_pi = 2.0 * std::f64::consts::PI;
        let scale = (two_pi / self.sample_rate as f64) as f32;
        let scale_t = Tensor::new(scale, dev)?.to_dtype(dtype)?;
        let phase_inc = f0_exp.mul(&h_mult_exp)?.broadcast_mul(&scale_t)?; // [batch, n_h, T]

        // cumulative sum over time (candle has no cumsum, use prefix sum loop)
        // For short T this is fine; for long sequences we pay O(T²) memory via scan.
        // Use a running accumulation approach instead: build slice by slice.
        let phase_slices: Vec<Tensor> = {
            let mut acc: Option<Tensor> = None;
            let mut slices = Vec::with_capacity(t);
            for i in 0..t {
                let step = phase_inc.i((.., .., i..i + 1))?.contiguous()?;
                let cumulated = match acc {
                    None => step.clone(),
                    Some(ref prev) => (prev + &step)?,
                };
                slices.push(cumulated.clone());
                acc = Some(cumulated);
            }
            slices
        };
        let phase = Tensor::cat(&phase_slices, 2)?; // [batch, n_h, T]

        debug_stats("harmonic_source: phase (after accumulation)", &phase);

        // sin of accumulated phase
        let harmonics_sin = phase.sin()?; // [batch, n_h, T]

        debug_stats("harmonic_source: harmonics_sin (after sin)", &harmonics_sin);

        // linear combination: [batch, T, n_h] @ [1, n_h, 1] → [batch, T, 1] → [batch, 1, T]
        let hs_t = harmonics_sin.transpose(1, 2)?.contiguous()?; // [batch, T, n_h]
        let w = self.l_linear_w.t()?.contiguous()?; // [n_h, 1]
        let w = w.unsqueeze(0)?.broadcast_as((batch, self.n_harmonics, 1))?; // [batch, n_h, 1]
        let out = hs_t.matmul(&w.contiguous()?)?; // [batch, T, 1]
        let b = self.l_linear_b.reshape((1, 1, 1))?;
        let out = out.broadcast_add(&b)?; // [batch, T, 1]
        let out = out.transpose(1, 2)?.contiguous()?; // [batch, 1, T]

        debug_stats("harmonic_source: out (after l_linear combination)", &out);

        Ok(out)
    }
}

// ── STFT (forward analysis + inverse synthesis) ────────────────────────────────
// Fixed STFT filterbank weights (not learned).
// forward: Conv1d(1, 11*2=22, k=20, stride=5) → real [11, T], imag [11, T]
// inverse: ConvTranspose1d(11, 1, k=20, stride=5) for real and imag parts separately
struct Stft {
    weight_fwd_real: Tensor, // [11, 1, 20]
    weight_fwd_imag: Tensor, // [11, 1, 20]
    weight_bwd_real: Tensor, // [11, 1, 20] — ConvTranspose weight [in_ch=11, out_ch=1, k=20]
    weight_bwd_imag: Tensor,
}

impl Stft {
    fn load(vb: VarBuilder) -> Result<Self> {
        let weight_fwd_real = vb.get((11, 1, 20), "weight_forward_real")?;
        let weight_fwd_imag = vb.get((11, 1, 20), "weight_forward_imag")?;
        let weight_bwd_real = vb.get((11, 1, 20), "weight_backward_real")?;
        let weight_bwd_imag = vb.get((11, 1, 20), "weight_backward_imag")?;
        Ok(Self { weight_fwd_real, weight_fwd_imag, weight_bwd_real, weight_bwd_imag })
    }

    // x: [batch, 1, T_audio] → [batch, 22, T_stft]
    // ONNX: edge-pad by 10, conv with pad=0, then polar form (log_amp, phase)
    fn forward_analysis(&self, x: &Tensor) -> Result<Tensor> {
        debug_stats("stft_forward_analysis: input x", x);
        let t = x.dim(2)?;
        let first = x.i((.., .., 0..1))?.contiguous()?; // [batch, 1, 1]
        let last = x.i((.., .., t - 1..t))?.contiguous()?;
        let pad_left = first.broadcast_as((x.dim(0)?, 1, 10))?.contiguous()?;
        let pad_right = last.broadcast_as((x.dim(0)?, 1, 10))?.contiguous()?;
        let padded = Tensor::cat(&[&pad_left, x, &pad_right], 2)?; // [batch, 1, T+20]

        // Conv1d with no padding (pad=0), stride=5
        let real = padded.conv1d(&self.weight_fwd_real, 0, 5, 1, 1)?; // [batch, 11, T_stft]
        let imag = padded.conv1d(&self.weight_fwd_imag, 0, 5, 1, 1)?;

        // Convert to polar: log_amp = log(sqrt(real² + imag²) + eps), phase = atan2(imag, real)
        let eps = 1e-7_f64;
        let mag = ((real.powf(2.0)? + imag.powf(2.0)?)? + eps)?.sqrt()?;
        let log_amp = mag.log()?;
        // atan2(imag, real) computed element-wise via f32 stdlib (no NaN edge cases)
        let phase = tensor_atan2(&imag, &real)?;

        let out = Tensor::cat(&[log_amp, phase], 1)?; // [batch, 22, T_stft]
        debug_stats("stft_forward_analysis: output (22ch)", &out);
        Ok(out)
    }

    // x: [batch, 22, T_stft] → [batch, 1, T_audio]
    // Splits into log_amp [11] and phase [11], then applies inverse STFT.
    //   amp = exp(log_amp)
    //   waveform = ConvTranspose(amp*cos(phase), w_bwd_real)
    //            - ConvTranspose(amp*sin(phase), w_bwd_imag)
    fn inverse_synthesis(&self, x: &Tensor) -> Result<Tensor> {
        let n_harm = 11usize;
        let (batch, _, _t_stft) = x.dims3()?;
        let _ = batch;

        let log_amp = x.i((.., ..n_harm, ..))?.contiguous()?; // [batch, 11, T_stft]
        let raw_phase = x.i((.., n_harm.., ..))?.contiguous()?;

        debug_stats("stft_inverse: log_amp (before exp)", &log_amp);

        let amp = log_amp.exp()?;

        debug_stats("stft_inverse: amp (after exp)", &amp);

        // ONNX: sin(raw_phase) first, then sin/cos of THAT
        let phase1 = raw_phase.sin()?;    // first sin
        let sin_p = phase1.sin()?;        // sin(sin(raw_phase))
        let cos_p = phase1.cos()?;        // cos(sin(raw_phase))

        debug_stats("stft_inverse: sin_p", &sin_p);
        debug_stats("stft_inverse: cos_p", &cos_p);

        let real_part = amp.mul(&cos_p)?; // [batch, 11, T_stft]
        let imag_part = amp.mul(&sin_p)?;

        // ConvTranspose1d: NO padding (pads=[0,0] in ONNX)
        let wv_real = real_part.conv_transpose1d(&self.weight_bwd_real, 0, 0, 5, 1, 1)?;
        let wv_imag = imag_part.conv_transpose1d(&self.weight_bwd_imag, 0, 0, 5, 1, 1)?;

        debug_stats("stft_inverse: wv_real (after ConvTranspose)", &wv_real);
        debug_stats("stft_inverse: wv_imag (after ConvTranspose)", &wv_imag);

        // waveform = real_component - imag_component (from ONNX trace)
        let waveform = (wv_real - wv_imag)?; // [batch, 1, T_out]

        // Trim 10 from each end (matching edge-padding added in forward_analysis)
        let t_out = waveform.dim(2)?;
        let start = 10.min(t_out);
        let end = t_out.saturating_sub(10);
        if end > start {
            Ok(waveform.i((.., .., start..end))?.contiguous()?)
        } else {
            Ok(waveform)
        }
    }

    // Convenience: T_stft from T_audio for kernel=20, stride=5, pad=5
    #[allow(dead_code)]
    fn stft_len(audio_len: usize) -> usize {
        (audio_len + 2 * 5 - 20) / 5 + 1
    }
}

// ── Generator ─────────────────────────────────────────────────────────────────
// HiFi-GAN-style generator with 2 upsample stages and noise injection.
//
// Stage 0: ups.0 ConvTranspose1d(256→128, k=20, stride=10) + 2×resblocks + noise
// Stage 1: ups.1 ConvTranspose1d(128→64, k=12, stride=6)  + 2×resblocks + noise
// conv_post: Conv1d(64→22, k=7, pad=3)
// STFT inverse synthesis → waveform
struct Generator {
    ups_w: Vec<Tensor>, // [in_ch, out_ch, kernel]
    ups_b: Vec<Tensor>,
    resblocks: Vec<AdaInResBlock>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock>,
    conv_post: Conv1d,
    stft: Stft,
    m_source: HarmonicSource,
    upsample_rates: Vec<usize>,
    upsample_kernels: Vec<usize>,
}

impl Generator {
    fn load(style_in: usize, cfg: &KittenConfig, vb: VarBuilder) -> Result<Self> {
        let rates = &cfg.generator_upsample_rates;     // [10, 6]
        let kernels = &cfg.generator_upsample_kernels; // [20, 12]
        let channels = &cfg.generator_channels;        // [256, 128, 64]

        let mut ups_w = Vec::new();
        let mut ups_b = Vec::new();
        let mut resblocks = Vec::new();
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();

        for i in 0..rates.len() {
            let in_ch = channels[i];
            let out_ch = channels[i + 1];

            // ConvTranspose1d weight stored as [in_ch, out_ch, kernel]
            ups_w.push(vb.get((in_ch, out_ch, kernels[i]), &format!("ups.{i}.weight"))?);
            ups_b.push(vb.get(out_ch, &format!("ups.{i}.bias"))?);

            // 2 AdaIN ResBlocks per stage (kernel=3 for both stages)
            for j in 0..2usize {
                let rb_idx = i * 2 + j;
                resblocks.push(AdaInResBlock::load(out_ch, 3, style_in, vb.pp(format!("resblocks.{rb_idx}")))?);
            }

            // Noise convs: stage 0 projects 22ch → 128ch with stride=6, kernel=12
            //              stage 1 projects 22ch → 64ch  with stride=1, kernel=1
            let (nk, ns, np) = if i == 0 {
                let k = kernels[1]; // kernel=12 (matches ups.1, not ups.0)
                let s = rates[1];   // stride=6
                let p = (k - s) / 2;
                (k, s, p)
            } else {
                (1usize, 1usize, 0usize)
            };
            let ncfg = Conv1dConfig { padding: np, stride: ns, ..Default::default() };
            noise_convs.push(conv1d(22, out_ch, nk, ncfg, vb.pp(format!("noise_convs.{i}")))?);

            // Noise ResBlocks: kernel=7 for 128ch stage, kernel=11 for 64ch stage
            let nrk = if out_ch == 128 { 7 } else { 11 };
            noise_res.push(AdaInResBlock::load(out_ch, nrk, style_in, vb.pp(format!("noise_res.{i}")))?);
        }

        let post_cfg = Conv1dConfig { padding: 3, ..Default::default() };
        let conv_post = conv1d(channels[channels.len() - 1], cfg.post_conv_channels, 7, post_cfg, vb.pp("conv_post"))?;

        let stft = Stft::load(vb.pp("stft"))?;
        // n_harmonics in config counts STFT bins (11), but harmonic source uses 9 sine waves.
        let n_harm_src = cfg.n_harmonics - 2; // 11 - 2 = 9
        let m_source = HarmonicSource::load(n_harm_src, cfg.sample_rate, vb.pp("m_source"))?;

        Ok(Self {
            ups_w, ups_b, resblocks, noise_convs, noise_res,
            conv_post, stft, m_source,
            upsample_rates: rates.clone(),
            upsample_kernels: kernels.clone(),
        })
    }

    // x:    [batch, 256, T]
    // f0:   [batch, 1, T]  — at the T frame rate entering the generator
    // style: [batch, style_in]
    // Returns: [batch, 1, T_audio]
    fn forward(&self, x: &Tensor, f0: &Tensor, style: &Tensor) -> Result<Tensor> {
        let total_up: usize = self.upsample_rates.iter().product(); // 10*6=60
        let (_batch, _, t) = x.dims3()?;
        let audio_t = t * total_up * 5; // ×60 acoustic → audio, then ×5 via STFT inverse?
        // Actually: total acoustic upsample = 10*6 = 60
        // Then STFT inverse adds another ×5 stride, but the generator outputs 22ch for STFT.
        // The STFT inverse ConvTranspose has stride=5.
        // So: acoustic T → (ups) → T*60 (64ch) → conv_post → 22ch at T*60 →
        //     STFT inverse (stride=5, ConvTranspose) → T*60*5 audio? No —
        // The STFT forward took audio at stride=5, so STFT channels are at T_stft = T_audio/5.
        // The generator's output at 64ch should be at the *STFT* frame rate, not audio rate.
        // That means: audio_T = T_stft * 5 where T_stft = T * 60.
        // audio_T = T * 60 * 5 = T * 300? But cfg says sr=24000, frame_rate=80Hz → ratio=300. ✓
        //
        // So the generator 64ch output IS at the STFT rate (T*60), and STFT inverse produces T*300.
        // Harmonic source needs audio rate = T * 300.
        let _ = audio_t;
        let harmonic_audio_t = t * total_up * 5; // T * 300
        let f0_audio = f0.upsample_nearest1d(harmonic_audio_t)?; // [batch, 1, T*300]
        let harmonic_src = self.m_source.forward(&f0_audio)?;     // [batch, 1, T*300]

        // Run forward STFT on harmonic source → [batch, 22, T_stft]
        // T_stft ≈ T * 60 (with stride=5, kernel=20: T_stft = (T*300 - 20 + 10) / 5 + 1 ≈ T*60)
        let noise_stft = self.stft.forward_analysis(&harmonic_src)?; // [batch, 22, T_stft]

        let mut h = x.clone();

        debug_stats("generator: input h (after leaky_relu will follow)", &h);

        for i in 0..self.upsample_rates.len() {
            h = leaky_relu_02(&h)?;

            // Upsample via ConvTranspose1d
            let stride = self.upsample_rates[i];
            let kernel = self.upsample_kernels[i];
            let padding = (kernel - stride) / 2;
            h = h.conv_transpose1d(&self.ups_w[i], padding, 0, stride, 1, 1)?;
            // add bias
            let b = self.ups_b[i].reshape((1, self.ups_b[i].dim(0)?, 1))?;
            h = h.broadcast_add(&b)?;

            debug_stats(&format!("generator: after ups[{i}] ConvTranspose"), &h);
            if i == 0 {
                let t = h.dim(2)?;
                let n = 5.min(t);
                let v = h.i((0, 0, ..n))?.to_vec1::<f32>()?;
                eprintln!("[CMP] GEN_UPS0_OUT shape={:?} ch0_first5={:.6?}", h.shape(), v);
            }

            // Noise injection
            let noise = self.noise_convs[i].forward(&noise_stft)?;
            // trim/pad noise to match h's time dim
            let noise = match_time(&noise, h.dim(2)?)?;
            let noise = self.noise_res[i].forward(&noise, style)?;

            debug_stats(&format!("generator: noise[{i}] after noise_res"), &noise);

            h = (h + noise)?;

            debug_stats(&format!("generator: h after noise injection [{i}]"), &h);

            // AdaIN ResBlocks: parallel (averaged), not sequential
            let rb0 = i * 2;
            let rb_in = h.clone();
            let rb0_out = self.resblocks[rb0].forward(&rb_in, style)?;
            let rb1_out = self.resblocks[rb0 + 1].forward(&rb_in, style)?;
            h = ((rb0_out + rb1_out)? / 2.0)?;
            debug_stats(&format!("generator: h after resblocks[{rb0}+{}] (parallel avg)", rb0 + 1), &h);
            if i == 0 {
                let t = h.dim(2)?;
                let n = 5.min(t);
                let v = h.i((0, 0, ..n))?.to_vec1::<f32>()?;
                eprintln!("[CMP] GEN_RESBLOCK_AVG0_OUT shape={:?} ch0_first5={:.6?}", h.shape(), v);
            }
        }

        h = leaky_relu_02(&h)?;
        h = self.conv_post.forward(&h)?; // [batch, 22, T_stft]

        debug_stats("generator: h after conv_post (22ch)", &h);
        {
            let t = h.dim(2)?;
            let n = 5.min(t);
            let v0 = h.i((0, 0, ..n))?.to_vec1::<f32>()?;
            let v11 = h.i((0, 11, ..n))?.to_vec1::<f32>()?;
            eprintln!("[CMP] CONV_POST_OUT shape={:?} ch0_first5={:.6?} ch11_first5={:.6?}", h.shape(), v0, v11);
        }

        // Split for inspection
        let log_amp_slice = h.i((.., ..11usize, ..))?.contiguous()?;
        debug_stats("generator: log_amplitude slice (before exp)", &log_amp_slice);

        let amp_after_exp = log_amp_slice.exp()?;
        debug_stats("generator: amp after exp(log_amplitude)", &amp_after_exp);

        let phase_slice = h.i((.., 11usize.., ..))?.contiguous()?;
        let sin_phase = phase_slice.sin()?;
        let cos_phase = phase_slice.cos()?;
        debug_stats("generator: sin(phase)", &sin_phase);
        debug_stats("generator: cos(phase)", &cos_phase);

        // STFT inverse synthesis → [batch, 1, T_audio]
        let waveform = self.stft.inverse_synthesis(&h)?;

        debug_stats("generator: final waveform", &waveform);
        {
            let n = 20.min(waveform.dim(2)?);
            let v = waveform.i((0, 0, ..n))?.to_vec1::<f32>()?;
            eprintln!("[CMP] FINAL_WAVEFORM shape={:?} first20={:.6?}", waveform.shape(), v);
        }

        Ok(waveform)
    }
}

/// Trim or pad a tensor along dim=2 to exactly `target_t` time steps.
fn match_time(x: &Tensor, target_t: usize) -> Result<Tensor> {
    let t = x.dim(2)?;
    if t == target_t {
        return Ok(x.clone());
    }
    if t > target_t {
        Ok(x.i((.., .., ..target_t))?.contiguous()?)
    } else {
        // zero-pad
        let pad = Tensor::zeros((x.dim(0)?, x.dim(1)?, target_t - t), x.dtype(), x.device())?;
        Ok(Tensor::cat(&[x, &pad], 2)?)
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────────

pub struct Decoder {
    /// Conv1d(128→64, k=1) — projects ASR features before concatenation
    asr_res: Conv1d,
    /// Conv1d(1→1, k=3, stride=2, pad=1) — downsample F0 from 2T→T
    f0_conv: Conv1d,
    /// Conv1d(1→1, k=3, stride=2, pad=1) — downsample N from 2T→T
    n_conv: Conv1d,
    encode: EncodeBlock,
    decode: Vec<DecodeBlock>,
    generator: Generator,
    style_half: usize, // 128
}

impl Decoder {
    pub fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let vb_d = vb.pp("decoder");
        let style_half = cfg.style_dim / 2; // 128

        let asr_res = conv1d(128, 64, 1, Default::default(), vb_d.pp("asr_res").pp("0"))?;

        let stride2 = Conv1dConfig { padding: 1, stride: 2, ..Default::default() };
        let f0_conv = conv1d(1, 1, 3, stride2, vb_d.pp("F0_conv"))?;
        let n_conv = conv1d(1, 1, 3, stride2, vb_d.pp("N_conv"))?;

        let encode = EncodeBlock::load(vb_d.pp("encode"), style_half)?;

        let mut decode = Vec::new();
        for i in 0..4usize {
            decode.push(DecodeBlock::load(vb_d.pp(format!("decode.{i}")), style_half, i == 3)?);
        }

        let generator = Generator::load(style_half, cfg, vb_d.pp("generator"))?;

        Ok(Self { asr_res, f0_conv, n_conv, encode, decode, generator, style_half })
    }

    /// Forward pass.
    ///
    /// - `shared_lstm_out`: [batch, 128, T] — from Predictor shared LSTM
    /// - `asr_features`:    [batch, 128, T] — CNN features (for asr_res projection)
    /// - `f0`:              [batch, 1, 2T]  — F0 at 2× frame rate (after Block1 pool)
    /// - `n_amp`:           [batch, 1, 2T]  — noise amplitude at 2× frame rate
    /// - `style`:           [batch, 256]    — full style vector
    ///
    /// Returns waveform [batch, 1, num_samples].
    pub fn forward(
        &self,
        shared_lstm_out: &Tensor,
        asr_features: &Tensor,
        f0: &Tensor,
        n_amp: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        // style_half = style[:, 128:]
        let style_half = style.i((.., self.style_half..))?; // [batch, 128]

        // Project ASR features: [batch, 128, T] → [batch, 64, T]
        let asr = self.asr_res.forward(asr_features)?;

        // Downsample F0/N: [batch, 1, 2T] → [batch, 1, T]
        let f0_down = self.f0_conv.forward(f0)?;
        let n_down = self.n_conv.forward(n_amp)?;
        // Align asr to the downsampled T (in case they differ by ±1 sample)
        let t = f0_down.dim(2)?;
        let asr = match_time(&asr, t)?;

        // EncodeBlock input: concat[shared_lstm_out(128), f0_down(1), n_down(1)] = 130ch
        // shared_lstm_out may be at T (matching after downsample) or at the original T.
        let lstm_aligned = match_time(shared_lstm_out, t)?;
        let enc_in = Tensor::cat(&[&lstm_aligned, &f0_down, &n_down], 1)?; // [batch, 130, T]
        let mut h = self.encode.forward(&enc_in, &style_half)?; // [batch, 256, T]
        debug_stats("ENCODE_OUT", &h);
        {
            let t = h.dim(2)?;
            let n = 10.min(t);
            let v = h.i((0, 0, ..n))?.to_vec1::<f32>()?;
            eprintln!("[CMP] ENCODE_OUT shape={:?} ch0_first10={:.6?}", h.shape(), v);
        }

        // Four DecodeBlocks
        for (block_idx, block) in self.decode.iter().enumerate() {
            // Align asr, f0, n to current h time dim (needed for block 3 which doubles T)
            let ht = h.dim(2)?;
            let asr_t = match_time(&asr, ht)?;
            let f0_t = match_time(&f0_down, ht)?;
            let n_t = match_time(&n_down, ht)?;
            let dec_in = Tensor::cat(&[&h, &asr_t, &f0_t, &n_t], 1)?; // [batch, 322, ht]
            h = block.forward(&dec_in, &style_half)?;
            debug_stats(&format!("DECODE_{block_idx}_OUT"), &h);
            {
                let t = h.dim(2)?;
                let n = 5.min(t);
                let v = h.i((0, 0, ..n))?.to_vec1::<f32>()?;
                eprintln!("[CMP] DECODE_{block_idx}_OUT shape={:?} ch0_first5={:.6?}", h.shape(), v);
            }
        }
        // After block 3: h is [batch, 256, 2T]

        // F0 upsampled to 2T for the generator
        let t2 = h.dim(2)?;
        let f0_gen = f0.upsample_nearest1d(t2)?; // [batch, 1, 2T]

        // Generator → waveform [batch, 1, num_samples]
        let waveform = self.generator.forward(&h, &f0_gen, &style_half)?;
        Ok(waveform)
    }
}
