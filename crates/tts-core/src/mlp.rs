use candle_core::{DType, Result, Tensor, D};
use candle_nn::{LayerNorm, LayerNormConfig, Module, VarBuilder};
use mimi_rs::gguf_loader::GgufTensors;
use mimi_rs::qlinear::QLinear;

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one_plus_scale = (scale + 1.0f64)?;
    x.broadcast_mul(&one_plus_scale)?.broadcast_add(shift)
}

/// Variance-based "RMSNorm" matching pocket-tts behavior:
/// Normalizes by sqrt(var(x) + eps) (variance computed with mean subtraction),
/// but does NOT subtract mean from the output. Then multiplies by weight.
///
/// This differs from standard RMSNorm (candle's LayerNorm::rms_norm) which
/// divides by sqrt(E[x^2] + eps). The pocket-tts model was trained with
/// variance-based normalization (E[(x-mean)^2]) in the denominator, so we
/// must match that behavior here for correct inference.
///
/// pocket_tts's `_rms_norm` (`pocket_tts/modules/mlp.py`) computes the
/// variance with `x.var(dim=-1, keepdim=True)`, which is PyTorch's *unbiased*
/// (Bessel-corrected) variance: divides the sum of squared deviations by
/// `n - 1`, not `n`. Match that here (biased/`n` variance was the bug).
fn variance_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let hidden_size = x.dim(D::Minus1)? as f64;
    let mean = (x.sum_keepdim(D::Minus1)? / hidden_size)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / (hidden_size - 1.0))?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv_std)?.broadcast_mul(weight)
}

// ---- TimestepEmbedder ----

pub struct TimestepEmbedder {
    linear1: QLinear,
    linear2: QLinear,
    rms_weight: Tensor,
    freqs: Tensor,
}

impl TimestepEmbedder {
    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        frequency_embedding_size: usize,
    ) -> Result<Self> {
        let mlp = vb.pp("mlp");
        let linear1 = QLinear::from_linear(candle_nn::linear(frequency_embedding_size, hidden_size, mlp.pp("0"))?);
        let linear2 = QLinear::from_linear(candle_nn::linear(hidden_size, hidden_size, mlp.pp("2"))?);

        // Load RMSNorm weight (stored under "3.alpha")
        let rms_weight = mlp.get((hidden_size,), "3.alpha")?;

        let freqs = vb.get((frequency_embedding_size / 2,), "freqs")?;

        Ok(Self { linear1, linear2, rms_weight, freqs })
    }

    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, _hidden_size: usize, _frequency_embedding_size: usize) -> Result<Self> {
        let mlp_prefix = format!("{prefix}.mlp");
        let linear1 = gguf.qlinear(&format!("{mlp_prefix}.0"))?;
        let linear2 = gguf.qlinear(&format!("{mlp_prefix}.2"))?;
        let rms_weight = gguf.tensor(&format!("{mlp_prefix}.3.alpha"))?;
        let freqs = gguf.tensor(&format!("{prefix}.freqs"))?;
        Ok(Self { linear1, linear2, rms_weight, freqs })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // t: [..., 1] -> frequency embedding
        let args = t.broadcast_mul(&self.freqs)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let last_dim = cos.rank() - 1;
        let embedding = Tensor::cat(&[&cos, &sin], last_dim)?;

        // MLP: linear -> silu -> linear -> variance_norm
        let x = self.linear1.forward(&embedding)?;
        let x = x.silu()?;
        let x = self.linear2.forward(&x)?;
        variance_norm(&x, &self.rms_weight, 1e-5)
    }
}

// ---- ResBlock ----

pub struct ResBlock {
    in_ln: LayerNorm,
    mlp_linear1: QLinear,
    mlp_linear2: QLinear,
    ada_ln_silu_linear: QLinear,
}

impl ResBlock {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let in_ln = candle_nn::layer_norm(
            channels,
            LayerNormConfig { eps: 1e-6, ..Default::default() },
            vb.pp("in_ln"),
        )?;
        let mlp = vb.pp("mlp");
        let mlp_linear1 = QLinear::from_linear(candle_nn::linear(channels, channels, mlp.pp("0"))?);
        let mlp_linear2 = QLinear::from_linear(candle_nn::linear(channels, channels, mlp.pp("2"))?);
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = QLinear::from_linear(candle_nn::linear(channels, 3 * channels, ada.pp("1"))?);
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, _channels: usize) -> Result<Self> {
        let in_ln_w = gguf.tensor(&format!("{prefix}.in_ln.weight"))?;
        let in_ln_b = gguf.tensor(&format!("{prefix}.in_ln.bias"))?;
        let in_ln = LayerNorm::new(in_ln_w, in_ln_b, 1e-6);
        let mlp_prefix = format!("{prefix}.mlp");
        let mlp_linear1 = gguf.qlinear(&format!("{mlp_prefix}.0"))?;
        let mlp_linear2 = gguf.qlinear(&format!("{mlp_prefix}.2"))?;
        let ada_prefix = format!("{prefix}.adaLN_modulation");
        let ada_ln_silu_linear = gguf.qlinear(&format!("{ada_prefix}.1"))?;
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor, y_silu: &Tensor) -> Result<Tensor> {
        let ada = self.ada_ln_silu_linear.forward(y_silu)?;
        let channels = x.dim(D::Minus1)?;
        let shift_mlp = ada.narrow(D::Minus1, 0, channels)?.contiguous()?;
        let scale_mlp = ada.narrow(D::Minus1, channels, channels)?.contiguous()?;
        let gate_mlp = ada.narrow(D::Minus1, 2 * channels, channels)?.contiguous()?;

        // h = modulate(ln(x), shift, scale)
        let h = self.in_ln.forward(x)?;
        let h = modulate(&h, &shift_mlp, &scale_mlp)?;

        // MLP
        let h = self.mlp_linear1.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp_linear2.forward(&h)?;
        x + &gate_mlp.broadcast_mul(&h)?
    }
}

// ---- FinalLayer ----

pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: QLinear,
    ada_ln_silu_linear: QLinear,
}

impl FinalLayer {
    pub fn load(vb: VarBuilder, model_channels: usize, out_channels: usize) -> Result<Self> {
        let dev = vb.device().clone();
        let dtype = DType::F32;
        let ones = Tensor::ones((model_channels,), dtype, &dev)?;
        let zeros = Tensor::zeros((model_channels,), dtype, &dev)?;
        let norm_final = LayerNorm::new(ones, zeros, 1e-6);
        let linear = QLinear::from_linear(candle_nn::linear(model_channels, out_channels, vb.pp("linear"))?);
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear =
            QLinear::from_linear(candle_nn::linear(model_channels, 2 * model_channels, ada.pp("1"))?);
        Ok(Self { norm_final, linear, ada_ln_silu_linear })
    }

    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, model_channels: usize, _out_channels: usize) -> Result<Self> {
        let device = gguf.device.clone();
        let ones = Tensor::ones((model_channels,), DType::F32, &device)?;
        let zeros = Tensor::zeros((model_channels,), DType::F32, &device)?;
        let norm_final = LayerNorm::new(ones, zeros, 1e-6);
        let linear = gguf.qlinear(&format!("{prefix}.linear"))?;
        let ada_prefix = format!("{prefix}.adaLN_modulation");
        let ada_ln_silu_linear = gguf.qlinear(&format!("{ada_prefix}.1"))?;
        Ok(Self { norm_final, linear, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor, c_silu: &Tensor) -> Result<Tensor> {
        let ada = self.ada_ln_silu_linear.forward(c_silu)?;
        let model_channels = x.dim(D::Minus1)?;
        let shift = ada.narrow(D::Minus1, 0, model_channels)?.contiguous()?;
        let scale = ada.narrow(D::Minus1, model_channels, model_channels)?.contiguous()?;

        let x = self.norm_final.forward(x)?;
        let x = modulate(&x, &shift, &scale)?;
        self.linear.forward(&x)
    }
}

// ---- SimpleMLPAdaLN ----

pub struct SimpleMLPAdaLN {
    time_embeds: Vec<TimestepEmbedder>,
    cond_embed: QLinear,
    input_proj: QLinear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
    pub num_time_conds: usize,
}

impl SimpleMLPAdaLN {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::load(
                vb.pp("time_embed").pp(i),
                model_channels,
                256,
            )?);
        }

        let cond_embed =
            QLinear::from_linear(candle_nn::linear(cond_channels, model_channels, vb.pp("cond_embed"))?);
        let input_proj =
            QLinear::from_linear(candle_nn::linear(in_channels, model_channels, vb.pp("input_proj"))?);


        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks
                .push(ResBlock::load(vb.pp("res_blocks").pp(i), model_channels)?);
        }

        let final_layer =
            FinalLayer::load(vb.pp("final_layer"), model_channels, out_channels)?;

        Ok(Self {
            time_embeds,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
            num_time_conds,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_gguf(
        gguf: &mut GgufTensors, prefix: &str,
        _in_channels: usize, model_channels: usize, out_channels: usize,
        _cond_channels: usize, num_res_blocks: usize, num_time_conds: usize,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::load_gguf(
                gguf, &format!("{prefix}.time_embed.{i}"),
                model_channels, 256,
            )?);
        }
        let cond_embed = gguf.qlinear(&format!("{prefix}.cond_embed"))?;
        let input_proj = gguf.qlinear(&format!("{prefix}.input_proj"))?;
        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::load_gguf(gguf, &format!("{prefix}.res_blocks.{i}"), model_channels)?);
        }
        let final_layer = FinalLayer::load_gguf(gguf, &format!("{prefix}.final_layer"), model_channels, out_channels)?;
        Ok(Self { time_embeds, cond_embed, input_proj, res_blocks, final_layer, num_time_conds })
    }

    /// Forward pass.
    /// c: conditioning from AR transformer [N, cond_channels]
    /// ts: list of time tensors (length = num_time_conds), each [..., 1]
    /// x: input tensor [N, in_channels]
    pub fn forward(&self, c: &Tensor, ts: &[&Tensor], x: &Tensor) -> Result<Tensor> {
        let mut x = self.input_proj.forward(x)?;
        let mut t_combined = self.time_embeds[0].forward(ts[0])?;
        for (embed, &t_input) in self.time_embeds[1..].iter().zip(ts[1..].iter()) {
            t_combined = (&t_combined + &embed.forward(t_input)?)?;
        }
        let scale = 1.0f64 / self.num_time_conds as f64;
        t_combined = (t_combined * scale)?;

        let c = self.cond_embed.forward(c)?;
        let y = (&t_combined + &c)?;
        let y_silu = y.silu()?;
        for block in &self.res_blocks {
            x = block.forward(&x, &y_silu)?;
        }
        self.final_layer.forward(&x, &y_silu)
    }
}

#[cfg(test)]
mod tests {
    use super::variance_norm;
    use candle_core::{Device, Tensor};

    /// Fixture dumped from the official `pocket_tts.modules.mlp._rms_norm`
    /// (unbiased variance, PyTorch's default `x.var(dim=-1, keepdim=True)`),
    /// with a fixed seed (`torch.manual_seed(42)`) and `eps=1e-5`. Guards
    /// against regressing to the biased (divide-by-n) variance this port
    /// used before, which silently mismatched pocket-tts's normalization
    /// scale (the TimestepEmbedder's RMSNorm feeds every flow-net ResBlock's
    /// AdaLN conditioning, so this error compounds into the generated latent).
    #[test]
    fn variance_norm_matches_official_unbiased_variance() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(
            vec![
                1.9269150495529175f32,
                1.4872841835021973,
                0.9007171988487244,
                -2.1055214405059814,
                0.6784184575080872,
                -1.2345449924468994,
                -0.043067481368780136,
                -1.6046669483184814,
                -0.7521361708641052,
                1.6487228870391846,
                -0.3924786448478699,
                -1.4036067724227905,
                -0.7278812527656555,
                -0.5594298839569092,
                -0.7688389420509338,
                0.7624453902244568,
            ],
            (2, 8),
            &dev,
        )?;
        let alpha = Tensor::from_vec(
            vec![
                0.5f32,
                0.6428571343421936,
                0.7857142686843872,
                0.9285714626312256,
                1.0714285373687744,
                1.2142857313156128,
                1.3571428060531616,
                1.5,
            ],
            (8,),
            &dev,
        )?;
        let expected: Vec<f32> = vec![
            0.6426977515220642,
            0.6377972364425659,
            0.47209271788597107,
            -1.3042150735855103,
            0.48488089442253113,
            -1.0000046491622925,
            -0.03898964077234268,
            -1.6056482791900635,
            -0.38093823194503784,
            1.07361900806427,
            -0.31236961483955383,
            -1.320227861404419,
            -0.7899723052978516,
            -0.6881049275398254,
            -1.0569367408752441,
            1.1584787368774414,
        ];

        let got = variance_norm(&x, &alpha, 1e-5)?.flatten_all()?.to_vec1::<f32>()?;
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "index {i}: got {g}, expected {e} (diff {})",
                (g - e).abs()
            );
        }
        Ok(())
    }
}
