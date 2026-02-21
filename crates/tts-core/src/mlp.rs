use candle_core::{DType, Result, Tensor, D};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

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
fn variance_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let hidden_size = x.dim(D::Minus1)? as f64;
    let mean = (x.sum_keepdim(D::Minus1)? / hidden_size)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden_size)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv_std)?.broadcast_mul(weight)
}

// ---- TimestepEmbedder ----

pub struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
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
        let linear1 = candle_nn::linear(frequency_embedding_size, hidden_size, mlp.pp("0"))?;
        let linear2 = candle_nn::linear(hidden_size, hidden_size, mlp.pp("2"))?;

        // Load RMSNorm weight (stored under "3.alpha")
        let rms_weight = mlp.get((hidden_size,), "3.alpha")?;

        let freqs = vb.get((frequency_embedding_size / 2,), "freqs")?;

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
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    ada_ln_silu_linear: Linear,
}

impl ResBlock {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let in_ln = candle_nn::layer_norm(
            channels,
            LayerNormConfig { eps: 1e-6, ..Default::default() },
            vb.pp("in_ln"),
        )?;
        let mlp = vb.pp("mlp");
        let mlp_linear1 = candle_nn::linear(channels, channels, mlp.pp("0"))?;
        let mlp_linear2 = candle_nn::linear(channels, channels, mlp.pp("2"))?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = candle_nn::linear(channels, 3 * channels, ada.pp("1"))?;
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let ada = self.ada_ln_silu_linear.forward(&y.silu()?)?;
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
    linear: Linear,
    ada_ln_silu_linear: Linear,
}

impl FinalLayer {
    pub fn load(vb: VarBuilder, model_channels: usize, out_channels: usize) -> Result<Self> {
        let dev = vb.device().clone();
        let dtype = DType::F32;
        let ones = Tensor::ones((model_channels,), dtype, &dev)?;
        let zeros = Tensor::zeros((model_channels,), dtype, &dev)?;
        let norm_final = LayerNorm::new(ones, zeros, 1e-6);
        let linear = candle_nn::linear(model_channels, out_channels, vb.pp("linear"))?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear =
            candle_nn::linear(model_channels, 2 * model_channels, ada.pp("1"))?;
        Ok(Self { norm_final, linear, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let ada = self.ada_ln_silu_linear.forward(&c.silu()?)?;
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
    cond_embed: Linear,
    input_proj: Linear,
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
            candle_nn::linear(cond_channels, model_channels, vb.pp("cond_embed"))?;
        let input_proj =
            candle_nn::linear(in_channels, model_channels, vb.pp("input_proj"))?;

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
        for block in &self.res_blocks {
            x = block.forward(&x, &y)?;
        }
        self.final_layer.forward(&x, &y)
    }
}
