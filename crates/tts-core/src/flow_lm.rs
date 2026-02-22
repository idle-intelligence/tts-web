use crate::mlp::SimpleMLPAdaLN;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};
use mimi_rs::qlinear::QLinear;
use mimi_rs::transformer::{Kind, StreamingTransformer, StreamingTransformerState};

pub trait Rng {
    fn sample(&mut self) -> f32;
}

// ---- LUTConditioner ----

/// Look-up-table conditioner: wraps an embedding table and looks up token embeddings.
/// Tokenization is handled externally (in JS); this struct just holds the weight matrix.
pub struct LUTConditioner {
    pub embed: Tensor,
    pub dim: usize,
    pub output_dim: usize,
}

impl LUTConditioner {
    pub fn load(vb: VarBuilder, n_bins: usize, dim: usize, output_dim: usize) -> Result<Self> {
        let embed = vb.get((n_bins + 1, dim), "embed.weight")?;
        Ok(Self { embed, dim, output_dim })
    }

    /// Get embeddings for token ids. Returns [1, num_tokens, dim].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            let dev = self.embed.device();
            return Tensor::zeros((1, 0, self.dim), DType::F32, dev);
        }
        let ids = Tensor::from_vec(
            token_ids.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            (token_ids.len(),),
            self.embed.device(),
        )?;
        let emb = self.embed.index_select(&ids, 0)?;
        emb.reshape((1, token_ids.len(), self.dim))
    }
}

// ---- LSD decode ----

/// Lagrangian Self Distillation decode.
/// Rebuilds the data sample from starting point x_0.
fn lsd_decode(
    flow_net: &SimpleMLPAdaLN,
    transformer_out: &Tensor,
    x_0: &Tensor,
    num_steps: usize,
) -> Result<Tensor> {
    let mut current = x_0.clone();
    let dev = x_0.device();
    let dtype = x_0.dtype();

    for i in 0..num_steps {
        let s_val = i as f64 / num_steps as f64;
        let t_val = (i + 1) as f64 / num_steps as f64;

        // Create s and t tensors matching x_0 shape but with last dim = 1
        let mut shape: Vec<usize> = x_0.dims()[..x_0.rank() - 1].to_vec();
        shape.push(1);

        let s = (Tensor::ones(shape.as_slice(), dtype, dev)? * s_val)?;
        let t = (Tensor::ones(shape.as_slice(), dtype, dev)? * t_val)?;

        let flow_dir = flow_net.forward(transformer_out, &[&s, &t], &current)?;
        let step_scale = 1.0f64 / num_steps as f64;
        current = (&current + &(flow_dir * step_scale)?)?;
    }
    Ok(current)
}

// ---- FlowLM ----

pub struct FlowLM {
    pub conditioner: LUTConditioner,
    flow_net: SimpleMLPAdaLN,
    pub transformer: StreamingTransformer,
    pub emb_std: Tensor,
    pub emb_mean: Tensor,
    pub bos_emb: Tensor,
    pub input_linear: QLinear,
    out_norm: LayerNorm,
    out_eos: QLinear,
    pub dim: usize,
    pub ldim: usize,
}

#[derive(Clone, Debug)]
pub struct FlowLMState {
    pub transformer_state: StreamingTransformerState,
}

impl FlowLM {
    pub fn load(vb: VarBuilder, cfg: &crate::config::FlowLMConfig) -> Result<Self> {
        let conditioner = LUTConditioner::load(
            vb.pp("conditioner"),
            cfg.n_bins,
            cfg.lut_dim,
            cfg.d_model,
        )?;

        let flow_net = SimpleMLPAdaLN::load(
            vb.pp("flow_net"),
            cfg.ldim,       // in_channels
            cfg.flow_dim,   // model_channels
            cfg.ldim,       // out_channels
            cfg.d_model,    // cond_channels
            cfg.flow_depth, // num_res_blocks
            2,              // num_time_conds
        )?;

        let transformer = StreamingTransformer::load(
            vb.pp("transformer"),
            cfg.d_model,
            cfg.num_heads,
            cfg.num_layers,
            None,
            cfg.dim_feedforward,
            None,
            cfg.max_period,
            Kind::FlowLm,
        )?;

        let emb_std = vb.get((cfg.ldim,), "emb_std")?;
        let emb_mean = vb.get((cfg.ldim,), "emb_mean")?;
        let bos_emb = vb.get((cfg.ldim,), "bos_emb")?;
        let input_linear =
            QLinear::from_linear(candle_nn::linear_no_bias(cfg.ldim, cfg.d_model, vb.pp("input_linear"))?);

        // Load out_norm weight/bias manually
        let out_norm_weight = vb.pp("out_norm").get((cfg.d_model,), "weight")?;
        let out_norm_bias = vb.pp("out_norm").get((cfg.d_model,), "bias")?;
        let out_norm = LayerNorm::new(out_norm_weight, out_norm_bias, 1e-5);

        let out_eos = QLinear::from_linear(candle_nn::linear(cfg.d_model, 1, vb.pp("out_eos"))?);

        Ok(Self {
            conditioner,
            flow_net,
            transformer,
            emb_std,
            emb_mean,
            bos_emb,
            input_linear,
            out_norm,
            out_eos,
            dim: cfg.d_model,
            ldim: cfg.ldim,
        })
    }

    pub fn init_state(&self) -> FlowLMState {
        let transformer_state = self.transformer.init_state();
        FlowLMState { transformer_state }
    }

    pub fn quantize_weights(&mut self, dtype: GgmlDType) -> Result<()> {
        self.input_linear.quantize_in_place(dtype)?;
        self.out_eos.quantize_in_place(dtype)?;
        self.flow_net.quantize_weights(dtype)?;
        self.transformer.quantize_weights(dtype)?;
        Ok(())
    }

    /// Run the backbone: concat text_embeddings + input, run transformer, strip prefix.
    fn backbone(
        &self,
        input: &Tensor,
        text_embeddings: &Tensor,
        seq_len: usize,
        state: &mut FlowLMState,
    ) -> Result<Tensor> {
        let input = Tensor::cat(&[text_embeddings, input], 1)?;
        let out = self.transformer.forward(&input, &mut state.transformer_state)?;
        let out = self.out_norm.forward(&out)?;
        // Remove prefix — keep only last seq_len positions
        let total = out.dim(1)?;
        let start = total - seq_len;
        out.narrow(1, start, seq_len)?.contiguous()
    }

    /// Sample next latent using flow matching.
    /// Returns (next_latent [B, 1, ldim], is_eos [B, 1]).
    pub fn sample_next_latent(
        &self,
        sequence: &Tensor,
        text_embeddings: &Tensor,
        state: &mut FlowLMState,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor, bool)> {
        let (b, s, _) = sequence.dims3()?;
        let dev = sequence.device();
        let dtype = sequence.dtype();

        let sequence = self.replace_nan_with_bos(sequence)?;
        let input = self.input_linear.forward(&sequence)?;
        let transformer_out = self.backbone(&input, text_embeddings, s, state)?;
        let t_len = transformer_out.dim(1)?;
        let transformer_out = transformer_out.narrow(1, t_len - 1, 1)?.contiguous()?;
        let transformer_out = transformer_out.reshape((b, self.dim))?;

        let eos_logit = self.out_eos.forward(&transformer_out)?;
        let eos_val = eos_logit.flatten_all()?.to_vec1::<f32>()?;
        let is_eos = eos_val[0] > eos_threshold;

        let noise_data: Vec<f32> = (0..b * self.ldim).map(|_| rng.sample()).collect();
        let noise = Tensor::from_vec(noise_data, (b, self.ldim), dev)?;
        let noise = noise.to_dtype(dtype)?;

        let latent = lsd_decode(&self.flow_net, &transformer_out, &noise, lsd_decode_steps)?;
        let latent = latent.reshape((b, 1, self.ldim))?;
        Ok((latent, is_eos))
    }

    /// Replace NaN values in sequence with bos_emb.
    fn replace_nan_with_bos(&self, sequence: &Tensor) -> Result<Tensor> {
        let shape = sequence.shape().clone();
        let dev = sequence.device();
        let dtype = sequence.dtype();
        // Flatten to 1D, patch NaNs, then reshape back.
        let flat = sequence.flatten_all()?;
        let mut data = flat.to_vec1::<f32>()?;
        let bos_data = self.bos_emb.to_vec1::<f32>()?;
        let ldim = self.ldim;

        for (i, val) in data.iter_mut().enumerate() {
            if val.is_nan() {
                *val = bos_data[i % ldim];
            }
        }
        Tensor::from_vec(data, shape, dev)?.to_dtype(dtype)
    }
}
