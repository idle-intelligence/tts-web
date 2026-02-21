use crate::config::TTSConfig;
use crate::flow_lm::{FlowLM, FlowLMState, Rng};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use mimi_rs::mimi::{MimiModel, MimiState};

pub struct TTSModel {
    pub flow_lm: FlowLM,
    pub mimi: MimiModel,
    lsd_decode_steps: usize,
    eos_threshold: f32,
}

#[derive(Clone, Debug)]
pub struct TTSState {
    pub flow_lm_state: FlowLMState,
}

impl TTSModel {
    pub fn load(vb: VarBuilder, cfg: &TTSConfig) -> Result<Self> {
        let flow_lm = FlowLM::load(vb.pp("flow_lm"), &cfg.flow_lm)?;
        let mimi = MimiModel::load(vb.pp("mimi"), &cfg.mimi)?;

        Ok(Self {
            flow_lm,
            mimi,
            lsd_decode_steps: cfg.lsd_decode_steps,
            eos_threshold: cfg.eos_threshold,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.mimi.sample_rate
    }

    /// Initialize flow LM state.
    pub fn init_flow_lm_state(&self) -> TTSState {
        TTSState {
            flow_lm_state: self.flow_lm.init_state(),
        }
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let dev = text_embeddings.device();
        let dtype = text_embeddings.dtype();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dtype, dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// Run one autoregressive generation step.
    /// Returns (next_latent [B, 1, ldim], is_eos).
    pub fn generate_step(
        &self,
        state: &mut TTSState,
        backbone_input: &Tensor,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, bool)> {
        let dev = backbone_input.device();
        let dtype = backbone_input.dtype();
        let empty_text =
            Tensor::zeros((1, 0, self.flow_lm.conditioner.dim), dtype, dev)?;

        let (latent, is_eos) = self.flow_lm.sample_next_latent(
            backbone_input,
            &empty_text,
            &mut state.flow_lm_state,
            self.lsd_decode_steps,
            rng,
            self.eos_threshold,
        )?;

        Ok((latent, is_eos))
    }

    /// Decode latent to audio using mimi (streaming).
    pub fn decode_latent(
        &self,
        latent: &Tensor,
        mimi_state: &mut MimiState,
    ) -> Result<Tensor> {
        {
            let d = latent.flatten_all()?.to_vec1::<f32>()?;
            let std = self.flow_lm.emb_std.to_vec1::<f32>()?;
            let mean = self.flow_lm.emb_mean.to_vec1::<f32>()?;
            eprintln!("[DECODE] latent: first4={:?} emb_std: first4={:?} emb_mean: first4={:?}",
                &d[..4.min(d.len())], &std[..4.min(std.len())], &mean[..4.min(mean.len())]);
        }
        let denorm = latent
            .broadcast_mul(&self.flow_lm.emb_std)?
            .broadcast_add(&self.flow_lm.emb_mean)?;
        {
            let d = denorm.flatten_all()?.to_vec1::<f32>()?;
            eprintln!("[DECODE] denorm: first8={:?}", &d[..8.min(d.len())]);
        }

        // [B, T, C] -> [B, C, T]
        let transposed = denorm.transpose(1, 2)?.contiguous()?;
        // DummyQuantizer: project latent [B, quantizer_dim, T] -> [B, output_dim, T]
        let quantized = self.mimi.quantizer.forward(&transposed)?;
        {
            let d = quantized.flatten_all()?.to_vec1::<f32>()?;
            eprintln!("[DECODE] quantized: shape={:?} first8={:?} last4={:?}",
                quantized.shape(), &d[..8.min(d.len())], &d[d.len().saturating_sub(4)..]);
        }
        self.mimi.decode_from_latent(&quantized, mimi_state)
    }

    /// Initialize mimi streaming state.
    pub fn init_mimi_state(&self, batch_size: usize, device: &Device) -> Result<MimiState> {
        self.mimi.init_state(batch_size, device)
    }

    fn run_backbone_and_increment(
        &self,
        state: &mut TTSState,
        text_embeddings: &Tensor,
        backbone_input_latents: &Tensor,
    ) -> Result<()> {
        let input = if backbone_input_latents.dim(1)? == 0 {
            text_embeddings.clone()
        } else {
            let projected = self.flow_lm.input_linear.forward(backbone_input_latents)?;
            Tensor::cat(&[text_embeddings, &projected], 1)?
        };
        let _out = self
            .flow_lm
            .transformer
            .forward(&input, &mut state.flow_lm_state.transformer_state)?;
        Ok(())
    }
}

pub const MAX_TOKENS_PER_CHUNK: usize = 50;

/// Prepare text for generation: capitalize, add punctuation, pad short text.
/// Returns (prepared_text, frames_after_eos).
pub fn prepare_text_prompt(text: &str) -> (String, usize) {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return (text, 3);
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

    let number_of_words = text.split_whitespace().count();
    let frames_after_eos = if number_of_words <= 4 { 3 } else { 1 };
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        text = first.to_uppercase().to_string() + chars.as_str();
    }
    if text.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        text.push('.');
    }
    if text.split_whitespace().count() < 5 {
        text = format!("        {text}");
    }
    (text, frames_after_eos)
}
