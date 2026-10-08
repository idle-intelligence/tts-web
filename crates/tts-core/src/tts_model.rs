use crate::config::TTSConfig;
use crate::flow_lm::{FlowLM, FlowLMState, Rng};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use mimi_rs::gguf_loader::GgufTensors;
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

    pub fn load_gguf(gguf: &mut GgufTensors, cfg: &TTSConfig) -> Result<Self> {
        let flow_lm = FlowLM::load_gguf(gguf, "flow_lm", &cfg.flow_lm)?;
        let mimi = MimiModel::load_gguf(gguf, "mimi", &cfg.mimi)?;
        Ok(Self {
            flow_lm, mimi,
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
        let denorm = latent
            .broadcast_mul(&self.flow_lm.emb_std)?
            .broadcast_add(&self.flow_lm.emb_mean)?;

        // [B, T, C] -> [B, C, T]
        let transposed = denorm.transpose(1, 2)?.contiguous()?;
        // DummyQuantizer: project latent [B, quantizer_dim, T] -> [B, output_dim, T]
        let quantized = self.mimi.quantizer_forward(&transposed)?;
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

const TERMINAL_PUNCTUATION: &str = ".!?\u{2026}";
const WEAK_PUNCTUATION: &str = ",;:-\u{2013}\u{2014}";
const CLOSERS: &str = "\"'\u{201d}\u{2019})]\u{00bb}";

/// Port of `_ensure_terminal_punctuation` (`text_chunking.py:61-79`).
fn ensure_terminal_punctuation(text: &str) -> String {
    let core = text.trim_end_matches(|c| CLOSERS.contains(c) || c == ' ');
    let closers = text[core.len()..].trim();
    match core.chars().last() {
        None => text.to_string(),
        Some(last) if TERMINAL_PUNCTUATION.contains(last) => text.to_string(),
        Some(last) if WEAK_PUNCTUATION.contains(last) => {
            let core = core.trim_end_matches(|c| WEAK_PUNCTUATION.contains(c) || c == ' ');
            format!("{core}.{closers}")
        }
        Some(_) => format!("{text}."),
    }
}

/// Port of the `re.sub(r"([.!?…])\s*[,;:]", r"\1", text)` cleanup applied
/// right after `replace_characters` (`text_chunking.py:28`): a terminal
/// punctuation mark immediately followed by (optional whitespace then) a
/// weak one collapses to just the terminal mark.
fn collapse_terminal_then_weak_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        out.push(c);
        if TERMINAL_PUNCTUATION.contains(c) {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && ",;:".contains(chars[j]) {
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Prepare text for generation: apply per-language character replacements,
/// capitalize, add terminal punctuation, pad short text. Returns
/// (prepared_text, frames_after_eos). Ports `prepare_text_prompt`
/// (`pocket_tts/models/text_chunking.py:15-53`).
///
/// `model_recommended_frames_after_eos` mirrors
/// `Config.model_recommended_frames_after_eos`: when a model config sets it,
/// it overrides the text-length-based guess entirely (`tts_model.py:720-731`).
/// None of the currently shipped language configs set it.
pub fn prepare_text_prompt(
    text: &str,
    model_recommended_frames_after_eos: Option<usize>,
    text_config: &crate::text_config::TextConfig,
) -> (String, usize) {
    let mut text = text.trim().to_string();

    if !text_config.replace_characters.is_empty() {
        let mut replaced = String::new();
        for c in text.chars() {
            match text_config.replace_characters.iter().find(|(k, _)| *k == c) {
                Some((_, rep)) => replaced.push_str(rep),
                None => replaced.push(c),
            }
        }
        text = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
        text = collapse_terminal_then_weak_punctuation(&text);
    }

    if text.is_empty() {
        return (text, model_recommended_frames_after_eos.unwrap_or(3));
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

    if text_config.remove_semicolons {
        text = text.replace(';', ",");
    }

    let number_of_words = text.split_whitespace().count();
    let frames_after_eos_guess = if number_of_words <= 4 { 3 } else { 1 };
    // tts_model.py:728: `frames_after_eos_guess += 2` before it is used, unless
    // the model config supplies its own recommended value.
    let frames_after_eos =
        model_recommended_frames_after_eos.unwrap_or(frames_after_eos_guess + 2);

    let mut chars = text.chars();
    if let Some(first) = chars.next()
        && !first.is_uppercase()
    {
        text = first.to_uppercase().to_string() + chars.as_str();
    }

    text = ensure_terminal_punctuation(&text);

    if text_config.pad_with_spaces_for_short_inputs && text.split_whitespace().count() < 5 {
        text = format!("        {text}");
    }
    (text, frames_after_eos)
}

#[cfg(test)]
mod prepare_text_prompt_tests {
    use super::prepare_text_prompt;
    use crate::text_config::TextConfig;

    #[test]
    fn english_short_input_is_not_padded() {
        // pad_with_spaces_for_short_inputs defaults to false for every
        // currently shipped language config, including English.
        let (text, _) = prepare_text_prompt("Hi", None, &TextConfig::ENGLISH);
        assert_eq!(text, "Hi.");
    }

    #[test]
    fn english_frames_after_eos_gets_the_plus_two() {
        let (_, frames) = prepare_text_prompt("Hello there, this is a test.", None, &TextConfig::ENGLISH);
        assert_eq!(frames, 1 + 2);
        let (_, frames) = prepare_text_prompt("Hi", None, &TextConfig::ENGLISH);
        assert_eq!(frames, 3 + 2);
    }

    #[test]
    fn model_recommended_frames_after_eos_overrides_the_guess() {
        let (_, frames) = prepare_text_prompt("Hi", Some(5), &TextConfig::ENGLISH);
        assert_eq!(frames, 5);
    }

    #[test]
    fn french_strips_quotes_and_parens_and_normalizes_apostrophes() {
        let (text, _) = prepare_text_prompt(
            "\u{201c}Bonjour\u{201d} (il dit) l\u{2019}heure est venue.",
            None,
            &TextConfig::FRENCH,
        );
        assert_eq!(text, "Bonjour il dit l'heure est venue.");
    }

    #[test]
    fn french_replaces_colon_with_comma() {
        let (text, _) = prepare_text_prompt("Voici: le résultat.", None, &TextConfig::FRENCH);
        assert_eq!(text, "Voici, le résultat.");
    }

    #[test]
    fn french_removes_semicolons() {
        let (text, _) = prepare_text_prompt(
            "Premier point; deuxième point.",
            None,
            &TextConfig::FRENCH,
        );
        assert_eq!(text, "Premier point, deuxième point.");
    }

    #[test]
    fn german_removes_semicolons_but_keeps_colon() {
        let (text, _) = prepare_text_prompt("Erstens; zweitens: fertig.", None, &TextConfig::GERMAN);
        assert_eq!(text, "Erstens, zweitens: fertig.");
    }

    #[test]
    fn spanish_strips_inverted_punctuation() {
        let (text, _) = prepare_text_prompt(
            "\u{00a1}Hola! \u{00bf}Qué tal?",
            None,
            &TextConfig::SPANISH,
        );
        assert_eq!(text, "Hola! Qué tal?");
    }

    #[test]
    fn terminal_punctuation_left_alone_when_already_present() {
        let (text, _) = prepare_text_prompt("Already done!", None, &TextConfig::ENGLISH);
        assert_eq!(text, "Already done!");
    }

    #[test]
    fn trailing_weak_punctuation_becomes_a_period() {
        let (text, _) = prepare_text_prompt("Wait for it,", None, &TextConfig::ENGLISH);
        assert_eq!(text, "Wait for it.");
    }

    #[test]
    fn trailing_comma_before_closing_quote_becomes_period() {
        // Matches text_chunking.py:73-78: a closing quote after weak
        // punctuation keeps its place after the inserted period.
        let (text, _) = prepare_text_prompt("She said \"wait for it,\"", None, &TextConfig::ENGLISH);
        assert_eq!(text, "She said \"wait for it.\"");
    }

    #[test]
    fn empty_input_returns_empty_with_default_frames() {
        let (text, frames) = prepare_text_prompt("   ", None, &TextConfig::ENGLISH);
        assert_eq!(text, "");
        assert_eq!(frames, 3);
    }

    #[test]
    fn removing_characters_can_empty_the_text() {
        let (text, frames) = prepare_text_prompt("()", None, &TextConfig::FRENCH);
        assert_eq!(text, "");
        assert_eq!(frames, 3);
    }
}
