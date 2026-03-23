/// Llama-style LLM backbone configuration.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<RopeScaling>,
}

/// RoPE scaling parameters for extended context (Llama 3.1-style).
#[derive(Clone, Debug)]
pub struct RopeScaling {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

/// Decoder (Mimi-style audio decoder) configuration.
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_attn_layers: usize,
    pub num_attn_heads: usize,
    pub attn_dim_feedforward: usize,
    pub wav_decoder_channels: usize,
    pub strides: Vec<usize>,
}

/// Top-level TADA model configuration combining the LLM backbone,
/// acoustic prediction head, and decoder.
#[derive(Clone, Debug)]
pub struct TadaConfig {
    pub llama: LlamaConfig,
    pub decoder: DecoderConfig,

    /// Acoustic latent dimension (512).
    pub acoustic_dim: usize,
    /// Number of discrete time classes for gray-code time encoding (256).
    pub num_time_classes: usize,
    /// Shift applied to acoustic token positions (5).
    pub shift_acoustic: usize,
    /// Number of transformer layers in the diffusion prediction head (6).
    pub head_layers: usize,
    /// FFN expansion ratio in the prediction head (4.0).
    pub head_ffn_ratio: f64,
    /// Mean for acoustic latent normalization.
    pub acoustic_mean: f64,
    /// Std for acoustic latent normalization.
    pub acoustic_std: f64,
}

impl TadaConfig {
    /// Number of bits needed to gray-code encode `num_time_classes` values.
    /// ceil(log2(num_time_classes)) — e.g. 8 for 256 classes.
    pub fn num_time_bits(&self) -> usize {
        if self.num_time_classes <= 1 {
            return 1;
        }
        // ceil(log2(n)) = 64 - (n-1).leading_zeros() for n>1
        let n = self.num_time_classes as u64;
        (64 - (n - 1).leading_zeros()) as usize
    }

    /// Dimension of the time embedding: 2 * num_time_bits (float {-1,1} encoding).
    /// 16 for 256 classes.
    pub fn time_dim(&self) -> usize {
        2 * self.num_time_bits()
    }

    /// Total latent dimension fed into the prediction head's final layer:
    /// acoustic_dim + time_dim = 528 for the default config.
    pub fn total_latent_dim(&self) -> usize {
        self.acoustic_dim + self.time_dim()
    }

    /// Factory for the TADA 1B model (`kyutai/tada-1b`).
    pub fn tada_1b() -> Self {
        Self {
            llama: LlamaConfig {
                hidden_size: 2048,
                intermediate_size: 8192,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                num_hidden_layers: 16,
                head_dim: 64,
                vocab_size: 128256,
                rms_norm_eps: 1e-5,
                rope_theta: 500000.0,
                max_position_embeddings: 131072,
                tie_word_embeddings: true,
                rope_scaling: Some(RopeScaling {
                    factor: 32.0,
                    high_freq_factor: 4.0,
                    low_freq_factor: 1.0,
                    original_max_position_embeddings: 8192,
                }),
            },
            decoder: DecoderConfig {
                embed_dim: 512,
                hidden_dim: 1024,
                num_attn_layers: 6,
                num_attn_heads: 8,
                attn_dim_feedforward: 4096,
                wav_decoder_channels: 1536,
                strides: vec![4, 4, 5, 6],
            },
            acoustic_dim: 512,
            num_time_classes: 256,
            shift_acoustic: 5,
            head_layers: 6,
            head_ffn_ratio: 4.0,
            acoustic_mean: 0.0,
            acoustic_std: 1.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tada_1b_derived_dims() {
        let cfg = TadaConfig::tada_1b();
        assert_eq!(cfg.num_time_bits(), 8);
        assert_eq!(cfg.time_dim(), 16);
        assert_eq!(cfg.total_latent_dim(), 528);
    }

    #[test]
    fn test_num_time_bits_edge_cases() {
        let mut cfg = TadaConfig::tada_1b();

        cfg.num_time_classes = 1;
        assert_eq!(cfg.num_time_bits(), 1);

        cfg.num_time_classes = 2;
        assert_eq!(cfg.num_time_bits(), 1);

        cfg.num_time_classes = 3;
        assert_eq!(cfg.num_time_bits(), 2);

        cfg.num_time_classes = 128;
        assert_eq!(cfg.num_time_bits(), 7);

        cfg.num_time_classes = 256;
        assert_eq!(cfg.num_time_bits(), 8);

        cfg.num_time_classes = 257;
        assert_eq!(cfg.num_time_bits(), 9);
    }
}
