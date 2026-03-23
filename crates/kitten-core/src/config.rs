#[derive(Clone, Debug)]
pub struct KittenConfig {
    pub n_token: usize,
    pub bert_embed_dim: usize,
    pub bert_hidden_dim: usize,
    pub bert_ffn_dim: usize,
    pub bert_n_heads: usize,
    pub bert_max_pos: usize,
    pub hidden_dim: usize,
    pub style_dim: usize,
    pub lstm_hidden: usize,
    pub predictor_conv_dim: usize,
    pub decoder_dim: usize,
    pub generator_channels: Vec<usize>,
    pub generator_upsample_rates: Vec<usize>,
    pub generator_upsample_kernels: Vec<usize>,
    pub n_harmonics: usize,
    pub post_conv_channels: usize,
    pub sample_rate: usize,
    pub max_duration: usize,
}

impl KittenConfig {
    pub fn nano() -> Self {
        Self {
            n_token: 179,
            bert_embed_dim: 128,
            bert_hidden_dim: 768,
            bert_ffn_dim: 2048,
            bert_n_heads: 12,
            bert_max_pos: 512,
            hidden_dim: 128,
            style_dim: 256,
            lstm_hidden: 64,
            predictor_conv_dim: 128,
            decoder_dim: 256,
            generator_channels: vec![256, 128, 64],
            generator_upsample_rates: vec![10, 6],
            generator_upsample_kernels: vec![20, 12],
            n_harmonics: 11,
            post_conv_channels: 22,
            sample_rate: 24000,
            max_duration: 50,
        }
    }
}
