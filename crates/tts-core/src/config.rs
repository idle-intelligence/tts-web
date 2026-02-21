use mimi_rs::config::MimiConfig;

pub struct FlowLMConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub max_period: f64,
    pub n_bins: usize,
    pub lut_dim: usize,
    pub flow_dim: usize,
    pub flow_depth: usize,
    pub ldim: usize,
}

pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
}

impl TTSConfig {
    pub fn v202601(temp: f32) -> Self {
        Self {
            flow_lm: FlowLMConfig {
                d_model: 1024,
                num_heads: 16,
                num_layers: 6,
                dim_feedforward: 4096,
                max_period: 10000.0,
                n_bins: 4000,
                lut_dim: 1024,
                flow_dim: 512,
                flow_depth: 6,
                ldim: 32,
            },
            mimi: MimiConfig::v202601(),
            temp,
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
        }
    }
}
