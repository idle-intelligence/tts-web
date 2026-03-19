use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::bert::AlbertEncoder;
use crate::config::KittenConfig;
use crate::decoder::Decoder;
use crate::predictor::Predictor;
use crate::text_encoder::TextEncoder;

pub struct KittenModel {
    bert: AlbertEncoder,
    text_encoder: TextEncoder,
    predictor: Predictor,
    decoder: Decoder,
    cfg: KittenConfig,
}

impl KittenModel {
    pub fn load(data: &[u8], cfg: &KittenConfig, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(data.to_vec(), DType::F32, device)?;

        let bert = AlbertEncoder::load(vb.clone(), cfg)?;
        let text_encoder = TextEncoder::load(vb.clone(), cfg)?;
        let predictor = Predictor::load(vb.clone(), cfg)?;
        let decoder = Decoder::load(vb, cfg)?;

        Ok(Self { bert, text_encoder, predictor, decoder, cfg: cfg.clone() })
    }

    /// Synthesize audio from phoneme IDs.
    ///
    /// - `phoneme_ids`: sequence of phoneme IDs
    /// - `style`: `[1, 256]` style vector
    /// - `speed`: speaking rate multiplier (1.0 = normal)
    ///
    /// Returns PCM samples at `cfg.sample_rate` Hz.
    pub fn synthesize(&self, phoneme_ids: &[i32], style: &Tensor, speed: f32) -> Result<Vec<f32>> {
        let device = style.device();

        // 1. Build input_ids tensor [1, seq] as U32
        let ids_u32: Vec<u32> = phoneme_ids.iter().map(|&x| x as u32).collect();
        let seq_len = ids_u32.len();
        let input_ids = Tensor::from_vec(ids_u32, (1, seq_len), device)?;

        // 2. BERT → [1, seq, 128]
        let bert_out = self.bert.forward(&input_ids)?;

        // 3. Text encoder → (lstm_features [1, seq, 256], cnn_features [1, 128, seq])
        let (lstm_features, cnn_features) = self.text_encoder.forward(&bert_out, style)?;

        // 4. Predictor → (durations, expanded_features, shared_lstm_out, f0, n_amp)
        let (durations, _expanded_features, shared_lstm_out, f0, n_amp) =
            self.predictor.forward(&lstm_features, style, speed)?;

        // 5. Duration-expand cnn_features [1, 128, seq] → [1, 128, T]
        //    cnn_features is NCL; expand each position by its duration.
        let asr_features = expand_cnn_features(&cnn_features, &durations)?;

        // 6. Decoder → waveform [1, 1, num_samples]
        // shared_lstm_out [1, 128, T] → encode block input
        // asr_features    [1, 128, T] → asr_res projection
        // f0/n_amp        [1, 1, T]   — from predictor F0/N branches
        let waveform = self.decoder.forward(&shared_lstm_out, &asr_features, &f0, &n_amp, style)?;

        // 7. Trim last 5000 samples and return as Vec<f32>
        let samples = waveform.i((0, 0, ..))?; // [num_samples]
        let n = samples.dim(0)?;
        let trim = n.saturating_sub(5000);
        let samples = if trim > 0 {
            samples.i(..trim)?
        } else {
            samples
        };

        Ok(samples.to_vec1::<f32>()?)
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.sample_rate
    }
}

/// Expand `cnn_features` [batch, C, seq] by `durations` [batch, seq] (i64).
/// Returns [batch, C, T] where T = sum(durations[b]).
fn expand_cnn_features(cnn_features: &Tensor, durations: &Tensor) -> Result<Tensor> {
    let (batch, c, _seq) = cnn_features.dims3()?;
    let device = cnn_features.device();
    let dtype = cnn_features.dtype();

    let dur_vec = durations.to_vec2::<i64>()?;

    let mut batch_outputs = Vec::with_capacity(batch);
    for (b, durs) in dur_vec.iter().enumerate().take(batch) {
        let feats = cnn_features.i(b)?; // [C, seq]
        let total: i64 = durs.iter().sum();
        let total = total as usize;

        let mut frames: Vec<Tensor> = Vec::with_capacity(total);
        for (s, &d) in durs.iter().enumerate() {
            let col = feats.i((.., s))?; // [C]
            for _ in 0..d as usize {
                frames.push(col.clone());
            }
        }

        let expanded = if frames.is_empty() {
            Tensor::zeros((c, 0), dtype, device)?
        } else {
            // Stack [C] columns along dim 1: [C, total]
            Tensor::stack(&frames, 1)?
        };
        batch_outputs.push(expanded.unsqueeze(0)?); // [1, C, total]
    }

    // Pad to max total length across batch
    let max_t = batch_outputs.iter().map(|t: &Tensor| t.dim(2).unwrap_or(0)).max().unwrap_or(0);
    let mut padded = Vec::with_capacity(batch);
    for out in batch_outputs {
        let t = out.dim(2)?;
        let p = if t < max_t {
            let pad = Tensor::zeros((1, c, max_t - t), dtype, device)?;
            Tensor::cat(&[out, pad], 2)?
        } else {
            out
        };
        padded.push(p);
    }

    Ok(Tensor::cat(&padded, 0)?) // [batch, C, max_t]
}
