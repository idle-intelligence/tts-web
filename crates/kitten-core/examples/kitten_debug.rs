/// Debug example: run KittenTTS with exact hardcoded inputs matching ONNX reference test.
///
/// Input IDs: [0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]
/// Style: voices.npz['expr-voice-2-m'][11] (from /tmp/test-voice.safetensors, row 11)
/// Speed: 1.0

use candle_core::{DType, Device, IndexOp, Tensor};
use kitten_core::config::KittenConfig;
use kitten_core::kitten_model::KittenModel;
use safetensors::SafeTensors;

fn main() -> anyhow::Result<()> {
    let model_path = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-nano.safetensors";
    let voices_path = "/tmp/test-voice.safetensors";

    let device = Device::Cpu;
    let cfg = KittenConfig::nano();

    eprintln!("[1] Loading model...");
    let model_data = std::fs::read(model_path)?;
    let model = KittenModel::load(&model_data, &cfg, &device)?;

    eprintln!("[2] Loading style (expr-voice-2-m row 11)...");
    let voices_data = std::fs::read(voices_path)?;
    let st = SafeTensors::deserialize(&voices_data)?;
    let tv = st.tensor("test-voice")?;
    let data = tv.data();
    let flat_f32: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let rows = tv.shape()[0];
    let cols = tv.shape()[1];
    let t = Tensor::from_vec(flat_f32, (rows, cols), &device)?;
    // Pick row 11
    let style = t.i(11)?.unsqueeze(0)?; // [1, 256]
    eprintln!("  style shape: {:?}", style.shape());
    let sv0 = style.i((0, ..5))?.to_vec1::<f32>()?;
    eprintln!("  style first 5: {:?}", sv0);
    let sv128 = style.i((0, 128..133))?.to_vec1::<f32>()?;
    eprintln!("  style [128:133]: {:?}", sv128);

    // Exact same input_ids as ONNX test
    let phoneme_ids: Vec<i32> = vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0];
    eprintln!("[3] phoneme_ids ({} tokens): {:?}", phoneme_ids.len(), &phoneme_ids);

    eprintln!("[4] Synthesizing...");
    let _samples = model.synthesize(&phoneme_ids, &style, 1.0)?;
    eprintln!("[5] Done. {} samples.", _samples.len());

    Ok(())
}
