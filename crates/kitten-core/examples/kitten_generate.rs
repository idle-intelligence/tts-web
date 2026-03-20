/// Kitten TTS generation example.
///
/// Usage:
///   cargo run --example kitten_generate -p kitten-core --release -- \
///     [--model /path/to/kitten-nano.safetensors] \
///     [--voices /path/to/kitten-voices.safetensors] \
///     [--voice jasper] \
///     [--text "Hello world"] \
///     [--speed 1.0] \
///     [--output /tmp/kitten_out.wav]
///
/// Writes a Float32 PCM WAV at 24000 Hz (mono).

use candle_core::{DType, Device, IndexOp, Tensor};
use kitten_core::config::KittenConfig;
use kitten_core::kitten_model::KittenModel;
use kitten_core::phoneme_map::map_phonemes_to_ids;
use safetensors::SafeTensors;

// ---------------------------------------------------------------------------
// Debug tensor writer
// ---------------------------------------------------------------------------

/// Write tensor data as raw f32 LE bytes.
/// Filename: `<dir>/<name>_<d0>x<d1>x...xdN.bin`
fn write_debug_tensor(dir: &str, name: &str, tensor: &Tensor) -> anyhow::Result<()> {
    use std::io::Write;
    let shape = tensor.shape().dims();
    let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
    let path = format!("{dir}/{name}_{shape_str}.bin");
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut f = std::fs::File::create(&path)?;
    for v in &data {
        f.write_all(&v.to_le_bytes())?;
    }
    eprintln!("  [debug] saved {path}  shape={shape:?}");
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI arg parsing (manual, no clap)
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    voices_path: String,
    voice_name: String,
    text: String,
    speed: f32,
    output_path: String,
    /// If Some, save all intermediate tensors as .bin files to this directory.
    debug_dir: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::from("/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-nano.safetensors");
    let mut voices_path = String::from("/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-voices.safetensors");
    let mut voice_name = String::from("jasper");
    let mut text = String::from("Hello, world.");
    let mut speed = 1.0f32;
    let mut output_path = String::from("/tmp/kitten_out.wav");
    let mut debug_dir: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"     => { i += 1; model_path  = args[i].clone(); }
            "--voices"    => { i += 1; voices_path = args[i].clone(); }
            "--voice"     => { i += 1; voice_name  = args[i].clone(); }
            "--text"      => { i += 1; text        = args[i].clone(); }
            "--speed"     => { i += 1; speed       = args[i].parse().expect("speed must be f32"); }
            "--output"    => { i += 1; output_path = args[i].clone(); }
            "--debug-dir" => { i += 1; debug_dir   = Some(args[i].clone()); }
            "--help" | "-h" => {
                eprintln!("Usage: kitten_generate [--model PATH] [--voices PATH] [--voice NAME] [--text TEXT] [--speed FLOAT] [--output PATH] [--debug-dir DIR]");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args { model_path, voices_path, voice_name, text, speed, output_path, debug_dir }
}

// ---------------------------------------------------------------------------
// WAV writer (Float32 PCM IEEE, mono)
// ---------------------------------------------------------------------------

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    let num_samples = samples.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size = num_samples * 4;
    let chunk_size = 36 + data_size;

    f.write_all(b"RIFF")?;
    f.write_all(&chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    f.write_all(b"fmt ")?;
    f.write_all(&18u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?; // IEEE_FLOAT
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?; // cbSize

    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phonemize via espeak-ng subprocess
// ---------------------------------------------------------------------------

fn phonemize(text: &str) -> anyhow::Result<String> {
    let output = std::process::Command::new("espeak-ng")
        .args(["--ipa", "-q", text])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to run espeak-ng: {e}. Is it installed?"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("espeak-ng failed: {stderr}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let args = parse_args();

    eprintln!("[1] Phonemizing: {:?}", args.text);
    let t0 = std::time::Instant::now();
    let ipa = phonemize(&args.text)?;
    eprintln!("  IPA: {:?}", ipa.trim());
    let phoneme_ids = map_phonemes_to_ids(&ipa);
    eprintln!("  IDs ({} tokens): {:?}", phoneme_ids.len(), &phoneme_ids[..phoneme_ids.len().min(20)]);

    let device = Device::Cpu;
    let cfg = KittenConfig::nano();

    eprintln!("[2] Loading model: {}", args.model_path);
    let t_load = std::time::Instant::now();
    let model_data = std::fs::read(&args.model_path)
        .map_err(|e| anyhow::anyhow!("cannot read model {}: {e}", args.model_path))?;
    let model = KittenModel::load(&model_data, &cfg, &device)?;
    eprintln!("  loaded in {:.2}s", t_load.elapsed().as_secs_f32());

    eprintln!("[3] Loading voices: {}", args.voices_path);
    let voices_data = std::fs::read(&args.voices_path)
        .map_err(|e| anyhow::anyhow!("cannot read voices {}: {e}", args.voices_path))?;

    eprintln!("[4] Selecting style for voice {:?}", args.voice_name);
    let style = load_style(&voices_data, &args.voice_name, args.text.len(), &device)?;
    eprintln!("  style shape: {:?}", style.shape());

    eprintln!("[5] Synthesizing (speed={:.2})...", args.speed);
    let t_gen = std::time::Instant::now();

    // If --debug-dir is set, run debug_forward and save all intermediates.
    if let Some(ref dir) = args.debug_dir {
        eprintln!("  [debug] saving intermediates to: {dir}");
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow::anyhow!("cannot create debug dir {dir}: {e}"))?;

        let dbg = model.debug_forward(&phoneme_ids, &style, args.speed)?;

        write_debug_tensor(dir, "bert_output",    &dbg.bert_output)?;
        write_debug_tensor(dir, "lstm_features",  &dbg.lstm_features)?;
        write_debug_tensor(dir, "cnn_features",   &dbg.cnn_features)?;
        write_debug_tensor(dir, "durations",      &dbg.durations.to_dtype(candle_core::DType::F32)?)?;
        write_debug_tensor(dir, "expanded_features", &dbg.expanded_features)?;
        write_debug_tensor(dir, "shared_lstm_out",&dbg.shared_lstm_out)?;
        write_debug_tensor(dir, "f0",             &dbg.f0)?;
        write_debug_tensor(dir, "n_amp",          &dbg.n_amp)?;
        write_debug_tensor(dir, "waveform",       &dbg.waveform)?;

        eprintln!("  [debug] done — run: python scripts/compare_pipeline.py --debug-dir {dir}");
    }

    let samples = model.synthesize(&phoneme_ids, &style, args.speed)?;
    let gen_elapsed = t_gen.elapsed().as_secs_f32();

    let sample_rate = model.sample_rate() as u32;
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    let rtf = gen_elapsed / duration_secs;
    eprintln!(
        "  {} samples, {:.2}s audio, gen={:.2}s, RTF={:.3}",
        samples.len(), duration_secs, gen_elapsed, rtf
    );

    eprintln!("[6] Writing WAV: {}", args.output_path);
    write_wav(&args.output_path, &samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("failed to write WAV: {e}"))?;
    eprintln!("  done. Total elapsed: {:.2}s", t0.elapsed().as_secs_f32());

    Ok(())
}

/// Load a style vector for the given voice name.
///
/// The voices safetensors contains tensors named `<voice_name>` with shape
/// `[400, 256]` (one style vector per text length bucket 0..=399).
/// We pick `min(text_len, 399)` and return `[1, 256]`.
fn load_style(
    voices_data: &[u8],
    voice_name: &str,
    text_len: usize,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let st = SafeTensors::deserialize(voices_data)
        .map_err(|e| anyhow::anyhow!("cannot parse voices safetensors: {e}"))?;

    let tensor_view = st.tensor(voice_name)
        .map_err(|_| {
            let names = st.names();
            anyhow::anyhow!("voice {:?} not found. Available: {:?}", voice_name, names)
        })?;

    // voices tensor: [400, 256] in F32 (or BF16 — convert)
    let data = tensor_view.data();
    let shape = tensor_view.shape();
    let dtype = tensor_view.dtype();

    let rows = shape[0];
    let cols = shape[1];

    // Build an F32 candle tensor
    let flat_f32: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        other => return Err(anyhow::anyhow!("unsupported voices dtype: {:?}", other)),
    };

    let t = Tensor::from_vec(flat_f32, (rows, cols), device)?
        .to_dtype(DType::F32)?;

    // Pick row index = min(text_len, rows - 1)
    let idx = text_len.min(rows.saturating_sub(1));
    let style = t.i(idx)?.unsqueeze(0)?; // [1, 256]

    Ok(style)
}
