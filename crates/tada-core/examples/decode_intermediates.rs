/// Decode acoustic intermediates saved from Python into audio via the TADA decoder.
///
/// Reads `samples/python_intermediates.bin` (binary format), loads only the
/// decoder portion of the TADA model from an F16 GGUF file, decodes to PCM,
/// and writes `samples/rust_decoder_from_intermediates.wav`.
///
/// Usage:
///   cargo run --example decode_intermediates -p tada-core --release

use candle_core::{Device, Result as CResult};
use tada_core::config::TadaConfig;
use tada_core::tada_model::TadaModel;

const MODEL_PATH: &str = "/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-f16.gguf";
const INPUT_PATH: &str = "samples/python_intermediates.bin";
const OUTPUT_PATH: &str = "samples/rust_decoder_from_intermediates.wav";

// ---------------------------------------------------------------------------
// WAV writer (Float32 PCM, format type 3, mono, 24000 Hz)
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
    f.write_all(&0u16.to_le_bytes())?; // cbSize = 0

    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Binary loader for python_intermediates.bin
// ---------------------------------------------------------------------------

struct Intermediates {
    acoustics: Vec<Vec<f32>>,
    times_before: Vec<u32>,
}

fn load_intermediates(path: &str) -> std::io::Result<Intermediates> {
    let data = std::fs::read(path)?;
    let mut offset = 0;

    let read_u32 = |data: &[u8], off: &mut usize| -> u32 {
        let val = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
        *off += 4;
        val
    };

    let read_f32 = |data: &[u8], off: &mut usize| -> f32 {
        let val = f32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
        *off += 4;
        val
    };

    // Binary format: num_frames(u32), num_times(u32), then acoustic data, then times.
    // acoustic_dim is implicit (512, from TadaConfig).
    let num_frames = read_u32(&data, &mut offset) as usize;
    let num_times = read_u32(&data, &mut offset) as usize;
    let acoustic_dim = 512; // implicit, matches TadaConfig::tada_1b()
    eprintln!("  num_frames={num_frames}, acoustic_dim={acoustic_dim}, num_times={num_times}");

    let mut acoustics = Vec::with_capacity(num_frames);
    for _ in 0..num_frames {
        let mut frame = Vec::with_capacity(acoustic_dim);
        for _ in 0..acoustic_dim {
            frame.push(read_f32(&data, &mut offset));
        }
        acoustics.push(frame);
    }

    // Read times_before (num_times already read from header)
    let mut times_before = Vec::with_capacity(num_times);
    for _ in 0..num_times {
        times_before.push(read_u32(&data, &mut offset));
    }

    assert_eq!(offset, data.len(), "did not consume all bytes");

    Ok(Intermediates {
        acoustics,
        times_before,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> CResult<()> {
    eprintln!("=== Decode Intermediates ===");

    // Load intermediates
    eprintln!("\n[1] Loading intermediates from {INPUT_PATH}...");
    let intermediates = load_intermediates(INPUT_PATH)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read intermediates: {e}")))?;

    eprintln!("  acoustics: {} frames", intermediates.acoustics.len());
    eprintln!("  times_before: {:?}", intermediates.times_before);

    // Acoustic stats
    let all_flat: Vec<f32> = intermediates
        .acoustics
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    let a_mean = all_flat.iter().sum::<f32>() / all_flat.len() as f32;
    let a_min = all_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let a_max = all_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let a_std = (all_flat
        .iter()
        .map(|x| (x - a_mean).powi(2))
        .sum::<f32>()
        / all_flat.len() as f32)
        .sqrt();
    eprintln!("  [acoustic stats] mean={a_mean:.4} std={a_std:.4} min={a_min:.4} max={a_max:.4}");

    // Load model
    eprintln!("\n[2] Loading model from {MODEL_PATH}...");
    let device = Device::new_metal(0).unwrap_or_else(|_| {
        eprintln!("Metal not available, falling back to CPU");
        Device::Cpu
    });
    eprintln!("  device: {:?}", device);

    let model_bytes = std::fs::read(MODEL_PATH)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read model: {e}")))?;
    eprintln!("  read {} MB", model_bytes.len() / (1024 * 1024));

    let cfg = TadaConfig::tada_1b();
    let model = TadaModel::load_gguf(&model_bytes, &cfg, &device)?;
    let sample_rate = TadaModel::sample_rate() as u32;
    eprintln!("  model loaded OK (sample_rate={sample_rate})");

    // Decode audio
    eprintln!("\n[3] Decoding audio...");
    let samples = model.decode_audio(&intermediates.acoustics, &intermediates.times_before)?;

    // Audio stats
    let duration = samples.len() as f64 / sample_rate as f64;
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
    eprintln!("\n[4] Audio stats:");
    eprintln!("  duration:  {duration:.3}s ({} samples @ {sample_rate} Hz)", samples.len());
    eprintln!("  peak:      {peak:.6}");
    eprintln!("  dc offset: {dc_offset:.6}");

    // Write WAV
    eprintln!("\n[5] Writing WAV to {OUTPUT_PATH}...");
    write_wav(OUTPUT_PATH, &samples, sample_rate)
        .map_err(|e| candle_core::Error::Msg(format!("failed to write WAV: {e}")))?;
    eprintln!("  wrote {} samples ({duration:.3}s)", samples.len());
    eprintln!("\nDone!");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
