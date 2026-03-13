//! Compare Burn (GPU) and candle (CPU) Llama forward pass outputs.
//!
//! Loads the same GGUF model in both backends, feeds identical inputs,
//! and checks that hidden states match within floating point tolerance.
//!
//! Run with: cargo test -p tada-wasm --test compare_burn_candle -- --nocapture

use std::fs;
use std::io::Cursor;

use burn::backend::wgpu::WgpuDevice;

use tada_core::config::TadaConfig;
use tada_wasm::gguf;
use tada_wasm::model;

const GGUF_PATH: &str = "/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf";

#[test]
fn compare_hidden_states() {
    let cfg = TadaConfig::tada_1b();
    let acoustic_dim = cfg.acoustic_dim;
    let hidden_size = cfg.llama.hidden_size;

    // Check that GGUF file exists
    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("Skipping test: GGUF file not found at {GGUF_PATH}");
        return;
    }

    eprintln!("[test] Loading GGUF file ({GGUF_PATH})...");
    let gguf_bytes = fs::read(GGUF_PATH).expect("Failed to read GGUF file");

    // --- Load candle model (CPU) ---
    eprintln!("[test] Loading candle model...");
    let candle_model = tada_core::tada_model::TadaModel::load_gguf(
        &gguf_bytes,
        &cfg,
        &candle_core::Device::Cpu,
    )
    .expect("Failed to load candle model");

    // --- Load Burn model (GPU) ---
    eprintln!("[test] Loading Burn model...");
    let device = WgpuDevice::default();
    let cursor = Cursor::new(gguf_bytes);
    let mut reader = gguf::GgufReader::open(cursor).expect("Failed to parse GGUF");
    let burn_model = model::load_tada_llama_gguf(&mut reader, &device)
        .expect("Failed to load Burn model");
    drop(reader);

    eprintln!("[test] Models loaded. Running comparison...");

    // --- Test inputs ---
    let token_id: u32 = 128000; // BOS token
    let acoustic = vec![0.0f32; acoustic_dim];
    let acoustic_mask: u32 = 0;
    let time_before: u32 = 0;
    let time_after: u32 = 0;

    // --- Candle forward ---
    let mut candle_model = candle_model;
    let candle_device = candle_core::Device::Cpu;

    let token_tensor = candle_core::Tensor::from_vec(vec![token_id], (1, 1), &candle_device)
        .unwrap();
    let acoustic_tensor = candle_core::Tensor::from_vec(
        acoustic.clone(),
        (1, 1, acoustic_dim),
        &candle_device,
    )
    .unwrap();
    let mask_tensor = candle_core::Tensor::from_vec(vec![acoustic_mask], (1, 1), &candle_device)
        .unwrap();
    let time_before_tensor = candle_core::Tensor::from_vec(vec![time_before], (1, 1), &candle_device)
        .unwrap();
    let time_after_tensor = candle_core::Tensor::from_vec(vec![time_after], (1, 1), &candle_device)
        .unwrap();

    let input_embeds = candle_model
        .build_input_embeds(
            &token_tensor,
            &acoustic_tensor,
            &mask_tensor,
            &time_before_tensor,
            &time_after_tensor,
        )
        .unwrap();

    let candle_hidden = candle_model.forward_step(&input_embeds).unwrap();
    let candle_vec: Vec<f32> = candle_hidden.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(candle_vec.len(), hidden_size);

    // --- Burn forward ---
    let mut cache = burn_model.create_cache(256);
    let burn_hidden = burn_model.forward_step(
        token_id,
        &acoustic,
        acoustic_mask,
        time_before,
        time_after,
        &mut cache,
    );
    let burn_data = burn_hidden.into_data();
    let burn_vec: Vec<f32> = burn_data.to_vec().unwrap();
    assert_eq!(burn_vec.len(), hidden_size);

    // --- Compare ---
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f32;

    for (i, (&c, &b)) in candle_vec.iter().zip(burn_vec.iter()).enumerate() {
        let abs_diff = (c - b).abs();
        let rel_diff = if c.abs() > 1e-6 {
            abs_diff / c.abs()
        } else {
            abs_diff
        };

        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        sum_abs_diff += abs_diff as f64;
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }

        if i < 8 {
            eprintln!("  [{i:4}] candle={c:12.6} burn={b:12.6} diff={abs_diff:.6e}");
        }
    }

    let mean_abs_diff = sum_abs_diff / hidden_size as f64;

    eprintln!("[test] Results:");
    eprintln!("  max_abs_diff = {max_abs_diff:.6e}");
    eprintln!("  mean_abs_diff = {mean_abs_diff:.6e}");
    eprintln!("  max_rel_diff = {max_rel_diff:.6e}");

    // Q4_0 dequantization introduces some error, so be generous.
    // Both models dequant Q4_0 independently so we expect small diffs
    // from float ordering and GPU vs CPU arithmetic.
    assert!(
        max_abs_diff < 0.1,
        "Hidden states diverge too much: max_abs_diff={max_abs_diff}"
    );
    eprintln!("[test] PASSED — hidden states match within tolerance.");
}
