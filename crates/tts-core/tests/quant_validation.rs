//! Integration test: validates INT8 dequantization produces results within
//! expected quantization error bounds compared to the original BF16 model.
//!
//! Run with:
//!   ORIGINAL_MODEL=/tmp/tts_original.safetensors \
//!   QUANTIZED_MODEL=model_int8.safetensors \
//!   cargo test --test quant_validation -- --nocapture
//!
//! Both env vars are required; the test is skipped if either is unset.

use std::collections::HashSet;
use std::path::PathBuf;

use half::bf16;
use safetensors::SafeTensors;

// ---------------------------------------------------------------------------
// Helpers: bf16 raw bytes → f32
// ---------------------------------------------------------------------------

fn bf16_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let raw = bf16::from_le_bytes([c[0], c[1]]);
            raw.to_f32()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dequantize: mirrors dequantize.rs logic exactly
// ---------------------------------------------------------------------------

fn dequantize(
    i8_bytes: &[u8],
    scale_bytes: &[u8],
    shape: &[usize],
) -> Vec<f32> {
    let out_channels = shape[0];
    let elements_per_channel: usize = shape[1..].iter().product();

    let scales = bf16_slice_to_f32(scale_bytes);
    assert_eq!(scales.len(), out_channels, "scale count mismatch");

    let i8_vals = i8_bytes;
    let total = out_channels * elements_per_channel;
    let mut result = vec![0f32; total];

    for ch in 0..out_channels {
        let s = scales[ch];
        let base = ch * elements_per_channel;
        for i in 0..elements_per_channel {
            let q = i8_vals[base + i] as i8;
            result[base + i] = (q as f32) * s;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Key remapping — mirrors dequantize.rs remap_key()
// ---------------------------------------------------------------------------

const SKIP_PATTERNS: &[&str] = &[
    "flow.w_s_t",
    "quantizer.vq",
    "quantizer.logvar_proj",
    "learnt_padding",
];

fn remap_key(name: &str) -> Option<String> {
    for pat in SKIP_PATTERNS {
        if name.contains(pat) {
            return None;
        }
    }

    let mut s = name.to_string();
    s = s.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    s = s.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    s = s.replace("flow_lm.backbone.", "flow_lm.transformer.");
    s = s.replace("flow_lm.flow.", "flow_lm.flow_net.");
    s = s.replace("mimi.model.", "mimi.");
    Some(s)
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct Stats {
    max_abs_err: f32,
    rmse: f32,
    sqnr_db: f32,
}

fn compute_stats(orig: &[f32], deq: &[f32]) -> Stats {
    assert_eq!(orig.len(), deq.len());
    let n = orig.len() as f32;

    let mut max_abs_err = 0f32;
    let mut sum_sq_err = 0f32;
    let mut sum_sq_signal = 0f32;

    for (&o, &d) in orig.iter().zip(deq.iter()) {
        let err = (o - d).abs();
        max_abs_err = max_abs_err.max(err);
        sum_sq_err += (o - d) * (o - d);
        sum_sq_signal += o * o;
    }

    let rmse = (sum_sq_err / n).sqrt();
    let signal_power = sum_sq_signal / n;
    let noise_power = sum_sq_err / n;
    let sqnr_db = if noise_power > 0.0 {
        10.0 * (signal_power / noise_power.max(1e-30)).log10()
    } else {
        f32::INFINITY
    };

    Stats { max_abs_err, rmse, sqnr_db }
}

// ---------------------------------------------------------------------------
// Expected key prefixes (model must have at least one key per prefix after remap)
// ---------------------------------------------------------------------------

const EXPECTED_PREFIXES: &[&str] = &[
    "flow_lm.conditioner.",
    "flow_lm.transformer.",
    "flow_lm.flow_net.",
    "flow_lm.input_linear.",
    "flow_lm.out_norm.",
    "flow_lm.out_eos.",
    "mimi.encoder.",
    "mimi.decoder.",
    "mimi.encoder_transformer.",
    "mimi.decoder_transformer.",
    "mimi.quantizer.",
    "mimi.downsample.",
    "mimi.upsample.",
];

const EXPECTED_SCALAR_KEYS: &[&str] = &[
    "flow_lm.emb_std",
    "flow_lm.emb_mean",
    "flow_lm.bos_emb",
    "flow_lm.speaker_proj_weight",
];

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn validate_int8_dequantization() {
    let original_path = match std::env::var("ORIGINAL_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: ORIGINAL_MODEL env var not set");
            return;
        }
    };
    let quantized_path = match std::env::var("QUANTIZED_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            // Default to the known location
            PathBuf::from("/Users/tc/Code/idle-intelligence/tts-web/model_int8.safetensors")
        }
    };

    if !original_path.exists() {
        eprintln!("SKIP: original model not found at {:?}", original_path);
        return;
    }
    if !quantized_path.exists() {
        eprintln!("SKIP: quantized model not found at {:?}", quantized_path);
        return;
    }

    println!("Original:  {:?}", original_path);
    println!("Quantized: {:?}", quantized_path);

    // Load both models
    let orig_bytes = std::fs::read(&original_path)
        .expect("failed to read original model");
    let quant_bytes = std::fs::read(&quantized_path)
        .expect("failed to read quantized model");

    let orig_st = SafeTensors::deserialize(&orig_bytes)
        .expect("failed to parse original safetensors");
    let quant_st = SafeTensors::deserialize(&quant_bytes)
        .expect("failed to parse quantized safetensors");

    // Build maps
    let orig_names: HashSet<String> = orig_st.iter().map(|(n, _)| n.to_string()).collect();

    // Find all INT8 tensors in the quantized file
    let scale_suffix = "_scale";
    let int8_names: Vec<String> = quant_st
        .iter()
        .filter(|(_, t)| t.dtype() == safetensors::Dtype::I8)
        .map(|(n, _)| n.to_string())
        .collect();

    println!("\nFound {} INT8 tensors in quantized model", int8_names.len());

    // --- Test 1: Dequantization accuracy ---
    println!("\n=== Test 1: Dequantization Accuracy ===\n");

    let sqnr_threshold = 30.0f32;
    let mut worst_sqnr = f32::INFINITY;
    let mut worst_name = String::new();
    let mut bad_tensors: Vec<String> = Vec::new();

    for name in &int8_names {
        if !orig_names.contains(name.as_str()) {
            println!("  [MISS] {name}: not in original!");
            bad_tensors.push(name.clone());
            continue;
        }

        let scale_name = format!("{name}{scale_suffix}");
        let scale_tensor = match quant_st.tensor(&scale_name) {
            Ok(t) => t,
            Err(_) => {
                println!("  [ERR ] {name}: scale tensor '{scale_name}' missing!");
                bad_tensors.push(name.clone());
                continue;
            }
        };

        let orig_tensor = orig_st.tensor(name).expect("tensor should exist");
        let quant_tensor = quant_st.tensor(name).expect("tensor should exist");

        let shape = orig_tensor.shape().to_vec();
        let orig_f32 = bf16_slice_to_f32(orig_tensor.data());
        let deq_f32 = dequantize(quant_tensor.data(), scale_tensor.data(), &shape);

        let stats = compute_stats(&orig_f32, &deq_f32);

        let status = if stats.sqnr_db < 20.0 {
            "BAD "
        } else if stats.sqnr_db < sqnr_threshold {
            "WARN"
        } else {
            "OK  "
        };

        println!(
            "  [{status}] {:60}  max_err={:.6}  rmse={:.7}  sqnr={:.1} dB",
            &name[..name.len().min(60)],
            stats.max_abs_err,
            stats.rmse,
            stats.sqnr_db
        );

        if stats.sqnr_db < worst_sqnr {
            worst_sqnr = stats.sqnr_db;
            worst_name = name.clone();
        }

        if stats.sqnr_db < 20.0 {
            bad_tensors.push(name.clone());
        }
    }

    println!("\n  Worst SQNR: {:.1} dB  ({})", worst_sqnr, worst_name);

    // --- Test 2: Key completeness after remapping ---
    println!("\n=== Test 2: Key Completeness After Remapping ===\n");

    let remapped_keys: HashSet<String> = quant_st
        .iter()
        .filter(|(name, _)| !name.ends_with(scale_suffix))
        .filter_map(|(name, _)| remap_key(name))
        .collect();

    // Also add INT8 keys (they map to their remapped names after dequant)
    let all_remapped: HashSet<String> = remapped_keys;

    let mut missing_prefixes: Vec<&str> = Vec::new();
    for prefix in EXPECTED_PREFIXES {
        let found = all_remapped.iter().any(|k| k.starts_with(prefix));
        if found {
            let count = all_remapped.iter().filter(|k| k.starts_with(prefix)).count();
            println!("  [OK] {prefix}  ({count} keys)");
        } else {
            println!("  [MISSING] {prefix}");
            missing_prefixes.push(prefix);
        }
    }

    for key in EXPECTED_SCALAR_KEYS {
        if all_remapped.contains(*key) {
            println!("  [OK] {key}");
        } else {
            println!("  [MISSING] {key}");
            missing_prefixes.push(key);
        }
    }

    // --- Test 3: A few key tensor spot checks ---
    println!("\n=== Test 3: Spot Check Key Tensors ===\n");

    // These are tensors we expect to be quantized (large linear weights in flow_lm).
    // The original model already uses internal names (HF == internal for this checkpoint).
    let spot_check_hf_names = [
        "flow_lm.transformer.layers.0.self_attn.in_proj.weight",
        "flow_lm.transformer.layers.0.linear1.weight",
        "flow_lm.flow_net.res_blocks.0.mlp.0.weight",
    ];

    for hf_name in &spot_check_hf_names {
        if !orig_names.contains(*hf_name) {
            println!("  [N/A] {hf_name}: not in original (check HF name)");
            continue;
        }
        match quant_st.tensor(hf_name) {
            Err(_) => {
                println!("  [MISS] {hf_name}: missing from quantized model!");
            }
            Ok(qt) => {
                if qt.dtype() != safetensors::Dtype::I8 {
                    println!(
                        "  [WARN] {hf_name}: expected INT8, got {:?}",
                        qt.dtype()
                    );
                } else {
                    let scale_name = format!("{hf_name}{scale_suffix}");
                    let st = quant_st.tensor(&scale_name).expect("scale should exist");
                    let ot = orig_st.tensor(hf_name).expect("orig should exist");
                    let shape = ot.shape().to_vec();
                    let orig_f32 = bf16_slice_to_f32(ot.data());
                    let deq_f32 = dequantize(qt.data(), st.data(), &shape);
                    let stats = compute_stats(&orig_f32, &deq_f32);

                    // INT8 max theoretical error is scale/2 per element
                    // Expected SQNR for INT8: ~48 dB theoretical, real-world ~40+ dB
                    println!(
                        "  [OK] {}: sqnr={:.1} dB, max_err={:.6}",
                        hf_name, stats.sqnr_db, stats.max_abs_err
                    );

                    // Assert reasonable bounds
                    assert!(
                        stats.sqnr_db > 30.0,
                        "Spot check tensor {hf_name} has SQNR {:.1} dB < 30 dB — dequantization error too high",
                        stats.sqnr_db
                    );
                }
            }
        }
    }

    // --- Final assertions ---
    println!("\n=== Final Assertions ===\n");

    assert!(
        bad_tensors.is_empty(),
        "BAD tensors found (SQNR < 20 dB): {:?}",
        bad_tensors
    );

    assert!(
        missing_prefixes.is_empty(),
        "Missing key prefixes after remapping: {:?}",
        missing_prefixes
    );

    if worst_sqnr < sqnr_threshold {
        println!(
            "  WARN: Some tensors below {sqnr_threshold:.0} dB SQNR (worst: {worst_sqnr:.1} dB for '{worst_name}')"
        );
    } else {
        println!("  All checks PASSED. Worst SQNR: {:.1} dB", worst_sqnr);
    }
}
