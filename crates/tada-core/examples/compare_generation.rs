/// Compare Python vs Rust TADA generation step-by-step.
///
/// Loads Python's per-step debug data from `samples/python_generation_debug.bin`,
/// runs the same generation in Rust (forcing Python's token IDs at every step),
/// and compares hidden states and acoustics at each step.
///
/// Usage:
///   cargo run --example compare_generation -p tada-core --release -- \
///     --model /path/to/tada-1b-f16.gguf \
///     --tokenizer tokenizer.json

use candle_core::{Device, Result as CResult, Tensor};
use tada_core::config::TadaConfig;
use tada_core::tada_model::{Rng, TadaModel};

// ---------------------------------------------------------------------------
// Python debug binary format reader
// ---------------------------------------------------------------------------

struct PyStepData {
    token_id: u32,
    hidden: Vec<f32>,
    has_acoustic: bool,
    acoustic: Option<Vec<f32>>,
    time_before: Option<u32>,
    time_after: Option<u32>,
    top_logits: Vec<(u32, f32)>,
}

struct PyDebugData {
    num_steps: usize,
    hidden_dim: usize,
    acoustic_dim: usize,
    steps: Vec<PyStepData>,
}

fn read_debug_bin(path: &str) -> std::io::Result<PyDebugData> {
    let data = std::fs::read(path)?;
    let mut pos = 0usize;

    let read_u32 = |data: &[u8], pos: &mut usize| -> u32 {
        let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        val
    };
    let read_f32 = |data: &[u8], pos: &mut usize| -> f32 {
        let val = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        val
    };

    let num_steps = read_u32(&data, &mut pos) as usize;
    let hidden_dim = read_u32(&data, &mut pos) as usize;
    let acoustic_dim = read_u32(&data, &mut pos) as usize;

    let mut steps = Vec::with_capacity(num_steps);

    for _ in 0..num_steps {
        let token_id = read_u32(&data, &mut pos);

        let mut hidden = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            hidden.push(read_f32(&data, &mut pos));
        }

        let has_acoustic = read_u32(&data, &mut pos) != 0;
        let (acoustic, time_before, time_after) = if has_acoustic {
            let mut ac = Vec::with_capacity(acoustic_dim);
            for _ in 0..acoustic_dim {
                ac.push(read_f32(&data, &mut pos));
            }
            let tb = read_u32(&data, &mut pos);
            let ta = read_u32(&data, &mut pos);
            (Some(ac), Some(tb), Some(ta))
        } else {
            (None, None, None)
        };

        let num_top = read_u32(&data, &mut pos) as usize;
        let mut top_logits = Vec::with_capacity(num_top);
        for _ in 0..num_top {
            let idx = read_u32(&data, &mut pos);
            let val = read_f32(&data, &mut pos);
            top_logits.push((idx, val));
        }

        steps.push(PyStepData {
            token_id,
            hidden,
            has_acoustic,
            acoustic,
            time_before,
            time_after,
            top_logits,
        });
    }

    Ok(PyDebugData {
        num_steps,
        hidden_dim,
        acoustic_dim,
        steps,
    })
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

struct SimpleRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, 1.0).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { inner: rng, distr }
    }
}

impl Rng for SimpleRng {
    fn sample_normal(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

// ---------------------------------------------------------------------------
// Tokenization (zero-shot, matching tada_generate.rs)
// ---------------------------------------------------------------------------

fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> CResult<Vec<u32>> {
    let enc = |s: &str| -> CResult<Vec<u32>> {
        tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))
    };
    let text_tokens = enc(text)?;
    let mut ids = vec![128000u32]; // BOS
    let prefix = enc("<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>")?;
    ids.extend(&prefix);
    ids.extend(&text_tokens);
    Ok(ids)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    model_path: String,
    tokenizer_path: String,
    debug_path: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = None;
    let mut tokenizer_path = None;
    let mut debug_path = String::from("samples/python_generation_debug.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = Some(args[i].clone()); }
            "--tokenizer" => { i += 1; tokenizer_path = Some(args[i].clone()); }
            "--debug" => { i += 1; debug_path = args[i].clone(); }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        model_path: model_path.expect("--model required"),
        tokenizer_path: tokenizer_path.expect("--tokenizer required"),
        debug_path,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> CResult<()> {
    let args = parse_args();

    // --- Load Python debug data ---
    eprintln!("[1] Loading Python debug data from {}...", args.debug_path);
    let py_data = read_debug_bin(&args.debug_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read debug bin: {e}")))?;
    eprintln!(
        "  {} steps, hidden_dim={}, acoustic_dim={}",
        py_data.num_steps, py_data.hidden_dim, py_data.acoustic_dim
    );
    eprintln!(
        "  First 10 Python tokens: {:?}",
        py_data.steps.iter().take(10).map(|s| s.token_id).collect::<Vec<_>>()
    );

    // --- Load model ---
    eprintln!("\n[2] Loading model from {}...", args.model_path);
    let device = Device::Cpu;
    let model_bytes = std::fs::read(&args.model_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read model: {e}")))?;
    eprintln!("  read {} MB", model_bytes.len() / (1024 * 1024));
    let cfg = TadaConfig::tada_1b();
    let mut model = TadaModel::load_gguf(&model_bytes, &cfg, &device)?;
    eprintln!("  model loaded OK");

    // --- Load tokenizer ---
    eprintln!("\n[3] Loading tokenizer from {}...", args.tokenizer_path);
    let tokenizer_bytes = std::fs::read(&args.tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to read tokenizer: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;

    let text = "The quick brown fox jumps over the lazy dog.";
    let rust_tokens = tokenize(&tokenizer, text)?;
    eprintln!(
        "  Rust tokens ({} total): {:?}",
        rust_tokens.len(),
        rust_tokens
    );

    // --- Generation loop forcing Python tokens ---
    eprintln!("\n[4] Running comparison ({} steps)...\n", py_data.num_steps);

    let acoustic_dim = cfg.acoustic_dim;
    let shift_acoustic = cfg.shift_acoustic;
    let noise_temp: f32 = 0.9;
    let flow_steps: usize = 10;

    model.clear_state();
    let mut rng = SimpleRng::new(42);

    let mut acoustic = vec![0.0f32; acoustic_dim];
    let mut acoustic_mask: u32 = 0;
    let mut time_before: u32 = 0;
    let mut time_after: u32 = 0;

    // Collect stats
    let mut hid_cos_all = Vec::new();
    let mut hid_l2_all = Vec::new();
    let mut ac_cos_all = Vec::new();
    let mut ac_l2_all = Vec::new();

    // Table header
    println!(
        "{:>4} | {:>6} | {:>15} | {:>10} | {:>8} | {:>10} | {:>8} | {:>5} | {:>5}",
        "step", "py_tok", "token", "hid_cos", "hid_l2", "ac_cos", "ac_l2", "py_tb", "rs_tb"
    );
    println!("{}", "-".repeat(95));

    for step in 0..py_data.num_steps {
        let py_step = &py_data.steps[step];
        let current_token = py_step.token_id;

        // Build input embeddings
        let token_tensor = Tensor::from_vec(vec![current_token], (1, 1), &device)?;
        let acoustic_tensor =
            Tensor::from_vec(acoustic.clone(), (1, 1, acoustic_dim), &device)?;
        let mask_tensor = Tensor::from_vec(vec![acoustic_mask], (1, 1), &device)?;
        let time_before_tensor = Tensor::from_vec(vec![time_before], (1, 1), &device)?;
        let time_after_tensor = Tensor::from_vec(vec![time_after], (1, 1), &device)?;

        let input_embeds = model.build_input_embeds(
            &token_tensor,
            &acoustic_tensor,
            &mask_tensor,
            &time_before_tensor,
            &time_after_tensor,
        )?;

        // Forward step
        let hidden = model.forward_step(&input_embeds)?;

        // Extract hidden state
        let rust_hidden: Vec<f32> = hidden.flatten_all()?.to_vec1()?;

        let hid_cos = cosine_sim(&py_step.hidden, &rust_hidden);
        let hid_l2 = l2_dist(&py_step.hidden, &rust_hidden);
        hid_cos_all.push(hid_cos);
        hid_l2_all.push(hid_l2);

        // Generate acoustics after shift_acoustic
        let mut ac_cos_str = format!("{:>10}", "n/a");
        let mut ac_l2_str = format!("{:>8}", "n/a");
        let mut py_tb_str = format!("{:>5}", "n/a");
        let mut rs_tb_str = format!("{:>5}", "n/a");

        if step >= shift_acoustic {
            let (acou, tb, ta) = model.generate_acoustic(
                &hidden,
                noise_temp,
                &mut rng,
                flow_steps,
                1.0,
            )?;
            let acou_vec = acou.squeeze(0)?.to_vec1::<f32>()?;

            // Denormalize for comparison (Python saves denormalized)
            let denorm: Vec<f32> = acou_vec
                .iter()
                .map(|&v| v * cfg.acoustic_std as f32 + cfg.acoustic_mean as f32)
                .collect();

            if let Some(ref py_ac) = py_step.acoustic {
                let ac_cos = cosine_sim(py_ac, &denorm);
                let ac_l2 = l2_dist(py_ac, &denorm);
                ac_cos_all.push(ac_cos);
                ac_l2_all.push(ac_l2);
                ac_cos_str = format!("{:10.6}", ac_cos);
                ac_l2_str = format!("{:8.2}", ac_l2);
            }

            py_tb_str = format!("{:5}", py_step.time_before.unwrap_or(0));
            rs_tb_str = format!("{:5}", tb);

            // Update state for next step (use raw normalized values)
            acoustic = acou_vec;
            acoustic_mask = 1;
            time_before = tb;
            time_after = ta;
        }

        // Decode token text for display
        let tok_text = tokenizer
            .decode(&[current_token], false)
            .unwrap_or_else(|_| format!("#{current_token}"));
        let tok_display = if tok_text.len() > 12 {
            format!("{}...", &tok_text[..9])
        } else {
            tok_text
        };

        println!(
            "{:4} | {:6} | {:>15} | {:10.6} | {:8.2} | {} | {} | {} | {}",
            step,
            current_token,
            tok_display,
            hid_cos,
            hid_l2,
            ac_cos_str,
            ac_l2_str,
            py_tb_str,
            rs_tb_str,
        );

        // Log top-5 logits at a few key steps
        if step < 3 || step == shift_acoustic || step == py_data.num_steps - 1 {
            let logits = model.lm_head_logits(&hidden)?;
            let logits_vec: Vec<f32> = logits.squeeze(0)?.squeeze(0)?.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> =
                logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!(
                "  [step {:3}] Rust   top5: {:?}",
                step,
                indexed[..5].iter().map(|(i, v)| (*i, format!("{:.3}", v))).collect::<Vec<_>>()
            );
            eprintln!(
                "  [step {:3}] Python top5: {:?}",
                step,
                py_step.top_logits.iter().map(|(i, v)| (*i, format!("{:.3}", v))).collect::<Vec<_>>()
            );
        }
    }

    // --- Summary statistics ---
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let min_f = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_f = |v: &[f32]| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "Hidden cosine sim:  mean={:.6}  min={:.6}  max={:.6}  (n={})",
        mean(&hid_cos_all), min_f(&hid_cos_all), max_f(&hid_cos_all), hid_cos_all.len()
    );
    println!(
        "Hidden L2 dist:     mean={:.4}  min={:.4}  max={:.4}",
        mean(&hid_l2_all), min_f(&hid_l2_all), max_f(&hid_l2_all)
    );
    if !ac_cos_all.is_empty() {
        println!(
            "Acoustic cosine sim: mean={:.6}  min={:.6}  max={:.6}  (n={})",
            mean(&ac_cos_all), min_f(&ac_cos_all), max_f(&ac_cos_all), ac_cos_all.len()
        );
        println!(
            "Acoustic L2 dist:    mean={:.2}  min={:.2}  max={:.2}",
            mean(&ac_l2_all), min_f(&ac_l2_all), max_f(&ac_l2_all)
        );
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
