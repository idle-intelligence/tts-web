use candle_core::{Device, Result, Tensor};

use crate::vibevoice::VibeVoiceDiffusionHead;

// ---------------------------------------------------------------------------
// Gray code conversion
// ---------------------------------------------------------------------------

/// Convert an integer to its Gray code representation.
pub fn int_to_gray(value: u32) -> u32 {
    value ^ (value >> 1)
}

/// Convert a Gray code value back to a standard integer.
pub fn gray_to_int(mut gray: u32) -> u32 {
    let mut mask = gray >> 1;
    while mask != 0 {
        gray ^= mask;
        mask >>= 1;
    }
    gray
}

/// Decode a tensor of Gray-coded float bits ({-1.0, 1.0}) into integer time values.
///
/// `gray_bits` has shape `[..., num_bits]` where each element is -1.0 (bit=0) or
/// 1.0 (bit=1). The MSB is at index 0. Returns a tensor of u32 time indices with
/// shape `[...]` (last dim consumed).
pub fn decode_gray_code_to_time(gray_bits: &Tensor, num_bits: usize) -> Result<Tensor> {
    let shape = gray_bits.dims().to_vec();
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    // Flatten to [batch, num_bits]
    let flat = gray_bits.reshape((batch_size, num_bits))?;
    let data = flat.to_vec2::<f32>()?;

    let mut result = Vec::with_capacity(batch_size);
    for row in &data {
        // Convert float bits to gray code integer
        let mut gray: u32 = 0;
        for (i, &bit) in row.iter().enumerate() {
            if bit > 0.0 {
                gray |= 1 << (num_bits - 1 - i);
            }
        }
        result.push(gray_to_int(gray));
    }

    let out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    let out_shape = if out_shape.is_empty() { vec![1] } else { out_shape };
    Tensor::from_vec(result, out_shape.as_slice(), gray_bits.device())
}

// ---------------------------------------------------------------------------
// Time schedule builders
// ---------------------------------------------------------------------------

/// Build a time schedule for the flow matching ODE solver.
///
/// Returns `num_steps + 1` time values in \[0, 1\].
///
/// Supported schedules:
/// - `"uniform"` — evenly spaced from 0 to 1.
/// - `"cosine"` — cosine spacing (denser near endpoints).
/// - `"logsnr"` — log-SNR spacing (default in TADA), linspace(5, -5), t = sigmoid(-logsnr/2).
pub fn build_time_schedule(
    num_steps: usize,
    schedule: &str,
    _device: &Device,
) -> Result<Vec<f32>> {
    let n = num_steps + 1;
    let ts = match schedule {
        "uniform" => {
            (0..n).map(|i| i as f32 / num_steps as f32).collect()
        }
        "cosine" => {
            (0..n)
                .map(|i| {
                    let frac = i as f64 / num_steps as f64;
                    let val = (frac * std::f64::consts::FRAC_PI_2).sin();
                    val as f32
                })
                .collect()
        }
        "logsnr" | _ => {
            // linspace(5, -5, n), then t = sigmoid(-logsnr / 2)
            let mut schedule = Vec::with_capacity(n);
            for i in 0..n {
                let logsnr = 5.0 - 10.0 * (i as f64 / num_steps as f64);
                let t = sigmoid(-logsnr / 2.0);
                schedule.push(t as f32);
            }
            // Clamp endpoints to exactly 0 and 1
            schedule[0] = 0.0;
            schedule[n - 1] = 1.0;
            schedule
        }
    };
    Ok(ts)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Flow matching ODE solver (Euler method)
// ---------------------------------------------------------------------------

/// Solve the flow matching ODE using the Euler method.
///
/// Starting from noise `speech` (sampled from N(0,1) scaled by acoustic_std),
/// integrates the velocity field predicted by the diffusion head over `num_steps`.
///
/// Euler update: x_{t+1} = x_t + dt * v(x_t, t)
///
/// No classifier-free guidance (acoustic_cfg_scale = 1.0 in TADA v1).
///
/// # Arguments
/// - `speech` — initial noisy acoustic latent, shape `[B, acoustic_dim]`
/// - `cond` — conditioning from the LLM backbone, shape `[B, hidden_size]`
/// - `head` — the VibeVoice diffusion prediction head
/// - `num_steps` — number of Euler steps (e.g. 32)
/// - `time_schedule` — one of "uniform", "cosine", "logsnr"
pub fn solve_flow_matching(
    speech: &Tensor,
    cond: &Tensor,
    head: &VibeVoiceDiffusionHead,
    num_steps: usize,
    time_schedule: &str,
) -> Result<Tensor> {
    let device = speech.device();
    let dtype = speech.dtype();
    let schedule = build_time_schedule(num_steps, time_schedule, device)?;

    let mut x = speech.clone();

    for i in 0..num_steps {
        let t_val = schedule[i];
        let t_next = schedule[i + 1];
        let dt = t_next - t_val;

        // Broadcast scalar time to [B, 1]
        let b = x.dim(0)?;
        let t_tensor = Tensor::full(t_val, (b, 1), device)?.to_dtype(dtype)?;

        // Predict velocity at current (x, t)
        let velocity = head.forward(&x, &t_tensor, cond)?;

        // Euler step: x = x + dt * velocity
        x = (&x + &(velocity * dt as f64)?)?;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_code_roundtrip() {
        for i in 0..256u32 {
            let gray = int_to_gray(i);
            let back = gray_to_int(gray);
            assert_eq!(i, back, "Gray code roundtrip failed for {i}");
        }
    }

    #[test]
    fn test_gray_code_known_values() {
        // First few Gray codes: 0,1,3,2,6,7,5,4,12,13,...
        assert_eq!(int_to_gray(0), 0);
        assert_eq!(int_to_gray(1), 1);
        assert_eq!(int_to_gray(2), 3);
        assert_eq!(int_to_gray(3), 2);
        assert_eq!(int_to_gray(4), 6);
        assert_eq!(int_to_gray(5), 7);
        assert_eq!(int_to_gray(6), 5);
        assert_eq!(int_to_gray(7), 4);
    }

    #[test]
    fn test_decode_gray_bits() -> Result<()> {
        let device = Device::Cpu;
        // Encode time=5 → gray=7 → bits=[1,1,1] (3-bit), as floats {-1,1}
        // Actually for 8 bits: gray(5) = 7 = 0b00000111
        let num_bits = 8;
        let gray = int_to_gray(5); // = 7 = 0b00000111
        let mut bits = vec![-1.0f32; num_bits];
        for b in 0..num_bits {
            if (gray >> (num_bits - 1 - b)) & 1 == 1 {
                bits[b] = 1.0;
            }
        }
        let tensor = Tensor::from_vec(bits, (1, num_bits), &device)?;
        let decoded = decode_gray_code_to_time(&tensor, num_bits)?;
        let val = decoded.to_vec1::<u32>()?;
        assert_eq!(val[0], 5);
        Ok(())
    }

    #[test]
    fn test_uniform_schedule() -> Result<()> {
        let s = build_time_schedule(4, "uniform", &Device::Cpu)?;
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[1] - 0.25).abs() < 1e-6);
        assert!((s[4] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_logsnr_schedule() -> Result<()> {
        let s = build_time_schedule(32, "logsnr", &Device::Cpu)?;
        assert_eq!(s.len(), 33);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[32], 1.0);
        // logsnr schedule should be monotonically increasing
        for i in 1..s.len() {
            assert!(
                s[i] >= s[i - 1],
                "logsnr schedule not monotonic at step {i}: {} < {}",
                s[i],
                s[i - 1]
            );
        }
        Ok(())
    }

    #[test]
    fn test_cosine_schedule() -> Result<()> {
        let s = build_time_schedule(4, "cosine", &Device::Cpu)?;
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[4] - 1.0).abs() < 1e-6);
        // Cosine should be monotonically increasing
        for i in 1..s.len() {
            assert!(s[i] >= s[i - 1]);
        }
        Ok(())
    }
}
