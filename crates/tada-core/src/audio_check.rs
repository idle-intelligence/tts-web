//! Audio sanity checks for generated TTS output.
//!
//! These checks catch obvious failures (noise, silence, DC offset, wrong
//! duration) that indicate model or pipeline bugs.  They are designed to
//! run cheaply on the final PCM buffer.

use std::fmt;

/// Summary statistics for a PCM audio buffer.
#[derive(Debug, Clone)]
pub struct AudioStats {
    pub num_samples: usize,
    pub sample_rate: usize,
    pub duration_secs: f64,
    pub rms: f32,
    pub peak: f32,
    pub mean: f32,
    pub zero_crossing_rate: f32,
    /// Spectral flatness of the first `sample_rate` samples (1 second).
    /// Close to 1.0 = white noise, close to 0.0 = tonal/speech.
    pub spectral_flatness: f32,
}

impl fmt::Display for AudioStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "samples={} duration={:.2}s rms={:.4} peak={:.4} mean={:.4} zcr={:.4} flatness={:.4}",
            self.num_samples, self.duration_secs, self.rms, self.peak, self.mean,
            self.zero_crossing_rate, self.spectral_flatness,
        )
    }
}

impl AudioStats {
    /// Compute statistics from raw PCM samples.
    pub fn from_samples(samples: &[f32], sample_rate: usize) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                num_samples: 0,
                sample_rate,
                duration_secs: 0.0,
                rms: 0.0,
                peak: 0.0,
                mean: 0.0,
                zero_crossing_rate: 0.0,
                spectral_flatness: 1.0,
            };
        }

        let mean = samples.iter().sum::<f32>() / n as f32;
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        // Zero crossing rate
        let mut crossings = 0usize;
        for i in 1..n {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        let zero_crossing_rate = crossings as f32 / (n - 1).max(1) as f32;

        // Spectral flatness on first 1 second (via DFT magnitude)
        let window = &samples[..n.min(sample_rate)];
        let spectral_flatness = compute_spectral_flatness(window);

        Self {
            num_samples: n,
            sample_rate,
            duration_secs: n as f64 / sample_rate as f64,
            rms,
            peak,
            mean,
            zero_crossing_rate,
            spectral_flatness,
        }
    }

    /// Run sanity checks. Returns a list of failure messages (empty = all OK).
    pub fn check(&self, expected_text_tokens: usize) -> Vec<String> {
        let mut failures = Vec::new();

        // 1. Duration sanity: speech is roughly 3-10 chars/second.
        //    For token-based: ~0.1-0.5s per text token is normal.
        //    If duration > 2s per text token, something is very wrong.
        let max_reasonable_secs = (expected_text_tokens as f64) * 2.0 + 2.0;
        if self.duration_secs > max_reasonable_secs {
            failures.push(format!(
                "DURATION: {:.1}s for {} tokens (max expected ~{:.0}s) — model likely didn't stop generating",
                self.duration_secs, expected_text_tokens, max_reasonable_secs,
            ));
        }

        // 2. DC offset: speech should have near-zero mean
        if self.mean.abs() > 0.01 {
            failures.push(format!(
                "DC_OFFSET: mean={:.4} (expected |mean| < 0.01) — indicates broken decoder output",
                self.mean,
            ));
        }

        // 3. Spectral flatness: speech < 0.2, noise > 0.4
        if self.spectral_flatness > 0.35 {
            failures.push(format!(
                "NOISE: spectral_flatness={:.3} (speech < 0.2, noise > 0.4) — output is noise, not speech",
                self.spectral_flatness,
            ));
        }

        // 4. Silence: RMS should be above some floor for speech
        if self.rms < 0.005 {
            failures.push(format!(
                "SILENCE: rms={:.4} — output is near-silent",
                self.rms,
            ));
        }

        // 5. Clipping: peak should be < 1.0 (we use tanh so this shouldn't happen)
        if self.peak > 0.99 {
            failures.push(format!(
                "CLIPPING: peak={:.4} — output is clipping",
                self.peak,
            ));
        }

        // 6. Peak too low: speech typically has peak > 0.2
        if self.peak < 0.15 && self.num_samples > self.sample_rate {
            failures.push(format!(
                "QUIET: peak={:.4} — output is abnormally quiet (expected > 0.15)",
                self.peak,
            ));
        }

        failures
    }
}

/// Check that generation stats (frame counts, time predictions) look reasonable.
pub fn check_generation(
    num_text_tokens: usize,
    num_acoustic_frames: usize,
    times_before: &[u32],
    hit_eos: bool,
) -> Vec<String> {
    let mut failures = Vec::new();

    // 1. EOS: model should have hit EOS
    if !hit_eos {
        failures.push(format!(
            "NO_EOS: model generated {num_acoustic_frames} frames without hitting EOS — likely degraded model quality",
        ));
    }

    // 2. Frame count: should be roughly proportional to text tokens
    //    In TADA, each text token gets ~1 acoustic frame. Some tokens might
    //    not produce acoustics, but we shouldn't get 5x more frames than tokens.
    let max_frames = num_text_tokens * 5 + 10;
    if num_acoustic_frames > max_frames {
        failures.push(format!(
            "TOO_MANY_FRAMES: {num_acoustic_frames} frames for {num_text_tokens} text tokens (max ~{max_frames})",
        ));
    }

    // 3. Time predictions: average should be reasonable (5-50 frames per token at 50fps)
    if !times_before.is_empty() {
        let avg_time: f64 =
            times_before.iter().map(|&t| t as f64).sum::<f64>() / times_before.len() as f64;
        let max_time = *times_before.iter().max().unwrap_or(&0);

        if avg_time > 80.0 {
            failures.push(format!(
                "HUGE_DURATIONS: avg time_before={avg_time:.1} frames (expected 5-50) — duration predictions are broken",
            ));
        }
        if max_time > 200 {
            failures.push(format!(
                "MAX_DURATION: max time_before={max_time} (expected < 200) — extreme outlier in duration",
            ));
        }
    }

    failures
}

/// Compute spectral flatness = geometric_mean(|FFT|) / arithmetic_mean(|FFT|).
///
/// Uses a simple DFT (no FFT library needed). For short windows this is fine.
fn compute_spectral_flatness(samples: &[f32]) -> f32 {
    let n = samples.len();
    if n == 0 {
        return 1.0;
    }

    // Compute magnitude spectrum via real DFT (only positive freqs)
    let num_bins = n / 2 + 1;
    let mut magnitudes = Vec::with_capacity(num_bins);

    for k in 0..num_bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let w = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        for (i, &s) in samples.iter().enumerate() {
            let angle = w * i as f64;
            re += s as f64 * angle.cos();
            im += s as f64 * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }

    // Geometric mean via log
    let log_sum: f64 = magnitudes.iter().map(|&m| (m + 1e-10).ln()).sum::<f64>();
    let geo_mean = (log_sum / num_bins as f64).exp();
    let arith_mean: f64 = magnitudes.iter().sum::<f64>() / num_bins as f64;

    (geo_mean / (arith_mean + 1e-10)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_detected() {
        let samples = vec![0.0f32; 48000];
        let stats = AudioStats::from_samples(&samples, 24000);
        let failures = stats.check(5);
        assert!(failures.iter().any(|f| f.starts_with("SILENCE")));
    }

    #[test]
    fn test_dc_offset_detected() {
        let samples = vec![0.5f32; 24000];
        let stats = AudioStats::from_samples(&samples, 24000);
        let failures = stats.check(5);
        assert!(failures.iter().any(|f| f.starts_with("DC_OFFSET")));
    }

    #[test]
    fn test_white_noise_detected() {
        // Generate pseudo-random noise
        let mut samples = Vec::with_capacity(24000);
        let mut x = 0.123456f32;
        for _ in 0..24000 {
            // Simple LCG for deterministic "noise"
            x = (x * 1664525.0 + 1013904223.0) % (u32::MAX as f32);
            samples.push(x / (u32::MAX as f32) * 2.0 - 1.0);
        }
        let stats = AudioStats::from_samples(&samples, 24000);
        assert!(
            stats.spectral_flatness > 0.3,
            "noise should have high flatness, got {}",
            stats.spectral_flatness,
        );
    }

    #[test]
    fn test_generation_no_eos() {
        let failures = check_generation(10, 50, &[30; 50], false);
        assert!(failures.iter().any(|f| f.starts_with("NO_EOS")));
    }

    #[test]
    fn test_generation_too_many_frames() {
        let failures = check_generation(5, 100, &[10; 100], true);
        assert!(failures.iter().any(|f| f.starts_with("TOO_MANY_FRAMES")));
    }

    #[test]
    fn test_generation_huge_durations() {
        let failures = check_generation(10, 10, &[150; 10], true);
        assert!(failures.iter().any(|f| f.starts_with("HUGE_DURATIONS")));
    }

    #[test]
    fn test_good_generation_passes() {
        let failures = check_generation(10, 8, &[15, 10, 12, 8, 20, 10, 15, 12], true);
        assert!(failures.is_empty(), "unexpected failures: {:?}", failures);
    }
}
