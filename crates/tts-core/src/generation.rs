/// Shared EOS/frame-count bookkeeping for autoregressive generation.
///
/// Mirrors the official `_autoregressive_generation` loop
/// (`pocket_tts/models/tts_model.py`): for each generation step, decide
/// whether this step's latent/audio should be produced at all, *before*
/// producing it. This is a "check then append" order, not "append then
/// check" — the two orders differ by one frame at the tail of generation.
///
/// Both the native example (`tts_generate.rs`) and the WASM binding
/// (`tts-wasm/src/lib.rs`) call this, so the EOS-timing rule lives in one
/// place instead of being reimplemented per call site.
/// EOS is ignored on the first frames: before speech starts, the EOS logit
/// of some voices can cross the threshold (`tts_model.py:89,884`).
pub const MIN_FRAMES_BEFORE_EOS: usize = 6;

pub struct EosGate {
    frames_after_eos: usize,
    min_frames_before_eos: usize,
    eos_step: Option<usize>,
}

impl EosGate {
    pub fn new(frames_after_eos: usize) -> Self {
        Self::with_min_frames_before_eos(frames_after_eos, MIN_FRAMES_BEFORE_EOS)
    }

    pub fn with_min_frames_before_eos(frames_after_eos: usize, min_frames_before_eos: usize) -> Self {
        Self { frames_after_eos, min_frames_before_eos, eos_step: None }
    }

    /// Call once per generation step, after computing `is_eos` for that
    /// step's latent but before producing (decoding/appending) it.
    ///
    /// Returns `true` if this step's latent should be produced, `false` if
    /// generation should stop now, without producing it.
    pub fn accept(&mut self, step: usize, is_eos: bool) -> bool {
        if is_eos && self.eos_step.is_none() && step >= self.min_frames_before_eos {
            self.eos_step = Some(step);
        }
        if self.eos_step.is_some_and(|eos_step| step >= eos_step + self.frames_after_eos) {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eos_always_accepts() {
        let mut gate = EosGate::new(3);
        for step in 0..20 {
            assert!(gate.accept(step, false));
        }
    }

    #[test]
    fn total_frames_from_eos_onward_is_frames_after_eos() {
        // Matches the diagnosis: total frames from eos_step onward (inclusive)
        // must equal frames_after_eos, not frames_after_eos + 1.
        let frames_after_eos = 3;
        let mut gate = EosGate::new(frames_after_eos);
        let eos_step = 10;
        let mut accepted = 0usize;
        for step in 0..30 {
            let is_eos = step == eos_step;
            if gate.accept(step, is_eos) {
                accepted += 1;
            } else {
                break;
            }
        }
        assert_eq!(accepted, eos_step + frames_after_eos);
    }

    #[test]
    fn min_frames_before_eos_gate_ignores_early_eos() {
        let mut gate = EosGate::with_min_frames_before_eos(3, 6);
        // Spurious EOS at step 2 must be ignored.
        assert!(gate.accept(2, true));
        assert!(gate.accept(3, false));
        // Real EOS at step 6 is accepted.
        assert!(gate.accept(6, true));
        // Frames 7, 8 still produced (frames_after_eos=3 → steps 6,7,8).
        assert!(gate.accept(7, false));
        assert!(gate.accept(8, false));
        assert!(!gate.accept(9, false));
    }
}
