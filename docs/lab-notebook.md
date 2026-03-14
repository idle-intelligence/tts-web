# lab-notebook.md

Experiment log. Each block corresponds to an entry in results.md (same experiment name). Contains purpose, parameters, and verbatim user feedback.

---

## first-python-ref

**Date**: 2026-03-12
**Commit**: unknown (pre-benchmark)
**Purpose**: First Python BF16 reference generation to establish a quality baseline.

**Parameters**:
- engine: python-bf16
- device: cpu
- seed: unknown
- flow_steps: 10
- voice: ljspeech
- noise_temp: unknown

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**: BF16 Python reference

**User feedback**: Reference quality.

---

## timing-v2

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: First timing instrumentation pass; also includes voice alignment V2 fix (match Python's zeros+mask=0 approach).

**Parameters**:
- engine: candle
- device: cpu / metal (no --features metal)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5 (trims all 5 voice prompt tokens → effectively zero-shot)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**:
- F16 CPU
- F16 Metal (no --features metal flag — not real GPU acceleration)
- Q4_0 CPU
- Q4_0 Metal (no --features metal flag)

**Notes**: F16 Metal with --features metal was OOM killed (3.3GB model too large for GPU buffer).

**User feedback**: "all Rust outputs sound comparable, possibly better than Python reference"

---

## metal-discovery

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: Discovered that --features metal flag enables real Metal GPU acceleration (vs. the no-feature Metal runs in timing-v2).

**Parameters**:
- engine: candle
- device: metal (with --features metal)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5
- model: Q4_0 baseline

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**: Q4_0 Metal with --features metal

**User feedback**: Not evaluated separately (same text and model as timing-v2).

---

## burn-hybrid-zeroshot

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: First Burn/wgpu LLM benchmark. LLM runs on GPU via Burn; VibeVoice remains on CPU via candle. Zero-shot (no voice prompt — example did not yet support --voice).

**Parameters**:
- device: wgpu+cpu (LLM on wgpu GPU, VibeVoice on candle CPU)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: none (zero-shot)
- model: Q4_0 baseline

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**:
- candle Q4_0 CPU (reference)
- burn+candle Q4_0 wgpu+cpu

**Notes**: LLM per-step: Burn GPU 146ms vs candle CPU 1660ms (11x speedup). VibeVoice 455ms/step (74% of total gen time) — becomes the new bottleneck.

**User feedback**: "BAD AUDIO — zero-shot, no voice. Not comparable."

---

## mixed-quant-discovery

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: Test mixed Q4+Q8 quantization variants for size/speed. Also caught a noise_temp=0.6 regression bug.

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.6 (run 1, bug) / 0.9 (runs 2–4, correct)
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**:
- Q4_0 baseline (2.64G), noise_temp=0.6
- Q4_0 baseline (2.64G), noise_temp=0.9
- Mixed v1 decoder-Q4_0 (1.48G)
- Mixed v2 decoder-Q8_0 (1.52G)

**User feedback**:
- noise_temp=0.6: "flat/dead audio"
- Q4_0 noise_temp=0.9: baseline reference
- Mixed v1 and v2: "Near-identical quality"

---

## variant-precision-sweep

**Date**: 2026-03-13
**Commit**: ec906e4
**Purpose**: Systematic sweep to identify which model components (VV head, embeddings) tolerate lower precision without gray-code mispredictions.

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"

**Variants**:
- baseline: VV F32, Embed Q4_0 (2.64G)
- A: VV F16, Embed Q8_0 (1.88G)
- B: VV F32, Embed Q8_0 (2.68G)
- C: VV Q8_0, Embed Q4_0 (1.38G)
- D: VV Q8_0, Embed Q8_0 (1.52G)
- E: VV F16, Embed Q4_0 (1.75G)

**Key metric**: times_before[12] gray-code prediction at a sensitive step.
- baseline: tb=4 (correct) on both texts
- A: tb=4 (fox only), tb=59 (time — misprediction)
- B: tb=4 (fox only), tb=59 (time — misprediction)
- C: tb=59 on both (misprediction everywhere)
- D: tb=59 on both (misprediction everywhere)
- E: tb=4 (correct) on both texts

**Notes**: No audio files logged for these runs — they were part of a systematic diagnostic comparison. Variant E selected as winner: smallest model with correct gray-code predictions on all tested phrases.

**User feedback**: Not individually evaluated. Variant E selected as winner.

---

## postfix-full-sweep

**Date**: 2026-03-14
**Commit**: 21d97e6
**Purpose**: Comprehensive post-bugfix benchmark across all model variants and 4 test phrases. Validates fixes for: autoregressive time feedback bug, trailing EOT frame trim, WASM EOT token.

**Parameters**:
- Rust runs: candle, metal, seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech, transition_steps=0
- Python run: python-bf16, cpu, acoustic_cfg_scale=1.6, transition_steps=5

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**:
- Python BF16 CPU
- F32 GGUF (6.5G)
- F16 GGUF (3.3G)
- Q4_0 baseline (2.6G)
- Var-B VV-F32 E-Q8 (2.5G)
- Var-A VV-F16 E-Q8 (1.9G)
- Var-E VV-F16 E-Q4 (1.8G)
- Mixed VV-Q8 E-Q8 (1.4G)
- Var-C VV-Q8 E-Q4 (1.3G)

**Notes**: All audio files in samples/bench_2026-03-13/ subdirectories.

**User feedback**:
- Python (all phrases): "voice is OK, output is complete, but quieter, more noise/wind"
- fox (all Rust variants): OK
- call (all Rust variants): "sounds good, just cut early — missing 'the middle of the night'"
- tyger (all Rust variants): universally bad, wrong phonemes
- wutang (all Rust variants): "sounds very good, voice switch mid-phrase"

---

## q8-llm-alignment

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Test Q8_0 LLM backbone with F16 VV head and Q4_0 embeddings to see if quantizing the LLM backbone further degrades output quality vs Var-E (F16 VV + Q4_0 embed).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0
- model file: tada-1b-llmq8-vvf16-eq4.gguf (2.24G)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**: Q8_0 LLM + F16 VV + Q4_0 Embed (2.24G)

**Notes**: Files in samples/bench_2026-03-13/llmq8_vvf16_eq4/

**User feedback**: Not individually evaluated (same perceptual quality as other Rust variants per user's earlier statement).

---

## burn-gpu-zeroshot-v2

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Burn/wgpu GPU benchmark with updated code. Zero-shot because the burn example does not support --voice. Intended to measure VibeVoice GPU acceleration potential.

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu
- noise_temp: 0.9
- flow_steps: 10
- voice: none (zero-shot — example limitation)
- model: Q4_0 baseline (2.6G)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**: Burn+candle wgpu+cpu Q4_0 baseline

**Notes**:
- Gen time breakdown (LLM / VibeVoice): fox 4.3s / 15.7s, call 11.6s / 65.2s, tyger 11.0s / 62.8s, wutang 3.6s / 12.8s
- call and tyger have no decode time logged (likely crash or skip)
- Files in samples/bench_2026-03-13/burn_gpu/

**User feedback**: "even worse, just noise and garbage" (expected — zero-shot)

---

## tyger-debug

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Investigate the tyger pronunciation issue — wrong phonemes on "Tyger". Vary transition_steps and voice conditioning to isolate cause.

**Parameters**:
- engine: candle
- seed: 42
- noise_temp: 0.9
- model: Q4_0 baseline (2.6G)
- voice: ljspeech (except novoice variant)

**Texts**:
- tyger: "Tyger Tyger, burning bright"

**Variants**:
- ts0 Metal: transition_steps=0, Metal
- ts5 Metal: transition_steps=5, Metal
- novoice Metal: no voice prompt, Metal
- ts0 CPU: transition_steps=0, CPU

**Notes**: No load/gen/decode timing captured for these runs.

**User feedback**:
- tyger_ts0: "shitty"
- tyger_ts5: "matches the python version, apart from this one being cut early"
- tyger_novoice: "nonsense 'aaaaaaha bnoeoeo'"
- tyger_cpu: "exactly equivalent to tyger_ts0"

---

## cross-variant-quality

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Verify that audio quality is consistent across all model sizes on a phrase known to be well-handled (not affected by the tyger/call known issues).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0

**Texts**:
- time: "Time is money, who can afford to pay attention?"

**Variants**:
- F32 GGUF (6.5G)
- Q4_0 baseline (2.6G)
- Var-E VV-F16 E-Q4 (1.8G)
- Mixed VV-Q8 E-Q8 (1.4G)
- Var-C VV-Q8 E-Q4 (1.3G)

**Notes**: No load/decode timing captured for these runs.

**User feedback**: "They all sound good"
