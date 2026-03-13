# TADA Inference Run Log

All runs use text: "The quick brown fox jumps over the lazy dog." with ljspeech voice, seed=42, noise_temp=0.9, text_temp=0.6, flow_steps=10 unless noted.

---

## Run 1 (2026-03-12) — First benchmark pass, before timing instrumentation

Files from before timing was added. bench3-6 were overwritten (lost).

| File | Config | Notes |
|------|--------|-------|
| `bench1_python_bf16.wav` | Python BF16 CPU | 5.58s audio, reference quality |

---

## Run 2 (2026-03-12) — With timing instrumentation + voice alignment fix (V2)

Voice alignment fix: match Python's zeros+mask=0 during prompt phase. `num_transition_steps=5` trims all 5 ljspeech voice tokens → text-only conditioning.

| File | Config | Load | Gen | Decode | Audio | RTF |
|------|--------|------|-----|--------|-------|-----|
| `bench3_run2_rust_f16_cpu.wav` | Rust F16 CPU | 7.4s | 19.0s | 2.4s | 3.68s | 5.17x |
| `bench4_run2_rust_f16_metal.wav` | Rust F16 Metal | 4.5s | 19.9s | 3.2s | 3.68s | 5.42x |
| `bench5_run2_rust_q4_cpu.wav` | Rust Q4_0 CPU | 1.4s | 14.2s | 1.6s | 3.04s | 4.66x |
| `bench6_run2_rust_q4_metal.wav` | Rust Q4_0 Metal | 0.9s | 14.2s | 1.6s | 3.04s | 4.66x |

**Key finding**: Metal provides no speedup for generation (QMatMul likely CPU-only in candle). Only helps loading.

**Quality notes**: User reported all Rust outputs sound comparable, possibly better than Python reference.

---

## Run 3 (2026-03-12) — Burn/wgpu hybrid vs candle-only (zero-shot)

First benchmark of the Burn+wgpu LLM port. Zero-shot generation (no voice prompt).
Burn runs LLM on GPU via wgpu, VibeVoice + decoder remain on candle/CPU.

| File | Config | Load | Gen | Decode | Audio | RTF |
|------|--------|------|-----|--------|-------|-----|
| `bench7_run3_candle_q4_cpu.wav` | candle Q4_0 CPU (all components) | 1.8s | 88.2s | 15.7s | 22.8s | 3.87x |
| `bench7_run3_burn_q4_gpu.wav` | Burn/wgpu LLM + candle VibeVoice/decoder | 1.8s | **14.7s** | 2.2s | 4.28s | **3.43x** |

**Generation breakdown (Burn hybrid):**
- LLM forward: 3.5s total, avg 146ms/step (first step 1430ms — GPU warmup)
- VibeVoice flow matching: 10.9s total, avg 455ms/step
- VibeVoice is 74% of generation time — now the bottleneck

**Key findings:**
- Burn GPU LLM is ~11x faster per-step than candle CPU (146ms vs ~1660ms)
- Different generation lengths (24 vs 53 steps) due to GPU/CPU float divergence in token sampling
- Audio durations differ accordingly (4.28s vs 22.8s)
- Quality: needs user evaluation

---

## Run 4 (2026-03-12) — Mixed quantization experiment (bench8)

Investigating mixed Q4+Q8 quantization to reduce model size while preserving quality. Also caught a bug: noise_temp was incorrectly defaulting to 0.6 instead of 0.9.

All runs: Metal, seed=42, text="The quick brown fox jumps over the lazy dog.", voice=ljspeech, flow_steps=10, temp=0.9.

| File | Config | noise_temp | Load | Gen | Decode | Audio | Samples | RTF | Notes |
|------|--------|-----------|------|-----|--------|-------|---------|-----|-------|
| `bench8_run1_q4_metal.wav` | Q4_0 (2.64GB) | 0.6 (wrong) | 3.7s | 6.9s | 2.6s | 3.18s | 76314 | 2.18x | Bad audio — wrong noise_temp |
| `bench8_run2_q4_metal.wav` | Q4_0 (2.64GB) | 0.9 | 3.7s | 7.1s | 2.6s | 3.04s | 72954 | 2.32x | Good audio — baseline reference |
| `bench8_run3_mixedv1_metal.wav` | Mixed Q4+Q8 v1 (1.48GB, decoder attn Q4_0) | 0.9 | 1.0s | 2.2s | 3.4s | 4.14s | 99354 | 0.54x | Near-identical voice quality, slightly longer, trailing noise |
| `bench8_run4_mixedv2_metal.wav` | Mixed Q4+Q8 v2 (1.52GB, decoder attn Q8_0) | 0.9 | 1.0s | 2.3s | 3.5s | 4.14s | 99354 | 0.56x | Identical to run3 — decoder quant type doesn't affect output |

**Key findings:**
- `noise_temp=0.9` is critical (correct reference default); `noise_temp=0.6` produces flat/dead audio
- Mixed quantization nearly halves model size (2.64GB → 1.52GB) with negligible quality loss
- Generation is 3x faster with the mixed model (2.2–2.3s vs 7.1s) — likely less data to load and process per step
- Decoder attention quant type (Q4_0 vs Q8_0) does not change output: acoustic features from VibeVoice are identical before the decoder, so decoder quant precision is irrelevant to voice quality
- Mixed model produces ~1.1s longer audio (99354 vs 72954 samples at 24kHz) — under investigation

---

## Earlier samples (pre-benchmark, 2026-03-12)

| File | Config | Notes |
|------|--------|-------|
| `rust_f16_ljspeech_foxdog.wav` | Rust F16 CPU, V1 (feeds voice feats) | "Clearer, first words garbled" |
| `rust_f16_ljspeech_v2.wav` | Rust F16 CPU, V2 (Python-matching zeros) | "Best" per user |
| `rust_f16_ljspeech_v3_noise07.wav` | Rust F16 CPU, noise_temp=0.7 | Clearer but lower volume |
| `rust_f16_ljspeech_v3_20steps.wav` | Rust F16 CPU, 20 flow steps | Not yet evaluated |
| `rust_f16_ljspeech_v4_20steps_gpu.wav` | Rust F16 Metal, 20 flow steps | Not yet evaluated |
| `python_bf16_ljspeech_foxdog.wav` | Python BF16 CPU reference | "More veiled" per user |
| `python_decoder_from_intermediates.wav` | Python decoder on Python intermediates | Reference decode |
| `rust_decoder_from_intermediates.wav` | Rust decoder on Python intermediates | Identical to Python decoder |
| `python_decoded_rust_intermediates.wav` | Python decoder on Rust intermediates | Confirmed Rust acoustics are valid |
