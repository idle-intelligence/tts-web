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

## Run 5 (2026-03-13) — Mixed-precision variant comparison

**Hypothesis going in**: The trailing noise seen in bench8 run3/run4 (4.14s audio vs 3.04s baseline) was suspected to come from VibeVoice quantization errors. The mixed model used Q8_0 for VibeVoice weights, so the plan was to test whether upgrading VV to F16 or F32 would fix the gray-code misprediction (`times_before[12]` = 59 instead of 4) that causes the extra ~1.1s.

All runs: seed=42, temp=0.9, noise_temp=0.9, flow_steps=10, voice=ljspeech, Metal GPU. Two texts tested: "The quick brown fox jumps over the lazy dog." (fox) and "Time is money, who can afford to pay attention?" (time).

| Variant | VibeVoice | Embeddings | GGUF Size | fox tb[12] | fox dur | time tb[12] | time dur |
|---------|-----------|------------|-----------|-----------|---------|------------|----------|
| baseline | F32 | Q4_0 | 2.64 GB | 4 | 3.04s | 4 | 3.88s |
| A | F16 | Q8_0 | 1.88 GB | 4 | 3.04s | 59 | 4.98s |
| B | F32 | Q8_0 | 2.68 GB | 4 | 3.04s | 59 | 4.98s |
| C | Q8_0 | Q4_0 | 1.38 GB | 59 | 4.10s | 59 | 4.98s |
| D | Q8_0 | Q8_0 | 1.52 GB | 59 | 4.14s | 59 | 4.98s |
| E | F16 | Q4_0 | 1.75 GB | 4 | 3.04s | 4 | 3.88s |

**The surprise**: Variant B (F32 VibeVoice + Q8_0 embeddings) still mispredicts the "time" text. VibeVoice precision is not the culprit — Q8_0 embeddings are sufficient to cause the bit flip, regardless of how precise the flow-matching computation is downstream.

**The real story**: The gray-code time token is predicted by the LLM, which attends over token embeddings. Q8_0 embedding quantization introduces enough error in the "time" text's context to flip bit 5 in the gray-coded duration prediction (gray(4)=00000110 → gray(59)=00100110). Q4_0 embeddings, despite being coarser in absolute terms, happen to preserve the critical activations correctly for both test texts.

**Conclusion**: Variant E (VibeVoice F16 + Embeddings Q4_0) is the winner. At 1.75 GB it is 34% smaller than the 2.64 GB baseline, loads faster, and produces identical audio for both test texts. VibeVoice at F16 is sufficient precision — F32 adds 0.89 GB with no measurable benefit.

---

## Run 6 (2026-03-13) — Post-bugfix full benchmark, all variants × 4 phrases

**Bugs fixed before this run:**
- Autoregressive time feedback bug: predictions now fed back correctly instead of re-feeding the input
- Trailing EOT frame trim: removes the final silent EOT padding frame from audio output
- WASM EOT token fix: padding token corrected in WASM bindings

**Test matrix**: 9 configs × 4 phrases = 36 runs. Phrases: fox ("The quick brown fox jumps over the lazy dog."), call ("I had to call you up in the middle of the night"), tyger ("Tyger Tyger, burning bright"), wutang ("Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds.").

**Setup**: seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech (5 tokens), transition_steps=0, Metal GPU.
**Python reference**: BF16 CPU (MPS has dtype assertion errors), CFG acoustic_cfg_scale=1.6, transition_steps=5.

### Summary (averages across 4 phrases)

| Variant | GGUF Size | Avg Load | Avg Gen | Avg Decode | Avg Audio | Avg RTF | Notes |
|---------|-----------|----------|---------|------------|-----------|---------|-------|
| F32 GGUF | 6.5 GB | 37.9s | 33.9s | 2.0s | 2.5s | 15.8x | Unusable — 38s load, 30-40s gen |
| F16 GGUF | 3.3 GB | 8.1s | 18.2s | 2.1s | 2.5s | 8.9x | 2× faster than F32, still 3× slower than Q4_0 |
| Q4_0 baseline | 2.6 GB | 2.6s | 7.6s | 2.1s | 2.5s | 3.3x | LLM compute dominates gen |
| Var-B VV-F32 E-Q8 | 2.5 GB | 2.9s | 7.6s | 2.1s | 2.6s | 3.2x | Same gen as Q4_0 — LLM Q4_0 dominates |
| Python BF16 CPU | — | 2.0s | 5.2s | — | 5.3s | 1.1x | CFG (2× passes) + no separate decode step |
| Var-A VV-F16 E-Q8 | 1.9 GB | 1.2s | 7.5s | 2.2s | 2.6s | 3.2x | |
| Var-E VV-F16 E-Q4 | 1.8 GB | 1.1s | 7.2s | 2.1s | 2.5s | 3.1x | Matches Q4_0 baseline quality (Run 5 finding) |
| Mixed VV-Q8 E-Q8 | 1.4 GB | 0.8s | 2.4s | 2.2s | 2.7s | 1.0x | ~3× faster gen than Q4_0 variants; sub-1× on fox/wutang |
| Var-C VV-Q8 E-Q4 | 1.3 GB | 0.8s | 2.4s | 2.1s | 2.5s | 1.1x | Smallest; sub-1× on fox/wutang |

### Key observations

- **Performance tiers**: VV-Q8 variants (Mixed, Var-C) average 2.4s gen vs 7.2–7.6s for Q4_0/VV-F16/VV-F32 variants — approximately 3× faster generation.
- **F32 GGUF is unusable**: 38s load + 30-40s gen per phrase. F16 GGUF is 2× faster than F32 but still 3× slower than Q4_0.
- **Q4_0, Var-B, Var-A, Var-E all have nearly identical gen times** (~6.5–7.5s for fox): the LLM backbone (Q4_0 in all) dominates generation time, not the VibeVoice precision.
- **Mixed and Var-C hit sub-1× RTF on fox and wutang** (realtime generation achieved).
- **Python BF16 CPU is surprisingly competitive**: 4.7s gen (fox) vs Rust Q4_0 Metal 7.1s. Python includes CFG (2× LLM passes per step) and no separate decode step — the apparent efficiency likely reflects CFG producing shorter-duration outputs in this comparison.
- **Audio duration asymmetry**: Python produces longer audio (fox 5.6s) vs Rust (fox 2.7–2.8s). CFG scale and transition_steps differences (Python: CFG=1.6, transition_steps=5; Rust: no CFG, transition_steps=0) are the likely cause.
- **"call" phrase consistently truncated across ALL variants**: audio 1.1–2.7s for a phrase expected ~4s. Observed in Q4_0, F32, F16, and all mixed variants. Model behavior, not a quantization artifact.
- **Decode time scales with audio length**: roughly 0.5–0.8s decode per second of audio across all Rust variants.

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
