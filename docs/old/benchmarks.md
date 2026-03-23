# TADA-1B Benchmarks

**Hardware**: Apple Silicon Mac (M-series), single machine
**Text**: "The quick brown fox jumps over the lazy dog."
**Voice**: ljspeech (5 tokens)
**Default settings**: seed=42, noise_temp=0.9, text_temp=0.6, flow_steps=10

**RTF** = Real-Time Factor (generation time / audio duration). Lower is better. RTF < 1.0 means faster than real-time.

## All Runs

| Run | Date | Config | Model Size | Load | Gen | Decode | Audio | RTF | WAV File | Notes |
|-----|------|--------|-----------|------|-----|--------|-------|-----|----------|-------|
| 1 | 2026-03-12 | Python BF16 CPU | 3.7 GB | ~10s | ~15s | ~1s | 5.58s | ~2.7x | `bench1_python_bf16.wav` | Reference. |
| 3r2 | 2026-03-12 | candle F16 CPU | 3.3 GB | 7.4s | 19.0s | 2.4s | 3.68s | 5.17x | `bench3_run2_rust_f16_cpu.wav` | V2 alignment fix |
| 4r2 | 2026-03-12 | candle F16 Metal (no feature) | 3.3 GB | 4.5s | 19.9s | 3.2s | 3.68s | 5.42x | `bench4_run2_rust_f16_metal.wav` | GPU not actually used for compute |
| 5r2 | 2026-03-12 | candle Q4_0 CPU | 2.6 GB | 1.4s | 14.2s | 1.6s | 3.04s | 4.66x | `bench5_run2_rust_q4_cpu.wav` | |
| 6r2 | 2026-03-12 | candle Q4_0 Metal (no feature) | 2.6 GB | 0.9s | 14.2s | 1.6s | 3.04s | 4.66x | `bench6_run2_rust_q4_metal.wav` | GPU not actually used — identical to CPU |
| 6r3 | 2026-03-12 | candle Q4_0 Metal | 2.6 GB | 2.6s | **6.8s** | 2.5s | 3.04s | **2.25x** | `bench6_run3_rust_q4_metal.wav` | **Real Metal GPU! 2x faster than CPU** |
| 4r3 | 2026-03-12 | candle F16 Metal | 3.3 GB | — | — | — | — | — | — | OOM killed (3.3GB too large for Metal) |
| 8r1 | 2026-03-12 | candle Q4_0 Metal | 2.64 GB | 3.7s | 6.9s | 2.6s | 3.18s | 2.18x | `bench8_run1_q4_metal.wav` | **BAD AUDIO** — noise_temp=0.6 (wrong default) |
| 8r2 | 2026-03-12 | candle Q4_0 Metal | 2.64 GB | 3.7s | 7.1s | 2.6s | 3.04s | 2.32x | `bench8_run2_q4_metal.wav` | Baseline reference (noise_temp=0.9 fixed) |
| 8r3 | 2026-03-12 | Mixed Q4+Q8 v1 Metal (decoder attn Q4_0) | 1.48 GB | 1.0s | **2.2s** | 3.4s | 4.14s | **0.54x** | `bench8_run3_mixedv1_metal.wav` | Near-identical quality; 3x faster gen; trailing noise |
| 8r4 | 2026-03-12 | Mixed Q4+Q8 v2 Metal (decoder attn Q8_0) | 1.52 GB | 1.0s | **2.3s** | 3.5s | 4.14s | **0.56x** | `bench8_run4_mixedv2_metal.wav` | Identical to 8r3 — decoder quant type irrelevant |

## Mixed-Precision Variant Comparison (2026-03-13)

Testing which model components tolerate lower precision without gray-code mispredictions.

**Setup**: seed=42, temp=0.9, noise_temp=0.9, flow_steps=10, voice=ljspeech, Metal GPU.
**Texts**: "fox" = "The quick brown fox jumps over the lazy dog." / "time" = "Time is money, who can afford to pay attention?"
**Key metric**: `times_before[12]` — gray-code time prediction. Correct value = 4 (gray(4) = 00000110). Misprediction = 59 (gray(59) = 00100110, bit 5 flipped), causing +1.1s trailing noise.

| Variant | VibeVoice | Embeddings | GGUF Size | fox tb[12] | fox dur | time tb[12] | time dur | Gray-code correct? |
|---------|-----------|------------|-----------|-----------|---------|------------|----------|-------------------|
| baseline | F32 | Q4_0 | 2.64 GB | 4 | 3.04s | 4 | 3.88s | Both correct |
| A | F16 | Q8_0 | 1.88 GB | 4 | 3.04s | 59 | 4.98s | fox only |
| B | F32 | Q8_0 | 2.68 GB | 4 | 3.04s | 59 | 4.98s | fox only |
| C | Q8_0 | Q4_0 | 1.38 GB | 59 | 4.10s | 59 | 4.98s | Neither |
| D | Q8_0 | Q8_0 | 1.52 GB | 59 | 4.14s | 59 | 4.98s | Neither |
| **E** | **F16** | **Q4_0** | **1.75 GB** | **4** | **3.04s** | **4** | **3.88s** | **Both correct** |

**Winner: Variant E** — VibeVoice F16 + Embeddings Q4_0. Matches baseline perfectly, 34% smaller (2.64 GB → 1.75 GB), zero gray-code errors.

**Key findings:**
- Embedding precision is the dominant factor: Q8_0 embeddings cause gray-code mispredictions even when VibeVoice is kept at F32 (variant B).
- VibeVoice F16 is sufficient: F16 VibeVoice with Q4_0 embeddings (variant E) matches the F32 baseline exactly.
- VibeVoice Q8_0 introduces independent errors: even with correct Q4_0 embeddings (variant C), VV Q8_0 mispredicts the fox text.
- Misprediction is always a single gray-code bit flip: gray(4)=00000110 → gray(59)=00100110 (bit 5).

## Post-Bugfix Full Benchmark (2026-03-13)

Post-bugfix run covering all variants × 4 phrases. Bugs fixed before this run: autoregressive time feedback bug (predictions now fed back correctly), trailing EOT frame trim (removes silent padding), WASM EOT token fix (padding token corrected).

**Setup**: Apple Silicon Mac, Metal GPU, seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech (5 tokens), transition_steps=0.
**Python reference**: BF16 on CPU (MPS has dtype assertion errors), transformers 4.57.6, CFG acoustic_cfg_scale=1.6, transition_steps=5.
**Note**: Python uses CFG (2× LLM passes per step) and transition_steps=5 (trims all 5 voice tokens → effectively zero-shot). Rust uses no CFG and transition_steps=0 (uses all 5 tokens). Audio duration differences (Python fox=5.58s vs Rust fox=2.7s) are likely due to CFG influence on duration predictions.
**"call" phrase note**: Consistently truncated across ALL variants (1.1–2.7s for a phrase expected ~4s) — model behavior, not a quantization or Rust artifact.

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

### Full Results

| Variant | Size | Phrase | Load | Gen | Decode | Audio | RTF |
|---------|------|--------|------|-----|--------|-------|-----|
| F32 GGUF | 6.5 GB | fox | 38.1s | 40.0s | 2.3s | 2.7s | 14.6x |
| F32 GGUF | 6.5 GB | call | 38.5s | 26.7s | 0.9s | 1.1s | 24.7x |
| F32 GGUF | 6.5 GB | tyger | 37.4s | 29.4s | 1.6s | 2.1s | 14.3x |
| F32 GGUF | 6.5 GB | wutang | 37.5s | 39.3s | 3.3s | 4.2s | 9.4x |
| F16 GGUF | 3.3 GB | fox | 10.6s | 19.1s | 2.2s | 2.7s | 7.0x |
| F16 GGUF | 3.3 GB | call | 8.0s | 17.1s | 1.0s | 1.1s | 15.9x |
| F16 GGUF | 3.3 GB | tyger | 7.2s | 16.0s | 1.6s | 2.1s | 7.8x |
| F16 GGUF | 3.3 GB | wutang | 6.6s | 20.5s | 3.4s | 4.2s | 4.9x |
| Q4_0 baseline | 2.6 GB | fox | 2.5s | 7.1s | 2.3s | 2.8s | 2.6x |
| Q4_0 baseline | 2.6 GB | call | 2.7s | 7.4s | 1.3s | 1.6s | 4.8x |
| Q4_0 baseline | 2.6 GB | tyger | 2.7s | 6.4s | 1.5s | 1.8s | 3.5x |
| Q4_0 baseline | 2.6 GB | wutang | 2.3s | 9.4s | 3.2s | 3.9s | 2.4x |
| Var-B VV-F32 E-Q8 | 2.5 GB | fox | 2.9s | 6.9s | 2.2s | 2.8s | 2.5x |
| Var-B VV-F32 E-Q8 | 2.5 GB | call | 3.0s | 7.5s | 1.3s | 1.6s | 4.8x |
| Var-B VV-F32 E-Q8 | 2.5 GB | tyger | 2.8s | 6.3s | 1.6s | 1.9s | 3.3x |
| Var-B VV-F32 E-Q8 | 2.5 GB | wutang | 2.7s | 9.6s | 3.3s | 4.2s | 2.3x |
| Python BF16 CPU | — | fox | 2.0s | 4.7s | — | 5.6s | 0.8x |
| Python BF16 CPU | — | call | 2.0s | 4.8s | — | 2.7s | 1.8x |
| Python BF16 CPU | — | tyger | 2.0s | 4.2s | — | 6.1s | 0.7x |
| Python BF16 CPU | — | wutang | 2.0s | 7.0s | — | 6.7s | 1.0x |
| Var-A VV-F16 E-Q8 | 1.9 GB | fox | 1.4s | 6.9s | 2.3s | 2.8s | 2.5x |
| Var-A VV-F16 E-Q8 | 1.9 GB | call | 1.1s | 7.4s | 1.3s | 1.6s | 4.8x |
| Var-A VV-F16 E-Q8 | 1.9 GB | tyger | 1.1s | 6.3s | 1.6s | 1.9s | 3.3x |
| Var-A VV-F16 E-Q8 | 1.9 GB | wutang | 1.1s | 9.4s | 3.4s | 4.2s | 2.2x |
| Var-E VV-F16 E-Q4 | 1.8 GB | fox | 1.3s | 6.6s | 2.2s | 2.8s | 2.4x |
| Var-E VV-F16 E-Q4 | 1.8 GB | call | 1.0s | 7.1s | 1.3s | 1.6s | 4.5x |
| Var-E VV-F16 E-Q4 | 1.8 GB | tyger | 1.0s | 5.9s | 1.5s | 1.8s | 3.3x |
| Var-E VV-F16 E-Q4 | 1.8 GB | wutang | 1.0s | 9.3s | 3.2s | 3.9s | 2.4x |
| Mixed VV-Q8 E-Q8 | 1.4 GB | fox | 1.0s | 2.3s | 2.3s | 2.7s | 0.8x |
| Mixed VV-Q8 E-Q8 | 1.4 GB | call | 0.8s | 2.4s | 1.3s | 1.6s | 1.5x |
| Mixed VV-Q8 E-Q8 | 1.4 GB | tyger | 0.8s | 2.1s | 1.6s | 1.9s | 1.1x |
| Mixed VV-Q8 E-Q8 | 1.4 GB | wutang | 0.7s | 2.9s | 3.5s | 4.4s | 0.7x |
| Var-C VV-Q8 E-Q4 | 1.3 GB | fox | 0.9s | 2.2s | 2.1s | 2.7s | 0.8x |
| Var-C VV-Q8 E-Q4 | 1.3 GB | call | 0.7s | 2.4s | 1.3s | 1.6s | 1.5x |
| Var-C VV-Q8 E-Q4 | 1.3 GB | tyger | 0.7s | 2.1s | 1.5s | 1.8s | 1.2x |
| Var-C VV-Q8 E-Q4 | 1.3 GB | wutang | 0.7s | 2.9s | 3.3s | 3.9s | 0.8x |

## Q8_0-LLM Alignment Test (2026-03-14)

**Setup**: candle Metal, seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech, transition_steps=0.
**Model**: Q8_0 LLM + F16 VV + Q4_0 Embed = 2.24 GB (file: `tada-1b-llmq8-vvf16-eq4.gguf`)
**Purpose**: Test whether Q8_0 LLM backbone produces F16-aligned output (Q4_0 had 480ms duration gaps on fox/call).

| Phrase | Load | Gen | Decode | Audio | RTF |
|--------|------|-----|--------|-------|-----|
| fox | 2.4s | 7.1s | 2.2s | 2.7s | 2.57x |
| call | 2.0s | 7.4s | 0.9s | 1.1s | 6.89x |
| tyger | 1.9s | 6.2s | 1.4s | 1.7s | 3.65x |
| wutang | 1.9s | 9.5s | 3.1s | 4.0s | 2.35x |

**Alignment comparison (sample counts)**:

| Phrase | F32/F16 | Q4_0 | Q8_0-LLM | Q8 vs F16 |
|--------|---------|------|-----------|-----------|
| fox | 65754 | 66234 (+480) | 65754 | exact match, corr=0.999 |
| call | 25914 | 37434 (+11520) | 25914 | exact match, corr=0.995 |
| tyger | 49434 | 43674 (-5760) | 40794 | closer but different length |
| wutang | 100794 | 94554 (-6240) | 96474 | closer but different length |

**Key findings**: Q8_0 LLM matches F16 sample-for-sample on fox and call (corr>0.995). On tyger/wutang the token sequence diverges at some point but durations are closer to F16 than Q4_0 was. The "call" truncation gap (480ms / 11520 samples) seen in Q4_0 is completely eliminated.

## Burn/wgpu GPU Benchmark — Zero-shot (2026-03-14)

**Setup**: Burn/wgpu LLM on GPU + candle VibeVoice/decoder on CPU. Q4_0 baseline GGUF (2.6 GB). Zero-shot (no voice prompt — the Burn example doesn't support `--voice` yet).
**Purpose**: Benchmark the GPU path (needed for web/WebGPU deployment).

| Phrase | Load | Gen | LLM | VibeVoice | Decode | Audio | RTF |
|--------|------|-----|-----|-----------|--------|-------|-----|
| fox | 2.5s | 20.7s | 4.3s | 15.7s | 1.6s | 2.86s | 7.23x |
| call | 2.4s | 80.6s | 11.6s | 65.2s | — | 6.06s | 13.30x |
| tyger | 2.3s | 77.5s | 11.0s | 62.8s | — | 4.04s | 19.19x |
| wutang | 2.3s | 16.7s | 3.6s | 12.8s | — | 2.48s | 6.72x |

Per-step averages: LLM=80–130ms (GPU), VibeVoice=455–463ms (CPU).

**Key findings**:
- LLM per-step 80–130ms on GPU — approximately 11x faster than candle CPU per step.
- VibeVoice 455–463ms/step on CPU — unchanged from Run 3; 76–81% of total gen time.
- Zero-shot generates far more steps than voice-prompted (143 frames for "call" vs ~25 with voice) — RTF not comparable to voice-prompted candle runs.
- The Burn GPU example does not support voice prompts, seed, or transition_steps — needs enhancement for proper comparison.
- The Burn WGSL shader only supports Q4_0 — cannot load Var-E or Q8_0-LLM GGUFs (errors on dtype code 8).

**Caveat**: These are zero-shot runs (no voice conditioning). Audio quality and step counts are not comparable to voice-prompted runs. Step counts inflated by `num_extra_steps=50` in zero-shot mode.

## Run 9: Parameter investigation — tyger debug (2026-03-14)

**Setup**: Commit 83c08ab. Q4_0 baseline (2.6 GB) except where noted. seed=42, noise_temp=0.9, Metal GPU unless noted.
**Text**: "Tyger Tyger, burning bright" with voice=ljspeech.

| File | Model | transition_steps | GPU | Audio | Quality |
|------|-------|-----------------|-----|-------|---------|
| `tyger_ts0.wav` | Q4_0 (2.6G) | 0 | Metal | 1.8s | Bad — wrong phonemes |
| `tyger_ts5.wav` | Q4_0 (2.6G) | 5 (all trimmed) | Metal | 1.2s | Matches Python but cut early |
| `tyger_novoice.wav` | Q4_0 (2.6G) | 0 (no voice) | Metal | 3.3s | Nonsense sounds |
| `tyger_cpu.wav` | Q4_0 (2.6G) | 0 | CPU | 1.8s | Identical to tyger_ts0 |

**Key findings**:
- Metal and CPU produce identical audio (same duration, same content).
- transition_steps=5 with 5-token voice prompt = effectively zero-shot = matches Python but truncated.
- No voice = nonsense output.
- "Tyger" is a model-level issue — no parameter combination produces correct pronunciation.

## Run 10: Cross-variant quality test — "time" phrase (2026-03-14)

**Setup**: Commit 83c08ab. voice=ljspeech, noise_temp=0.9, transition_steps=0, seed=42, Metal GPU.
**Text**: "Time is money, who can afford to pay attention?"

| File | Model | GGUF Size | Gen | Audio | RTF | Quality |
|------|-------|-----------|-----|-------|-----|---------|
| `time_f32.wav` | F32 | 6.5G | 30.9s | 2.5s | 12.4x | Good |
| `time_q4.wav` | Q4_0 baseline | 2.6G | 7.1s | 2.5s | 2.9x | Good |
| `time_varE.wav` | Var-E (VV-F16 E-Q4) | 1.8G | 6.7s | 2.5s | 2.7x | Good |
| `time_mixed.wav` | Mixed (VV-Q8 E-Q8) | 1.4G | 2.3s | 2.5s | 0.9x | Good |
| `time_varC.wav` | Var-C (VV-Q8 E-Q4) | 1.3G | 2.3s | 2.5s | 0.9x | Good |

**Key findings**:
- All 5 model sizes produce identical audio duration (2.5s); user confirmed good quality on all.
- No audible difference between F32 (6.5G) and Var-C (1.3G) for this phrase.
- Mixed and Var-C achieve sub-1× RTF (realtime) at 1.3–1.4 GB.
- Post-bugfix, quantization variant does not affect audio quality for phrases the model handles well.
- Problematic phrases (tyger, call, wutang) are model-level weaknesses, not quantization or pipeline issues.

### Skipped / Invalid

| Run | Date | Config | Notes |
|-----|------|--------|-------|
| 2 | 2026-03-12 | Python Q4_0 CPU | Skipped: transformers 5.3.0 bug |
| 7r3a | 2026-03-12 | candle Q4_0 CPU (zero-shot) | **BAD AUDIO** — zero-shot, no voice. Not comparable. |
| 7r3b | 2026-03-12 | Burn/wgpu+candle Q4_0 (zero-shot) | **BAD AUDIO** — zero-shot, no voice. Not comparable. |
| 8r1 | 2026-03-12 | candle Q4_0 Metal | **BAD AUDIO** — noise_temp=0.6 (wrong default; should be 0.9). |

### Pending

- [ ] Burn/wgpu hybrid with ljspeech voice (proper comparison to 6r3)
- [ ] candle Q4_K with ljspeech voice
- [ ] Burn/wgpu Q4_K with ljspeech voice
- [ ] Investigate mixed model longer audio output (99354 vs 72954 samples)

## Key Observations

- **Metal with `--features metal` gives 2x speedup** — Q4_0 gen went from 14.2s (CPU) to 6.8s (Metal).
- **Q4_0 is ~25% faster than F16** on CPU (14.2s vs 19.0s) with smaller model.
- **Quality**: All Rust outputs sound comparable to Python, possibly better. Python reference described as "more veiled."
- **Rust generates shorter audio** (3.0-3.7s) than Python (5.58s) for the same text — different RNG produces different durations.
- **Run 7 (zero-shot) produced garbage** — need to investigate zero-shot generation path, or only compare voice-prompted runs.
- **noise_temp=0.9 is critical** — noise_temp=0.6 (wrong default caught in bench8) produces flat/dead audio.
- **Mixed Q4+Q8 quantization nearly halves model size** (2.64GB → 1.52GB) with negligible quality loss and 3x faster generation (7.1s → 2.3s on Metal).
- **Decoder attention quant precision doesn't affect output** — acoustic features arrive from VibeVoice already fixed; Q4_0 vs Q8_0 in the decoder produces identical audio.
- **Mixed model produces longer audio** (~4.14s / 99354 samples vs ~3.04s / 72954) for the same text and seed — cause under investigation.

### Run 7 Timing Notes (invalid audio, but timing data still useful)

Despite bad audio, the Burn/wgpu per-step timing is informative:
- LLM per-step: Burn GPU avg **146ms** (first step 1430ms GPU warmup) vs candle CPU avg **~1660ms** → **11x speedup**
- VibeVoice per-step: 455ms (candle CPU, same in both — this is the bottleneck)
- VibeVoice = 74% of generation time in the hybrid pipeline
