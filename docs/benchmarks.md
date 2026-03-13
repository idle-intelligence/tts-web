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
