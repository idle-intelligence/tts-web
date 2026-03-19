# TADA WASM Performance Plan

## Current State (2026-03-15)

Browser (WebGPU + WASM): ~93s for 5.4s audio = **17x RTF** (terrible)
Native Burn GPU (Var-C, Metal): ~5.4s for 2.7s audio = **2x RTF**
Native candle Metal (Var-C): ~2.3s for 2.7s audio = **0.85x RTF** (realtime!)

## Where Time Goes (Browser)

Per-step breakdown from native Burn Var-C:
- **LLM** (Burn GPU): 134-140ms/step
- **VibeVoice** (candle CPU): 85ms/step (Q8_0 VV)
- Total per step: ~220ms × 30 steps ≈ 6.6s

In WASM, everything is slower:
- WebGPU dispatch overhead (vs native Metal)
- WASM CPU is ~3-5x slower than native CPU
- Async readback overhead (into_data_async round-trip)
- No SIMD for candle QMatMul in WASM (or limited simd128)

Estimated WASM breakdown:
- LLM: ~500ms/step? (WebGPU warmup + dispatch)
- VibeVoice: ~2000ms/step? (CPU WASM, no Metal)
- Total: ~2500ms × 34 steps ≈ 85s → matches observed 93s

## Optimization Priorities

### 1. VibeVoice is the bottleneck (~80% of gen time)
- Q8_0 VV on native CPU: 85ms/step
- Q8_0 VV on WASM CPU: ~2000ms/step (estimated 20-25x slower)
- Options:
  - **Move VibeVoice to WebGPU** — would need Burn or custom WGSL shaders for VV
  - **Use wasm-simd128** — candle may already use simd128, but verify
  - **Reduce flow_steps** — 5 instead of 10 (halves VV time, quality tradeoff)
  - **Reduce VV precision** — Q4_0 VV instead of Q8_0 (but we know Q8_0 is the sweet spot)

### 2. LLM WebGPU overhead
- First step has ~1400ms warmup (shader compilation)
- Add warmup pass after loading (stt-web pattern: 10 dummy frames, then reset_keep_buffers)
- Subsequent steps: ~130-170ms native, likely ~300-500ms in WebGPU

### 3. Async readback latency
- Each step does into_data_async → GPU→CPU transfer for hidden state
- This is a round-trip through WebGPU's mapAsync → adds ~5-10ms per step
- Not the bottleneck but worth optimizing later

### 4. Model loading
- Currently loads 1.38 GB into WASM memory
- Could split into multiple shards to avoid single large allocation
- stt-web pattern: ShardedCursor for >2GB models

### 5. Streaming audio
- Currently generates all audio, then decodes, then plays
- Could stream chunks as they're generated (like pocket-tts does)
- Requires partial decode support in the DAC decoder

## Quick Wins (Do First)
1. Add GPU warmup after model load (~saves 1400ms on first gen)
2. Reduce flow_steps to 5 for browser (halves VibeVoice time)
3. Verify simd128 is enabled for candle WASM

## Medium-term
4. Move VibeVoice to WebGPU (Burn or custom shaders)
5. Streaming audio output

## Long-term
6. Q4_K WGSL shader (currently only Q4_0)
7. Model distillation / smaller TADA variant
