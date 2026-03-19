# TADA WASM Performance Plan

## Measured Performance (2026-03-19, Chrome trace)

**56 generation steps in 40.1s** (34.2s compute + 5.9s idle/await gaps)

### Per-step breakdown:
- **Step 0** (shader warmup): **1409ms**
- **Step 1**: 781ms + **1721ms gap** (GPU readback stall)
- **Step 2**: 559ms + **1356ms gap** (GPU readback stall)
- **Steps 3-54** (steady state): **~488ms/step** + ~50ms gap
- **Step 55** (decode): **6171ms** (DAC decoder, single step)

### Where the 488ms/step goes:
Native Burn Var-C: LLM 140ms + VV 85ms = 225ms/step
WASM: 488ms/step = **2.2x slower than native**

Estimated split (need per-op instrumentation to confirm):
- LLM (WebGPU Q4 matmul): ~200-250ms
- VibeVoice (WASM CPU Q8 matmul): ~200-250ms
- Async readback (into_data_async): ~50ms gap between steps

**Note**: The initial estimate of "VV = 80% of time" was WRONG. The perf plan assumed candle WASM would be 20-25x slower than native. Actual slowdown is only ~2.2x overall. The Q8_0 VibeVoice is not as slow as feared — likely because it's simpler ops (6 layers × 10 ODE steps vs 16 transformer layers).

### Key bottlenecks:

1. **Shader warmup: 1409ms** on first step + 3077ms readback stalls on steps 1-2
   - Fix: add warmup pass after loading (stt-web pattern)
   - Expected savings: ~4.5s on first generation

2. **Step count: 56 steps** for a ~5s audio
   - With voice prompt (32 tokens): 8 prefix + 32 voice + ~12 target + 6 EOT = 58 tokens
   - VibeVoice runs on steps 5-52 = **47 VibeVoice calls**
   - BUT: steps 5-39 are prompt phase — their acoustic output is STRIPPED
   - **~34 VibeVoice calls are wasted** on prompt-phase steps
   - Fix: skip VibeVoice during prompt phase (feed zeros for acoustic, keep time feedback)
   - Expected savings: 34 × 488ms = **~16.6s** (42% reduction!)

3. **DAC decode: 6171ms** (step 55)
   - This is the audio decoder running on WASM CPU
   - Native: ~1.5s for same decode
   - ~4x slower in WASM
   - Lower priority — runs once at end

4. **Async readback gap: ~50ms/step** (steps 3+)
   - WebGPU mapAsync round-trip
   - Total: 50ms × 56 = 2.8s
   - Reducible but not the main bottleneck

### Comparison with stt-web:
stt-web (1.2GB Q4_0) runs the same Burn/wgpu pipeline ~2-3x faster.
Key differences:
- stt-web has GPU warmup (10 dummy frames on load)
- stt-web uses non-blocking readback (spawn_local + future_to_promise)
- stt-web doesn't have a VibeVoice diffusion head

## Action Plan (Priority Order)

### Immediate (big wins, no architecture change)

1. **Skip VibeVoice during prompt phase** — save ~16s (42%)
   - Steps where `step < prompt_phase_len + shift_acoustic` don't need VibeVoice
   - Acoustic features come from voice prompt, not VibeVoice
   - Need to verify: are time_before/time_after predictions needed during prompt phase?
   - If yes, can feed zeros (they get stripped anyway)

2. **GPU warmup after model load** — save ~4.5s on first gen
   - Pattern from stt-web: run 5-10 dummy forward passes, then reset_keep_buffers
   - Pre-compiles all WGSL shader pipelines

3. **Reduce flow_steps to 5 in browser** — save ~50% of VibeVoice time
   - flow_steps=10 → 10 ODE steps per VibeVoice call
   - flow_steps=5 → 5 ODE steps, ~half the VV compute
   - Quality tradeoff: less accurate ODE integration
   - Test: does flow_steps=5 produce acceptable audio?

### Medium-term

4. **Non-blocking GPU readback** (stt-web pattern)
   - Use spawn_local + future_to_promise for async readback
   - Pipeline: dispatch step N+1 while reading back step N
   - Could overlap LLM and readback

5. **Reduce step count** (architecture)
   - Shorter voice prompts need fewer steps
   - Or: prefill (process all prompt tokens in one batched forward pass)

### Long-term

6. **Move VibeVoice to WebGPU**
7. **Model distillation / smaller variant**
8. **Streaming audio decode**

## Expected Outcome After Fixes 1-3

Current: 56 steps × 488ms + 6.2s decode + 4.5s warmup = **40s**

After fix 1 (skip VV in prompt): 56 steps, but only 22 need VV:
- 34 prompt steps × ~200ms (LLM only) = 6.8s
- 22 content steps × 488ms = 10.7s
- Decode: 6.2s
- Total: **~24s** (40% faster)

After fix 2 (warmup): subtract 4.5s from first gen = **~19s**

After fix 3 (flow_steps=5): VV portion halved in content steps:
- 34 prompt steps × 200ms = 6.8s
- 22 content steps × ~350ms = 7.7s
- Decode: 6.2s
- Total: **~21s** (first gen), **~16s** (subsequent)

That gets us from 40s → ~16-21s. Still not realtime, but 2-2.5x improvement.
