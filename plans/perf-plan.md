# TADA WASM Performance Plan

## Latest Trace (2026-03-20, Q8_0 VV on GPU + tasks_max=512)

**Total: 50.6s** (29 steps, Steps=10, CFG=1.0, long voice)

| Phase | Time | % | Status |
|-------|------|---|--------|
| Warmup (LLM + VV shaders) | 10.6s | 21% | Fix: add VV warmup pass |
| First VV compile (step 4 gap) | 5.5s | 11% | Fix: VV warmup covers this |
| Prompt steps (1-3) | 0.5s | 1% | ✓ VV skip working |
| Content steps (5-27, 23 steps × 551ms) | 12.7s | 25% | Acceptable |
| **Decode** | **17.8s** | **35%** | **NEW BOTTLENECK** |
| Gaps/overhead | 3.3s | 7% | |

### Second generation (shaders cached): ~32s
- Content: 13.8s
- Decode: 17.8s

## Per-step comparison across all configs

| Config | Content step | Total gen | Decode |
|--------|------------|-----------|--------|
| Original (no opts) | 486ms | 40.1s | 6.2s |
| VV skip, CPU VV | 554ms | 31.6s | 9.7s |
| tasks_max=512, CPU VV | 540ms | 53.1s* | 8.2s |
| **Q8_0 VV on GPU** | **551ms** | **50.6s** | **17.8s** |
| CPU VV, CFG=1.6, Steps=20 | 4253ms | 74.2s | 3.6s |

*53.1s had longer voice prompt (45 steps vs 29)

## The decode problem

DAC decoder: 50+ conv1d layers on WASM CPU. Native: 1.5s. WASM: 8-18s.
Decode is now 35% of total time. Getting worse with GPU VV (memory pressure?).

Options:
1. **Move decoder to GPU** — same dispatch overhead concern, but conv1d is fewer ops than VV
2. **WebCodecs API** — browser-native audio decoding (not applicable, custom codec)
3. **WASM SIMD** — verify candle uses simd128 for conv1d
4. **Streaming decode** — decode chunks as generated, overlap with next step

## Remaining optimizations

### Immediate
1. ✅ VV skip during prompt phase
2. ✅ GPU warmup (LLM)
3. ✅ tasks_max=512 (command batching)
4. ✅ Q8_0 VV on GPU
5. **VV warmup** — building now, saves 16s on first gen
6. **Investigate decode** — why 17.8s vs 8.2s?

### Medium-term
7. Streaming decode
8. Decoder on GPU (Q8_0 shader for conv1d)
9. Prefill (batch prompt tokens)

### Long-term
10. Fused LLM+VV kernel
11. Model distillation
