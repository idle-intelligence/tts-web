# TADA WASM Performance Plan

## Measured Performance (2026-03-20)

### Trace 1: VV skip + warmup, VV on CPU (31.6s)
- 25 steps, 31.6s total
- Warmup: 3.3s, Decode: 9.7s
- Steady state: 554ms/step (LLM GPU + VV CPU parallel)

### Trace 2: VV on GPU via Burn F32 (98.7s — WORSE)
- 61 steps, 98.7s total (3x slower!)
- VV steps: ~5000ms each (F32 matmul on WebGPU is 20x slower than Q8_0 on CPU)
- GPU contention: even LLM-only steps slowed from 480ms to 657ms
- **Conclusion: F32 GPU VV is not viable. Need Q4_0/Q8_0 WGSL shaders for VV.**

### Current best: VV skip + warmup + VV on CPU
Expected: ~25-30s for voice-prompted generation

## Where Time Goes (best config)

Per content step (~554ms):
- LLM forward (WebGPU Q4_0): ~200ms dispatch + ~300ms GPU execution
- VV flow matching (WASM CPU Q8_0): ~250ms (parallel with GPU)
- Async readback: ~50ms
- Net: max(GPU, CPU) + readback ≈ 350-550ms

The GPU and CPU work overlap — the bottleneck alternates.

## Remaining Optimizations

### Quick wins
1. ✅ Skip VV during prompt phase — done, saves ~10s
2. ✅ GPU warmup — done, saves ~3s on first gen
3. ❌ VV on GPU (F32) — tried, 3x WORSE. Need Q4_0/Q8_0 shader
4. **flow_steps=5** — halves VV CPU time from 250ms to ~125ms, saves ~2.5s
5. **Reduce readback gaps** — pipeline GPU dispatch with async readback

### Medium-term
6. **Q8_0 WGSL shader** — would make VV on GPU viable (same speed as Q4_0 LLM)
7. **Prefill** — batch all prompt tokens in one GPU dispatch (not one per step)
8. **Streaming decode** — play audio while generating

### Why VV on GPU failed
Burn's generic F32 matmul on WebGPU is extremely slow because:
- No workgroup-level tiling (naive 1-thread-per-element)
- F32 tensors are 4x larger than Q4_0 (more memory bandwidth)
- 10 ODE steps × 6 layers × 3 matmuls = 180 GPU dispatches per VV call
- Each dispatch has ~2ms overhead on WebGPU
- Total: 180 × 2ms overhead + compute = 5000ms

To make VV on GPU work, we need:
- Q8_0 WGSL shader (like our Q4_0 shader_naive.wgsl but for Q8_0 blocks)
- Or a tiled F32 matmul shader (shared memory, proper workgroup sizes)
