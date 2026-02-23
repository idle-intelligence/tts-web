# Candle QMatMul performance on WASM: brief for upstream

## Problem

`QMatMul::QTensor` (quantized matmul) is **~1.7x slower** than `Linear` (F32 matmul via `gemm`) on WASM with simd128 enabled, for the matrix sizes typical in transformer inference (1024×3072, 512×2048, etc.).

This makes it counterproductive to keep weights quantized at runtime — the memory bandwidth savings from Q8_0 (1 byte vs 4 bytes per element) are overwhelmed by the naive matmul kernel.

## Benchmark

Model: [pocket-tts](https://github.com/kyutai-labs/pocket-tts) (118M params, 6-layer 1024-dim transformer + 2-layer 512-dim decoder)

| Approach | Realtime factor (higher = better) |
|---|---|
| F32 `Linear` via `gemm` | 1.8x |
| Q8_0 `QMatMul::QTensor` | 1.08x |

Target: `wasm32-unknown-unknown` with `target-feature=+simd128`, Chrome, M-series Mac.

## Root cause

`candle_core::quantized::matmul` (in `k_quants.rs`) uses a **naive triple loop**:

```rust
for row_idx in 0..m {
    for col_idx in 0..n {
        dst[...] = T::vec_dot(k, rhs_col, lhs_row);
    }
}
```

While `vec_dot_q8_0_q8_0` in `simd128.rs` is well-vectorized (using `i32x4_dot_i16x8`), the outer loop has:

1. **No tiling/blocking** — poor cache utilization for large matrices
2. **Per-call LHS quantization** — `T::VecDotType::from_float()` converts the entire activation tensor from F32 to Q8_0 blocks on every forward pass
3. **No parallelism** — single-threaded, no work decomposition

By contrast, `gemm` (used by `candle_nn::Linear`) has WASM SIMD kernels with L1/L2-aware tiling, micro-kernel packing, and efficient register blocking.

## What would help

1. **Tiled quantized matmul kernel** — block the M/N/K dimensions for cache locality, similar to how `gemm` tiles F32. Even a simple 64×64 tile would help significantly.

2. **Amortize LHS quantization** — when the same activation tensor is multiplied against multiple weight matrices (e.g., Q/K/V projections in attention), quantize LHS once and reuse. Currently `from_float()` is called per matmul.

3. **Consider `gemm` integration for quantized types** — `gemm` already supports custom micro-kernels. A Q8_0 micro-kernel that reads packed int8 blocks and accumulates i32 dot products would get the tiling infrastructure for free.

## Workaround

We dequantize Q8_0 → F32 at load time and use `candle_nn::Linear`. The GGUF file stays compact for download (178 MB Q8_0 vs 236 MB F32), but runtime uses F32 matmuls.

This sacrifices the memory bandwidth advantage of quantized inference, which matters on mobile devices with limited bandwidth.

## Relevant code paths

- `candle-core/src/quantized/k_quants.rs` — `matmul()` function (~line 2268)
- `candle-core/src/quantized/simd128.rs` — `vec_dot_q8_0_q8_0()` (~line 54)
- `candle-core/src/quantized/mod.rs` — `QTensor::matmul_t()` and `QMatMul::forward()`
