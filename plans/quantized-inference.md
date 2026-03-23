# Quantized Inference for Mobile Performance

## Context

The TTS model dequantizes INT8 weights to F32 at load time, so ~388MB of F32 weights sit in memory. Every inference step reads all of them. On mobile phones (~20GB/s bandwidth vs M2's 100GB/s), this caps inference at 0.5-0.65x realtime.

**Goal**: Keep weights as Q8_0 in memory (~103MB), dequantize on-the-fly during matmul via candle's `QMatMul`. ~4x less memory traffic per inference step.

## Why this works (no architectural blockers)

- All weight dimensions are multiples of 32 (Q8_0 block size) — verified every Linear
- Candle has WASM SIMD128-optimized `vec_dot_q8_0_q8_0` (already enabled in our build)
- The matmul auto-quantizes F32 input vectors on-the-fly per row (standard GGML approach)
- Non-quantized weights (SEANet convs, LayerNorm, biases, embeddings) stay F32 — no issue
- Mimi weights are BF16 in the file -> F32 at load. SEANet convs stay F32. Only Mimi transformer Linears get quantized.

## Approach: Post-load F32 -> Q8_0 conversion

```
INT8 safetensors -> dequantize_and_remap() -> BF16 -> VarBuilder -> F32 Linear
  -> model.quantize_weights() -> QLinear(QMatMul::QTensor) with Q8_0
```

The F32 roundtrip is a one-time load cost (~1-2s extra, ~613MB peak memory) that's fine on mobile. The runtime benefit (every step, forever) dwarfs it. We can optimize the loading path later (direct INT8->Q8_0 or GGUF) if needed.

## Implementation

### Branch: `quantized-inference`

Work on a dedicated branch, commit often. Merge to main when verified.

### Team

| Role | Model | Scope |
|---|---|---|
| **Lead** (opus) | Orchestration | Overall context, task assignment, review, integration |
| **mimi-engineer** (opus) | mimi-rs changes | qlinear.rs, transformer.rs, mimi.rs |
| **tts-engineer** (opus) | tts-web changes | mlp.rs, flow_lm.rs, tts_model.rs, lib.rs |
| **build-runner** (sonnet) | Build & test | cargo check, wasm-pack build, verify in browser |

### Step 1: `mimi-rs` — QLinear + transformer changes (mimi-engineer)

**New file: `mimi-rs/src/qlinear.rs`**

`QLinear` struct = `QMatMul` + optional bias `Tensor`. Methods:
- `from_linear_f32(Linear)` — wraps weight as `QMatMul::Tensor` (passthrough, no quantization)
- `quantize_in_place(&mut self, GgmlDType)` — converts weight to Q8_0 QTensor in-place via `QTensor::quantize()`
- `impl Module` — delegates to `QMatMul::forward()` + bias broadcast_add

Expose via `pub mod qlinear` in `mimi-rs/src/lib.rs`.

**`mimi-rs/src/transformer.rs`** — Linear -> QLinear

8 fields across 4 structs change type:
- `MimiStreamingMHA`: `in_proj`, `out_proj`
- `StreamingMultiheadAttention`: `in_proj`, `out_proj`
- `StreamingTransformerLayer`: `linear1`, `linear2`
- `ProjectedTransformer`: `input_proj`, `output_projs`

Each `load()` wraps with `QLinear::from_linear_f32()`. Forward methods unchanged (QLinear implements Module).

Add `quantize_weights(&mut self, GgmlDType)` to `StreamingTransformerLayer`, `StreamingTransformer`, `ProjectedTransformer`.

**`mimi-rs/src/mimi.rs`** — expose quantize methods

Add to `MimiModel`:
- `quantize_decoder_transformer(&mut self, GgmlDType)`
- `quantize_encoder_transformer(&mut self, GgmlDType)`

### Step 2: `tts-core` changes (tts-engineer)

**`tts-core/src/mlp.rs`** — Linear -> QLinear

11 fields across 4 structs:
- `TimestepEmbedder`: `linear1`, `linear2`
- `ResBlock`: `mlp_linear1`, `mlp_linear2`, `ada_ln_silu_linear`
- `FinalLayer`: `linear`, `ada_ln_silu_linear`
- `SimpleMLPAdaLN`: `cond_embed`, `input_proj`

Same pattern: wrap in `load()`, add `quantize_weights()`.

**`tts-core/src/flow_lm.rs`** — Linear -> QLinear

2 fields: `input_linear`, `out_eos`. Add `quantize_weights()` that delegates to `self.transformer`, `self.flow_net`.

**`tts-core/src/tts_model.rs`** — top-level quantize

Add `TTSModel::quantize_weights(&mut self)` calling `flow_lm.quantize_weights()` + `mimi.quantize_decoder_transformer()` + `mimi.quantize_encoder_transformer()`.

### Step 3: `tts-wasm` integration (tts-engineer)

**`tts-wasm/src/lib.rs`** — one line

After `TTSModel::load(vb, &cfg)?`, call `inner.quantize_weights()?`.

### Step 4: Build & verify (build-runner)

1. `cargo check -p tts-wasm`
2. `wasm-pack build crates/tts-wasm --target web --release`
3. Deploy to gh-pages, test on mobile

## What stays F32

- LayerNorm parameters (tiny, not matmul)
- Embedding tables (LUTConditioner — index_select, not matmul)
- Biases (tiny, additive)
- SEANet conv/deconv weights (audio-critical, directly shape waveform)
- All activations (F32, only weight storage changes)

## Memory estimate

| | F32 (current) | Q8_0 (proposed) |
|---|---|---|
| Quantized params (~97M) | 388 MB | 103 MB |
| Non-quantized params (~21M) | 84 MB | 84 MB |
| **Total weights** | **472 MB** | **187 MB** |
| Load-time peak | 472 MB | ~613 MB (transient) |

## Verification

1. `cargo check -p tts-wasm` — compiles
2. `wasm-pack build crates/tts-wasm --target web --release` — WASM builds
3. `node web/serve.mjs` — model loads, voice plays, audio sounds correct
4. Compare output quality before/after on desktop (should be near-identical)
5. Benchmark on mobile — target >1x realtime
