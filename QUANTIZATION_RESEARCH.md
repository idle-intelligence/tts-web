# INT8 Scale Factor Dtype: BF16 vs F32

Research notes on whether per-channel INT8 quantization scale factors can be
stored and used in BF16 instead of F32.

## TL;DR

BF16 scales are fine. The industry standard for INT8 quantization uses fp16
or bf16 scales. The 0.3 dB SQNR loss vs f32 scales is negligible compared to
the INT8 quantization error itself.

## Industry Practice

### llama.cpp / GGUF — fp16 scales

The Q8_0 format stores one **fp16** scale per 32-element block:

```c
typedef struct {
    ggml_half d;        // scale -- float16
    int8_t qs[QK8_0];  // quants (32 elements)
} block_q8_0;
```

Q8_0 is described as "essentially lossless" with ~0.01 perplexity increase
vs the fp16 baseline. All K-quant formats (Q4_K, Q5_K, Q6_K) also use fp16
scales.

Source: [llama.cpp ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h),
[GGUF quantization types (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/7.3-distributed-inference-and-rpc)

### GPTQ / AutoGPTQ — fp16 scales

GPTQ checkpoints store `scales` and `qzeros` tensors in **fp16** in the
safetensors files. Inference kernels (Marlin, ExLlama) perform int4/int8 × fp16
GEMM with dequantization fused into the kernel. The multiply happens in fp16,
dot-product accumulation in fp32.

Source: [GPTQ Checkpoint Format (Daniel de Kok)](https://danieldk.eu/GPTQ-Checkpoint-Format)

### AWQ — fp16 scales

AWQ stores per-group scales in **fp16** for symmetric quantization. The
documentation notes: "the (average) size of quantization metadata per group
is dominated by the quantization scale size which can be a half precision
floating point number (16 bits)."

Source: [AWQ paper (arxiv:2306.00978)](https://arxiv.org/abs/2306.00978)

### TensorRT-LLM — fp16 or bf16

Dequantization formula: `x = static_cast<FP>(q) * s` where FP matches the
compute dtype (fp16 or bf16). Scales are stored in the model's native dtype.

Source: [TensorRT-LLM Numerical Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)

### compressed-tensors / vLLM — bf16 scales

FP8 `weight_scale` and `input_scale` tensors are stored as **BF16** in the
safetensors files.

Source: [HuggingFace compressed-tensors docs](https://huggingface.co/docs/transformers/quantization/compressed_tensors)

### bitsandbytes — f32 (the outlier)

bitsandbytes LLM.int8() stores absmax scaling factors as **f32** internally.
Optional "double quantization" compresses these f32 values into 8-bit floats
to save memory, implicitly acknowledging f32 is overkill.

Source: [bitsandbytes functional.py](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py),
[HuggingFace bitsandbytes blog](https://huggingface.co/blog/hf-bitsandbytes-integration)

### PyTorch native — f32/f64 (reference implementation)

`torch.quantize_per_channel()` uses f64 scales by default. Conservative
choice for a reference implementation, not representative of production
inference.

Source: [PyTorch torch.quantize_per_channel](https://docs.pytorch.org/docs/stable/generated/torch.quantize_per_channel.html)

## Why BF16 Scales Work

**BF16 has 7 mantissa bits** (~0.78% relative precision). The INT8
quantization grid has 254 levels, giving ~0.39% relative error per element.
BF16 scale error is on the same order — it roughly doubles the noise in the
worst case, but in practice the compounding is sublinear because scale error
is systematic (same direction across all elements in a channel).

**BF16 has 8 exponent bits** — same range as f32 (±3.4×10³⁸). No risk of
overflow or underflow for scale values. This makes BF16 strictly better than
fp16 (5 exponent bits, max ~65504) for scale storage, even though fp16 has
3 more mantissa bits.

**INT8 values up to 127 are exactly representable in BF16** (bf16 can
represent integers up to 256 exactly). So `bf16(i8_value)` is lossless, and
the only rounding happens in the single multiply `bf16_int × bf16_scale`.

## Empirical Measurement

Tested on `flow_lm.transformer.layers.0.self_attn.in_proj.weight` [3072, 1024]
from `kyutai/pocket-tts-without-voice-cloning`:

| Configuration | Max error | RMSE | SQNR |
|---|---|---|---|
| f32 scale, f32 multiply | 0.002168 | 0.000275 | 41.6 dB |
| bf16 scale, bf16 multiply | 0.002197 | 0.000286 | 41.3 dB |
| bf16 scale, f32 multiply | 0.002319 | 0.000280 | 41.5 dB |

**0.3 dB difference** between all-f32 and all-bf16. The bf16 scales have
0.14% mean relative error vs f32 scales. Both paths are well above the 30 dB
safety threshold.

## References

- "Give Me BF16 or Give Me Death" — Shumailov et al., ACL 2025.
  [arxiv:2411.02355](https://arxiv.org/abs/2411.02355). Found INT8 (W8A8-INT)
  achieves 1-3% accuracy degradation vs BF16 baseline; FP8 is "effectively
  lossless."
- PyTorch Int8-Mixed-BF16 RFC — validates mixing int8 quantized ops with bf16
  compute on BERT, DistilBERT, Stable Diffusion.
  [GitHub issue #111640](https://github.com/pytorch/pytorch/issues/111640)
- AWQ: Activation-aware Weight Quantization — Lin et al., 2023.
  [arxiv:2306.00978](https://arxiv.org/abs/2306.00978)
