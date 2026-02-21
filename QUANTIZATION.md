# Weight-Only INT8 Quantization for Pocket-TTS

Weight-only INT8 quantization that reduces the safetensors file size by ~41%
from the BF16 original while keeping all inference in F32. Only **storage** is
quantized — at model load the Rust/WASM runtime dequantizes INT8 weights back
to BF16, producing a buffer identical in dtype to the original model. The VB
loader then promotes BF16 to F32 for compute as usual.

## Pipeline Overview

```
 HuggingFace safetensors (BF16, ~225 MB)
          │
          ▼
   quantize.py          ← offline, runs once
          │
          ▼
 quantized safetensors   (INT8 weights + BF16 scales + BF16 skip-layers, ~133 MB)
          │
          ▼
 dequantize.rs           ← at model load in WASM, transparent to model code
          │
          ▼
 safetensors (all BF16)  ← VB loader promotes BF16 → F32 for compute
```

## Which Layers Are Quantized

### Quantized (INT8 with per-channel BF16 scale)

| Component | Tensor pattern | Shape per instance | Count | Why safe to quantize |
|---|---|---|---|---|
| FlowLM transformer self-attention | `*.self_attn.in_proj.weight` | [3072, 1024] | 6 | Large matmul weight; errors average out over 16 heads |
| FlowLM transformer self-attention | `*.self_attn.out_proj.weight` | [1024, 1024] | 6 | Output projection, well-conditioned |
| FlowLM transformer FFN | `*.linear1.weight` | [4096, 1024] | 6 | Expansion layer; GELU activation smooths small errors |
| FlowLM transformer FFN | `*.linear2.weight` | [1024, 4096] | 6 | Contraction layer |
| FlowLM input projection | `input_linear.weight` | [1024, 32] | 1 | Small but 2-D; projects latent to model dim |
| FlowLM EOS head | `out_eos.weight` | [1, 1024] | 1 | Binary threshold decision, tolerant |
| FlowLM speaker projection | `speaker_proj_weight` | [1024, 512] | 1 | Linear projection of speaker embedding |
| Flow net (SimpleMLPAdaLN) linears | `flow_net.*.weight` | various 2-D | ~20 | MLP weights; residual connections and adaptive LN limit error propagation |
| Mimi encoder/decoder transformer | `*_transformer.transformer.layers.*.weight` | various 2-D | 8 | Standard transformer layers, same reasoning as FlowLM |

These represent ~82% of total parameters and account for all of the file size
reduction.

### Kept in BF16

| Component | Tensor pattern | Why |
|---|---|---|
| **Embedding table** | `*.embed.weight` [4001, 1024] | Integer lookup — no accumulation to average out error. Each token's vector is used directly as transformer input; distortion here propagates through all 6 layers without being dampened. Only ~4M params so savings are minimal. |
| **LayerNorm / RMSNorm** | `*.norm*.weight`, `*.norm*.bias`, `*.alpha` | 1-D vectors (tiny). Normalization parameters are scale/shift applied element-wise — even small errors change the distribution seen by downstream layers. |
| **LayerScale** | `*.layer_scale_*.scale` | 1-D per-channel scaling in Mimi transformer residuals. Initialized to 0.01; quantization to INT8 would collapse most values to zero. |
| **All bias vectors** | `*.bias` | 1-D additive terms, negligible size. |
| **Mimi SEANet decoder convolutions** | `mimi.decoder.model.{N}.*` | These ConvTranspose1d / Conv1d layers directly synthesize the audio waveform. Even small weight perturbations produce audible clicks, tonal distortion, and high-frequency artifacts. The kernels are small (3–12 taps) so per-channel quantization has very few elements to average over, making the error especially noticeable. |
| **Mimi SEANet encoder convolutions** | `mimi.encoder.model.{N}.*` | Used for voice-conditioning encoding. Same sensitivity as decoder convolutions. Also relatively small. |
| **Mimi quantizer output projection** | `quantizer.output_proj.*` | Conv1d kernel_size=1 that bridges the 32-d latent bottleneck to 512-d. Sits at the narrowest point of the information path — errors here are amplified by the decoder. Shape [512, 32, 1] is small. |
| **Resampling convolutions** | `downsample.*`, `upsample.*` | Frame-rate conversion layers. Depthwise with groups=dimension; very small, and errors cause temporal aliasing artifacts. |
| **Small buffers** | `emb_mean`, `emb_std`, `bos_emb`, `*.freqs` | 1-D vectors of 32–128 elements. `emb_mean`/`emb_std` are latent normalization statistics — must be exact. `freqs` are precomputed sinusoidal frequencies for timestep embeddings. |

## Quantization Method

**Per-channel symmetric INT8:**

For a weight tensor with shape `[out_channels, ...]`:

```
scale[c]     = max(|weight[c, ...]|) / 127
quantized[c] = round(weight[c, ...] / scale[c])   clamped to [-127, 127]
```

Stored in safetensors as:
- Original tensor name, dtype `I8` (the quantized weights)
- `{name}_scale`, dtype `BF16`, shape `[out_channels]`

Dequantization (Rust, at load time):
```
bf16_weight[c, ...] = bf16(i8_weight[c, ...]) * bf16_scale[c]
```

Everything stays in BF16 — the dequantized buffer is identical in dtype to
the original non-quantized model. BF16 scales are standard practice (llama.cpp
Q8_0 uses fp16 scales, GPTQ/TensorRT-LLM use fp16/bf16). The ~0.3 dB SQNR
difference vs f32 scales is negligible compared to the INT8 quantization
error itself.

Using the output-channel axis for per-channel scaling gives each row of the
weight matrix its own dynamic range, which is critical for transformer
attention projections where different heads can have very different magnitude
distributions.

## Measured Size Reduction

For `kyutai/pocket-tts-without-voice-cloning` (`tts_b6369a24.safetensors`):

| | Parameters | Bytes |
|---|---|---|
| Original (all BF16) | ~118M | 225 MB |
| Quantized weights (INT8) | ~97M | ~97 MB |
| Scale factors (BF16) | ~0.1M | ~0.2 MB |
| Kept BF16 layers | ~21M | ~42 MB |
| **Quantized total** | | **133 MB** |
| **Reduction** | | **~41%** |

## Measured Quantization Quality

All quantized layers achieve SQNR > 37 dB (well above the 30 dB safety
threshold). Worst real layer: `flow_lm.transformer.layers.5.linear2.weight`
at 37.2 dB. Simulated matmul errors are negligible.

Note: the encoder transformer layers in the `without-voice-cloning` model
variant report `-inf` SQNR because they were zeroed out when voice cloning
was removed. This is expected and harmless.

## Gotchas Specific to CALM / Flow Matching

### 1. Flow net sensitivity to scale

The `SimpleMLPAdaLN` flow network is conditioned on the transformer output
**and** two timestep embeddings (s, t). The adaptive LayerNorm modulation
(`adaLN_modulation`) produces shift/scale/gate triples that directly multiply
activations. Small errors in these weights can compound across the 6 residual
blocks. In practice per-channel INT8 is fine here because:
- The residual connections limit error accumulation
- The AdaLN gate values are bounded by SiLU activation
- The flow is evaluated for only 1 LSD step in production (not iterated many times)

If you increase `lsd_decode_steps` beyond 1, quantization error in the flow
net accumulates with each step. Monitor SQNR of `flow_net.res_blocks.*`
weights if using multi-step decoding.

### 2. EOS detection threshold

The `out_eos` linear layer produces a single logit compared against
`eos_threshold = -4.0`. Since this is a hard threshold on a scalar output,
quantization error in `out_eos.weight` could shift the logit by a fraction
and cause early/late EOS. In practice the 1x1024 weight is well within INT8
precision (SQNR > 40 dB typically), but if you observe EOS timing issues,
this layer can be moved to the skip list.

### 3. BOS embedding and latent normalization

`bos_emb`, `emb_mean`, and `emb_std` are kept in their original dtype. These
are small 1-D vectors but they're used in very sensitive positions:
- `bos_emb` replaces NaN markers at the start of generation — it seeds the
  entire autoregressive loop
- `emb_mean` / `emb_std` denormalize every generated latent before Mimi
  decoding — errors here shift/scale the entire latent distribution

### 4. Mimi decoder is the audio bottleneck

The Mimi VAE decoder (SEANet architecture) converts latents to raw PCM audio.
Its convolutional layers operate at progressively higher temporal resolution
(from 200 Hz latent rate up to 24 kHz audio). The transposed convolutions
that perform upsampling are especially sensitive:
- Kernel sizes are 2x the stride (e.g., kernel=12 for stride=6)
- Each output sample depends on very few kernel weights
- Quantization noise in these weights becomes directly audible as periodic
  tonal artifacts at the upsampling ratio frequencies

This is why all `mimi.{encoder,decoder}.model.*` convolutions stay BF16.
The Mimi **transformer** layers (at `mimi.*_transformer.transformer.*`) are
safe to quantize since they operate in the latent space, not on raw audio.

### 5. Streaming KV cache is unaffected

The KV cache states loaded for voice conditioning (`add_voice()`) are always
F32. The quantization only affects weight tensors — activations and cached
key/value projections remain in full precision. This means voice quality is
not affected by switching to a quantized model file.

## Usage

### Python: Create quantized weights

```bash
pip install torch safetensors huggingface_hub numpy

# From HuggingFace
python quantize.py --model-id kyutai/pocket-tts-without-voice-cloning -o model_int8.safetensors --validate

# From local file
python quantize.py -i model.safetensors -o model_int8.safetensors --validate
```

### Rust/WASM: Load quantized weights

No code changes needed — the WASM port's `Model::new()` constructor
automatically detects and dequantizes INT8 tensors at load time via
`dequantize::dequantize_if_needed()`. Non-quantized safetensors files
continue to work unchanged.

```javascript
// Works with either original or quantized safetensors
const model = new Model(quantizedWeightsBuffer);
```
