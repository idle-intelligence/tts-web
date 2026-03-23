# KittenTTS Architecture

StyleTTS 2 distilled model, Apache 2.0 license. Two variants: nano (14M params, F32) and mini (73M params, INT8 quantized). Both are ONNX2 format. Sample rate: 24kHz.

## Model Variants

| Property | Nano | Mini |
|---|---|---|
| Parameters | 14M | 73M |
| Dtype | F32 | INT8 (dynamic quantization) |
| LSTM layers | 2 | 3 |
| Hidden dim | 128 | 512 |
| Decoder channels | 256→128→64 | 1024→512 |

## Architecture Overview

The pipeline is: phoneme IDs + style embedding → BERT/ALBERT encoder → text encoder → duration/F0 predictor → HiFi-GAN vocoder → waveform.

## Components

### BERT/ALBERT Encoder

- 178-token phoneme vocabulary
- 128-dim token embedding → 768-dim hidden
- Single shared ALBERT layer group (parameter sharing across layers)
- LayerNormalization (29 instances) + Softmax attention

### Text Encoder

- 2-layer BiLSTM (nano) / 3-layer BiLSTM (mini)
  - Nano: hidden=64, total output=128
  - Mini: hidden=256, total output=512
- 128-channel CNN blocks (nano) / 512-channel (mini)

### Duration and F0 Predictor

- Conv + AdaIN blocks with InstanceNorm
- Channel tapering: 128→64 (nano)
- Style conditioning via AdaIN: takes 256-dim style input, projects to 128-dim internal style_dim
- AdaIN = InstanceNorm + style-projected scale/bias (`norm.fc` Linear projections)

### Decoder / Vocoder (HiFi-GAN with Harmonic Source)

Sinusoidal harmonic source generator (`l_sin_gen`) produces 11 harmonics as the excitation signal, then refines through convolutional upsampling stages.

**Encode stage**: 256-channel convolutions (nano) / 1024-channel (mini).

**Upsampling** (nano, total 60× factor):
- ConvTranspose1d stride=10: 256→128 channels
- ConvTranspose1d stride=6: 128→64 channels

**ResBlocks**: kernels [3, 7, 11] with dilations, interleaved with noise injection ResBlocks.

**Output projection**: 64→22 channels (22 harmonics), reduce to 1-channel waveform.

Activation: LeakyReLU throughout (not Snake like TADA). InstanceNormalization (57 instances in nano) + AdaIN style conditioning in conv blocks. Final waveform clipped with Tanh.

## Inputs and Outputs

| Tensor | Type | Shape | Description |
|---|---|---|---|
| `input_ids` | int64 | [1, seq_len] | Phoneme token IDs, boundary tokens: `[0, ...tokens..., 10, 0]` |
| `style` | float32 | [1, 256] | Voice embedding, selected by text length from voices.npz |
| `speed` | float32 | [1] | Speed multiplier (scaled by per-voice `speed_prior` from config) |
| `waveform` (out) | float32 | [num_samples] | 24kHz audio — trim last 5000 samples |
| `duration` (out) | int64 | — | Predicted token durations |

## Voices

8 voices stored in `voices.npz`. Each voice is a [400, 256] float32 matrix.

Style embedding selection: `style[min(len(text), 399)]` — the embedding is length-dependent, encoding different prosody styles by utterance length.

| Friendly Name | Internal Name |
|---|---|
| Bella | expr-voice-2-f |
| Jasper | expr-voice-2-m |
| Luna | expr-voice-3-f |
| Bruno | expr-voice-3-m |
| Rosie | expr-voice-4-f |
| Hugo | expr-voice-4-m |
| Kiki | expr-voice-5-f |
| Leo | expr-voice-5-m |

## Normalization Summary

- **InstanceNormalization**: 57 instances (nano) — conv blocks in decoder and predictor
- **LayerNormalization**: 29 instances — ALBERT encoder layers
- **AdaIN**: InstanceNorm + style-projected scale/bias — style conditioning throughout

## Activations

- **LeakyReLU**: primary non-linearity in all conv blocks
- **Tanh**: LSTM gates + final waveform clipping
- **Softmax**: ALBERT attention

## Mini Quantization

Mini uses ONNX dynamic quantization:
- `ConvInteger` — quantized convolutions
- `MatMulInteger` — quantized matrix multiplications
- `DynamicQuantizeLSTM` — quantized LSTM layers

## Differences from TADA Decoder

| Property | KittenTTS | TADA |
|---|---|---|
| Activation | LeakyReLU | Snake1d |
| Vocoder | HiFi-GAN + harmonic source | DAC codec decoder |
| Style conditioning | InstanceNorm + AdaIN (extensive) | None |
| Architecture source | StyleTTS 2 distilled | Llama 3.2 + VibeVoice diffusion head |

These are different enough architecturally that there is minimal shared code opportunity — keep them separate.
