# KittenTTS Implementation Log

Each iteration generates audio in `samples/` with the naming convention `iter{N}-{voice}-{text_slug}.wav`.

| Iter | Commit | Change | Audio | Status |
|------|--------|--------|-------|--------|
| 01 | 2ad223f | First working e2e (broken decoder, noise) | iter01-jasper-hello.wav | NOISE — RMS 0.59, values hit tanh saturation |
| 02 | 341d12c | Pool ConvTranspose stride=2 for T→2T upsampling | iter02-jasper-hello.wav | NOISE — better length (24985 vs 8185 samples) |
| 03 | 83d36c3 | AdaIN: layer norm axis + gamma=fc+1 | iter03-jasper-hello.wav | NOISE — durations now match ONNX (sum=50 vs 51) |
| 04 | 2325c82 | Decoder AdaIN gamma+1, remove debug prints | - | NOISE |
| 05 | bcdbeb7 | Add tanh clamp to waveform output | - | NOISE — output clamped to [-1,1] but content still wrong |
| 06 | 957ef7e | STFT polar form, double-sin inverse, parallel resblocks, 1/√2 | - | NOISE |
| 07 | ea7965f | CNN path uses own embedding (not BERT output) | - | NOISE — CNN corr improved 0.37→0.87 |
| 08 | 91a4184 | CNN LayerNorm (not InstanceNorm), decoder style[:,:128] | - | NOISE — decoder injection test passes |
| 09 | 100692d | Predictor √2, Block1 rewrite, shared norm, both decoder args = expanded_cnn | iter09-jasper-hello.wav | NOISE — decode blocks now match ONNX ~5%, RMS still 0.65 |
| ONNX ref | — | ONNX reference (correct audio) | kitten-reference-hello.wav | GOOD — RMS 0.106, range [-0.38, 0.54] |

## Reference metrics (ONNX)
- "Hello world" Jasper: 25600 samples, RMS=0.106, range=[-0.39, 0.54], no tanh saturation
- "Fox" Bella: ~95800 samples, RMS~0.10

## Current metrics (iter09)
- "Hello world" Jasper: 24995 samples, RMS=0.647, range=[-1.0, 1.0], tanh-saturated
- Correlation vs ONNX: 0.145
