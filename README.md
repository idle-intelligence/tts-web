# tts-web

Text-to-speech that runs entirely in the browser, with Rust compiled to WebAssembly. No server: the text never leaves the machine.

[**Try the demo**](https://idle-intelligence.github.io/tts-web/web/)

## Models

| Model | Size | Params | Architecture | License |
|-------|------|--------|-------------|---------|
| [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) | ~134MB per language (Q8_0) | ~97M | Autoregressive + Mimi codec | MIT (code), CC-BY-4.0 (weights) |
| [KittenTTS](https://github.com/KittenML/KittenTTS) | ~56MB (F32) | 14M | StyleTTS 2 distilled, single forward pass | Apache 2.0 |

Weights are on Hugging Face: [Pocket TTS GGUF](https://huggingface.co/idle-intelligence/pocket-tts-gguf), [KittenTTS safetensors](https://huggingface.co/idle-intelligence/kitten-tts-nano-safetensors).

## Languages

Pocket TTS speaks six languages. Each is a separate Q8_0 quantization of Kyutai's checkpoint for that language, from [kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) at revision `4e1e0a3`.

| Language | Default voice |
|----------|---------------|
| English | alba |
| French | estelle |
| German | juergen |
| Spanish | lola |
| Portuguese | rafael |
| Italian | giovanni |

A Pocket TTS voice is a KV cache computed by one specific model, so voices are always fetched from the same Kyutai revision as the weights. Voices from a different revision produce broken speech.

## Command line

Pick a language and a voice by name. The weights, tokenizer and voice are downloaded on first use and cached under `~/.cache/tts-web`.

```bash
git clone https://github.com/idle-intelligence/tts-web.git
cd tts-web

# Italian, voice giovanni
cargo run --example tts_generate -p tts-core --release -- \
  --language italian --voice giovanni \
  --text "Ciao, questo è un test." \
  --output italian.wav

# Every language and voice available
cargo run --example tts_generate -p tts-core --release -- --list
```

`--model`, `--tokenizer` and `--voice` also accept local file paths, for offline use.

KittenTTS (English) works the same way, with 8 voices: bella, jasper, luna, bruno, rosie, hugo, kiki, leo. The model and voices (~56MB) are downloaded into `models/` on first run.

```bash
cargo run --example kitten_generate -p kitten-core --release --features espeak -- \
  --voice bruno --text "Hello, this is a test." --output bruno.wav
```

The `espeak` feature bundles a pure-Rust port of espeak-ng with English data, so text to phonemes works with no system dependency. The built binary, `target/release/examples/kitten_generate`, runs on its own.

### KittenTTS without the espeak feature

The espeak-ng port is GPL. To avoid it, use a system espeak-ng or pass IPA directly:

```bash
# System espeak-ng (brew install espeak-ng)
cargo run --example kitten_generate -p kitten-core --release -- --text "Hello world"

# IPA input
cargo run --example kitten_generate -p kitten-core --release -- --ipa "həlˈəʊ wˈɜːld"
```

### Converting the original KittenTTS ONNX weights

To convert the weights yourself instead of using the safetensors on Hugging Face:

```bash
hf download KittenML/KittenTTS-nano --local-dir models/kitten-nano
python scripts/kitten/convert_kitten_to_safetensors.py models/kitten-nano   # needs onnx, safetensors, numpy
```

This writes `kitten-nano.safetensors` and `kitten-voices.safetensors` next to the ONNX file.

## Browser demo

```bash
scripts/build.sh                     # builds both WASM packages and assembles the site
python3 scripts/serve.py --port 8082
```

Open http://localhost:8082/web/.

## Architecture

```
crates/
  kitten-core/     # KittenTTS inference (BERT, text encoder, predictor, decoder) on Candle
  kitten-wasm/     # KittenTTS WASM bindings
  tts-core/        # Pocket TTS inference
  tts-wasm/        # Pocket TTS WASM bindings

web/
  index.html       # Demo page (model and language selector)
  kitten-worker.js # KittenTTS Web Worker
  worker.js        # Pocket TTS Web Worker
  tts-client.js    # Shared client class
```

- **Pocket TTS**: autoregressive, with the Mimi codec decoder; audio is streamed in chunks for real-time playback.
- **KittenTTS**: a single forward pass. Text goes to espeak IPA, phoneme IDs, the model, then 24kHz audio.
- **[mimi-rs](https://github.com/idle-intelligence/mimi-rs)**: the shared Mimi audio codec library.

## Performance (M-series Mac)

| Model | Native RTF | WASM (Chrome) |
|-------|-----------|---------------|
| Pocket TTS | 0.23 | 2.28x realtime (first audio after 0.41s) |
| KittenTTS | 0.24 | 1.81x realtime |

RTF is generation time divided by audio duration (lower is faster). WASM speed is audio duration divided by wall time (higher is faster).

## Building from source

Dependencies come straight from git; no other checkout is needed:

- **[candle](https://github.com/ilnmtlbnm/candle)**, branch `wasm-simd-opt`: a fork with an optimized WASM SIMD128 quantized matmul, applied through `[patch.crates-io]` in `Cargo.toml`.
- **[mimi-rs](https://github.com/idle-intelligence/mimi-rs)**: the Mimi codec (encoder, decoder, streaming transformer).

The Rust toolchain is pinned in `rust-toolchain.toml`.
