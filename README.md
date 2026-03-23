# tts-web

Browser-native text-to-speech running 100% client-side via Rust/WASM.

[**Try the demo →**](https://idle-intelligence.github.io/tts-web/web/)

## Models

| Model | Size | Params | Architecture | License |
|-------|------|--------|-------------|---------|
| [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) | ~130MB (Q8_0) | ~97M | Autoregressive + Mimi codec | MIT |
| [KittenTTS](https://github.com/KittenML/KittenTTS) | ~56MB (F32) | 14M | StyleTTS 2 distilled, single forward pass | Apache 2.0 |

Weights are on HuggingFace: [Pocket TTS GGUF](https://huggingface.co/idle-intelligence/pocket-tts-gguf), [KittenTTS safetensors](https://huggingface.co/idle-intelligence/kitten-tts-nano-safetensors).

## Quick Start — KittenTTS CLI

Generate speech from text with zero system dependencies:

```bash
# Clone
git clone https://github.com/idle-intelligence/tts-web.git
cd tts-web

# Download model weights (~60MB)
hf download idle-intelligence/kitten-tts-nano-safetensors --local-dir models/kitten-nano

# Build (one-time)
cargo build --example kitten_generate -p kitten-core --release --features espeak

# Generate speech — no system dependencies needed
./target/release/examples/kitten_generate \
  --model models/kitten-nano/kitten-nano.safetensors \
  --voices models/kitten-nano/kitten-voices.safetensors \
  --voice jasper \
  --text "Hello, this is a test of the text-to-speech system." \
  --output hello.wav
```

The `--features espeak` flag bundles a pure-Rust espeak-ng port with English data, so text → IPA phonemization works out of the box. The built binary at `target/release/examples/kitten_generate` is standalone — use it directly without `cargo run`.

8 built-in voices: bella, jasper, luna, bruno, rosie, hugo, kiki, leo.

### Without the espeak feature

If you prefer not to pull the GPL espeak-ng crate, you can use system espeak-ng or pass IPA directly:

```bash
# Option A: system espeak-ng (brew install espeak-ng)
cargo run --example kitten_generate -p kitten-core --release -- --text "Hello world"

# Option B: pass IPA directly
cargo run --example kitten_generate -p kitten-core --release -- --ipa "həlˈəʊ wˈɜːld"
```

### Using the original KittenTTS ONNX weights

If you prefer to convert the weights yourself instead of using the pre-converted safetensors from HuggingFace:

```bash
# Download the original ONNX model from KittenML
hf download KittenML/KittenTTS-nano --local-dir models/kitten-nano

# Convert ONNX → safetensors (requires: pip install onnx safetensors numpy)
python scripts/kitten/convert_kitten_to_safetensors.py models/kitten-nano
```

This produces `kitten-nano.safetensors` and `kitten-voices.safetensors` in the same directory.

## Quick Start — Browser Demo

```bash
# Build KittenTTS WASM
wasm-pack build crates/kitten-wasm --target web --release -- --features wasm --no-default-features

# Start dev server
node web/serve.mjs --port 8082
```

Open http://localhost:8082/web/, select KittenTTS, click a voice.

## Architecture

```
crates/
  kitten-core/     # Pure candle inference (BERT → text encoder → predictor → decoder)
  kitten-wasm/     # WASM bindings (2.8MB binary)
  tts-core/        # Pocket TTS inference
  tts-wasm/        # Pocket TTS WASM bindings

web/
  index.html       # Shared demo UI (model selector)
  kitten-worker.js # KittenTTS Web Worker
  worker.js        # Pocket TTS Web Worker
  tts-client.js    # Shared client class
```

- **KittenTTS**: Single forward pass (no autoregressive loop). Text → espeak IPA → phoneme IDs → model → 24kHz audio. 0.24 RTF native, 1.81x realtime WASM.
- **Pocket TTS**: Autoregressive with Mimi codec decoder. Streams audio chunks for real-time playback.
- **[mimi-rs](https://github.com/idle-intelligence/mimi-rs)**: Shared Mimi audio codec library.

## Performance (WASM, Chrome on M-series Mac)

| Model | Audio | Wall time | Speed |
|-------|-------|-----------|-------|
| Pocket TTS | 3.28s | 1.64s (TTFB 0.41s) | 2.28x realtime |
| KittenTTS | 3.57s | 1.97s | 1.81x realtime |

Speed = audio duration / generation time. Higher = faster.
