# tts-web

Browser-native text-to-speech running 100% client-side via Rust/WASM.

[**Try the demo →**](https://idle-intelligence.github.io/tts-web/web/)

> **Disclaimer:** Experimental port. Model from [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai Labs.

## Requirements

- Modern browser with WebAssembly support

## Quick Start

```bash
# 1. Build WASM package
wasm-pack build crates/tts-wasm --target web

# 2. Start dev server
bun web/serve.mjs
```

## Architecture

- **TTS model** (`crates/tts-wasm/`): Pocket TTS compiled to WebAssembly via Candle. Generates speech from text tokens using a voice embedding.
- **[mimi-rs](https://github.com/idle-intelligence/mimi-rs)**: Shared Rust library for the Mimi audio codec (encoder + decoder + streaming transformer). Used by both tts-web and stt-web.
- **Web UI** (`web/`): Web Worker orchestrates model loading and generation, streams audio chunks back to the main thread for real-time playback.

## Quantization

The model ships as a [GGUF Q8\_0 file](https://huggingface.co/idle-intelligence/pocket-tts-gguf) (~130MB). Weights are loaded directly as Q8\_0 via candle's `QMatMul`, keeping ~97M quantized parameters in memory (~103MB vs ~388MB F32) and reducing memory bandwidth ~4x per inference step.
