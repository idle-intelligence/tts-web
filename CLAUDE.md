# CLAUDE.md — tts-web project conventions

## Overview

Browser TTS inference engine. Two models supported:
- **Pocket TTS** (Kyutai, 97M params) — autoregressive + Mimi codec, Q8_0 GGUF
- **KittenTTS** (StyleTTS 2 distilled, 14M params) — non-autoregressive, safetensors

Shared infrastructure: candle (Rust ML), mimi-rs (shared codec), WASM.

## Workflow
- **Commit early and often**: Atomic commits, one logical change per commit.
- **Don't add Co-Authored-By for trivial commits**

## Development
- Always use venv for Python ("venv mon ami")
- Prefer candle over other Rust ML frameworks
- No premature optimization — get it working first
- Open audio files separately — one at a time, not all in one command
- Audio metrics (RMS, peak, flatness) are unreliable quality indicators — let the user listen
- Build + assemble the deployed site: `ENGINE_BUILD=<tag> scripts/build.sh` (bumps the `?v=` build tag on every loading URL)
- Dev server: `python3 scripts/serve.py` (serves `_site/`, port 8030 by default)
- Never open the user's personal browser for testing — use Playwright headless Chromium

## Project Structure
```
crates/
  tts-core/ + tts-wasm/         — Pocket TTS (Kyutai)
  kitten-core/ + kitten-wasm/   — KittenTTS
scripts/
  pocket-tts/                   — Kyutai quantization utilities
  kitten/                       — KittenTTS conversion, ONNX tools
  build.sh, serve.py            — build + local server
docs/
  kitten/                       — architecture, iteration log
web/                            — frontend (HTML, JS, workers for both models)
```

## Shared Dependencies
- **mimi-rs**: `git = "https://github.com/idle-intelligence/mimi-rs.git"` — shared audio codec (encoder + decoder + streaming transformer + QLinear). Used by Pocket TTS.
- **candle**: ML inference framework (CPU + Metal), patched fork for WASM SIMD.

## Build Commands

### Pocket TTS (Kyutai)
```bash
wasm-pack build crates/tts-wasm --target web --release
python scripts/pocket-tts/quantize_to_gguf.py  # safetensors → GGUF Q8_0
```

### KittenTTS
```bash
wasm-pack build crates/kitten-wasm --target web --release -- --features wasm
cargo run --example kitten_generate -p kitten-core --release -- --model model.safetensors --text "Hello"
```

### Deployment
```bash
ENGINE_BUILD=<tag> scripts/build.sh   # builds both wasm packages, assembles _site/, bumps the build tag
```
