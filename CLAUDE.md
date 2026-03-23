# CLAUDE.md — tts-web project conventions

## Overview

Multi-model browser TTS inference engine. Three models supported:
- **Pocket TTS** (Kyutai, 97M params) — autoregressive + Mimi codec, Q8_0 GGUF
- **KittenTTS** (StyleTTS 2 distilled, 14M params) — non-autoregressive, safetensors
- **TADA-1B** (HumeAI, 2.2B params) — Llama 3.2 + VibeVoice diffusion + DAC decoder, GGUF

Shared infrastructure: candle (Rust ML), mimi-rs (shared codec), WASM + WebGPU.

## Workflow
- **Use sub-agents and teams**: Lead (opus) orchestrates. All research, code reading, and code writing delegated to sub-agents (sonnet). Multiple independent tasks → parallel agents.
- **Commit early and often**: Atomic commits, one logical change per commit.
- **Log EVERY inference run**: (1) `docs/tada/lab-notebook.md` — experiment name, commit, params, user feedback, (2) `docs/tada/results.md` — data row (ID, Engine, Device, Model, Size, Text, Load, Gen, Decode, Audio, RTF, File). No exceptions.
- **Don't add Co-Authored-By for trivial commits**

## Development
- Always use venv for Python ("venv mon ami")
- Prefer candle over other Rust ML frameworks
- No premature optimization — get it working first
- Open audio files separately — one at a time, not all in one command
- Audio metrics (RMS, peak, flatness) are unreliable quality indicators — let the user listen
- Dev server: `node web/serve.mjs` (port 8081)
- Never open the user's personal browser for testing — use Playwright headless Chromium

## Project Structure
```
crates/
  tts-core/ + tts-wasm/         — Pocket TTS (Kyutai)
  kitten-core/ + kitten-wasm/   — KittenTTS
  tada-core/ + tada-wasm/       — TADA-1B (Burn/wgpu LLM + candle VV/decoder)
scripts/
  pocket-tts/                   — Kyutai quantization utilities
  kitten/                       — KittenTTS conversion, ONNX tools
  tada/                         — TADA quantization, reference gen, benchmarks
  shared/                       — cross-model analysis
docs/
  tada/                         — lab-notebook, results, quantization strategy
  kitten/                       — architecture, iteration log
voices/                         — TADA voice prompts (.safetensors + .json)
  matrix/                       — speaker × style grid (Expresso)
web/                            — frontend (HTML, JS, workers for all models)
patches/cubecl-wgpu-0.9.0/     — WebGPU workgroup size cap for TADA
```

## Shared Dependencies
- **mimi-rs**: `git = "https://github.com/idle-intelligence/mimi-rs.git"` — shared audio codec (encoder + decoder + streaming transformer + QLinear). Used by both Pocket TTS and TADA.
- **candle**: v0.9 — ML inference framework (CPU + Metal)
- **burn**: v0.20 — ML framework with WebGPU backend (TADA LLM only)

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

### TADA-1B
```bash
# Native (candle Metal)
cargo run --example tada_generate -p tada-core --release --features metal -- \
  --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-C-vvq8-eq4.gguf \
  --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json \
  --voice voices/matrix/ex04_whisper.safetensors \
  --noise-temp 0.9 --temperature 0.6 --transition-steps 5 --seed 42 \
  --cfg-scale 1.6 --flow-steps 10 --top-p 0.9 --repetition-penalty 1.1 \
  --text "The quick brown fox jumps over the lazy dog."

# WASM (Burn/wgpu + candle)
wasm-pack build crates/tada-wasm --target web --release -- --features wasm --no-default-features

# Quantize
python scripts/tada/quantize_tada.py --format mixed --llm-type q4_0 --vv-type q8_0 --embed-type q4_0

# Precompute voice
python scripts/tada/precompute_voice.py --audio clip.wav --text "transcript" --output voices/name.safetensors

# Python reference
python scripts/tada/tada_reference_generate.py --text "Hello" --voice voices/matrix/ex04_whisper.safetensors --output /tmp/test.wav
```

### Deployment
```bash
# Build both WASM packages, then copy to gh-pages branch
wasm-pack build crates/tts-wasm --target web --release
wasm-pack build crates/kitten-wasm --target web --release -- --features wasm
# gh-pages: pkg/ (Kyutai), kitten-pkg/ (Kitten), web/ (frontend)
```

## Local Paths
- Llama tokenizer: `/Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json`
- TADA models: `/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-*.gguf`
- TADA codec decoder: `/Users/tc/Code/idle-intelligence/hf/tada-codec/decoder/model.safetensors`

## TADA Key Parameters (matching Python reference)
- `noise_temp=0.9` (flow matching noise — critical, lower = flat audio)
- `temperature=0.6` (text token sampling)
- `cfg_scale=1.6` (classifier-free guidance — amplifies voice conditioning)
- `flow_steps=10` (ODE solver steps, reference default is 20)
- `top_p=0.9` (nucleus sampling)
- `repetition_penalty=1.1`
- `transition_steps=5` (trims last N voice tokens for smooth transition)

## TADA Architecture Notes
- CFG requires dual KV caches (pos + neg paths with independent histories)
- RoPE uses split-half convention (NOT interleaved pairs)
- WASM: `tasks_max=512` for GPU command batching, SIMD128 enabled for candle CPU
- VV on GPU (Q8_0 WGSL shader) available but currently slower than candle CPU due to dispatch overhead
- Voice prompts need 25+ acoustic tokens for stable conditioning (5 is too few)
