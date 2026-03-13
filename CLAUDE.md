# CLAUDE.md — tts-web project conventions

## Workflow
- **Use sub-agents and teams**: The lead (opus) keeps context and orchestrates. All research, code reading, and code writing is delegated to sub-agents (sonnet). For multiple independent tasks, use teams of sonnet agents in parallel. Never do sequential manual edits from the lead.
- **Commit early and often**: Don't let large amounts of work pile up untracked. Commit atomically — small, focused commits, one logical change per commit.
- **Track benchmark runs**: Keep a research log of every test/benchmark run. Record parameters, results, and observations. Maintain a separate performance table (benchmarks table = data only, analysis goes in separate docs).
- **Don't add Co-Authored-By for trivial commits** (e.g. README updates, config changes)

## Development
- Always use venv for Python ("venv mon ami")
- Prefer candle over other Rust ML frameworks
- No premature optimization — get it working first
- Open audio files separately — one at a time, not all in one command
- Audio metrics (RMS, peak, flatness) are unreliable quality indicators — let the user listen
- Dev server: `node web/serve.mjs` (port 8081)

## Project Structure
- `crates/tada-core/` — TADA-1B model (config, llama, vibevoice, flow_matching, decoder)
- `crates/tada-wasm/` — WASM bindings for browser deployment
- `scripts/` — Python utilities (quantization, voice precompute, etc.)
- `voices/` — precomputed voice prompts (.safetensors + .json)
- `samples/` — generated audio samples (untracked, for local testing)
- `docs/` — benchmarks, run logs, architecture docs
- `web/` — frontend (HTML, JS, worker)

## Local Paths
- Llama tokenizer: `/Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json`
- TADA-1B model (BF16 source): `/Users/tc/Code/idle-intelligence/hf/tada-1b/model.safetensors`
- TADA-1B GGUF (Q4_0): `/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf`
- TADA-1B GGUF (mixed Q4+Q8): `/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-mixed.gguf`
- TADA codec decoder: `/Users/tc/Code/idle-intelligence/hf/tada-codec/decoder/model.safetensors`
- Voice prompt: `voices/ljspeech.safetensors` (repo-relative)

## Build Commands
- Native example:
  ```
  cargo run --example tada_generate -p tada-core --release [--features metal] -- \
    --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-mixed.gguf \
    --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json \
    --voice voices/ljspeech.safetensors \
    --noise-temp 0.9 \
    --text "The quick brown fox jumps over the lazy dog."
  ```
  Note: `--noise-temp 0.9` is a critical parameter (reference default); omitting it significantly degrades output quality.
- WASM build: `wasm-pack build crates/tada-wasm --target web --release`
- Quantize model: `python scripts/quantize_tada.py [--format mixed|q4_0]`
- Precompute voice: `python scripts/precompute_voice.py --audio <wav> --text <transcript> --output voices/<name>.safetensors`
