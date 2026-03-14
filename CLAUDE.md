# CLAUDE.md — tts-web project conventions

## Workflow
- **Use sub-agents and teams**: The lead (opus) keeps context and orchestrates. All research, code reading, and code writing is delegated to sub-agents (sonnet). For multiple independent tasks, use teams of sonnet agents in parallel. Never do sequential manual edits from the lead.
- **Commit early and often**: Don't let large amounts of work pile up untracked. Commit atomically — small, focused commits, one logical change per commit.
- **Log EVERY inference run**: Every time audio is generated, log it in TWO places: (1) `docs/run_log.md` — scientific-style entry with full parameters (commit, model file, seed, noise_temp, transition_steps, voice, text, output file) and observations, (2) `docs/benchmarks.md` — results row (timing, audio duration, RTF, quality). No exceptions. Unlogged runs are useless.
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
    --transition-steps 0 \
    --text "The quick brown fox jumps over the lazy dog."
  ```
  Notes:
  - `--noise-temp 0.9` is a critical parameter (reference default); omitting it significantly degrades output quality.
  - `--transition-steps 0` is recommended for the current ljspeech voice prompt (only 5 acoustic tokens). With transition_steps=5 (default), all tokens are trimmed → zero-shot mode. With 0, all 5 tokens are used for voice conditioning.
- WASM build: `wasm-pack build crates/tada-wasm --target web --release`
- Quantize model: `python scripts/quantize_tada.py [--format mixed|q4_0]`
- Precompute voice: `python scripts/precompute_voice.py --audio <wav> --text <transcript> --output voices/<name>.safetensors`

## TADA Architecture (verified findings)

### Config (verified against checkpoint)
- `num_time_classes: 256` (NOT 1024 — Python class default is 1024 but checkpoint overrides to 256)
- `num_time_bits: 8`, `time_dim: 16`, `total_latent_dim: 528` (512 acoustic + 16 gray-code bits)
- `head_layers: 6`, `head_ffn_ratio: 4.0` (NOT Python class defaults of 4 and 3.0)
- All Rust config values in `TadaConfig::tada_1b()` verified exact match against HF `config.json`

### Pipeline differences: Python vs Rust
- **CFG scale**: Python default `acoustic_cfg_scale=1.6`, Rust uses 1.0 (no CFG). Rust audio sounds cleaner/less grainy — user prefers no-CFG quality. CFG confirmed as the cause of Python reference's less-clean endings (tested: Python without CFG also has trailing noise).
- **`times_before` collection**: Python collects the *input* fed to each step (from previous step's prediction). Rust collects the *prediction* from each step. This difference is intentional and self-consistent — investigated as a suspect for trailing noise, but the fix made things worse and was reverted. The collection difference is not the root cause.
- **Autoregressive time feedback bug**: Fixed. Was a known source of timing errors in earlier versions.
- **EOS handling**: Python runs flow matching on ALL steps including EOT frames. Rust skips flow matching during EOS countdown.
- **`num_transition_steps`**: Python trims last N voice prompt acoustic frames for smooth voice→text transition. Our voice prompt only has 5 frames so transition_steps=5 removes all conditioning → zero-shot. Use transition_steps=0 to retain all 5 tokens.

### Known issues
- **"call" phrase truncated early**: "I had to call you up in the middle of the night" is consistently truncated in both Q4_0 and F32. Model produces very short durations (1,1,1) for middle words — model behavior, not a Rust bug.
- **"tyger" phrase wrong pronunciation**: Starts with "tiiigger" and non-English sounds in both Q4_0 and F32. Model issue, not quantization.
- **Voice changes mid-utterance on long phrases**: Voice conditioning only has 5 tokens (too short for stable voice cloning), causing voice drift on longer inputs (e.g., wutang).
- **Voice prompt too short**: `voices/ljspeech.safetensors` has only 5 acoustic tokens. With `transition_steps=0`, all 5 are used but voice conditioning is still minimal. With `transition_steps=5` (default), all 5 are trimmed → effectively zero-shot. Need a longer voice prompt for real voice cloning.
- **NO_EOS on all outputs**: Model never produces true EOS (128001), always hits token budget via EOT (128009).

### Previously known issues (fixed or documented)
- **"dzouib" trailing noise**: FIXED by autoregressive time feedback bug fix + trailing EOT frame trim.
- **Quantization finding (documented)**: Q8_0 embeddings cause gray-code mispredictions. Q4_0 embeddings work fine. Variant E (VV F16 + Embed Q4_0, 1.75 GB) matches baseline. Still a valid finding for future quantization work.

### Gray code time prediction
- Flow matching outputs 528-dim vector: 512 acoustic features + 16 gray-code bits (8 for time_before, 8 for time_after)
- `time_before` drives `expand_durations()` — determines how many frames each acoustic token spans
- `time_after` is fed back as autoregressive input for next step only
- A single gray-code bit flip can cause large integer changes (e.g., 4→59)
- Anomalous time_before values (59, 196) cause decoder to insert many zero frames → noise
