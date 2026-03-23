# Next Session Handoff — TADA-1B TTS

## What This Is

You are continuing work on a browser-based TTS engine for the TADA-1B model.
The stack is: Rust (candle ML framework) for inference + WASM bindings for browser
deployment. The frontend lives in `web/`.

Repository: `/Users/tc/Code/idle-intelligence/tts-web`
Branch: `feat/burn-wgpu-llama`

---

## State of the Pipeline (as of 2026-03-13)

The end-to-end native generation pipeline is working and producing good-quality audio.
Two major bugs were found and fixed this session:

### Bug 1: Autoregressive time feedback (FIXED)

The generation loop was feeding zeros for `time_before` / `time_after` at every
autoregressive step instead of the model's own predictions. The fix was to check
whether the *next* step's prompt_idx is still within the prompt phase (not the
current step's). The corrected logic is in the update block at the end of each
generation step in both:

- `crates/tada-core/examples/tada_generate.rs` (lines ~645–677)
- `crates/tada-wasm/src/lib.rs` (lines ~347–383)

### Bug 2: Trailing EOT frame (FIXED)

Voice-prompted mode produced a junk final acoustic frame — the frame that encodes
the EOT token itself rather than a shifted text token. Fixed by popping the last
acoustic frame after generation in both files above.

### Bug 3: WASM tokenize() used wrong EOT token (FIXED)

`tada-wasm/src/lib.rs` `tokenize()` was using `EOS_TOKEN_ID` (128001) for trailing
padding instead of `EOT_TOKEN_ID` (128009). Fixed. The trailing tokens must be
`128009` (`<|eot_id|>`) to match Python's reference exactly.

### Pipeline differences between Rust and Python that are intentional

- **No CFG in Rust**: Python uses `acoustic_cfg_scale=1.6`. Rust uses 1.0 (no CFG).
  User has evaluated both and *prefers* Rust's cleaner sound without CFG. Do NOT
  implement CFG unless explicitly asked.
- **`times_before` collection order**: Python collects the *input* fed to each step.
  Rust collects the *prediction* from each step. This is intentional and
  self-consistent; changing it made things worse and was reverted.
- **EOS handling**: Python runs flow matching on ALL steps including EOT frames.
  Rust skips flow matching during EOS countdown. This is deliberate.

---

## Current Configuration

All config values in `TadaConfig::tada_1b()` have been verified exact-match against
the HF `config.json`. Critical verified values:

- `num_time_classes: 256` (NOT 1024 — checkpoint overrides Python class default)
- `num_time_bits: 8`, `time_dim: 16`, `total_latent_dim: 528`
- `head_layers: 6`, `head_ffn_ratio: 4.0`
- `shift_acoustic: 5`

Do NOT change these without re-verifying against the checkpoint.

---

## Recommended Build Command

```bash
cargo run --example tada_generate -p tada-core --release --features metal -- \
  --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-mixed.gguf \
  --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json \
  --voice voices/ljspeech.safetensors \
  --noise-temp 0.9 \
  --transition-steps 0 \
  --output /tmp/tada_test.wav \
  --text "The quick brown fox jumps over the lazy dog."
```

Critical flags:
- `--noise-temp 0.9` is mandatory. Without it audio is flat/dead.
- `--transition-steps 0` is required for the current ljspeech voice prompt. The
  prompt only has 5 acoustic tokens. With `--transition-steps 5` (old default), all
  5 tokens are trimmed → zero-shot mode (no voice conditioning at all).

Available models:
- `tada-1b-q4_0.gguf` (2.64 GB) — Q4_0 baseline
- `tada-1b-mixed.gguf` (1.75 GB) — **Variant E: VV F16 + Embed Q4_0** — recommended

---

## Quantization Finding (Variant E Is the Winner)

From systematic mixed-precision testing, the only quantization config that produces
correct gray-code time predictions for all test phrases is:

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| LLM (Llama) | Q4_0 | Fine |
| Embeddings | Q4_0 | **Critical** — Q8_0 causes gray-code bit flips |
| VibeVoice | F16 | Sufficient. F32 adds 0.9 GB with no measurable benefit |
| Decoder | Q4_0 | Doesn't affect output — acoustics arrive already fixed |

Variant E (VV F16 + Embed Q4_0) = 1.75 GB vs 2.64 GB baseline = 34% smaller,
identical audio for all tested phrases.

Gray-code misprediction signature: `times_before[N]` = 59 instead of 4
(gray(4)=0b00000110 → gray(59)=0b00100110, single bit 5 flip). This causes
`expand_durations()` to insert ~55 zero frames → trailing noise.

The `--max-time-before 40` clamp in `tada_generate.rs` is a guard against this
producing audible damage even if a misprediction slips through.

**Variant E has NOT been re-tested after the pipeline bug fixes.** This is a
pending task — the pre-fix benchmarks may not be comparable.

---

## Known Remaining Issues (Priority Order)

### 1. Voice prompt too short (most impactful)

`voices/ljspeech.safetensors` contains only 5 acoustic tokens. This is barely any
voice conditioning. The TADA encoder (a separate model from the LLM) processes raw
audio waveforms into acoustic tokens. A longer audio clip would produce more tokens
and significantly improve voice stability.

To regenerate:
```bash
# Activate venv first ("venv mon ami")
source .venv/bin/activate
python scripts/precompute_voice.py \
  --audio /path/to/longer_ljspeech_clip.wav \
  --text "The transcript of the clip." \
  --output voices/ljspeech_long.safetensors
```

Then use `--voice voices/ljspeech_long.safetensors` and set `--transition-steps 5`
(or whatever is appropriate for the new prompt length).

Note: `precompute_voice.py` needs the TADA encoder model. Check its `--help` for the
required model path argument.

### 2. "call" phrase truncation

Text: "I had to call you up in the middle of the night"

This phrase consistently generates duration values of 1, 1, 1 for middle words,
causing the audio to cut short. The model predicts almost-zero durations for "call",
"you", "up". Likely a gray-code prediction issue specific to this phoneme context.

Investigate by adding `--debug-dump /tmp/call_debug` to the generation command and
examining `step_NNN/scalars.json` for the affected steps. Compare `times_before`
values with a phrase that generates correctly.

### 3. "tyger" mispronunciation

Text: "Tyger Tyger, burning bright"

Starts with the wrong pronunciation. Likely a tokenization artifact — "Tyger" is
not a standard English word and may get split into unusual tokens that the model
hasn't seen in TTS context. Check what token IDs are produced and whether prepending
a phonetic respelling helps.

### 4. Voice instability on long phrases

Voice changes mid-utterance (heard on wutang-style long text). This is directly
related to the weak voice conditioning from the 5-token ljspeech prompt. Fixing
issue #1 (longer voice prompt) should help significantly.

### 5. Re-test Variant E with fixed pipeline

The benchmarks in `docs/benchmarks.md` and `docs/run_log.md` for Variant E were
recorded before the autoregressive time feedback bug was fixed. Re-run with the same
test phrases and record results in `docs/run_log.md` as a new run entry.

Benchmark command (fox phrase, Metal):
```bash
cargo run --example tada_generate -p tada-core --release --features metal -- \
  --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-mixed.gguf \
  --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json \
  --voice voices/ljspeech.safetensors \
  --noise-temp 0.9 --transition-steps 0 --seed 42 \
  --output /tmp/tada_varE_retest.wav \
  --text "The quick brown fox jumps over the lazy dog."
```

Record: load time, gen time, decode time, audio duration, RTF, times_before array,
any clamping events.

### 6. WASM path testing

All three bug fixes (time feedback, trailing frame, EOT token) have been applied to
`crates/tada-wasm/src/lib.rs` but the WASM build has NOT been tested end-to-end
since the fixes. Build and test in browser:

```bash
wasm-pack build crates/tada-wasm --target web --release
node web/serve.mjs  # port 8081
# Open http://localhost:8081
```

Known WASM architecture: Burn/wgpu runs the LLM on GPU (WebGPU), candle handles
VibeVoice + decoder on CPU. The JS worker (`web/tada-worker.js`) drives the
step-by-step generation loop.

### 7. CFG implementation (low priority / optional)

Python uses `acoustic_cfg_scale=1.6` for classifier-free guidance. This requires
running the LLM twice per step (conditional + unconditional) and combining the
logits. User has explicitly said they prefer the current Rust output quality without
CFG. Only implement if asked.

---

## Key Files

| File | Purpose |
|------|---------|
| `crates/tada-core/examples/tada_generate.rs` | Main native generation example. All pipeline fixes are here. |
| `crates/tada-core/src/tada_model.rs` | Model: `load_gguf`, `build_input_embeds`, `generate_acoustic`, `decode_audio` |
| `crates/tada-core/src/flow_matching.rs` | ODE solver, gray code decode, `expand_durations` |
| `crates/tada-core/src/config.rs` | `TadaConfig::tada_1b()` — verified config values |
| `crates/tada-wasm/src/lib.rs` | WASM bindings: mirrors `tada_generate.rs` logic, Burn/wgpu LLM |
| `scripts/quantize_tada.py` | GGUF quantization, mixed precision support |
| `scripts/precompute_voice.py` | Precompute voice prompt from audio |
| `scripts/save_generation_debug.py` | Dump Python intermediates for comparison |
| `voices/ljspeech.safetensors` | Current 5-token voice prompt (too short) |
| `voices/ljspeech.json` | Companion metadata (prompt transcript) |
| `docs/benchmarks.md` | All benchmark data (data only — no analysis) |
| `docs/run_log.md` | Detailed run notes and findings |

---

## Local Model Paths

```
Tokenizer:    /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json
Model BF16:   /Users/tc/Code/idle-intelligence/hf/tada-1b/model.safetensors
Model Q4_0:   /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf
Model mixed:  /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-mixed.gguf
Codec decoder: /Users/tc/Code/idle-intelligence/hf/tada-codec/decoder/model.safetensors
```

---

## Workflow Reminders

- Use sub-agents for all research and code writing. Lead orchestrates, sub-agents
  execute. Never do sequential manual edits from the lead.
- Commit atomically. One logical change per commit.
- Audio quality: do NOT judge by flatness/peak/RMS metrics. Have the user listen.
- Benchmarks go in `docs/benchmarks.md` (data only). Analysis goes in separate docs.
- Always use venv for Python scripts: `source .venv/bin/activate`
- Dev server: `node web/serve.mjs` (port 8081)
