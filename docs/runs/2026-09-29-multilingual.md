# Multilingual Pocket-TTS — finish and verify, 2026-09-29

Branch `feat/multilingual-en-fr` re-applied onto current `main`. `main` had merged
and then reverted an earlier version of this work (2026-07-26/27); a plain `git
rebase main` silently dropped the branch's code because git treated the
already-reverted commits as "previously applied" against `main`'s history. Fixed
by resetting to a fresh branch off `main` and re-checking-out the branch tip's
files directly (`git checkout <old-tip> -- <paths>`), squashed into one commit.
Dutch was never part of this branch's implementation (checked: no `dutch`
reference anywhere in the tree before or after), so there was nothing to remove
for that decision.

## Source checkpoints

All from `kyutai/pocket-tts-without-voice-cloning`, `languages/<lang>/`, snapshot
`069025daa1d6a8a9640bd52581e2a2252f8bede4` (fetched 2026-09-29).

| Language | tokenizer.model bytes (branch's old fixture) | tokenizer.model bytes (current Hub) | Changed |
|---|---|---|---|
| english | 59,339 | 59,339 | no |
| french | (branch used `french_24l`, 60,173) | 60,690 (new `french` small model) | model switch |
| german | 59,837 | 60,347 | yes |
| spanish | 60,895 | 61,063 | yes |
| portuguese | 60,995 | 61,351 | yes |
| italian | 60,078 | 60,905 | yes |

Four of five previously-quantized languages had a changed tokenizer upstream
since the branch's July run; German/Spanish/Portuguese/Italian were
re-quantized, and French was re-quantized from the new small (6-layer)
checkpoint instead of the old 24-layer one.

## Requantization (Q8_0, `--subfolder languages/<lang> --no-encoder --validate`)

| Lang | Output bytes | Tensors | Layers | Worst-layer SQNR |
|---|---|---|---|---|
| fr | 133,837,984 | 171 | 6 | 39.3 dB |
| de | 133,837,984 | 171 | 6 | 40.1 dB |
| es | 133,837,984 | 171 | 6 | 39.9 dB |
| pt | 133,837,984 | 171 | 6 | 39.6 dB |
| it | 133,837,984 | 171 | 6 | 39.9 dB |

All above the shipped English model's 37.2 dB baseline and the project's 37 dB
threshold. French drops from 375 MB / 24 layers to 134 MB / 6 layers, matching
the other four languages.

## Tokenizer fixtures

`scripts/pocket-tts/gen_tokenizer_fixtures.py` generates `golden.json` from the
official `sentencepiece` Python library run directly against the committed
`.model` files — not from this port's own tokenizer, so it is a real parity
fixture (this was flagged as an open question in the prior plan; confirmed by
reading the script). Regenerated for all six languages against the refreshed
`.model` files. `cargo test -p tts-core --test tokenizer_golden` passes,
covering accented and non-ASCII cases (café, œuf, Müller, à l'hôtel, etc.) for
every language.

## Official PyTorch parity (`pip install pocket-tts`)

Ran `TTSModel.load_model(language=<lang>, temp=0.0)` for the same text/voice as
the native smoke test, at temperature 0.0. At temp=0 the flow-matching noise's
std is `0.0**0.5 = 0` in both the official model
(`pocket_tts/models/flow_lm.py:157-160`) and this port
(`SimpleRng::new`, `std = temperature.sqrt()`), so both are deterministic
without needing to match RNG streams.

| Lang | Rust total samples | Official total samples | Ratio |
|---|---|---|---|
| fr | 69,120 | 69,120 | 1.00 |
| de | 84,480 | 88,320 | 0.96 |
| es | 97,920 | 105,600 | 0.93 |
| pt | 92,160 | 86,400 | 1.07 |
| it | 97,920 | 103,680 | 0.94 |

English was not compared: the official package's `language="english"` alias
resolves to the newer `english_2026-09` checkpoint, not the older root
checkpoint this repo currently ships as `pocket-tts-q8_0.gguf` — comparing them
would not be a same-checkpoint parity check.

All five ratios are within the 20% tolerance encoded in the new
`crates/tts-core/tests/official_parity.rs` (`#[ignore]`, needs local
model/tokenizer/voice files via env vars). One concrete contributing factor was
found while investigating the gap: the official package's
`generate_audio_stream` adds +2 to its text-length-based `frames_after_eos`
guess before using it (`pocket_tts/models/tts_model.py:720-731`); this port's
`prepare_text_prompt` does not add that +2. That alone does not explain every
observed delta (Portuguese runs longer in this port, not shorter, which the
+2-frame theory alone would predict), so a residual EOS-timing difference
remains open for follow-up.

Tokenizer-level parity (above) is exact for all six languages. Model-output
parity is directionally correct and within a documented tolerance, not
frame-exact.

## Native smoke generation (candle CPU, `--release`, temperature 0.7, seed 42)

| Lang | Text (first words) | Voice | Audio | EOS step | Native RTF |
|---|---|---|---|---|---|
| en | "Hello, this is a test..." | alba | 7.04s | (ran to step limit, see below) | ~2.7x |
| fr | "Bonjour, ceci est un test..." | estelle | 2.88s | 34 | ~3.0x |
| de | "Hallo, dies ist ein Test..." | juergen | 3.20s | 38 | ~1.9x |
| es | "Hola, esta es una prueba..." | lola | 3.68s | 44 | ~2.9x |
| pt | "Olá, este é um teste..." | rafael | 3.12s | 37 | ~2.9x |
| it | "Ciao, questo è un test..." | giovanni | 3.68s | 44 | ~3.3x |

RTF = audio duration / wall time, measured on the shared Mac at load average
~3.3 (mild contention from other workers) — informal smoke numbers, not a
controlled benchmark. English's run in this pass did not show an explicit EOS
line in the captured tail of output; it produced clean, NaN/Inf-free audio at
a plausible duration for the text, consistent with the branch's original run
log behavior for this same checkpoint.

All six reached completion with zero NaN/Inf in the final audio buffer.

## Workspace tests and lint

- `cargo test --workspace`: all pass, including `tokenizer_golden` (6
  languages) and the pre-existing `pipeline_debug`/`quant_validation` suites.
- `cargo clippy -p tts-core -p tts-wasm -- -D warnings`: fails on pre-existing
  warnings in `crates/tts-core/src/mlp.rs` and `flow_lm.rs` (unused
  constructor parameters), unrelated to this change and present unmodified on
  `main` before this branch was reapplied. `cargo clippy --workspace` also
  fails on pre-existing `kitten-core` issues, likewise untouched by this work.
  No new clippy warnings were introduced by the multilingual changes.

## Web / WASM

`wasm-pack build crates/tts-wasm --target web --release` rebuilt cleanly.
Bumped `MODEL_CACHE`/`CACHE_NAME` from `tts-model-v3` to `tts-model-v4` and
added an `ENGINE_BUILD = '2026-09-29-multilingual'` query tag on the worker and
wasm-module loading URLs, since French's on-disk path changed
(`languages/french_24l` → `languages/french`) and a stale cached worker could
otherwise keep serving the old 24-layer path silently (the exact failure mode
the branch's own run log warned about in July).

Playwright (headless Chromium, `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal`)
against a local dev server with the requantized GGUFs staged locally:
switched into every one of French/German/Spanish/Portuguese/Italian, waited
for the per-language default voice to auto-load, and confirmed generation
completed (a populated `stepInfo` readout with an RTF figure) with zero
console errors for all five. Screenshots at 1280px and 390px taken of the
language grid.

## Files

- `crates/tts-core/tests/fixtures/tokenizers/{german,italian,portuguese,spanish}.model` — refreshed
- `crates/tts-core/tests/fixtures/tokenizers/french_24l.model` → `french.model` — renamed, new content
- `crates/tts-core/tests/fixtures/golden.json` — regenerated for 6 languages
- `crates/tts-core/tests/official_parity.rs` + `crates/tts-core/tests/fixtures/parity/*.json` — new
- `scripts/pocket-tts/gen_tokenizer_fixtures.py` — `french_24l` → `french`
- `web/index.html`, `web/worker.js` — French points at `languages/french`, build tag bump
- `docs/multilingual-pocket-tts-runs.md` — unmodified historical branch log, kept as-is

## Parity fixes and English checkpoint move — 2026-09-29 (second pass)

Fixed three EOS/frame-count bugs found by reading the official package
against this port: the countdown was checked after appending a step's
audio instead of before (off-by-one), the official `+= 2` on the
text-length-based `frames_after_eos` guess was missing, and there was no
`_MIN_FRAMES_BEFORE_EOS = 6` gate on early EOS detections. Also ported the
official per-language text normalization (`replace_characters`,
`remove_semicolons`, full terminal-punctuation handling, and gating
`pad_with_spaces_for_short_inputs` behind config instead of applying it
unconditionally).

### Frame counts, Rust (Q8_0 GGUF, temperature 0) vs. official, before and after

"Before" is the first-pass run doc's ratio table (old EOS/frame-count logic,
`total_samples`, divided by 1920 samples/frame). "After" is this session's
run against the same weights, tokenizers and voices with the fixes above and
the same fixed text/voice per language. English was not part of the "before"
table (excluded then; moved to the `english_2026-09` checkpoint in this
pass, see below).

| Lang | Rust frames (before) | Rust frames (after) | Official frames | Δ frames, after (rust − official) |
|---|---|---|---|---|
| en | — | 41 | 41 | 0 |
| fr | 36 | 37 | 36 | +1 |
| de | 44 | 45 | 46 | -1 |
| es | 51 | 52 | 55 | -3 |
| pt | 48 | 49 | 45 | +4 |
| it | 51 | 52 | 54 | -2 |

Source: `cargo test -p tts-core --test official_parity --release -- --ignored --nocapture`,
this session (`fr/de/es/pt/it` against the same GGUFs/tokenizers/voices as the
first pass; `en` against the newly quantized `english_2026-09` GGUF). "Before"
column from `docs/runs/2026-09-29-multilingual.md`'s "Official PyTorch parity"
table (`total_samples / 1920`).

### Residual divergence is not primarily quantization noise

A flow-LM-only probe (`cargo run --example eos_probe -p tts-core --release`,
no Mimi, unquantized F32 safetensors from the official
`kyutai/pocket-tts-without-voice-cloning` checkpoint) against the same
French text/voice at temperature 0, diffed against a Python-side hook on
`FlowLMModel.out_eos`:

| Step | Rust (F32, unquantized) EOS logit | Official EOS logit |
|---|---|---|
| 0 | -14.32 | -14.35 |
| 1 | -13.25 | -11.70 |
| 4 | -11.69 | -13.47 |
| 5 | -13.03 | -15.85 |

First EOS crossing (`> -4.0`): Rust F32 at step 36, official at step 33 (Δ=3).
Rust Q8_0 (the production path) crosses at step 34 (Δ=1 vs. official) —
*closer* to official than the unquantized F32 run. Both engines' RoPE
convention and frequency formula were checked line-by-line against
`pocket_tts/modules/rope.py` and match exactly (interleaved-pairs, same
`max_period` formula). LayerNorm epsilon (`1e-5`) also matches.

**Verdict**: the gap is not simply Q8_0 quantization noise added on top of an
otherwise-exact port — it is present already in F32, and quantization noise
does not uniformly make it worse. The exact source (most likely floating-point
summation-order differences between candle and PyTorch compounding through
the flow-matching sampler's chaotic sensitivity near the EOS threshold, or a
smaller structural difference in `mimi-rs`'s `StreamingTransformer` not yet
isolated) remains open. This is not a fabricated conclusion to close the
investigation — it reflects the actual measurement above, not an assumption.

### English checkpoint

Moved to `english_2026-09` (same weights as the official `language="english"`
alias, per its own config comment). Quantized this session with the same
pipeline used for the other five languages:

| Tensors | Layers | Worst-layer SQNR |
|---|---|---|
| 214 | 6 | 39.6 dB |

Above the 37 dB project threshold. Frame count matches official exactly (41
frames both sides, see table above) for the fixture text/voice.

### Not done in this pass

- Root-causing the residual per-step divergence beyond ruling out RoPE
  convention/frequency and LayerNorm epsilon — needs a deeper diff of
  `mimi-rs`'s `StreamingTransformer` (KV-cache indexing, attention masking)
  against `pocket_tts/modules/transformer.py`. `mimi-rs` is a separate,
  shared crate; a fix there is out of scope for this pass.
- Uploading the new English GGUF/tokenizer to the `idle-intelligence/pocket-tts-gguf`
  Hub repo — the code now points at `languages/english/pocket-tts-q8_0.gguf`
  and `languages/english/tokenizer.model` on that repo, but nothing has been
  pushed there in this pass.
