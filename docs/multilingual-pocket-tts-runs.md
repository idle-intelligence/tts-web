# Multilingual Pocket-TTS — run log

Branch `feat/multilingual-pocket-tts`. Voices, checkpoints and tokenizers all from
`kyutai/pocket-tts-without-voice-cloning`, `languages/<lang>/`.

## Quantization (Q8_0, `--no-encoder --validate`)

Produced by `scripts/pocket-tts/quantize_to_gguf.py --subfolder languages/<lang>`.
Source checkpoints are 219,029,196 B BF16 (French: 672,178,676 B).

| Lang | Output | Bytes | GGUF tensors | Layers | Worst-layer SQNR |
|---|---|---|---|---|---|
| de | `pocket-tts-de-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.3 dB |
| es | `pocket-tts-es-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.2 dB |
| pt | `pocket-tts-pt-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.7 dB |
| it | `pocket-tts-it-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.5 dB |
| en2 | `pocket-tts-en2-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.6 dB |
| fr | `pocket-tts-fr-q8_0.gguf` | 374,792,736 | **315** | **24** | 38.5 dB |

Worst tensor is `flow_lm.out_eos.weight` in every case except en2
(`flow_lm.transformer.layers.0.linear2.weight`).

**en2 and fr are on branch `feat/multilingual-en-fr`**, the rest on `feat/multilingual-pocket-tts`.
`en2` is `languages/english` — the new-generation English, NOT the older root checkpoint the
shipped Pocket TTS tab uses. French is the only 24-layer model upstream publishes; its tensor
count checks out exactly (171 + 8 per-layer tensors × 18 extra layers = 315), and every shape
outside the layer count matches the 6-layer models, so `remap_key()` and the quantize allowlist
needed no changes at all. All six beat the shipped English model's 37.2 dB. `remap_key()` covers every tensor — no `None` returns, no collisions.
All GGUFs have distinct sha256s; the identical byte size is a consequence of identical
shapes, and was checked rather than assumed.

Artifacts live in `hf/pocket-tts/` (outside the repo). **Nothing has been uploaded.**

## Native runs (candle CPU, `cargo run --example tts_generate --release`)

Voice `anna` from `languages/<lang>/embeddings/`. All reached EOS naturally.

| Lang | Text | Audio | Notes |
|---|---|---|---|
| de | "Hallo, dies ist ein Test des Text-zu-Sprache-Systems." | 2.88 s | EOS step 34, no NaN/inf |
| es | "Hola, esta es una prueba del sistema de texto a voz. ¿Cómo estás hoy?" | 3.92 s | EOS step 47 |
| pt | "Olá, este é um teste do sistema de texto para voz. Como você está?" | 3.92 s | EOS step 47 |
| it | "Ciao, questo è un test del sistema da testo a voce. Come stai oggi?" | 4.00 s | EOS step 48 |
| en2 | "Hello, this is a test of the text to speech system." | 3.12 s | EOS step 37, voice `anna` |
| fr | "Bonjour, ceci est un test du système de synthèse vocale. Comment allez-vous aujourd'hui ?" | 3.68 s | EOS step 44, voice `estelle`, **24 layers auto-detected**, 48 voice tensors |

WAVs kept at `hf/pocket-tts/samples/` for listening. **Not yet listened to by a human** —
per project convention, audio metrics are not a quality signal, so these are unjudged.

## Browser runs (WASM, headless Chromium, local dev server)

Voice `alba`, user-initiated generation, fresh page per language.

| Lang | Audio | Wall | TTFB | RTF |
|---|---|---|---|---|
| de | 2.64 s | 1.32 s | 0.36 s | 2.26× |
| es | 2.40 s | 1.17 s | 0.34 s | 2.30× |
| pt | 3.12 s | 1.51 s | 0.34 s | 2.27× |
| it | 4.08 s | 1.92 s | 0.34 s | 2.27× |
| en2 | 2.80 s | 1.36 s | 0.34 s | 2.26× |
| fr | 3.92 s | 3.56 s | 1.06 s | 1.37× |

French runs at 1.37× against ~2.3× for the 6-layer models — expected for 4× the depth, and still
faster than realtime. **Rebuilding the WASM package is required** after any change to layer
detection: a stale `pkg/` silently loaded 6 of French's 24 layers and produced 0.16 s of audio
with no error at all.

Zero console errors. Correct assets fetched per language (verified by request interception).
Save-wav link produces the right per-language filename.

## Resolved — repeated in-session language switching

**Was:** cycling de → es → pt → it in one page session gave a ~0.24 s Spanish clip instead of ~2.4 s,
and left Portuguese/Italian with a stale `pocket-ml-es-...` download filename. Only a fresh page per
language was correct. Three speculative fixes (headless autoplay policy, synchronously disabling
controls, serialising `applyModel` on a promise chain) each failed and were reverted.

**Now:** does not reproduce. Cycling every language in a single session yields correct durations and
correct per-language filenames, zero console errors, on both branches.

**Why, probably:** the dropdown's change handler called `applyModel(...)` fire-and-forget and left the
user to re-pick a voice, so a click could land mid-teardown. The button grid replaced it with
`switchLanguage()`, which *awaits* `applyModel(...)` and then selects the voice itself — nothing
external can race the reload. That matches the original hypothesis (a late teardown cancelling
generation, a stale `onDone` closure building the blob), but the fix was a side effect of the UX
change rather than a targeted repair, so the underlying shared-state fragility in `applyModel` has
not been audited. Treat this as "symptom gone", not "root cause proven".

## 2026-10-07 — Italian gibberish: voice embeddings from a different model revision

Report: Italian on the page sounds like gibberish ("de de de"), even on simple text.

### Upstream revisions

`kyutai/pocket-tts-without-voice-cloning` on 2026-10-01 (commits `cede6cf`, `3463ec7`, `1e08e6a`)
replaced `model.safetensors` **and** every voice embedding for it, es, de, pt, fr and nl. English
weights did not change. The latest PyPI release, pocket-tts 3.3.0 (2026-09-24), pins weights,
tokenizers and predefined voices to `4e1e0a3e611c51c0b4ed8174fc10f32a54644303`.

Our GGUFs (2026-09-29) were quantized from `main` before that upload. Dequantized GGUF tensor vs
upstream weights, SQNR in dB:

| Lang | Tensor | GGUF vs 4e1e0a3 | GGUF vs main | 4e1e0a3 vs main |
|---|---|---|---|---|
| it | layers.0.self_attn.in_proj.weight | 44.7 | -2.6 | -1.3 |
| it | conditioner.embed.weight | 347.1 | 9.0 | 9.1 |
| es | layers.0.self_attn.in_proj.weight | 44.9 | -1.2 | -0.4 |
| de | layers.0.self_attn.in_proj.weight | 44.6 | -1.5 | -1.1 |
| pt | layers.0.self_attn.in_proj.weight | 45.1 | -2.5 | -1.3 |
| fr | layers.0.self_attn.in_proj.weight | 44.3 | -0.9 | -0.7 |
| en2 | layers.0.self_attn.in_proj.weight | 45.4 | 45.4 | 335.1 |

Every GGUF matches `4e1e0a3`. The page fetched voices from `resolve/main`, i.e. KV caches computed
by the 2026-10-01 models, for fr/de/es/pt/it. English voices on main are byte-identical to `4e1e0a3`.

Tokenizers: all six local `tokenizer-<code>.model` sha256s equal the upstream `tokenizer.model` at
both `4e1e0a3` and main. Token ids for the Italian sentences equal the reference's
(`tokenizer.json` via `tokenizers`, and `sentencepiece`) id for id:

| Text | Ids (ours = reference) |
|---|---|
| Ciao, questo è un test del sistema da testo a voce. | 801 271 273 261 260 438 319 277 1845 293 260 1733 392 292 1845 273 267 659 262 |
| Buongiorno, come stai? | 2614 471 261 303 546 276 306 |
| Oggi il cielo è azzurro. | 3384 269 938 319 267 2413 262 |

Text preparation (strip, replace_characters, capitalisation, terminal punctuation, no padding) gives
the same string as the reference's `prepare_text_prompt` for all three.

### Temperature-0 parity, Italian, "Ciao, questo è un test del sistema da testo a voce."

Reference: pocket-tts 3.3.0 on the GPU box. Rust F32: `eos_probe` (flow LM only, unquantized
safetensors). Rust Q8: `tts_generate --temperature 0`. "Weights / voices" names the revision of each.

| Weights / voices | Voice | Ref frames | Rust F32 frames | Rust Q8 frames | Step-0 EOS-logit diff F32 | Step-0 latent[..8] max diff F32 |
|---|---|---|---|---|---|---|
| 4e1e0a3 / 4e1e0a3 | giovanni | 54 | 55 | 49 | 3.7e-3 | 1.8e-3 |
| 4e1e0a3 / 4e1e0a3 | anna | 44 | 44 | 44 | 1.9e-3 | 6.5e-4 |
| 4e1e0a3 / main | giovanni | 81 | 77 | 63 | 2.7e-3 | 2.7e-4 |
| 4e1e0a3 / main | anna | 30 | 30 | 40 | 3.3e-3 | 5.6e-4 |
| main / main | giovanni | 43 | 43 | — | 2.4e-3 | 8.8e-5 |
| main / main | anna | 44 | 44 | — | 2.0e-3 | 7.7e-4 |

The reference itself, given 4e1e0a3 weights and main voices (the page's combination), runs to 81
frames for a 54-frame sentence; Rust F32 tracks the reference in every combination.

Spanish, reference only, "Hola, esta es una prueba del sistema de texto a voz.", temperature 0:
lola 55 frames with 4e1e0a3 voices, 51 with main voices; giovanni 57 and 54.

### Listening sets (temperature 0.7, seed 42)

`hf/pocket-tts/samples/it-compare-2026-10-07/` (outside the repo): 3 sentences × giovanni/anna/marius
for ours (Q8, main voices), ourspin (Q8, 4e1e0a3 voices), ref (3.3.0 defaults), refmain (main
weights + main voices), refmismatch (4e1e0a3 weights + main voices). Native EOS steps, page sentence:
ours giovanni 64, anna 28, marius 42; ourspin giovanni 46, anna 41, marius 32. Reference durations,
page sentence: giovanni 3.36 s, anna 3.20 s, marius 2.96 s.

### Fix

Voice base URL pinned to `resolve/4e1e0a3e611c51c0b4ed8174fc10f32a54644303`, build tag
`2026-10-07-voice-pin` (branch `multilingual-voice-pin`).

### Gates with the pin (GPU box)

Native, Q8, page text and page default voice, 4e1e0a3 voices, temperature 0.7:

| Lang | Voice | EOS step | Steps | Audio | NaN/inf |
|---|---|---|---|---|---|
| en2 | alba | 41 | 44 | 3.52 s | 0/0 |
| fr | estelle | 47 | 50 | 4.00 s | 0/0 |
| de | juergen | 45 | 48 | 3.84 s | 0/0 |
| es | lola | 41 | 44 | 3.52 s | 0/0 |
| pt | rafael | 37 | 40 | 3.20 s | 0/0 |
| it | giovanni | 46 | 49 | 3.92 s | 0/0 |

Reference (3.3.0, same text/voice, temperature 0.7): en2 3.84 s, fr 4.32 s, de 3.44 s, es 3.52 s,
pt 3.44 s, it 3.36 s. WAVs in `hf/pocket-tts/samples/gates-2026-10-07/`.

Browser, headless Chromium (CPU), `scripts/build.sh` with `ENGINE_BUILD=2026-10-07-voice-pin`, fresh
page per language, explicit voice click: every language generated, zero console errors, every voice
request went to `.../resolve/4e1e0a3e.../languages/<lang>/embeddings/<voice>.safetensors`.

| Lang | Voice | Result |
|---|---|---|
| en2 | alba | 3.52 s audio |
| fr | estelle | 3.76 s audio |
| de | juergen | 3.76 s audio |
| es | lola | 3.52 s audio |
| pt | rafael | 3.20 s audio |
| it | giovanni | 3.76 s audio |
