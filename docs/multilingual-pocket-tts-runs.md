# Multilingual Pocket-TTS — run log

Branch `feat/multilingual-pocket-tts`. Voices, checkpoints and tokenizers all from
`kyutai/pocket-tts-without-voice-cloning`, `languages/<lang>/`.

## Quantization (Q8_0, `--no-encoder --validate`)

Produced by `scripts/pocket-tts/quantize_to_gguf.py --subfolder languages/<lang>`.
Source checkpoints are 219,029,196 B BF16; all four are structurally identical.

| Lang | Output | Bytes | GGUF tensors | Layers | Worst-layer SQNR |
|---|---|---|---|---|---|
| de | `pocket-tts-de-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.3 dB |
| es | `pocket-tts-es-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.2 dB |
| pt | `pocket-tts-pt-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.7 dB |
| it | `pocket-tts-it-q8_0.gguf` | 133,837,984 | 171 | 6 | 39.5 dB |

Worst tensor is `flow_lm.out_eos.weight` in every case except en2
(`flow_lm.transformer.layers.0.linear2.weight`).

**en2 and fr are on branch `feat/multilingual-en-fr`**, the rest on `feat/multilingual-pocket-tts`.
`en2` is `languages/english` — the new-generation English, NOT the older root checkpoint the
shipped Pocket TTS tab uses. French is the only 24-layer model upstream publishes; its tensor
count checks out exactly (171 + 8 per-layer tensors × 18 extra layers = 315), and every shape
outside the layer count matches the 6-layer models, so `remap_key()` and the quantize allowlist
needed no changes at all. All four beat the shipped English
model's 37.2 dB. `remap_key()` covers every tensor — no `None` returns, no collisions.
All four GGUFs have distinct sha256s; the identical byte size is a consequence of identical
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

Zero console errors. Correct assets fetched per language (verified by request interception).
Save-wav link produces the right per-language filename.

## Open issue — repeated in-session language switching

**Repro:** load the page once, select Pocket Multilingual, then cycle de → es → pt → it
without reloading, generating in each.

**Observed:** German is correct. Spanish returns ~0.24 s of audio (≈3 frames) instead of
~2.4 s. Portuguese and Italian produce correct-length audio but their save-wav filename
stays `pocket-ml-es-...`.

**Not reproducible** with a fresh page per language (all four correct, table above), nor
when switching only de → es.

**Ruled out:** headless autoplay policy (persists with
`--autoplay-policy=no-user-gesture-required`); an interaction race during the async switch
(persists after disabling controls synchronously); overlapping `applyModel` calls (persists
after serialising them on a promise chain). All three speculative fixes were reverted since
none changed the behaviour — the committed tree contains no unvalidated changes.

**Best remaining hypothesis, untested:** a stale `onDone` closure from the previous
language's client building the blob, which would explain the correct-length audio paired
with the wrong filename, and a late teardown cancelling generation, which would explain the
truncated Spanish clip. `tts.destroy()` terminates the worker but module-scope state
(`allChunks`, `activeVoice`, the pending callbacks) is shared across clients.

**Impact:** the tab works correctly on a normal page load. A user who cycles several
languages in one session may get a truncated clip or a mislabelled download. Worth fixing
before this ships, not blocking local evaluation.
