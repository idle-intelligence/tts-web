# Multilingual Pocket-TTS — Onboarding Plan

**Status:** scoping only — nothing below has been implemented.
**Date:** 2026-07-25
**Origin:** [tts-web#2](https://github.com/idle-intelligence/tts-web/issues/2) (@ldenoue) → [kyutai-labs/pocket-tts#118](https://github.com/kyutai-labs/pocket-tts/issues/118)
**Goal:** add French / German / Spanish / Portuguese / Italian Pocket-TTS to the browser demo, following the exact path the existing English model already took.

Legend: **[EXISTS]** artifact already published — just consume it · **[CREATE]** we must produce it · **[CHECK]** needs verification before it can be scheduled.

---

## 0. Headline findings (read this first)

1. **Upstream already shipped multilingual.** #118 is an *announcement issue*, still open, but the work landed directly on `main`. Per-language checkpoints, tokenizers, and voice embeddings are all published today. We are not blocked on Kyutai.
2. **Each language is a separate checkpoint, not one multilingual model.** `kyutai/pocket-tts-without-voice-cloning` has a `languages/<lang>/` folder per variant, each with its own `model.safetensors`, `tokenizer.model`, and `embeddings/`. So this is "add N models", each following our single-model pipeline — not "swap one model for a bigger one".
3. **🔴 The tokenizer is per-language and NOT shared with English.** Every language ships a distinct SentencePiece `tokenizer.model` (different byte sizes: en 59,339 · de 59,837 · it 60,078 · fr 60,173 · es 60,895 · pt 60,995). This invalidates the assumption in the tracker note that "Pocket-TTS uses a standard text tokenizer" implies we can reuse the English one. Good news: our JS tokenizer is a **generic** SentencePiece-unigram implementation, so this is a *URL* change, not an algorithm change (see step 4).
4. **No phonemizer, confirmed.** Every language YAML sets `flow_lm.lookup_table.tokenizer: sentencepiece`. No G2P anywhere in the pipeline. Our `prepare_text_prompt()` is language-neutral.
5. **🔴 French has no small model.** Upstream publishes only `french_24l` (24-layer, ~641 MB BF16, labelled a not-yet-distilled preview). DE/ES/PT/IT each ship both a small (~209 MB BF16) and a `_24l` variant. See "Blockers" — French is the one language that does not drop cleanly into the current browser budget.
6. **Architecture is otherwise identical** across languages — same Mimi codec block, same `n_bins: 4000`, same `dim: 1024`. Only weights + tokenizer vocab differ. That means the small (non-24l) variants need **zero Rust model-architecture work**.

---

## 1. The established pipeline (what English did) — this is the template

Traced from the current tree (`convergence/tada-wasm-perf`, HEAD `68cf170`) plus the Kitten/TADA onboarding commits.

| # | Stage | Artifact / entry point |
|---|---|---|
| 1 | Quantize | `scripts/pocket-tts/quantize_to_gguf.py --model-id kyutai/pocket-tts-without-voice-cloning -o model.gguf --no-encoder` → Q8_0 GGUF (225 MB BF16 → 128 MB). An alternative INT8-safetensors path exists (`scripts/pocket-tts/quantize.py`, → `idle-intelligence/pocket-tts-int8`) but **the browser fetches the GGUF**, so GGUF is the live path. |
| 2 | Voices | Not vendored. Fetched at runtime from Kyutai: `${VOICE_BASE}/embeddings_v2/${name}.safetensors`. Roster is a hardcoded JS array, `web/index.html:441`. |
| 3 | Host | `idle-intelligence/pocket-tts-gguf` — holds `pocket-tts-q8_0.gguf` + `tokenizer.model`. Uploaded with `hf upload`. |
| 4 | Rust | `crates/tts-core` (model) + `crates/tts-wasm` (wasm-bindgen surface: `Model::new`, `add_voice`, `prepare_text`, `start_generation`, `generation_step`). Config is a hardcoded factory, `TTSConfig::v202601()` at `crates/tts-core/src/config.rs:25`. **No model registry/enum exists** — each model is its own crate pair + its own worker. |
| 5 | Test native | `cargo test -p tts-core`; baseline test at `crates/tts-core/tests/baseline_original.rs`. |
| 6 | Demo | `web/worker.js` (fetch + Cache API `tts-model-v3` + JS SentencePiece tokenizer) ← driven by `makePocketClient()` in `web/index.html`; radio-button model switch at `web/index.html:325-329`. |
| 7 | Build/release | `wasm-pack build crates/tts-wasm --target web --release` → commit `pkg/` + `web/` onto the orphan `gh-pages` branch. No CI, no `package.json`. E2E: `node scripts/test_demo_e2e.mjs` (Playwright headless Chromium). |

**Crucial simplification for this task:** adding a *language* is not adding a *model*. Same architecture, same crate, same worker code — only weights, tokenizer, and voice URLs change. This is materially cheaper than the Kitten/TADA onboarding, which each needed a new `crates/<model>-{core,wasm}` pair and a new worker.

---

## 2. Upstream artifact audit

| Artifact | Status | Location / evidence |
|---|---|---|
| Per-language checkpoints FR/DE/ES/PT/IT | **[EXISTS]** | `kyutai/pocket-tts-without-voice-cloning` → `languages/{german,italian,spanish,portuguese,french_24l,german_24l,italian_24l,portuguese_24l,spanish_24l}/model.safetensors` |
| Small-variant size | **[EXISTS]** | 219,029,196 B (~209 MB) for de/it/es/pt — same ballpark as English 225 MB → expect ~128 MB Q8_0 |
| `_24l` variant size | **[EXISTS]** | 672,178,676 B (~641 MB) → expect ~370 MB Q8_0 |
| Per-language SentencePiece tokenizers | **[EXISTS]** | `languages/<lang>/tokenizer.model`, distinct per language (sizes above) |
| Per-language voice embeddings | **[EXISTS]** | `languages/<lang>/embeddings/*.safetensors`, same 25-name roster as English, cross-cloned into every language. Note path is `embeddings/`, **not** the `embeddings_v2/` our English worker uses. |
| Per-language configs | **[EXISTS]** | `kyutai-labs/pocket-tts` → `pocket_tts/config/<lang>.yaml` |
| Default voice per language | **[EXISTS]** | estelle (fr), juergen (de), lola (es), rafael (pt), giovanni (it) |
| **Small French model** | **🔴 does not exist** | only `french_24l` published |
| Our multilingual GGUF quant | **[CREATE]** | HF org `idle-intelligence` has only `pocket-tts-int8` / `pocket-tts-gguf`, both tagged `en` |
| Redistribution license for re-hosted quants | **[CHECK]** | Kyutai repos are **gated** (HF auth required to download). Confirm the licence permits us re-hosting derived quants publicly, as we already do for English. |

---

## 3. Ordered task list

### Step 1 — Quantize *(per language)*

- [ ] **1.1 [CHECK]** Confirm HF gated-repo access: `hf auth whoami`, then accept terms on `kyutai/pocket-tts-without-voice-cloning`. Everything downstream blocks on this.
- [ ] **1.2 [CREATE]** Teach `scripts/pocket-tts/quantize_to_gguf.py` to read a **subfolder**. Today `--model-id` calls `hf_hub_download(repo_id=..., filename=...)` trying only root-level `tts_b6369a24.safetensors` / `model.safetensors` (`scripts/pocket-tts/quantize_to_gguf.py:325-335`) — it cannot reach `languages/<lang>/model.safetensors`. Add `--subfolder`, pass through to `hf_hub_download(subfolder=...)`. *(Workaround if we want zero script change: `hf download` the file manually and use `--input`. Prefer the flag — we run this 4-5×.)*
- [ ] **1.3 [CHECK]** Verify `remap_key()` and the quantize-allowlist in that script still match the per-language tensor names. Architecture is identical upstream, so expect a clean pass — but assert tensor-name coverage rather than assuming, and confirm the quantizer doesn't silently leave the (per-language, differently-sized) lookup-table embedding unquantized in a way that changes file layout.
- [ ] **1.4 [CREATE]** Produce `pocket-tts-{de,es,pt,it}-q8_0.gguf` with `--no-encoder`. Record output sizes in the research log.
- [ ] **1.5 [CHECK]** French: decide whether to ship `french_24l` at all (see Blockers). If yes, quantize it too and record the real size.
- [ ] **1.6 [CREATE]** Validate each quant offline (`--validate`, plus an SQNR spot-check like the English 37.2 dB worst-layer figure in `docs/pocket-tts/QUANTIZATION.md`).

### Step 2 — Voices

- [ ] **2.1 [EXISTS]** No voice extraction needed — upstream ships baked per-language embeddings. This step is nearly free.
- [ ] **2.2 [CHECK]** Confirm the runtime voice URL shape per language: English uses `.../resolve/main/embeddings_v2/<name>.safetensors`; language folders expose `.../resolve/main/languages/<lang>/embeddings/<name>.safetensors`. Confirm the `embeddings/` files are the same format `Model::add_voice()` expects (they are the same 25-voice roster; verify one loads before wiring all).
- [ ] **2.3 [CREATE]** Pick the per-language voice subset for the UI grid. English shows 8 of 25 (`web/index.html:441`); mirror that, leading with each language's upstream default voice (estelle/juergen/lola/rafael/giovanni).
- [ ] **2.4 [CHECK]** Kyutai's repo is gated — confirm anonymous browser `fetch()` of `resolve/main/...` still works for voices (English already does this today from the same repo, so this most likely holds; verify it holds under `languages/`).

### Step 3 — Host on HF

- [ ] **3.1 [CREATE]** Decide repo layout. **Recommended:** extend the existing `idle-intelligence/pocket-tts-gguf` with a `languages/<lang>/` prefix mirroring upstream, rather than one repo per language — one repo, one licence/model card, and the worker's URL builder stays trivial.
- [ ] **3.2 [CREATE]** Upload per language: `hf upload idle-intelligence/pocket-tts-gguf <local.gguf> languages/<lang>/pocket-tts-q8_0.gguf` and the matching `languages/<lang>/tokenizer.model` (copied from upstream — the tokenizer is per-language and must be co-hosted, exactly as we co-host the English one).
- [ ] **3.3 [CREATE]** Update the model card: language tags (`fr`,`de`,`es`,`pt`,`it`), base-model links, quant method, sizes, and credit to Kyutai.

### Step 4 — Rust support

Expected to be the *smallest* step for the non-24l languages.

- [ ] **4.1 [EXISTS]** `TTSConfig::v202601()` (`crates/tts-core/src/config.rs:25`) applies unchanged to all small variants — identical `d_model: 1024`, `n_bins: 4000`, same Mimi block. **No new config, no new crate, no new worker** for de/es/pt/it.
- [ ] **4.2 [EXISTS]** Tokenization is entirely JS-side (`UnigramTokenizer` + `decodeSentencepieceModel` in `web/worker.js:45-160`) and is a generic SentencePiece-unigram implementation with no English-specific vocab assumptions. A different `tokenizer.model` just works. **No Rust tokenizer work.**
- [ ] **4.3 [CHECK]** `prepare_text_prompt()` (`crates/tts-core/src/tts_model.rs:135-152`) is language-neutral (Unicode-aware capitalize, whitespace collapse, append `.` if the text ends alphanumeric, pad if <5 words). Review two cases before declaring it done: Spanish `¿…?` / `¡…!` and French spacing before `?!;:`. Likely fine as-is; confirm by ear.
- [ ] **4.4 [CREATE, French only]** If we ship `french_24l`, add a second config factory (e.g. `TTSConfig::v202601_24l()`) with the 24-layer dims and plumb a variant selector into `Model::new`. Read the layer/dim values from `pocket_tts/config/french_24l.yaml` rather than inferring them.
- [ ] **4.5 [CHECK]** Confirm `GgufTensors::from_bytes` + `TTSModel::load_gguf` accept the new file without shape assertions tripping on a differently-sized lookup table.

### Step 5 — Test locally against the Rust code

- [ ] **5.1 [CREATE]** Native smoke per language via the existing example/CLI path used for English; render one sentence per language to WAV and listen.
- [ ] **5.2 [CREATE]** Add a per-language baseline test alongside `crates/tts-core/tests/baseline_original.rs`, or at minimum assert non-silent audio + expected sample rate. Keep it lightweight — these need the gated weights, so gate the test behind an env var / `#[ignore]` like the existing baseline if it downloads.
- [ ] **5.3 [CREATE]** Log RTF per language natively (Metal) for comparison with the English 1.97× median, into the research log.

### Step 6 — Wire into + test the browser demo

- [ ] **6.1 [EXISTS]** `web/worker.js` **already** honours `config.tokenizerUrl` and `config.modelUrl` overrides (`web/worker.js:206,213`), falling back to the hardcoded English `HF_BASE`. So the worker needs no structural change — the language just has to be passed in the `load` config.
- [ ] **6.2 [CREATE]** Parameterize the voice URL. `web/worker.js:231` hardcodes `${VOICE_BASE}/embeddings_v2/${name}.safetensors`; add a `voiceBaseUrl` config field with the current value as the default, and pass `.../languages/<lang>/embeddings` for non-English.
- [ ] **6.3 [CREATE]** Add a language selector to the UI. Extend `makePocketClient()` (`web/index.html`, ~line 763) to take a language and build the three URLs; add a language dropdown next to the pocket-tts radio at `web/index.html:325`; make `VOICES` (`:441`) a per-language map instead of one flat array.
- [ ] **6.4 [CREATE]** Bump the Cache API name (`caches.open('tts-model-v3')`, `web/worker.js:16`) → `v4`, or key entries by URL-per-language so switching languages doesn't collide. Verify a language switch tears down and reloads the model cleanly via `switchModel()`.
- [ ] **6.5 [CREATE]** Set a per-language default test phrase. The canonical English phrase is "Hello, this is a test of the text to speech system"; add translated equivalents so the demo isn't feeding English text to a French model.
- [ ] **6.6 [CREATE]** Extend `scripts/test_demo_e2e.mjs` to cover each new language (it currently asserts ≥10 KB audio for the 3 models — add a case per language). Note this script imports Playwright from the absolute path `/Users/tc/node_modules/playwright/index.mjs`.
- [ ] **6.7 [CREATE]** Measure browser RTF per language and record it — the 209 MB→~128 MB quants should behave like English; French 24l will not.

### Step 7 — Package & release

- [ ] **7.1 [EXISTS]** `wasm-pack build crates/tts-wasm --target web --release`. Unchanged unless 4.4 (French 24l) lands — the WASM binary is language-agnostic since weights are fetched at runtime. **This is the big win: one binary serves all languages.**
- [ ] **7.2 🚨 [BLOCKER, pre-existing]** Push `main` — it is **101 commits ahead of origin** and exists only on this laptop. Deploy is blocked behind this and it is already the top item in the tracker.
- [ ] **7.3 [CREATE]** Deploy: rebuild `pkg/`, commit `pkg/` + `web/` onto the orphan `gh-pages` branch (manual — there is no `.github/` workflow). Note `gh-pages` was last deployed 2026-03-23 and is pre-TADA, so this deploy carries TADA too; sequence it with the pending TADA work rather than treating it as an isolated push.
- [ ] **7.4 [CREATE]** Update `README.md` + `CLAUDE.md` with the supported language list.
- [ ] **7.5 [CREATE]** Reply to [issue #2](https://github.com/idle-intelligence/tts-web/issues/2), credit @ldenoue's port (ldenoue.github.io/xn-ptts), link the deployed demo.

---

## 4. Blockers & open questions

1. **🔴 French model size.** Only `french_24l` exists (~641 MB BF16 → ~370 MB Q8_0). That is ~3× the English browser download and may be unusable on mobile. Options: (a) ship French anyway with a clear size warning; (b) ship DE/ES/PT/IT now and wait for Kyutai's distilled French; (c) attempt a more aggressive quant (Q4_K_M) for French only — which overlaps the existing backlog item "quantize pocket-tts to Q4_K_M/Q5_K_M/Q6_K". **Recommendation: (b) + (c) in parallel — ship the four small languages first, treat French as its own follow-up.**
2. **🔴 Gated weights.** Both Kyutai repos require accepting terms. Blocks step 1 entirely; also means the *runtime* voice fetch from Kyutai must be re-verified for the `languages/` paths (English already fetches from this repo anonymously today, so the mechanism is proven — the path is what's new).
3. **[CHECK] Redistribution licence** for our re-hosted per-language quants — we already do this for English, so precedent exists; confirm it extends.
4. **[CHECK] Text normalization** for ES/FR punctuation conventions (task 4.3). Low risk, quick to check by ear.
5. **[CHECK] `_24l` for the other four languages.** Upstream calls them higher-quality but slower. Out of scope for v1; revisit after the small variants ship.
6. **Not a blocker:** tokenizer differences. Distinct per-language vocabs are already handled by our generic JS SentencePiece implementation.

---

## 5. Effort shape

- **Cheapest path (DE/ES/PT/IT, small variants):** one script flag (1.2), four quantize+upload runs, three JS wiring changes (6.2/6.3/6.4), zero Rust changes, zero new crates, one shared WASM binary.
- **French:** genuinely more work — new config factory, variant plumbing, and an unresolved size problem.
- **Critical-path unlock:** HF gated-repo access (1.1). Nothing in step 1 onward can start without it.
