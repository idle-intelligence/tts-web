# Multilingual Pocket-TTS — Plan

**Status:** scoped, not started. No implementation yet.
**Last updated:** 2026-07-25
**Origin:** [tts-web#2](https://github.com/idle-intelligence/tts-web/issues/2) (@ldenoue) → [kyutai-labs/pocket-tts#118](https://github.com/kyutai-labs/pocket-tts/issues/118)
**Goal:** add French / German / Spanish / Portuguese / Italian Pocket-TTS.

Legend: **[EXISTS]** already published, just consume it · **[CREATE]** we must produce it · **[CHECK]** verify before scheduling.

---

## 0. What you need to know before reading anything else

1. **Upstream already shipped multilingual.** #118 is an announcement issue (still open), but the work landed on `main`. Per-language checkpoints, tokenizers and voice embeddings are all published today. We are not blocked on Kyutai.
2. **Each language is a separate checkpoint**, not one multilingual model. `kyutai/pocket-tts-without-voice-cloning` has `languages/<lang>/` per variant, each with its own `model.safetensors`, `tokenizer.model`, and `embeddings/`.
3. **🔴 Tokenizers are per-language and NOT shared with English.** Distinct SentencePiece models — en 59,339 B · de 59,837 · it 60,078 · fr 60,173 · es 60,895 · pt 60,995. Independently trained vocabs, not an extension of English. This kills any "reuse the English tokenizer" assumption.
4. **No phonemizer, confirmed.** Every language YAML sets `flow_lm.lookup_table.tokenizer: sentencepiece`. Kyutai's Python calls the official `sentencepiece` package — C++ core, so not WASM-portable, which is why our browser path reimplements it.
5. **🔴 French has no small model.** Only `french_24l` (24-layer, ~641 MB BF16 → ~370 MB Q8_0), labelled a not-yet-distilled preview. DE/ES/PT/IT each ship a ~209 MB small variant.
6. **`n_bins: 4000` is hard-asserted for every language.** Upstream does `assert nbins == sp.vocab_size()`. Valid IDs are 0–3999; the LUT has 4001 rows (row 4000 is padding, per their `nn.Embedding(n_bins + 1, dim)  # n_bins + 1 for padding`). Architecture is otherwise identical across languages — only weights and tokenizer vocab differ.

**Consequence:** the four small languages need **zero Rust model changes and no new crate**. Weights are fetched at runtime, so one WASM binary serves every language.

---

## 1. How Pocket-TTS actually works (orientation)

```
text ──► prepare_text_prompt()          crates/tts-core/src/tts_model.rs:135
     ──► SentencePiece unigram Viterbi  web/worker.js:46-191   [JS — the ONLY tokenizer that exists]
     ──► token_ids: &[u32]  ═══ JS/WASM boundary ═══
     ──► LUTConditioner.embed_tokens()  crates/tts-core/src/flow_lm.rs:34
           index_select into embed[4001, 1024]        ← the ENTIRE text path
     ──► [1, T, 1024] straight into the transformer (lut_dim == d_model, no projection)
     ──► FlowLM: 6-layer streaming transformer, appends to the voice's KV cache

per frame:
  prev_latent [1,1,32] ──► input_linear 32→1024 ──► transformer (+1 KV position)
     ├──► out_eos → scalar logit, EOS if > -4.0
     └──► lsd_decode: ONE Euler step of flow matching → next_latent [1,1,32]
  ──► denormalize (×emb_std +emb_mean)
  ──► DummyQuantizer Conv1d k=1, 32→512    [no VQ codebook — encoder is stripped]
  ──► decoder transformer (2 layers) ──► SEANet decoder (ratios 6,5,4 = ×120)
  ──► PCM f32 @ 24 kHz, ~2000 samples/frame, ~12.5 Hz
```

Two facts that make multilingual cheap:

- **There is no text encoder.** Text conditioning is one embedding-table lookup. Nothing downstream knows text exists, so swapping languages = different LUT weights (in the checkpoint) + different tokenizer.
- **Voices are pre-baked KV caches, not embeddings.** `add_voice()` (`crates/tts-wasm/src/lib.rs:74`) deserializes `transformer.layers.{i}.self_attn/cache` for each of the 6 layers and injects it — no forward pass. So **voice and language are orthogonal**, and upstream cross-clones the same 25-voice roster into every language.

---

## 2. Upstream artifact audit

| Artifact | Status | Where |
|---|---|---|
| Per-language checkpoints FR/DE/ES/PT/IT | **[EXISTS]** | `kyutai/pocket-tts-without-voice-cloning` → `languages/{german,italian,spanish,portuguese,french_24l,…}/model.safetensors` |
| Small-variant size | **[EXISTS]** | 219,029,196 B (~209 MB) for de/it/es/pt |
| `_24l` variant size | **[EXISTS]** | 672,178,676 B (~641 MB) |
| Per-language tokenizers | **[EXISTS]** | `languages/<lang>/tokenizer.model`, distinct per language |
| Per-language voice embeddings | **[EXISTS]** | `languages/<lang>/embeddings/*.safetensors`, same 25 names as English. Note path is `embeddings/`, **not** the `embeddings_v2/` our English worker uses. |
| Per-language configs | **[EXISTS]** | `kyutai-labs/pocket-tts` → `pocket_tts/config/<lang>.yaml` |
| Default voice per language | **[EXISTS]** | estelle (fr), juergen (de), lola (es), rafael (pt), giovanni (it) |
| **Small French model** | **🔴 does not exist** | only `french_24l` published |
| Our multilingual quant | **[CREATE]** | HF org `idle-intelligence` has only `pocket-tts-int8` / `pocket-tts-gguf`, both tagged `en` |
| Re-hosting licence | **[CHECK]** | Kyutai repos are gated. We already re-host English, so precedent exists — confirm it extends. |

Our English `tokenizer.model` (59,339 B, sha256 `d461765a…`) came from the **repo-root** path. Upstream's configs now point English at `languages/english/tokenizer.model@d29db797…` while their `DEFAULT_TOKENIZER_PATH` constant still points at the root `@d4fdd22a…`. Two paths, two pins. **[CHECK]** whether they're byte-identical — if they diverged, our English demo runs a different tokenizer than upstream's current English config specifies.

---

## 3. The tokenizer decision

To give Pocket-TTS a real `--text` CLI and enable per-language native testing, we need SentencePiece unigram encoding in Rust. Today there is **none** — `crates/tts-core/examples/tts_generate.rs:256` has 15 hardcoded token IDs pasted from a one-off Python run, so the native binary speaks exactly one hardcoded English sentence.

**Decision: port the existing ~145-line JS Viterbi (`web/worker.js:46-191`) to Rust.** Rejected alternative: the `tokenizers` crate.

Why:
- **Artifact story.** The port consumes `tokenizer.model` directly — the exact file upstream ships and we already host. The `tokenizers` crate needs `tokenizer.model` → `tokenizer.json` conversion, i.e. a derived artifact per language: six more things to generate, host, and keep in sync, right as the language count multiplies.
- **Repo convention.** There is no repo-wide tokenizer standard — each model consumes whatever its upstream ships. TADA used the `tokenizers` crate because Llama's upstream artifact *is* `tokenizer.json`. Kitten shells to espeak-ng. Pocket's artifact is `tokenizer.model`.
- **Size.** `crates/tts-wasm` currently has no serde/serde_json/regex — the leanest wasm crate. `tokenizers` drags in serde_json for a job 145 lines already does.
- **Unification.** Only the port lets us **delete the JS tokenizer** and have one implementation shared by browser and native.
- It's our own code — no licensing or dependency question.

**🔴 The real risk, and the one genuine argument for the other route:** the `.model` file carries a `NormalizerSpec` (precompiled NFKC-ish) that our JS **ignores** — `web/worker.js:149` only does the `▁` substitution. Invisible in ASCII English, but French with *decomposed* Unicode (`e` + combining acute vs precomposed `é`) misses the vocab and hits byte-fallback, producing wrong tokens and mispronounced audio. The `tokenizers` crate gets `Precompiled` normalization free. Mitigation: NFC-normalize input deliberately, and let golden tests arbitrate.

**🔴 Possible existing divergence, in English, today.** `prepare_text_prompt()` prepends 8 spaces for short text (`crates/tts-core/src/tts_model.rs:151`), which our JS encodes as 8 separate `▁` tokens. SentencePiece's default `remove_extra_whitespaces=true` would collapse them. If upstream collapses and we don't, short English utterances have been getting different conditioning than Kyutai's reference all along. **Unverified — the golden tests settle it.**

Both risks are answered by the same oracle: encode a fixture corpus through Python `sentencepiece` and through our implementation, assert identical ID sequences, per language, **including a short-text case**.

---

## 4. Ordered task list

Sequencing note: the tokenizer work is not infrastructure *preceding* the multilingual work — it is the **first work item of it**. Per-language tokenization is the highest-risk unknown, and the native CLI is the cheapest place to validate each checkpoint (quantize → run → listen; no browser, no worker, no cache plumbing).

### Step 1 — Rust tokenizer + golden tests
- [ ] **1.1 [CREATE]** Port `decodeSentencepieceModel` + `UnigramTokenizer` (`web/worker.js:46-191`) into a new `crates/tts-core/src/tokenizer.rs`. Protobuf varint parse + Viterbi. No new crates.
- [ ] **1.2 [CREATE]** NFC-normalize input (see §3 risk).
- [ ] **1.3 [CREATE]** Golden tests vs Python `sentencepiece` for EN + all 5 languages. Fixtures generated once by a script under `scripts/pocket-tts/` (venv). **Must include a short-text case** to settle the 8-space question.
- [ ] **1.4 [CHECK]** Assert `max(token_id) < 4001` on load — cheap guard, upstream enforces vocab==4000 so this should never fire.

### Step 2 — Real Pocket CLI
- [ ] **2.1 [CREATE]** Bring `crates/tts-core/examples/tts_generate.rs` up to its siblings' standard: `--model --tokenizer --voice --text --output`. Drop the hardcoded IDs at :256. Match the arg style of `kitten_generate.rs` / `tada_generate.rs` (hand-rolled parsing, no clap).
- [ ] Keep it an `example`, **not** a `[[bin]]` — CLAUDE.md documents `cargo run --example` as the convention for the other models.

### Step 3 — Quantize + host, per language
- [ ] **3.1 [CHECK]** HF gated-repo access (`hf auth whoami`; terms accepted on `kyutai/pocket-tts-without-voice-cloning`). Owner believes this is already done.
- [ ] **3.2 [CREATE]** Add `--subfolder` to `scripts/pocket-tts/quantize_to_gguf.py`. Today `--model-id` calls `hf_hub_download(repo_id, filename)` trying only root-level `tts_b6369a24.safetensors` / `model.safetensors` (`:325-335`) — it cannot reach `languages/<lang>/model.safetensors`.
- [ ] **3.3 [CHECK]** Verify `remap_key()` and the quantize allowlist still cover the per-language tensor names. Identical architecture, so expect a clean pass — assert rather than assume. For `french_24l`, check nothing hardcodes the 6-layer count.
- [ ] **3.4 [CREATE]** Produce `pocket-tts-{de,es,pt,it}-q8_0.gguf` with `--no-encoder`. Validate (`--validate` + SQNR spot-check against English's 37.2 dB worst-layer figure in `docs/pocket-tts/QUANTIZATION.md`).
- [ ] **3.5 [CREATE]** Upload to the **existing** `idle-intelligence/pocket-tts-gguf` under a `languages/<lang>/` prefix mirroring upstream — one repo, one model card, trivial URL builder. Co-host each `tokenizer.model` alongside its checkpoint; they are a matched pair and a mismatch yields plausible-sounding garbage, not an error.
- [ ] **3.6 [CREATE]** Update the model card: language tags, base-model links, quant method, sizes, credit to Kyutai.

### Step 4 — Native test per language
- [ ] **4.1 [CREATE]** Render a sentence per language via the Step-2 CLI and **listen**. Iterate here, not in the browser. (Audio metrics are unreliable — per CLAUDE.md, the owner listens.)
- [ ] **4.2 [CREATE]** Log RTF per language natively (Metal) against the English 1.97× median. Record in the research log.

### Step 5 — Unify on the Rust tokenizer
- [ ] **5.1 [CREATE]** Expose `tokenize()` from `crates/tts-wasm`, call it from `web/worker.js`, and **delete the JS tokenizer** (`web/worker.js:46-191`). Do this *before* the browser goes multilingual, so there's one implementation before it's exercised in six languages.
- [ ] **5.2 [CHECK]** Regression: English demo unchanged.

### Step 6 — Browser multilingual
- [ ] **6.1 [EXISTS]** `web/worker.js` already honours `config.tokenizerUrl` and `config.modelUrl` overrides (`:206`, `:213`) — no structural worker change needed.
- [ ] **6.2 [CREATE]** Parameterize the voice URL. `web/worker.js:231` hardcodes `${VOICE_BASE}/embeddings_v2/${name}.safetensors`; add a `voiceBaseUrl` config field defaulting to today's value, pass `.../languages/<lang>/embeddings` for non-English.
- [ ] **6.3 [CREATE]** Language selector. Extend `makePocketClient()` to take a language and build the three URLs; add the control near the pocket-tts radio; make the `VOICES` array a per-language map. **Note:** the view is already shared — `updateUIForModel()` toggles visibility, it does not rebuild the DOM — so this lands inside the existing `voiceSection` without touching Kitten.
- [ ] **6.4 [CREATE]** Bump the Cache API name (`caches.open('tts-model-v3')`, `web/worker.js:16`) or key per language so switching doesn't collide. Verify a language switch tears down and reloads cleanly.
- [ ] **6.5 [CREATE]** Per-language default test phrase (canonical English is "Hello, this is a test of the text to speech system").
- [ ] **6.6 [CHECK]** E2E. `scripts/test_demo_e2e.mjs` was **removed with the TADA cleanup** and currently lives only on `feat/tada-burn-wgpu`; it also imports Playwright from the hardcoded absolute path `/Users/tc/node_modules/playwright/index.mjs`, which is not installed. Decide whether to restore + fix it or test manually.

### Step 7 — Ship
- [ ] **7.1 [EXISTS]** `wasm-pack build crates/tts-wasm --target web --release`. **One binary serves all languages** — weights are fetched at runtime.
- [ ] **7.2 [CREATE]** Deploy: rebuild `pkg/`, commit `pkg/` + `web/` onto the orphan `gh-pages` branch (manual; no `.github/` workflow). **⚠️ trucs.ai vendors from `gh-pages` (`pkg/`, `kitten-pkg/`, `web/`)** — any deploy changes what it serves.
- [ ] **7.3 [CREATE]** Update `README.md` / `CLAUDE.md` with the supported language list.
- [ ] **7.4 [CREATE]** Reply to issue #2, credit @ldenoue's port (ldenoue.github.io/xn-ptts), link the demo.

---

## 5. Blockers & open questions

1. **🔴 French size.** Only `french_24l` (~370 MB Q8_0), ~3× the English download. Options: (a) ship with a size warning; (b) wait for Kyutai's distilled French; (c) more aggressive quant (Q4_K_M) — overlaps the existing backlog item. **Recommendation: (b)+(c) — ship the four small languages first, treat French as its own follow-up.** Decide after Step 4 produces native French audio to judge. Note French is the language the requester led with.
2. **🔴 Unicode normalization** (§3) — the highest-risk correctness unknown.
3. **🔴 The 8-space / `remove_extra_whitespaces` question** (§3) — may already affect English.
4. **[CHECK]** Re-hosting licence for derived quants from a gated upstream repo.
5. **[CHECK]** English root vs `languages/english/` tokenizer byte-identity (§2).
6. **[CHECK]** `_24l` variants for the other four languages — upstream calls them higher-quality but slower. Out of scope for v1.

## 6. Effort shape

- **DE/ES/PT/IT:** one Python flag, four quantize+upload runs, three JS wiring changes, one Rust tokenizer, zero new crates, one shared WASM binary.
- **French:** genuinely more — new config factory for 24 layers, variant plumbing, and an unresolved size problem.
- **Critical path:** the Rust tokenizer. It gates native testing, which gates per-language validation.

---

## Appendix — repo state as of 2026-07-25

- `main` = `ef7cb0f`, pushed. TADA fully removed (93 files, −27.5k lines); deps 536→203; `cargo check --workspace` ~16s; test suite green.
- TADA parked on `feat/tada-burn-wgpu` (= `68cf170`, also `convergence/tada-wasm-perf`) on origin.
- `gh-pages` = `df4713e` (2026-03-23), pre-cleanup, still what trucs.ai serves. `origin/main` used to be byte-identical to it; that is no longer true.
- The `test_segment_mask_v2_simple` failure that used to break `cargo test --workspace` was TADA's and is gone with it. It was never a regression — it failed identically on the old `origin/main`.
- Pocket TTS gained a WAV download link in the demo (`web/index.html`, `saveWavLink`).
