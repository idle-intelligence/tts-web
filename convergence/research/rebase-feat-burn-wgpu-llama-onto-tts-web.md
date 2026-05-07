# Survey: rebase-feat-burn-wgpu-llama-onto-tts-web

## Goal

Rebase the `feat/burn-wgpu-llama` branch onto current `main`, merge it, and deliver: (a) a benchmark suite that generates audio for all 32 voice-matrix voices via `tada_generate` and records RTF measurements, (b) a working local dev server demo with the TADA voice selector UI functional, and (c) verified no regression on Pocket TTS / KittenTTS frontend paths. No gh-pages deploy.

## Project Context

`tts-web` is a multi-model browser TTS inference engine hosting three models: Pocket TTS (Kyutai, WASM/GGUF), KittenTTS (StyleTTS 2 distilled, WASM), and TADA-1B (HumeAI Llama 3.2 + VibeVoice flow matching + DAC, Burn/wgpu + candle). Lead agent (Opus) orchestrates; all research and code writing is delegated to sub-agents (Sonnet). Commits are atomic, early, and often. No Co-Authored-By on trivial commits. All TADA inference runs must be logged to `docs/tada/lab-notebook.md` (experiment details) and `docs/tada/results.md` (data row). Dev server: `node web/serve.mjs` (port 8081). No personal browser — use Playwright headless only.

## Closest Analogous Work

- **Rebase precedent**: The `kitten` branch was merged into main via `7c81706 Merge kitten branch`. That was a clean fast-forward after divergence; the divergence here is larger.
- **Voice matrix creation**: `82c4c1b Add script to generate all speaker × style voice matrix` and `92539ce Add Expresso voice prompts` — these scripts already generated the 26 safetensors in `voices/matrix/` on the feature branch. The matrix currently has 26 entries (not 32 — see Open Questions).
- **RTF benchmarking pattern**: `docs/tada/results.md` + `docs/tada/lab-notebook.md` are the established log format. See `baece4c Log all missing experiments`.
- **TADA WASM front-end**: feature branch commits `a820c8f Add TADA voice selector UI`, `ff4dde6 Split voice selector into Speaker + Style grids`, `636a7d8 Run neg CFG forward on ALL steps`, `abaad42 Full VV warmup` — all live only on `feat/burn-wgpu-llama`.

## Hard Rules Surfaced

- **Never modify files in `refs/`** — read-only reference material.
- **Never open PRs on external repos without explicit user authorization.**
- **Never open the user's personal browser** — use Playwright headless Chromium for any browser testing.
- **Never skip hooks (`--no-verify`, etc.)** unless user explicitly asks.
- **Log EVERY inference run**: both `docs/tada/lab-notebook.md` (experiment details) and `docs/tada/results.md` (data row). No exceptions.
- **Audio metrics (RMS, peak, flatness) are unreliable** — record RTF timing only; let user listen for quality.
- **Never delete samples or benchmark data** — user keeps all audio for blog posts.
- **Benchmark tables = data only** — analysis/strategy goes in separate docs.
- **Open audio files separately** — one at a time, never batch-open.
- **Separate benchmarks from analysis** — results.md is pure data rows.
- **No gh-pages deploy** — explicitly out of scope.
- **Use sub-agents for all code writing** — lead never does sequential manual edits.

## Prior-Run Carryover

No prior `convergence/notes/*-next.md` files exist in this repo (notes directory is empty). No carryover.

## Open Questions for Mesh

1. **Voice count discrepancy**: The goal says "32 voices" but `voices/matrix/` contains 26 `.safetensors` files on main (ex01–ex04 × 7 styles, minus ex04_happy and ex04_default which are absent). The feature branch adds JSON metadata files but no additional safetensors. Mesh must decide: run benchmarks against the 26 existing voices, or is the 32-voice target aspirational and requires generating missing voices first?

2. **Rebase conflict risk**: The feature branch diverged from `70015d5e` (commit "Fix WASM: async generation, CPU embeddings…"), and main has advanced 20 commits since — including Cargo dependency changes (`1b19c26 Switch candle and mimi-rs patches from local paths to git deps`) which likely conflict with Cargo.lock and possibly Cargo.toml on the feature branch. Mesh should plan a conflict-resolution step before any build verification.

3. **WASM build scope**: The goal says "WASM builds clean." Does this mean `tada-wasm` only, or all three (tts-wasm + kitten-wasm + tada-wasm)? Regression check on Pocket TTS / Kitten suggests all three must compile, but only tada-wasm carries new changes.

4. **Native benchmark runner**: There is no existing shell script that runs `tada_generate` across all 32 (or 26) voices and emits RTF rows. Mesh needs to plan creation of a benchmark script (Bash or Python) that: iterates voices, runs `cargo run --example tada_generate`, parses timing output, and appends rows to `docs/tada/results.md`.

5. **Local demo voice selector data source**: The feature-branch UI (`web/index.html`, `web/tada-worker.js`) uses voice files at specific relative paths. After merge, the voice matrix JSON files need to be reachable from the dev server. Mesh must verify whether `serve.mjs` already serves `voices/` or needs updating.

## Estimated Shape

`multi-phase` — the work spans at least three sequential gates: (1) rebase + conflict resolution + Cargo build verification, (2) native benchmark sweep across all voices with RTF logging, (3) WASM build clean check + local dev server smoke test.
