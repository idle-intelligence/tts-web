STATUS: ACTIVE — started 2026-05-07T19:02:55Z

# rebase-feat-burn-wgpu-llama-onto-tts-web

**Goal:** Rebase `feat/burn-wgpu-llama` onto main, merge, and ship a working local demo + RTF benchmark across all 32 TADA voices (6 base in `voices/` + 26 in `voices/matrix/`). Success = a single Playwright run on a freshly-started dev server, all three models (Pocket TTS, KittenTTS, TADA) generate real audio in the browser, AND `docs/tada/results.md` has RTF rows for all 32 voices. No gh-pages deploy.

**Default posture:** Ship the fix. Sub-agents implement, build, smoke-test, commit. Doc-only fallback only when (a) judgment call needs the owner, (b) cannot verify autonomously, (c) too large for one iteration. Walls are not stop signals — document, work around, continue.

## Assumptions

- **32 voices = 6 base + 26 matrix.** Base voices in `voices/` (tracked, gitignore exception `!voices/*.safetensors`): `amazement`, `amusement`, `ljspeech`, `ljspeech_long`, `rickie`, `rickie_trimmed`. Matrix in `voices/matrix/` (untracked — gitignore exception is top-level only): 4 speakers × 7 styles = 28 design points, 2 missing (ex04_default, ex04_happy), 26 present. Total target: 32 RTF rows.
- **Matrix `.safetensors` are local-only by design.** Gitignored by `*.safetensors`; the `!voices/*.safetensors` exception covers only `voices/`, not `voices/matrix/`. The 26 matrix files exist on the dev machine only. Do not change `.gitignore`. Base voices ARE tracked and reproducible from a fresh clone.
- **Merge commit, not fast-forward.** Kitten precedent `7c81706` is a true merge commit (`Merge: a7c89de 23a2784`). Match it.
- **Cargo dependency conflict is the dominant rebase risk.** Main moved candle/mimi-rs to git deps (`1b19c26`); the feature branch likely has local-path overrides. Take main's git deps unconditionally.
- **"No regression"** = the existing models generate real audio in the browser after merge, not button presence in the DOM. Tested by Playwright actually clicking Generate.
- **Rebase fallback exists.** If rebase becomes intractable (3+ files with non-mechanical conflicts), abort and merge `feat/burn-wgpu-llama` directly into main with conflict resolution on the merge commit. Document choice in commit message.

## Hard rules

- **Never modify files in `refs/`** — read-only reference material.
- **Never open PRs on external repos without explicit user authorization.**
- **Never open the user's personal browser** — Playwright headless Chromium only.
- **Never skip hooks (`--no-verify`, etc.)** unless user explicitly asks.
- **Log every inference run** in `docs/tada/lab-notebook.md` (experiment block) and `docs/tada/results.md` (data row). Skipping a voice on retry because it already succeeded is fine — that is not a new run.
- **Audio metrics (RMS, peak, flatness) are unreliable** — record RTF only; let user listen for quality.
- **Never delete samples or benchmark data.**
- **Benchmark tables = data only.** Analysis goes in separate docs.
- **No gh-pages deploy.**
- **Use sub-agents for code writing** — lead never does sequential manual edits.

## Stop conditions
- All non-BLOCKED tasks `[x]` → halt with self-review.
- Wall-clock > 12h → halt.
- 3 remeshes across run → halt + `DIVERGENCE.md`.

## Adding tasks mid-run
Three triggers — Discovery (necessary unanticipated work), Split (task is 2+ subproblems → Na/Nb/Nc), Remesh (stuck 2+ steps → output a NEW solution task, max 3 across run).

---

## Tasks

### 1. [x] Land the merge with end-to-end sanity (iterate, ~2.5h)

Checkout `feat/burn-wgpu-llama`, rebase onto current main. Resolve all conflicts in favor of main's git-deps (`1b19c26` and successors). After rebase, on the rebased tip:
- `cargo check -p tada-core --features metal` exits 0
- Run `cargo run --example tada_generate -p tada-core --release --features metal -- --voice voices/matrix/ex01_default.safetensors --text "The quick brown fox jumps over the lazy dog." --output /tmp/sanity.wav` — exit 0, output ≥ 50KB.

Only then merge to main: `git checkout main && git merge --no-ff feat/burn-wgpu-llama`.

If rebase becomes intractable, take the documented fallback: abort rebase, merge feature branch directly into main, resolve conflicts on the merge commit.

**Convergence criteria**: `git log --first-parent main -1 --oneline` shows a merge commit (two parents); `cargo check -p tada-core --features metal` exits 0; `/tmp/sanity.wav` exists and is ≥ 50KB.

### 2. [x] Build all three WASM packages (single-shot, ~45m)

Run all three on merged main:
- `wasm-pack build crates/tts-wasm --target web --release`
- `wasm-pack build crates/kitten-wasm --target web --release -- --features wasm`
- `wasm-pack build crates/tada-wasm --target web --release -- --features wasm --no-default-features`

If any fails, delegate fix to a sub-agent and retry within this task. Do not silently disable features to make builds pass.

**Convergence criteria**: All three exit 0; `ls crates/tts-wasm/pkg/*.wasm crates/kitten-wasm/pkg/*.wasm crates/tada-wasm/pkg/*.wasm` lists three `.wasm` files.

### 3. [x] End-to-end demo verification (single-shot, ~1h)

This is the regression test. Probe `web/serve.mjs` first — if it does not already serve `voices/`, add a static route. Otherwise leave it alone.

Write `scripts/test_demo_e2e.mjs` (Playwright headless Chromium):
1. Start `node web/serve.mjs` as a child process; tear it down at end.
2. Load `http://localhost:8081`.
3. **Pocket TTS**: switch to its tab, fill a short text, click Generate, wait for audio blob (or `<audio>` `src` set), assert ≥ 10KB.
4. **KittenTTS**: switch tab, click Generate (default text OK), assert audio blob ≥ 10KB.
5. **TADA**: switch tab, assert speaker selector populated (≥ 4 options) and style selector populated (≥ 7 options), select `ex01_default`, click Generate, assert audio blob ≥ 10KB.

Commit the test script.

**Convergence criteria**: `node scripts/test_demo_e2e.mjs` exits 0 with all three audio assertions passing.

### 4. [x] Benchmark sweep across all 32 voices (iterate, ~2h)

Write `scripts/tada/benchmark_voices.sh` that iterates BOTH `voices/*.safetensors` (6 base) AND `voices/matrix/*.safetensors` (26 matrix) — 32 total. For each, runs `tada_generate` with default CLAUDE.md params, captures wall-clock + audio duration → RTF, appends one row to `docs/tada/results.md` and one experiment block to `docs/tada/lab-notebook.md`. On per-voice failure: log to lab notebook with reproducer command, continue sweep.

Idempotency: skip a voice only if its prior `results.md` row has a non-empty RTF value. Re-run failed voices on retry. Output WAVs go to `/tmp/tada_bench/`.

Commit updated `results.md`, `lab-notebook.md`, and the script.

**Convergence criteria**: `awk -F'|' '/^\| voice-sweep-/ && $12 ~ /[0-9]+\.[0-9]+x/' docs/tada/results.md | wc -l` outputs ≥ 30, AND every failed voice (if any) has a logged reproducer command in `docs/tada/lab-notebook.md`. (≥ 30 of 32 tolerates 2 transient failures; below that, investigate and re-sweep before passing.)

### 5. [x] Acceptance check (criterion-driven)

Independently verify the goal on merged main:
1. `node scripts/test_demo_e2e.mjs` exits 0.
2. `awk -F'|' '/^\| voice-sweep-/ && $12 ~ /[0-9]+\.[0-9]+x/' docs/tada/results.md | wc -l` outputs ≥ 30.

If either fails while upstream tasks are `[x]`, decomposition was incomplete — Discovery or Remesh.

**Acceptance criterion**: Both checks pass on `main` HEAD.

### 6. Global review (single-shot, adversarial, criterion-blind)

Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/global-review.md`. Inputs: RUN_NAME=`rebase-feat-burn-wgpu-llama-onto-tts-web`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, GOAL=run goal as stated above (verbatim), CONV_HOME=`/Users/tc/Code/convergence`. The reviewer reads the goal and the acceptance evidence — never the criterion text — and tries to falsify the run's claimed success. On FAIL: reviewer appends a Discovery block (re-fix + re-acceptance + re-global-review) to this queue and the loop continues. On PASS: proceed to Self-review.

### 7. Self-review (single-shot, ~30m)

Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/self-review.md`. Inputs: RUN_NAME=`rebase-feat-burn-wgpu-llama-onto-tts-web`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, COMMIT_PREFIX=`[rebase-feat-burn-wgpu-llama-onto-tts-web]`. Output: `convergence/queues/rebase-feat-burn-wgpu-llama-onto-tts-web-self-review.md`.
