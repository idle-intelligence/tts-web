STATUS: DRAFT — pending owner review

# rebase-feat-burn-wgpu-llama-onto-tts-web

**Goal:** Rebase feat/burn-wgpu-llama onto tts-web main, merge, and ship a benchmark suite + working local demo. Acceptance: native tada_generate works across voice matrix, WASM builds clean, RTF measurements recorded for all 32 voices, local dev server runs the full demo with working voice selector, no pocket-tts / KittenTTS regression on the frontend. No gh-pages deploy.

**Default posture:** Ship a fix, not a doc. Sub-agents are capable — let them implement, rebuild, smoke-test, and commit. Fall back to a documentation-only outcome only when (a) the change needs a judgment call an owner should make, (b) it would regress known-working behavior and we can't verify autonomously, or (c) it's too large for one iteration — land the largest clearly-safe increment and document the rest.

Walls are not stop signals: document the wall, attempt a workaround, continue. Documenting is *fallback*, not default.

## Assumptions made by mesh

- **Voice count is 26, not 32.** The feature branch's `voices/matrix/` contains 26 `.safetensors` files (ex01–ex04 × 7 styles, minus ex04_happy and ex04_default). No additional voices exist on any branch. Benchmarks will be run against all 26 existing voices; "32 voices" in the goal is aspirational and will be documented as a gap (the 6 missing ex04 entries can be precomputed post-merge as a follow-up). If the owner wants the missing 6 generated as part of this run, this task must be added manually before kickoff.
- **All three WASM packages must compile clean** (tts-wasm, kitten-wasm, tada-wasm). The goal says "no pocket-tts / KittenTTS regression," which requires compilation to pass, even though only tada-wasm carries new changes.
- **The benchmark runner will be a Bash script** (not Python) since `tada_generate` is a `cargo run --example` invocation. It will iterate voice files, parse timing from stderr, and append rows to `docs/tada/results.md` and experiment details to `docs/tada/lab-notebook.md`.
- **`serve.mjs` does not currently serve `voices/`** on main (voices/ was only added on the feature branch). A one-line fix to extend the static file root or add a route will be needed. This is a minor edit, not a new file.
- **Rebase, not merge-commit.** The feature branch will be rebased onto main and then merged as a fast-forward, matching the precedent set by the kitten branch merge (`7c81706`).
- **Conflict resolution is the dominant risk.** The 20-commit divergence includes `1b19c26 Switch candle and mimi-rs patches from local paths to git deps`, which likely conflicts with Cargo.toml and Cargo.lock on the feature branch. Phase 1 is dedicated to resolving this before any build work begins.

## Hard rules

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

## Stop conditions
- All non-BLOCKED `[x]` → halt with self-review.
- Wall-clock > 12h → halt.
- 3 remeshes across run → halt + `DIVERGENCE.md`.

## Adding tasks mid-run
Three triggers only — Discovery (necessary unanticipated work), Split (task is 2+ subproblems → Na/Nb/Nc), Remesh (stuck 2+ steps → output a NEW solution task, max 3 across run).

---

## Tasks

### Phase 1 — Rebase + Conflict Resolution

- [ ] **1. Rebase feat/burn-wgpu-llama onto main** (iterate, ~2h). Checkout `feat/burn-wgpu-llama`, rebase onto current `main`. Resolve all conflicts — prioritize main's Cargo dependency changes (`1b19c26`: git deps replacing local paths) over the feature branch's local-path overrides. After rebase, `cargo check -p tada-core --features metal` must succeed on the rebased tip. Commit the resolved rebase as a merge commit to main.
    **Convergence criteria**: `git log --oneline main | head -1` shows the rebase merge commit; `cargo check -p tada-core --features metal 2>&1 | grep -c '^error'` outputs `0`.

- [ ] **2. Verify all three crates build** (single-shot, ~30m). Run `cargo build -p tts-core`, `cargo build -p kitten-core --features wasm`, and `cargo check -p tada-core --features metal`. If any fail, delegate conflict/dependency fix to a sub-agent and retry within this task.
    **Convergence criteria**: All three `cargo build`/`cargo check` invocations exit 0 with no `error[` lines in output.

### Phase 2 — Native Benchmark Sweep

- [ ] **3. Write benchmark runner script** (single-shot, ~30m). Create `scripts/tada/benchmark_voice_matrix.sh`. It must: iterate all `.safetensors` files in `voices/matrix/`, run `cargo run --example tada_generate -p tada-core --release --features metal` with default parameters from CLAUDE.md for each voice, capture wall-clock time, compute RTF = generation_time / audio_duration, append one data row to `docs/tada/results.md`, and append one experiment block to `docs/tada/lab-notebook.md`. Audio output files go to `/tmp/tada_bench/`. Script must be idempotent (skip if output file already exists).
    **Convergence criteria**: `scripts/tada/benchmark_voice_matrix.sh` exists, is executable (`chmod +x`), and `bash -n scripts/tada/benchmark_voice_matrix.sh` exits 0 (syntax check passes).

- [ ] **4. Run benchmark across all 26 voices and log results** (iterate, ~3h). Execute `scripts/tada/benchmark_voice_matrix.sh`. For each of the 26 voices, a successful audio file must be generated and a row appended to `docs/tada/results.md`. If any voice fails, record the failure in the lab notebook and continue (do not abort the sweep). After completion, commit the updated `docs/tada/results.md` and `docs/tada/lab-notebook.md`.
    **Convergence criteria**: `wc -l docs/tada/results.md` shows at least 26 new data rows (header + 26 voice rows); `ls /tmp/tada_bench/*.wav | wc -l` outputs at least 20 (allowing up to 6 voice failures before the criterion demands investigation).

### Phase 3 — WASM Build + Local Demo

- [ ] **5. WASM builds clean for all three packages** (iterate, ~1h). Run `wasm-pack build crates/tts-wasm --target web --release`, `wasm-pack build crates/kitten-wasm --target web --release -- --features wasm`, and `wasm-pack build crates/tada-wasm --target web --release -- --features wasm --no-default-features`. If any fail, delegate the fix to a sub-agent. Repeat until all three exit 0.
    **Convergence criteria**: All three `wasm-pack build` commands exit 0; `ls crates/tts-wasm/pkg/*.wasm crates/kitten-wasm/pkg/*.wasm crates/tada-wasm/pkg/*.wasm` lists three `.wasm` files.

- [ ] **6. Fix dev server to serve voices/ and verify voice selector** (single-shot, ~30m). Inspect `web/serve.mjs` to confirm whether `voices/` is served. If not, add a static route. Then start the dev server (`node web/serve.mjs`), use Playwright headless Chromium to load `http://localhost:8081`, navigate to the TADA tab, and verify the voice selector dropdowns (Speaker + Style grids) are populated and at least one voice option is selectable.
    **Convergence criteria**: Playwright script exits 0 and reports that both Speaker and Style select elements have `options.length > 1`.

- [ ] **7. Smoke test: Pocket TTS and KittenTTS frontend paths** (single-shot, ~30m). Using Playwright headless Chromium on the running dev server, verify: (a) the Pocket TTS tab loads and the Generate button is present and enabled, (b) the KittenTTS tab loads and the Generate button and speed slider are present and enabled. No audio generation needed — UI presence is sufficient for regression check.
    **Convergence criteria**: Playwright script exits 0 and reports Pocket TTS Generate button present; KittenTTS Generate button + speed slider present.

- [ ] **Acceptance check** (iterate, criterion-driven). Independently verify the run's acceptance criterion by direct observation of the goal-as-stated — NOT by re-checking the conjunction of upstream tasks. If this fails while upstream tasks are [x], the decomposition was incomplete; use Discovery / Remesh to address the gap and retry.
    **Acceptance criterion**: Run `cargo run --example tada_generate -p tada-core --release --features metal -- --voice voices/matrix/ex01_default.safetensors --text "The quick brown fox jumps over the lazy dog." --output /tmp/acceptance_check.wav` on the merged main branch; exit code is 0 and `/tmp/acceptance_check.wav` exists and is non-empty. AND `docs/tada/results.md` contains ≥ 20 rows with non-empty RTF values (grep `'^[0-9]'` on data lines). AND `ls crates/tts-wasm/pkg/*.wasm crates/kitten-wasm/pkg/*.wasm crates/tada-wasm/pkg/*.wasm` lists three files. AND Playwright on `http://localhost:8081` reports Speaker and Style selectors have `options.length > 1`, Pocket TTS Generate button present, KittenTTS Generate button + speed slider present.

- [ ] **Global review** (single-shot, adversarial, criterion-blind). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/global-review.md`. Inputs: RUN_NAME=`rebase-feat-burn-wgpu-llama-onto-tts-web`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, GOAL=run goal as stated above (verbatim), CONV_HOME=`/Users/tc/Code/convergence`. The reviewer reads the goal and the acceptance evidence — never the criterion text — and tries to falsify the run's claimed success. On FAIL: the reviewer appends a Discovery block (re-fix + re-acceptance + re-global-review) to this queue and the loop continues. On PASS: proceed to Self-review.

- [ ] **Self-review** (single-shot, ~30m). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/self-review.md`. Inputs: RUN_NAME=`rebase-feat-burn-wgpu-llama-onto-tts-web`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, COMMIT_PREFIX=`[rebase-feat-burn-wgpu-llama-onto-tts-web]`. Output: `convergence/queues/rebase-feat-burn-wgpu-llama-onto-tts-web-self-review.md`.
