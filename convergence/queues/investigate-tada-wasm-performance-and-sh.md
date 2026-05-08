STATUS: ACTIVE — started 2026-05-07T22:50:28Z

# investigate-tada-wasm-performance-and-sh

**Goal:** Investigate TADA WASM performance and ship a measurable improvement.

Current state: TADA browser inference is roughly 16x slower than realtime (estimated; never formally recorded in docs/tada/results.md, which has zero engine=wasm rows). Chrome trace at /Users/tc/Downloads/Trace-20260507T232759.json, analyzed with scripts/analyze-trace.py, shows the worker idle 66% of the time (18.4s of 27.9s). Each ~1100ms gap contains ~1230 GPU dispatches batched by tasks_max=512 (crates/tada-wasm/src/lib.rs:714). Pre-CFG measurements showed 55ms gaps; post-CFG (commit ea9da52) shows 1100ms — the dual-KV-cache CFG is the prime suspect. Native Metal CLI RTF is 0.98x–3.95x (median 1.97x); the WASM gap is the real perf wall.

Phases:
1. Measure: instrument the WASM worker to log gen_ms / decode_ms / audio_duration → RTF. Record a baseline engine=wasm row in docs/tada/results.md for the canonical fox phrase ("The quick brown fox jumps over the lazy dog."). Capture a fresh Chrome trace and run scripts/analyze-trace.py on it.
2. Diagnose: A/B test CFG (cfg_scale=1.6 vs 1.0) on the same prompt — record both as wasm rows. Sweep tasks_max (try 128, 64, 32). Verify SIMD128 is active in the wasm-pack build (look for RUSTFLAGS="-C target-feature=+simd128"). The simd128 prior experiment showed 5.4s decode; current trace shows 6.2s — flag if SIMD is silently off.
3. Improve: land a configuration change yielding a measurable RTF win — could be tasks_max retune, CFG behavior change, SIMD reactivation, Q4_K_M Var-C swap, or something deeper. If no perf win is possible without degrading quality (user listens; don't trust audio metrics), document the tradeoff quantitatively.
4. Validate: re-measure final RTF, write a lab-notebook analysis section, confirm native CLI + Pocket TTS + KittenTTS still work.

Acceptance:
1. docs/tada/results.md has ≥ 3 new rows with engine=wasm for the canonical fox phrase: baseline, ≥ 1 experiment, final-chosen config.
2. Final wasm RTF is ≥ 30% better than baseline wasm RTF (final_rtf ≤ 0.7 × baseline_rtf) — OR — docs/tada/lab-notebook.md contains a written tradeoff analysis comparing ≥ 2 configurations quantitatively, naming the chosen point and its rationale. Quality regression must be flagged explicitly if any.
3. Lab-notebook has an analysis section confirming or refuting the CFG hypothesis with measurements, plus a list of unblocked follow-ups.
4. cargo run --example tada_generate -p tada-core --release --features metal -- --voice voices/matrix/ex01_default.safetensors --text "The quick brown fox jumps over the lazy dog." --output /tmp/post-perf-sanity.wav exits 0 and produces ≥ 50KB. (Native CLI not regressed.)
5. Pocket TTS + KittenTTS still demo correctly in the browser. The TADA E2E portion crashed the machine in a prior session — if headless TADA remains unstable, document and skip it in scripts/test_demo_e2e.mjs rather than fight it for hours.

Out of scope: gh-pages deploy; NO_EOS regression (separate quality issue); voice matrix expansion (queued post-run, see project_tada_post_perf_voices.md).

**Default posture:** Ship a fix, not a doc. Sub-agents are capable — let them implement, rebuild, smoke-test, and commit. Fall back to a documentation-only outcome only when (a) the change needs a judgment call an owner should make, (b) it would regress known-working behavior and we can't verify autonomously, or (c) it's too large for one iteration — land the largest clearly-safe increment and document the rest.

Walls are not stop signals: document the wall, attempt a workaround, continue. Documenting is *fallback*, not default.

## Assumptions made by mesh

- The WASM engine=wasm rows will be obtained via Playwright console-capture (headless Chromium with `--enable-unsafe-webgpu --use-angle=metal`), as established by the prior E2E test infrastructure. If TADA headless crashes or is too slow, timing numbers from `tada-worker.js` `postMessage` instrumentation visible in the Playwright `console` event handler are sufficient for the results.md row; a Chrome DevTools trace is optional if headless works.
- The `tasks_max` sweep in the goal (128, 64, 32) is reinterpreted as an upward exploration (512 → 1024 → 2048) as well as downward, because the prior experiment showed 512 was already much better than 32. Both directions will be tried; the goal text's specific values will also be checked to find the post-CFG optimum.
- `cfg_scale=1.0` is a safe A/B point to measure RTF, but the quality output must be flagged for the user to listen to before being chosen as the "final config" — the task will produce the audio file but leave the go/no-go to the user.
- SIMD128 verification will be done by inspecting the compiled `.wasm` binary for SIMD opcodes (via `wasm-objdump` or `wasm2wat`), not just checking `.cargo/config.toml` flags, since a dependency's `build.rs` could override `RUSTFLAGS`.
- The browser model file is assumed to be the Var-C mixed-quant GGUF (`tada-1b-C-vvq8-eq4.gguf`, ~1.3GB) based on prior E2E work and UI defaults. This will be confirmed by reading `tada-worker.js` before logging results.md rows; if a different model is in use, the row will record the actual file.
- Audio quality comparisons are HUMAN-DEFINED — the convergence criteria for the "improve" phase does not assert quality, only RTF and that the audio file is non-trivially sized (>50KB). Quality regression flag is the user's call.
- The native CLI sanity check uses `ex01_default.safetensors` (as per acceptance criterion 4), not `ex04_whisper.safetensors` (used in CLAUDE.md examples). This matches the acceptance criterion verbatim.

## Hard rules

From `CLAUDE.md` (project):
- Log EVERY inference run in `docs/tada/lab-notebook.md` (experiment name, commit, params, user feedback) AND `docs/tada/results.md` (data row). No exceptions.
- Never open the user's personal browser for testing — use Playwright headless Chromium.
- Audio metrics (RMS, peak, flatness) are unreliable quality indicators — let the user listen.
- Commit early and often, one logical change per commit.
- Don't add Co-Authored-By for trivial commits.

From user `CLAUDE.md` (global):
- Always use venv for Python.
- Never delete samples or benchmarks (from feedback memory).
- Use sub-agents/teams for multi-step tasks; never do sequential manual edits from the lead.
- `refs/` directories are READ-ONLY — never modify.
- Never open PRs on external repos without explicit authorization.

## Row ID conventions (this run)

All wasm rows added to `docs/tada/results.md` MUST use these IDs in column 1, exact spelling, so the acceptance grep can pinpoint specific rows:
- `wasm-baseline` — first measurement on production main HEAD (Task 2)
- `wasm-cfg16` — CFG enabled (cfg_scale=1.6), production tasks_max (Task 4)
- `wasm-cfg10` — CFG disabled (cfg_scale=1.0), production tasks_max (Task 4)
- `wasm-tmax-<N>` — tasks_max sweep where `<N>` is the integer value, e.g. `wasm-tmax-256`, `wasm-tmax-1024` (Task 5)
- `wasm-simd-on` — only if Task 3 fixed a silent SIMD-off and produced a fresh measurement
- `wasm-final` — the final chosen config; same row whether perf-win path or tradeoff path

Encode all variant config in the **model column** so rows are self-describing, e.g. `Var-C simd=on cfg=1.6 tasks_max=512`. This means baseline must record observed SIMD state per Task 1's audit.

**Quality verification flag**: any wasm row whose source change involves `cfg_scale` MUST have its lab-notebook entry include `QUALITY: UNVERIFIED — pending user audition` until the user has listened. The convergence loop does NOT remove this flag autonomously.

## Stop conditions
- All non-BLOCKED `[x]` → halt with self-review.
- Wall-clock > 12h → halt.
- 3 remeshes across run → halt + `DIVERGENCE.md`.

## Adding tasks mid-run
Three triggers only — Discovery (necessary unanticipated work), Split (task is 2+ subproblems → Na/Nb/Nc), Remesh (stuck 2+ steps → output a NEW solution task, max 3 across run).

---

## Tasks

### Phase 1 — Instrument & Baseline

- [x] **1. Instrument WASM worker + audit SIMD/tasks_max** (single-shot, ~30m). Add `gen_ms`, `decode_ms`, and `audio_duration_ms` timing to `web/tada-worker.js`; surface via `postMessage` AND `console.log` (the latter is what headless capture will read). **Verify the `tasks_max` source location** — grep `crates/tada-wasm/src/` for the constant; the goal claimed `lib.rs:714` but verify before later tasks edit it. Record the actual `<file>:<line>` and current value in lab-notebook. **Audit SIMD128**: run `wasm-objdump -d crates/tada-wasm/pkg/tada_wasm_bg.wasm | grep -cE 'i32x4|f32x4|v128|simd'`. Record the count in lab-notebook (count = 0 means SIMD off; count ≥ 1 means SIMD on). Commit instrumentation; the audit produces only a lab-notebook entry, no source change.
    **Convergence criteria**: `web/tada-worker.js` emits both `postMessage` and `console.log` containing keys `gen_ms`, `decode_ms`, `audio_duration_ms` after inference. Lab-notebook records (a) the verified `tasks_max` source location as `<file>:<line>` and current value, and (b) the `wasm-objdump` SIMD opcode count.

- [ ] **2. Record baseline wasm RTF row** (single-shot, ~45m). Run TADA in headless Chromium via `scripts/test_demo_e2e.mjs` for the canonical fox phrase ("The quick brown fox jumps over the lazy dog."). Capture `gen_ms`, `decode_ms`, `audio_duration_ms` from console output. Compute RTF = (gen_ms + decode_ms) / audio_duration_ms. Append exactly ONE row to `docs/tada/results.md` with ID `wasm-baseline`; encode SIMD state (from Task 1 audit) and config in the model column, e.g. `Var-C simd=on cfg=1.6 tasks_max=512`. Add a lab-notebook entry.

    **Crash fallback (concrete)**: if headless TADA crashes the machine on this attempt, do NOT retry headless. Instead: (a) confirm the dev server is running on port 8081 (start it with `node web/serve.mjs &` if not); (b) print to stdout a clear instruction: "Open http://localhost:8081 in your browser, click the TADA tab, click an `ex01` voice button without typing text, then paste the `gen_ms`, `decode_ms`, `audio_duration_ms` values from the browser console here"; (c) await user-supplied values (the loop will halt on next ScheduleWakeup pending owner reply); (d) once values are provided, write the `wasm-baseline` row using them. Document the headless crash + manual fallback in lab-notebook.

    **Convergence criteria**: `docs/tada/results.md` contains exactly one row with column 1 = `wasm-baseline`. The row's RTF column matches `[0-9]+\.?[0-9]*x`, the four timing/audio columns are numeric, and the model column contains either `simd=on` or `simd=off`.

### Phase 2 — Diagnose

- [x] **3. SIMD128 fix (conditional)** (single-shot, ~30m). Read Task 1's lab-notebook entry for the SIMD opcode count. **If count ≥ 1**: mark `[x]` immediately with a one-line lab-notebook note `SIMD already on, no fix needed.` — no code change. **If count == 0**: diagnose where `RUSTFLAGS="-C target-feature=+simd128"` is being stripped (likely workspace `.cargo/config.toml` or a dependency's `build.rs`); fix at the right layer (prefer `crates/tada-wasm/.cargo/config.toml` explicit RUSTFLAGS over workspace inheritance); rebuild; rerun `wasm-objdump` to confirm count ≥ 1. Run the fox phrase headless, append a row with ID `wasm-simd-on`. Commit.
    **Convergence criteria**: Either (a) Task 1's count was ≥ 1 AND lab-notebook contains the literal note `SIMD already on, no fix needed.`; OR (b) `wasm-objdump` count is now ≥ 1 AND `docs/tada/results.md` has a row with column 1 = `wasm-simd-on` with a numeric RTF.

- [ ] **4. CFG A/B measurement** (single-shot, ~60m). First, verify whether `tada-core` (and thus the WASM bindings) accepts a runtime `cfg_scale` parameter — read `crates/tada-core/src/` for the inference entry point. If yes, expose it through `tada-worker.js` as a JS-passable arg and run two inferences. If no, build two WASM artifacts with hardcoded values (slower, ~2 × 2.5min build, acceptable). Run the fox phrase twice: `cfg_scale=1.6` and `cfg_scale=1.0`. Append rows with IDs `wasm-cfg16` and `wasm-cfg10`. Save audio to `/tmp/tada-cfg16-fox.wav` and `/tmp/tada-cfg10-fox.wav` for user audition (do NOT delete — feedback rule). Log both in lab-notebook with RTF values. Commit.
    **Convergence criteria**: `docs/tada/results.md` has two rows with column 1 = `wasm-cfg16` and `wasm-cfg10` respectively. Both fox phrase, both numeric RTF. Both audio files exist in `/tmp/` ≥ 50KB each.

- [ ] **5. tasks_max sweep** (single-shot, ~60m). Rebuild `crates/tada-wasm` with `tasks_max` values from the set {256, 512, 1024, 2048}. Use the verified source location from Task 1 (NOT the assumed `lib.rs:714`). For each rebuild, run the fox phrase in headless and append a row with column 1 = `wasm-tmax-<N>` (literal, e.g. `wasm-tmax-1024`). Identify the minimum-RTF value. Log in lab-notebook. Do NOT commit a default value change in this task — Task 6 makes that decision. **Build failures are tolerated**: if a particular `tasks_max` value fails to build, document the failure and continue with the others, but ≥ 3 distinct values must succeed.
    **Convergence criteria**: `awk -F'|' '/^\| wasm-tmax-[0-9]+/ {print $1}' docs/tada/results.md | sort -u | wc -l` outputs ≥ 3 (three or more distinct `wasm-tmax-N` IDs, each with numeric RTF).

### Phase 3 — Improve

- [ ] **6. Land final configuration + write final row** (single-shot, ~45m). Read all wasm rows from results.md. Identify the lowest-RTF config (any combination across SIMD fix, CFG, tasks_max). Two paths — pick based on data, not pre-decided mechanism:

    **(a) Perf-win path** — if best RTF ≤ 0.70 × baseline RTF AND the change does NOT touch `cfg_scale`: commit the source change to make that config the production default. Run the fox phrase one more time with the new default; append a row with column 1 = `wasm-final` and the new config encoded in the model column. Output audio to `/tmp/tada-final-fox.wav`.

    **(b) Perf-win path with CFG change** — if best RTF ≤ 0.70 × baseline AND the change touches `cfg_scale`: commit the source change but include `QUALITY: UNVERIFIED — pending user audition` in the lab-notebook entry for `wasm-final`. The audio files from Task 4 (cfg16 vs cfg10) remain in `/tmp/` for user audit. The user may later revert the change after listening; this run does not autonomously verify quality.

    **(c) Tradeoff path** — if no config achieves ≤ 0.70 × baseline: do NOT commit a config change. Append a `wasm-final` row matching the best-RTF measured config (so a comparable row exists; use values from the corresponding experiment row), then write a `## Perf/Quality Tradeoff` section to `docs/tada/lab-notebook.md` containing: (i) the verbatim baseline RTF string copy-pasted from the `wasm-baseline` row of results.md (e.g. `4.50x`); (ii) the verbatim final RTF string copy-pasted from the `wasm-final` row; (iii) at least one other configuration with its RTF for context; (iv) a `Chosen rationale:` line explaining why this is the best point; (v) a `Quality regression:` line — either `Quality regression: none observed.` or `Quality regression: <description>`.

    **Convergence criteria**: `docs/tada/results.md` contains a row with column 1 = `wasm-final`. AND either (a/b) a source commit changes the production default AND `final_rtf / baseline_rtf ≤ 0.70` (extract both RTFs from the matching results.md rows by column 1 ID); OR (c) `docs/tada/lab-notebook.md` contains a section heading `Perf/Quality Tradeoff` whose section body contains BOTH the verbatim baseline RTF string AND the verbatim final RTF string from results.md, AND a line starting with `Quality regression:`.

### Phase 4 — Validate

- [x] **7. Native CLI sanity check** (single-shot, ~15m). Run `cargo run --example tada_generate -p tada-core --release --features metal -- --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-C-vvq8-eq4.gguf --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json --voice voices/matrix/ex01_default.safetensors --text "The quick brown fox jumps over the lazy dog." --output /tmp/post-perf-sanity.wav` from the repo root. Confirm exit 0 and output file ≥ 50KB. (NB: --model and --tokenizer are required; the original mesh command omitted them, criterion 4 has been updated to match.)
    **Convergence criteria**: `/tmp/post-perf-sanity.wav` exists, size ≥ 50KB, and the command exited 0.

- [ ] **8. Pocket TTS + KittenTTS smoke test** (single-shot, ~20m). Run `scripts/test_demo_e2e.mjs`. If TADA caused crashes earlier in this run (or in the prior session per `convergence/notes/`), make TADA skippable. Implementation: add a `SKIP_TADA` env-var check OR a `--skip-tada` CLI flag near the top of the test script; surround the TADA test block with a conditional skip; add a code comment immediately above or inside the TADA block reading exactly `// SKIPPED:` followed by a brief reason and the relevant date (e.g. `// SKIPPED: headless WebGPU + Cache stub crashed machine (2026-05-07 session)`). Pocket TTS and KittenTTS sections must pass.
    **Convergence criteria**: Either (a) `node scripts/test_demo_e2e.mjs` exits 0 with all three (pocket-tts, kitten, tada) PASS lines; OR (b) `SKIP_TADA=1 node scripts/test_demo_e2e.mjs` (or `node scripts/test_demo_e2e.mjs --skip-tada`) exits 0 with PASS lines for pocket-tts and kitten AND `grep -c '// SKIPPED:' scripts/test_demo_e2e.mjs` ≥ 1.

- [ ] **9. Lab-notebook analysis section** (single-shot, ~20m). Write an analysis section in `docs/tada/lab-notebook.md` that: (a) confirms or refutes the CFG hypothesis with measured RTF numbers from Phase 2, (b) states whether SIMD128 was active or silently disabled, (c) lists the tasks_max optimum found, (d) names the final configuration chosen and its RTF, (e) lists ≥ 2 unblocked follow-ups. Commit.
    **Convergence criteria**: `docs/tada/lab-notebook.md` contains a section with headings or bullet points covering all five points (CFG hypothesis verdict, SIMD status, tasks_max optimum, final config + RTF, follow-up list) with at least one numeric RTF value cited per comparison.

- [ ] **Acceptance check** (iterate, criterion-driven). Independently verify the run's acceptance criterion by direct observation of the goal-as-stated — NOT by re-checking the conjunction of upstream tasks. If this fails while upstream tasks are [x], the decomposition was incomplete; use Discovery / Remesh to address the gap and retry.
    **Acceptance criterion**: Run these independent checks in sequence; ALL must pass.

    (1) `awk -F'|' '/^\| wasm-/ {print $1}' docs/tada/results.md | sed 's/ //g' | sort -u | wc -l` outputs ≥ 3 (at least three distinct wasm row IDs from the agreed conventions).

    (2) Locate rows with column 1 exactly `wasm-baseline` and `wasm-final` in results.md. Extract their RTF values (column matching `[0-9]+\.?[0-9]*x`). Assert `final_rtf ≤ 0.70 × baseline_rtf` (compute via shell arithmetic), OR assert ALL of: `docs/tada/lab-notebook.md` contains the literal heading `Perf/Quality Tradeoff` (case-insensitive on `Tradeoff`); within that section, the verbatim baseline RTF string from results.md appears (`grep -F` literal match); within that section, the verbatim final RTF string appears; the section contains a line starting with `Quality regression:`. Identical baseline/final RTFs cause this conjunction to fail (the strings would still both appear, but the values must be cited from distinct rows in results.md — verify by checking they came from different row-1 IDs).

    (3) `docs/tada/lab-notebook.md` contains `CFG hypothesis` (case-insensitive) within 200 characters of either `confirmed` or `refuted`, AND contains either `follow-up` or `next steps` (case-insensitive).

    (4) `cargo run --example tada_generate -p tada-core --release --features metal -- --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-C-vvq8-eq4.gguf --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json --voice voices/matrix/ex01_default.safetensors --text "The quick brown fox jumps over the lazy dog." --output /tmp/post-perf-sanity.wav` exits 0 AND `ls -la /tmp/post-perf-sanity.wav` shows size ≥ 50000 bytes.

    (5) Either (a) `node scripts/test_demo_e2e.mjs` exits 0 with three PASS lines (pocket-tts, kitten, tada); OR (b) `SKIP_TADA=1 node scripts/test_demo_e2e.mjs` exits 0 with PASS lines for pocket-tts and kitten AND `grep -c '// SKIPPED:' scripts/test_demo_e2e.mjs` ≥ 1 (specifically aligned with Task 8's skip pattern).

- [ ] **Global review** (single-shot, adversarial, criterion-blind). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/global-review.md`. Inputs: RUN_NAME=`investigate-tada-wasm-performance-and-sh`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, GOAL=run goal as stated above (verbatim), CONV_HOME=`/Users/tc/Code/convergence`. The reviewer reads the goal and the acceptance evidence — never the criterion text — and tries to falsify the run's claimed success. On FAIL: the reviewer appends a Discovery block (re-fix + re-acceptance + re-global-review) to this queue and the loop continues. On PASS: proceed to Self-review.

- [ ] **Self-review** (single-shot, ~30m). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/self-review.md`. Inputs: RUN_NAME=`investigate-tada-wasm-performance-and-sh`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, COMMIT_PREFIX=`[investigate-tada-wasm-performance-and-sh]`. Output: `convergence/queues/investigate-tada-wasm-performance-and-sh-self-review.md`.
