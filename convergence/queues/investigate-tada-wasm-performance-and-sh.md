STATUS: DRAFT — pending owner review

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

## Stop conditions
- All non-BLOCKED `[x]` → halt with self-review.
- Wall-clock > 12h → halt.
- 3 remeshes across run → halt + `DIVERGENCE.md`.

## Adding tasks mid-run
Three triggers only — Discovery (necessary unanticipated work), Split (task is 2+ subproblems → Na/Nb/Nc), Remesh (stuck 2+ steps → output a NEW solution task, max 3 across run).

---

## Tasks

### Phase 1 — Instrument & Baseline

- [ ] **1. Instrument WASM worker for timing** (single-shot, ~30m). Add `gen_ms`, `decode_ms`, and `audio_duration_ms` timing to `web/tada-worker.js` and surface them via `postMessage` to the page (or log via `console.log`). Confirm the build target is `crates/tada-wasm` and that `tasks_max=512` is the current value in `crates/tada-wasm/src/lib.rs:714`. Verify SIMD128 is active by inspecting the compiled `.wasm` for SIMD opcodes (`wasm-objdump -d` or `wasm2wat`). Commit the instrumentation with message `[investigate-tada-wasm-performance-and-sh] instrument wasm worker for timing`.
    **Convergence criteria**: `web/tada-worker.js` emits a `postMessage` or `console.log` with keys `gen_ms`, `decode_ms`, `audio_duration_ms` after each inference, AND `wasm-objdump`/`wasm2wat` output on the built `.wasm` confirms SIMD opcodes are present (or explicitly records that they are absent).

- [ ] **2. Record baseline wasm RTF row** (single-shot, ~45m). Run TADA in headless Chromium via `scripts/test_demo_e2e.mjs` (with existing Cache stub) for the canonical fox phrase. Capture `gen_ms`, `decode_ms`, `audio_duration_ms` from console output. Compute RTF = (gen_ms + decode_ms) / audio_duration_ms. Append one row to `docs/tada/results.md` with `engine=wasm`, model=Var-C (confirm filename from `tada-worker.js`), all measured fields. Add a lab-notebook entry in `docs/tada/lab-notebook.md`. If headless TADA crashes the machine, document the failure in the lab-notebook, skip this step's convergence criteria, and add a Discovery task for a manual-timing fallback.
    **Convergence criteria**: `docs/tada/results.md` contains ≥ 1 row with `engine=wasm` and the fox phrase, with numeric values in `gen_ms`, `decode_ms`, `audio_duration_ms`, and `RTF` columns.

### Phase 2 — Diagnose

- [ ] **3. SIMD128 audit and fix if off** (single-shot, ~30m). Based on the wasm-objdump result from Task 1: if SIMD128 opcodes are confirmed present, mark done immediately. If absent, diagnose which crate's `build.rs` is stripping `RUSTFLAGS`, fix it (likely adding `RUSTFLAGS` to `crates/tada-wasm/.cargo/config.toml` explicitly rather than relying on workspace inheritance), rebuild, and confirm SIMD opcodes are now present. Record finding in lab-notebook. Commit any fix.
    **Convergence criteria**: `wasm-objdump -d` (or `wasm2wat`) on the production `tada_wasm_bg.wasm` shows at least one SIMD instruction (e.g. `i32x4`, `f32x4`, `v128`), OR lab-notebook records "SIMD confirmed present, no fix needed."

- [ ] **4. CFG A/B measurement** (single-shot, ~60m). Expose `cfg_scale` as a runtime JS parameter in `tada-worker.js` (or hardcode two separate test builds). Run the fox phrase twice: once with `cfg_scale=1.6` (baseline), once with `cfg_scale=1.0` (CFG disabled). Record both as `engine=wasm` rows in `docs/tada/results.md`. Save both output audio files to `/tmp/tada-cfg16-fox.wav` and `/tmp/tada-cfg10-fox.wav` for the user to listen to. Log both in lab-notebook with RTF values. Commit.
    **Convergence criteria**: `docs/tada/results.md` has 2 rows for the fox phrase with `engine=wasm`, one with `cfg_scale=1.6` and one with `cfg_scale=1.0`, both with numeric RTF values.

- [ ] **5. tasks_max sweep** (single-shot, ~45m). Rebuild `crates/tada-wasm` with `tasks_max` values of 256, 512, 1024, and 2048 (edit `crates/tada-wasm/src/lib.rs:714`). For each, run the fox phrase in headless and record RTF in `docs/tada/results.md`. Identify the minimum-RTF value. Log in lab-notebook. Commit the sweep results (keep the best value in the source as the new default, or leave at 512 if no improvement found).
    **Convergence criteria**: `docs/tada/results.md` has ≥ 4 rows with `engine=wasm` for the fox phrase with different `tasks_max` values (256, 512, 1024, 2048), all with numeric RTF.

### Phase 3 — Improve

- [ ] **6. Land best configuration** (single-shot, ~30m). From Phase 2 findings, select the configuration with best RTF that the user can evaluate for quality. If CFG disable (`cfg_scale=1.0`) wins RTF by ≥30% and audio quality is not obviously degraded (>50KB file, non-silent), set it as a default option in the UI or worker and commit. If tasks_max tuning wins ≥30%, commit the new value. If neither wins 30% alone but together they do, commit both. If no combination reaches 30% RTF reduction, write the tradeoff analysis (configurations tested, RTF values, quality notes) to `docs/tada/lab-notebook.md` and commit the best partial improvement found. Output the final audio file to `/tmp/tada-final-fox.wav`.
    **Convergence criteria**: Either (a) the chosen configuration is committed to source with its new default value AND final RTF ≤ 0.7 × baseline RTF, OR (b) `docs/tada/lab-notebook.md` contains a tradeoff section listing ≥ 2 configurations with numeric RTF values, naming the chosen point and its rationale, with any quality regression explicitly flagged.

### Phase 4 — Validate

- [ ] **7. Native CLI sanity check** (single-shot, ~15m). Run `cargo run --example tada_generate -p tada-core --release --features metal -- --voice voices/matrix/ex01_default.safetensors --text "The quick brown fox jumps over the lazy dog." --output /tmp/post-perf-sanity.wav` from the repo root. Confirm exit 0 and output file ≥ 50KB.
    **Convergence criteria**: `/tmp/post-perf-sanity.wav` exists, size ≥ 50KB, and the command exited 0.

- [ ] **8. Pocket TTS + KittenTTS smoke test** (single-shot, ~20m). Run `scripts/test_demo_e2e.mjs` for the Pocket TTS and KittenTTS demo paths only (skip TADA E2E if it caused machine crashes in prior runs — add a `--skip-tada` flag or inline skip, and document the skip in a comment). Confirm both Pocket TTS and KittenTTS tests pass.
    **Convergence criteria**: `scripts/test_demo_e2e.mjs` exits 0 for the Pocket TTS and KittenTTS sections, OR the TADA section is explicitly skipped with a documented comment and the other two sections pass.

- [ ] **9. Lab-notebook analysis section** (single-shot, ~20m). Write an analysis section in `docs/tada/lab-notebook.md` that: (a) confirms or refutes the CFG hypothesis with measured RTF numbers from Phase 2, (b) states whether SIMD128 was active or silently disabled, (c) lists the tasks_max optimum found, (d) names the final configuration chosen and its RTF, (e) lists ≥ 2 unblocked follow-ups. Commit.
    **Convergence criteria**: `docs/tada/lab-notebook.md` contains a section with headings or bullet points covering all five points (CFG hypothesis verdict, SIMD status, tasks_max optimum, final config + RTF, follow-up list) with at least one numeric RTF value cited per comparison.

- [ ] **Acceptance check** (iterate, criterion-driven). Independently verify the run's acceptance criterion by direct observation of the goal-as-stated — NOT by re-checking the conjunction of upstream tasks. If this fails while upstream tasks are [x], the decomposition was incomplete; use Discovery / Remesh to address the gap and retry.
    **Acceptance criterion**: Run the following independent checks in sequence and assert ALL pass: (1) `grep -c "engine=wasm" docs/tada/results.md` returns ≥ 3 (baseline + ≥1 experiment + final config, all fox phrase rows). (2) Compute `final_rtf / baseline_rtf` from the two wasm rows tagged "baseline" and "final" — assert ≤ 0.70 OR assert `docs/tada/lab-notebook.md` contains the string "tradeoff" (case-insensitive) AND at least two numeric RTF values in the same paragraph AND an explicit quality-regression flag ("no regression" or "regression: ..."). (3) `docs/tada/lab-notebook.md` contains "CFG hypothesis" (or "CFG" within 200 chars of "confirmed" or "refuted") AND contains "follow-up" or "next steps". (4) `ls -la /tmp/post-perf-sanity.wav` shows size ≥ 50000 bytes. (5) `scripts/test_demo_e2e.mjs` Pocket TTS and KittenTTS sections pass (TADA section may be skipped with documented comment).

- [ ] **Global review** (single-shot, adversarial, criterion-blind). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/global-review.md`. Inputs: RUN_NAME=`investigate-tada-wasm-performance-and-sh`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, GOAL=run goal as stated above (verbatim), CONV_HOME=`/Users/tc/Code/convergence`. The reviewer reads the goal and the acceptance evidence — never the criterion text — and tries to falsify the run's claimed success. On FAIL: the reviewer appends a Discovery block (re-fix + re-acceptance + re-global-review) to this queue and the loop continues. On PASS: proceed to Self-review.

- [ ] **Self-review** (single-shot, ~30m). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/convergence/prompts/self-review.md`. Inputs: RUN_NAME=`investigate-tada-wasm-performance-and-sh`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/tts-web`, COMMIT_PREFIX=`[investigate-tada-wasm-performance-and-sh]`. Output: `convergence/queues/investigate-tada-wasm-performance-and-sh-self-review.md`.
