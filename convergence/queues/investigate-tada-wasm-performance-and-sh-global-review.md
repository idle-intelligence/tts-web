---
run: investigate-tada-wasm-performance-and-sh
verdict: PASS
reviewed: 2026-05-08T19:28:30Z
---

# Goal (verbatim)
Investigate TADA WASM performance and ship a measurable improvement.

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

# Evidence considered
- `convergence/queues/investigate-tada-wasm-performance-and-sh.md` — queue, tasks list, Row ID conventions, hard rules
- `convergence/steps/.../10-acceptance/1.md` — acceptance sub-agent outcome report
- `convergence/steps/.../9-analysis/1.md` — analysis step outcome
- `convergence/steps/.../8-pocket-kitten-smoke/1.md` — E2E smoke test outcome
- `docs/tada/results.md` — all benchmark rows including wasm-* section (8 distinct IDs verified by shell)
- `docs/tada/lab-notebook.md` — full content: Task 4 raw entry (line 1930 CONFIRMED), Analysis Task 9 (line 2135 REFUTED), Perf/Quality Tradeoff section (lines 2072-2094), Root cause and follow-up section (lines 2096-2131), SIMD audit (lines 1876-1886), sub-step profiling sections (lines 1957-2068)
- `/tmp/post-perf-sanity.wav` — confirmed present, 159,382 bytes
- Shell verification: distinct wasm row IDs = 8, `wasm-tmax-*` count = 4, `// SKIPPED:` count in test_demo_e2e.mjs = 0

# Falsification attempt

**Strongest falsification candidate: contradictory CFG hypothesis verdicts in the lab-notebook.**

The lab-notebook contains two different verdicts on the CFG hypothesis:
1. Line 1930 (Task 4 raw log): `CFG hypothesis: CONFIRMED — CFG-off is 3.3% faster`
2. Line 2135 (Task 9 Analysis): `**(a) CFG hypothesis: REFUTED.**`

The goal asks for "an analysis section confirming or refuting the CFG hypothesis." If a stakeholder read the raw Task 4 entry and the canonical Analysis section they would see conflicting signals. This is a real documentation defect. The acceptance check itself acknowledged it: "a stale CONFIRMED wording from Task 4's sub-agent entry persists, but the canonical Task 9 Analysis section gives the corrected verdict."

However: these two entries are measuring different things. Task 4's "CONFIRMED" is a sub-agent's shorthand for "yes, disabling CFG is 3.3% faster on wall-time" — a narrow factual claim. Task 9's "REFUTED" addresses the actual goal-text hypothesis: that CFG dual-KV-cache is "the prime suspect" for the large performance regression (~16-20× slower than realtime). The Analysis section explains the discrepancy correctly — the apparent 40% wall-time gain at cfg=1.0 is illusory because the audio output is 36% shorter (2.04s vs 3.20s), so per-second RTF cost is flat at 6.30x vs 6.49x (3.3% noise). This is the rigorous diagnosis of the root-cause hypothesis, not a contradiction.

The sub-step profiling (Task 5b) further supports the REFUTED verdict: the bottleneck is AR VibeVoice dispatch fragmentation (797 GPU dispatches/step, 354 readback stalls/step), NOT CFG dual-KV-cache overhead. This is backed by a 121.4 MB Chrome trace + `wasm-objdump` SIMD audit + per-step JS instrumentation.

**Secondary falsification: wasm-final RTF = wasm-baseline RTF (both 6.49x) — is the OR-branch tradeoff rigorous?**

The Perf/Quality Tradeoff section does compare ≥2 configurations quantitatively (7 rows in table), names the chosen point ("wasm-final = wasm-baseline"), gives a rationale ("no measured improvement justified landing"), and contains a `Quality regression:` line. The verbatim RTF strings `6.49x` appear in both their respectively labeled positions. The fact that both RTFs are identical is explicitly acknowledged by the run and is correct — no config change was committed.

# Verdict: PASS

The falsification attempt around contradictory CFG verdicts does not hold. The Task 4 "CONFIRMED" line is a sub-agent's imprecise shorthand in a raw measurement log ("yes, cfg=1.0 is 3.3% faster by wall-clock"); the canonical Analysis section (Task 9) correctly identifies this as noise (RTF is flat when controlling for audio duration) and correctly refutes the goal's hypothesis (CFG is not the prime suspect for the large regression). A stakeholder reading the Analysis section gets the load-bearing verdict with quantitative evidence, meeting the goal's requirement for "confirming or refuting the CFG hypothesis with measurements." The sub-step profiling provides independent supporting evidence for REFUTED through dispatch fragmentation analysis. All other acceptance items are met: 8 distinct wasm-* rows (≥3 required), Perf/Quality Tradeoff section with ≥2 configs, Quality regression line, native CLI sanity WAV at 159KB (≥50KB), and full E2E pass for all 3 models. The "QUALITY: UNVERIFIED" flag on the cfg=1.0 audio is correctly preserved rather than autonomously resolved — consistent with the goal's "user listens" hard rule.
