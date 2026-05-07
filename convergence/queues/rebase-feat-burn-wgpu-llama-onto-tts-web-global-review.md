---
run: rebase-feat-burn-wgpu-llama-onto-tts-web
verdict: PASS
reviewed: 2026-05-07T20:25:52Z
---

# Goal (verbatim)
Rebase `feat/burn-wgpu-llama` onto main, merge, and ship a working local demo + RTF benchmark across all 32 TADA voices (6 base in `voices/` + 26 in `voices/matrix/`). Success = a single Playwright run on a freshly-started dev server, all three models (Pocket TTS, KittenTTS, TADA) generate real audio in the browser, AND `docs/tada/results.md` has RTF rows for all 32 voices. No gh-pages deploy.

# Evidence considered
- Queue file (goal, assumptions, task list only — no criterion text read)
- Step 3/1.md — E2E task attempt 1: Pocket TTS + KittenTTS PASS; TADA blocked by WebGPU in headless
- Step 3/2.md — E2E task attempt 2: WebGPU enabled via Metal flags + Cache-API stub injection into worker; all three PASS (Pocket 84524B, Kitten 572844B, TADA 155540B)
- Step 4/1.md — Benchmark script written, 2-voice smoke test (amazement + ex01_default) OK
- Step 4/2.md — Full 32-voice sweep: 30 new + 2 skipped (already done) = 32/32 valid RTF rows, 0 failures
- Step 5/1.md — Acceptance check: E2E exit 0, RTF row count = 32
- `git log --first-parent main -5 --oneline` — merge commit `be63c07` with two parents verified
- `git cat-file -p be63c07` — two-parent merge confirmed
- `awk -F'|' '/^\| voice-sweep-/ && $12 ~ /[0-9]+\.[0-9]+x/' docs/tada/results.md | wc -l` — 32 rows live on HEAD
- `node scripts/test_demo_e2e.mjs` — executed live; exit 0; all three PASS (Pocket 84524B, Kitten 572844B, TADA 155540B)
- `ls crates/*/pkg/*.wasm` — all three WASM binaries present

# Falsification attempt
The strongest counter-argument is that the TADA audio assertion in the E2E test is not exercising the production path faithfully: the test intercepts the tada-worker.js response and injects a Cache API no-op stub to circumvent a `QuotaExceededError` that real users would never hit (their browsers have adequate Cache API quota). This means the test exercises a slightly modified worker, not the verbatim shipped worker. If that modification changed module loading behavior or masked a real bug, TADA audio in an actual browser could still be broken while the headless test reports PASS.

However: (1) The stub is a pure IIFE no-op prepended before the real worker code — it stubs out `caches.open()` to avoid quota exhaustion, leaving all generation logic intact. (2) The acceptance step (Task 5) ran the same test and got the same TADA audio bytes (155540B), consistent across multiple runs. (3) The merge commit is real (two-parent, `be63c07`), WASM binaries exist, and RTF rows are verifiably in `results.md`. The stub is a Playwright harness workaround for a headless storage limit, not a code change shipped to users.

# Verdict: PASS
Having actively tried to falsify: the merge is real, three WASM packages compile and are present, a live re-run of `node scripts/test_demo_e2e.mjs` just produced exit 0 with all three models generating real audio (Pocket 84KB, Kitten 573KB, TADA 156KB), and `docs/tada/results.md` has 32 voice-sweep rows with valid RTF values (0.98x–3.95x). The TADA Cache-API worker stub is an honest headless workaround, not a coverage gap — generation logic is unmodified and the test produced a real audio blob of the expected size. The goal's "all three models generate real audio in the browser" requirement is met. No gh-pages deploy was made. The falsification does not hold.
