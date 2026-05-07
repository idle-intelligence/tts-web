---
run: rebase-feat-burn-wgpu-llama-onto-tts-web
reviewed: 2026-05-07
verdict: PASS (per global-review)
---

# Self-Review: rebase-feat-burn-wgpu-llama-onto-tts-web

## Summary table

| Repo | OK | CONCERN | REGRESSION |
|------|----|---------|------------|
| tts-web (main) | 20 | 1 | 0 |

## Per-commit verdicts

| SHA | Message | Verdict | Rationale |
|-----|---------|---------|-----------|
| `878503f` | plan: survey + DRAFT queue | OK | Survey + queue initialization; no code changes |
| `1290941` | step 1 stub: kick off Task 1 | OK | `.gitignore` updated, log bootstrapped, queue fleshed out |
| `be63c07` | Merge feat/burn-wgpu-llama: TADA-1B ... | OK | Genuine two-parent merge commit (parents: `1290941`, `3c202db`). No `.wasm` build artifacts committed. `voices/*.safetensors` additions (14 new base voices) are correctly tracked under the `!voices/*.safetensors` gitignore exception. `refs/` untouched. |
| `f8f8fbd` | step 1 OUTCOME: fallback merge, cargo check OK, sanity WAV 186KB | OK | Honest report: acknowledges rebase was aborted after first conflict in `web/index.html`, fallback merge used per plan. Cargo check exit 0 and 186KB WAV documented. |
| `031bf29` | mark Task 1 [x]: merge landed, sanity verified | OK | Queue update only; convergence criteria documented in step report |
| `cd979e8` | step 2 stub: WASM builds for all three packages | OK | Step stub only |
| `3979bf4` | step 2 OUTCOME: all three WASM builds pass, no fixes needed | OK | Accurately reports pass for all three (tts 1.7M, kitten 2.8M, tada 13M); no source fixes needed |
| `8a7d416` | mark Task 2 [x]: all three WASM builds clean | OK | Queue update |
| `c3b63ea` | step 3 stub: E2E demo verification (Playwright) | OK | Step stub only |
| `3d99c2c` | Task 3: E2E Playwright test, Pocket TTS + KittenTTS PASS | OK | Test script committed; honest about TADA being WebGPU-blocked in first attempt; does not false-claim TADA passed |
| `25017c3` | step 3.2 stub: try WebGPU flags for TADA in headless Chromium | OK | Step stub for iterate-task second attempt |
| `49d204b` | Enable WebGPU in headless Chromium for TADA E2E | CONCERN | Cache API worker-interception stub is sound engineering for a headless quota limit, but the commit message does not mention the stub or the QuotaExceededError root cause. A reader inspecting the commit cold would not know why `tada-worker.js` is being intercepted. The technique itself is honest (stub is a no-op for generation logic), but the commit message is underspecified for the complexity of what changed. |
| `495eeea` | mark Task 3 [x]: all 3 models PASS audio in headless | OK | Queue update; audio sizes consistent with step report |
| `db4e68c` | step 4.1 stub: write benchmark script + smoke 2 voices | OK | Step stub only |
| `a655e79` | Task 4: benchmark script + smoke test (amazement + ex01_default) | OK | Script committed, 2-voice smoke rows in results.md and lab-notebook.md. Format matches existing rows. Idempotency verified. |
| `272074c` | Step 4.1 report: script written, smoke test passed | OK | Accurate step report |
| `290e602` | step 4.2 stub: run full 32-voice sweep | OK | Step stub |
| `e8e17bc` | Task 4 sweep: 32 voices benched (Var-C VV-Q8 E-Q4, Metal) | OK | 32/32 RTF rows in results.md, 960 lines in lab-notebook.md. bash 3.2 `mapfile` fix committed alongside. No WAVs committed (gitignored, in `/tmp/tada_bench/`). |
| `b4856ea` | Step 4.2 OUTCOME: sweep complete, criterion met | OK | Accurate: 30 new + 2 skipped (idempotent) = 32/32 |
| `960e5bc` | mark Task 4 [x]: 32/32 voices benched | OK | Queue update |
| `a64fe7b` | Task 5 [x]: acceptance criterion met (E2E exit 0, 32/32 RTF rows) | OK | Acceptance criterion awk pattern updated from incorrect original (matched bare voice names) to `voice-sweep-` prefix form matching actual row IDs. Queue updated to reflect corrected criterion. Both checks pass. |
| `2867ba6` | step 6 stub: global review (adversarial) | OK | Step stub only |
| `bc1e69c` | Task 6 [x]: global review PASS | OK | Global-review verdict PASS attached; step 3/2.md (WebGPU fix report) appended retroactively here |

## Convergence criteria audit

| Task | Claimed [x] | Criteria met? |
|------|-------------|---------------|
| 1. Land the merge | Yes | Yes — two-parent merge commit verified (`be63c07`), `cargo check -p tada-core --features metal` exit 0 documented, `/tmp/sanity.wav` 186KB ≥ 50KB |
| 2. WASM builds | Yes | Yes — all three `.wasm` artifacts present per step report; `ls crates/*/pkg/*.wasm` confirmed by global-review re-run |
| 3. E2E demo | Yes | Yes — `node scripts/test_demo_e2e.mjs` exit 0, all three models ≥ 10KB (Pocket 84KB, Kitten 572KB, TADA 155KB) |
| 4. Benchmark sweep | Yes | Yes — 32/32 voice-sweep rows with valid RTF (`awk` criterion = 32 ≥ 30); 0 failures |
| 5. Acceptance check | Yes | Yes — both acceptance commands run and verified on `main` HEAD |
| 6. Global review | Yes | Yes — fresh agent reviewed; verdict PASS with falsification attempt documented |

## Hard rules audit

| Rule | Status |
|------|--------|
| Never modify `refs/` | CLEAR — no `refs/` path in any run commit |
| Never open PRs on external repos | CLEAR — no PR activity |
| Never use personal browser | CLEAR — Playwright headless Chromium throughout |
| Never skip hooks | CLEAR — no `--no-verify` in any commit |
| Log every inference run in lab-notebook + results.md | CLEAR — every `tada_generate` invocation (sanity + smoke + 32-voice sweep) has both a lab-notebook block and a results.md row |
| Audio metrics = RTF only, no flatness/peak | CLEAR — RTF and timing only in results.md |
| Never delete samples or benchmark data | CLEAR — no deletions observed |
| Benchmark tables = data only | CLEAR — results.md contains only data rows |
| No gh-pages deploy | CLEAR — no gh-pages activity |
| Sub-agents for code writing | CLEAR — step reports describe sub-agent delegation throughout |

## Discovery / Split / Remesh usage

No Discovery, Split, or Remesh triggers were invoked. Task 3 iterated naturally (step 3.1 → step 3.2) as a designed iterate task responding to a WebGPU blocker — this is normal iterate-task cadence, not a Remesh.

## global-review reconciliation

**CONCUR** — artifact-level evidence supports global-review's PASS verdict. The merge commit is genuine (two parents confirmed), all three WASM binaries exist, `node scripts/test_demo_e2e.mjs` exited 0 with real audio from all three models, and `docs/tada/results.md` has exactly 32 voice-sweep rows with valid RTF values. No hard rules were violated. The single CONCERN (commit `49d204b` message underspecifies the worker-interception technique) does not affect correctness — the technique is sound and the global-review independently verified that the Cache API stub does not alter generation logic. No artifact-level evidence contradicts global-review's PASS.
