# Survey: investigate-tada-wasm-performance-and-sh

## Goal

Profile TADA browser (WASM/WebGPU) inference, establish a formal baseline RTF row in `docs/tada/results.md`, diagnose the dominant bottleneck (suspected: dual-KV-cache CFG from commit ea9da52 causing ~1100ms inter-step gaps vs ~55ms pre-CFG), land a measurable configuration improvement (≥30% RTF reduction or a documented tradeoff analysis), and confirm native CLI + Pocket TTS + KittenTTS are not regressed.

## Project Context

`tts-web` is a browser-based multi-model TTS demo (Pocket TTS, KittenTTS, TADA-1B). The user works in an orchestrator/sub-agent pattern: lead (opus) plans, sonnet sub-agents write code and run tasks. Commits are atomic and small. TADA uses a hybrid architecture: Burn/wgpu for the Llama LLM on WebGPU and candle (CPU) for VibeVoice diffusion + DAC decoder. WASM builds use `.cargo/config.toml` with `+simd128` target-feature at both the root and `crates/tada-wasm/` levels. The dev server runs on port 8081. Chrome tracing is analyzed via `scripts/analyze-trace.py`. The benchmark doc (`docs/tada/results.md`) has zero `engine=wasm` rows — all existing rows are `candle` (native Metal) or `burn+candle` (native wgpu+cpu).

## Closest Analogous Work

- `simd128` experiment (results.md): Decode 17.8s → 5.4s (3.3x). Chrome trace: 23 steps, avg 402ms/step (90ms compute + 312ms gap).
- `tasks-max-512` experiment (results.md): Gaps 340ms → 55ms per step. `crates/tada-wasm/src/lib.rs:714` — `tasks_max: 512`.
- `burn-varC` sweep (results.md): native burn+candle RTF ~1.4x–2.4x, same model variant used in production.
- `cfg-negcond-dualcache-whisper` (results.md): dual KV cache introduced at commit ea9da52; pre-CFG traces showed 55ms gaps, post-CFG 1100ms gaps per the goal statement.
- Prior E2E test work (commit `49d204b`): headless Chromium requires `--enable-unsafe-webgpu --use-angle=metal`; Cache API quota 865MB workaround already in `scripts/test_demo_e2e.mjs` (worker-script interception).

## Hard Rules Surfaced

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

## Prior-Run Carryover

From `convergence/notes/rebase-feat-burn-wgpu-llama-onto-tts-web-next.md`:

- **Cache API quota in headless (1.3GB GGUF hits 865MB limit)**: already papered over with a no-op Cache stub injected into `tada-worker.js` in `scripts/test_demo_e2e.mjs`. If TADA headless crashes the machine again, skip it and document.
- **`scripts/analyze-trace.py` committed but undocumented**: confirmed usable standalone; takes Chrome trace JSON as argument. The `simd128` result row shows it was used before.
- **Acceptance criteria must reference actual artifact formats**: column names and row ID prefixes should be verified against the benchmark script's real output before writing the criterion — do not assume formats.
- **E2E Cache stub is a headless workaround, not production**: real-browser users on disk-constrained machines will still hit `QuotaExceededError`. Out of scope for this run, but document if encountered.
- **Commit message underspecification for non-trivial harness changes**: when adding timing instrumentation to `tada-worker.js`, put the "why" in the commit message, not just the "what".

## Open Questions for Mesh

1. **How to get wasm RTF numbers without live browser interaction**: The goal asks for `engine=wasm` rows with gen_ms, decode_ms, audio_duration → RTF. These require running TADA in the browser and reading JS `console.log` timing. The E2E Playwright test can capture console output, but it has crashed the machine in the prior session. Mesh needs to decide: (a) capture timings via Playwright console-capture harness, (b) instrument the worker to postMessage timings back to the page (already partially done — confirm), or (c) capture timings from headless server logs. The goal says if TADA E2E remains unstable to skip/document rather than fight for hours.

2. **Which model file does the browser currently load**: The WASM frontend likely uses a Var-C model (1.3G, VV-Q8 E-Q4) based on UI defaults — but the worker may reference a different GGUF. This affects whether the RTF baseline is comparable to the native Var-C rows in results.md.

3. **Is `cfg_scale=1.0` (disable CFG) a safe A/B test without quality risk**: The goal says to A/B test CFG. Disabling CFG changes voice conditioning behavior. Mesh should plan for the user to listen to the cfg=1.0 output before committing it as a production change. RTF win without quality regression → keep; regression → document tradeoff.

4. **tasks_max sweep values**: Goal suggests 128, 64, 32 — but the prior experiment showed 512 was already a win over the default 32. Going lower than current 512 would likely regress. Mesh should interpret the sweep as potentially trying values between 512 and 2048 (higher batching) rather than lower. Or the goal may intend trying lower to find the optimal point if 512 is no longer ideal post-CFG.

5. **SIMD128 verification**: Both `.cargo/config.toml` files have `+simd128` in rustflags, but the goal says the prior 5.4s decode regressed to 6.2s and SIMD may be "silently off." SIMD can be stripped by a dependency's build.rs overriding RUSTFLAGS. The built `.wasm` binary can be inspected with `wasm-objdump -d` or `wasm2wat` for SIMD opcodes. Mesh should verify before assuming SIMD is active.

## Estimated Shape

`multi-phase` — four distinct phases (measure, diagnose, improve, validate) with real browser execution required for measurement, plus potential code changes (worker instrumentation, tasks_max tuning, possible CFG bypass) and native CLI sanity check at the end.
