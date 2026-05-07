---
run: rebase-feat-burn-wgpu-llama-onto-tts-web
completed: 2026-05-07
verdict: PASS
---

# Walls hit

- **Rebase was immediately intractable**: `web/index.html` conflicted on the very first cherry-pick attempt (40 commits diverged, same UI file modified by both branches). The fallback merge strategy documented in the queue was the right call and went smoothly, but this consumed the expected rebase time anyway.
- **Headless Chromium WebGPU + Cache API quota**: `requestAdapter()` requires explicit Metal backend flags (`--enable-unsafe-webgpu --use-angle=metal --enable-features=Vulkan --enable-webgpu-developer-features`) in headless mode; even with those, the 1.3GB GGUF hit the 865MB Cache API quota in headless Chromium requiring a worker-script interception to inject a no-op Cache stub. `CDP Storage.overrideQuotaForOrigin` does not propagate to dedicated workers (service workers only).
- **benchmark_voices.sh bash 3.2 incompatibility**: `mapfile` (bash 4+) was used; macOS ships bash 3.2. Required a while-read loop substitution mid-sweep (committed alongside results without breaking idempotency).

# Anti-patterns observed

- **Commit message underspecification for non-trivial test harness changes** (commit `49d204b`): The Cache API worker-interception technique is subtle (route interception, IIFE prepend, Content-Type preservation) but the commit message just says "Enable WebGPU in headless Chromium for TADA E2E." A reader cold can't reconstruct why `tada-worker.js` is intercepted. Sub-agents should be prompted to put the "why" (root cause + workaround summary) in commit messages for anything involving test harness tricks. Happened in cycle Task 3.
- **Acceptance criterion awk pattern written for pre-sweep row ID format** (Task 5): The queue's acceptance criterion used a pattern matching bare voice names (`/^\| (ex0|amazement|...)`) but the benchmark script actually emitted `voice-sweep-<name>` row IDs. The acceptance step caught and fixed this, but the criterion should have been derived from the benchmark script's actual output format, not assumed. Next planner should require that acceptance criteria reference actual artifact formats (column names, row prefixes) explicitly.

# Latent items noticed

- **14 newly-merged base voices not yet swept**: The merge commit brought in 14 additional voices in `voices/` (angry, confused, default, disgusted, enunciated, happy, jeff, karen, laughing, sad, sad_f, sleepy, walt, whisper) — these are tracked and valid but have no RTF rows in `docs/tada/results.md`. The run's 32-voice sweep targeted the original 6+26 design points; the new voices were out of scope. A small follow-up run could sweep these 14 to complete the picture (total would then be 46 rows).
- **`scripts/analyze-trace.py` committed but undocumented**: The merge brought in `scripts/analyze-trace.py` (146-line trace analysis script) with no entry in any docs or README. Unclear if it's usable standalone or requires specific trace format input.
- **E2E test Cache API stub is a headless workaround, not shipped code**: `scripts/test_demo_e2e.mjs` now contains a worker-interception patch that bypasses Cache API quota in headless Chromium. This is correct for CI, but if a real browser user hits the same quota (e.g. disk-constrained machine), the production code will fail. Consider chunked cache writes or a quota fallback in `tada-worker.js` proper.
- **`web/serve.mjs` has no `voices/matrix/` explicit mention in documentation**: It happens to serve `voices/matrix/` because `voices/` is served recursively, but CLAUDE.md only lists `voices/` as a static route. This is fine but worth noting if someone changes `serve.mjs`.

# Suggested next goals

1. **Sweep the 14 newly-merged base voices for RTF** — A small single-session run. Run `benchmark_voices.sh` (it's already idempotent) and it will pick up the 14 unswept voices (angry, confused, default, disgusted, enunciated, happy, jeff, karen, laughing, sad, sad_f, sleepy, walt, whisper). Commit the updated `docs/tada/results.md` and `docs/tada/lab-notebook.md`. Success = `docs/tada/results.md` has RTF rows for all 20 base voices + 26 matrix = 46 total voice-sweep rows. No code changes needed.

2. **Fix TADA production worker for Cache API quota on disk-constrained machines** — `tada-worker.js` loads a 1.3GB GGUF into the Cache API (`caches.open()`). On machines with limited disk quota for web storage, this throws `QuotaExceededError`. The E2E test papers over this with a no-op stub injected at test time, but real users on constrained machines would see the worker fail silently. Success = `tada-worker.js` catches `QuotaExceededError` on `cache.put()`, falls back to in-memory (no caching), logs a console warning, and generation continues. The E2E test Cache stub should then be removable (or kept as a speed optimization to skip the cache write).

3. **TADA WASM performance profiling and optimization** — The 32-voice sweep showed RTF 0.98x–3.95x on Metal (native). The WASM build exists but has no RTF measurements in `docs/tada/results.md`. The next natural step is a WASM RTF sweep (same 32 voices, same text, Playwright-driven) to establish the browser performance baseline. Success = a results.md section `wasm-sweep` with RTF rows for all 32 voices in headless Chromium with WebGPU Metal backend, plus at least one note in `docs/tada/` about which bottleneck (LLM / VV / decode) dominates WASM latency.
