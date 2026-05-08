# lab-notebook.md

Experiment log. Each block corresponds to an entry in results.md (same experiment name). Contains purpose, parameters, and verbatim user feedback.

---

## first-python-ref

**Date**: 2026-03-12
**Commit**: unknown (pre-benchmark)
**Purpose**: First Python BF16 reference generation to establish a quality baseline.

**Parameters**:
- engine: python-bf16
- device: cpu
- seed: unknown
- flow_steps: 10
- voice: ljspeech
- noise_temp: unknown

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**: BF16 Python reference

**User feedback**: Reference quality.

---

## timing-v2

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: First timing instrumentation pass; also includes voice alignment V2 fix (match Python's zeros+mask=0 approach).

**Parameters**:
- engine: candle
- device: cpu / metal (no --features metal)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5 (trims all 5 voice prompt tokens → effectively zero-shot)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**:
- F16 CPU
- F16 Metal (no --features metal flag — not real GPU acceleration)
- Q4_0 CPU
- Q4_0 Metal (no --features metal flag)

**Notes**: F16 Metal with --features metal was OOM killed (3.3GB model too large for GPU buffer).

**User feedback**: "all Rust outputs sound comparable, possibly better than Python reference"

---

## metal-discovery

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: Discovered that --features metal flag enables real Metal GPU acceleration (vs. the no-feature Metal runs in timing-v2).

**Parameters**:
- engine: candle
- device: metal (with --features metal)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5
- model: Q4_0 baseline

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**: Q4_0 Metal with --features metal

**User feedback**: Not evaluated separately (same text and model as timing-v2).

---

## burn-hybrid-zeroshot

**Date**: 2026-03-12
**Commit**: unknown
**Purpose**: First Burn/wgpu LLM benchmark. LLM runs on GPU via Burn; VibeVoice remains on CPU via candle. Zero-shot (no voice prompt — example did not yet support --voice).

**Parameters**:
- device: wgpu+cpu (LLM on wgpu GPU, VibeVoice on candle CPU)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: none (zero-shot)
- model: Q4_0 baseline

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**:
- candle Q4_0 CPU (reference)
- burn+candle Q4_0 wgpu+cpu

**Notes**: LLM per-step: Burn GPU 146ms vs candle CPU 1660ms (11x speedup). VibeVoice 455ms/step (74% of total gen time) — becomes the new bottleneck.

**User feedback**: "BAD AUDIO — zero-shot, no voice. Not comparable."

---

## bench8-noise-discovery

**Date**: 2026-03-13
**Commit**: around 4d86056
**Purpose**: Mixed Q4+Q8 quantization test. Also caught noise_temp=0.6 bug (wrong default).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.6 (run 1, bug) / 0.9 (runs 2–4, correct)
- flow_steps: 10
- voice: ljspeech

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**:
- Q4_0 baseline (2.64G), noise_temp=0.6
- Q4_0 baseline (2.64G), noise_temp=0.9
- Mixed v1 decoder-Q4_0 (1.48G)
- Mixed v2 decoder-Q8_0 (1.52G)

**User feedback**:
- noise_temp=0.6: "flat/dead audio"
- Mixed v1 and v2: "almost identical voice quality, but slightly longer, and right after the voice, it introduces some noise"

---

## bench8-phrases

**Date**: 2026-03-13
**Commit**: around 4d86056
**Purpose**: Multi-phrase testing with Q4_0 and Mixed models.

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech

**Texts**:
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- time: "Time is money, who can afford to pay attention?"
- universe: (text not recorded)

**Variants**:
- Q4_0 baseline (2.64G)
- Mixed Q4+Q8 (1.52G)

**Notes**: No timing data captured for these runs.

**User feedback**: "All the examples, but the silence ones, include some noise/parasite voice in the end"

---

## variant-precision-sweep

**Date**: 2026-03-13
**Commit**: ec906e4
**Purpose**: Systematic sweep to identify which model components (VV head, embeddings) tolerate lower precision without gray-code mispredictions.

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"

**Variants**:
- baseline: VV F32, Embed Q4_0 (2.64G)
- A: VV F16, Embed Q8_0 (1.88G)
- B: VV F32, Embed Q8_0 (2.68G)
- C: VV Q8_0, Embed Q4_0 (1.38G)
- D: VV Q8_0, Embed Q8_0 (1.52G)
- E: VV F16, Embed Q4_0 (1.75G)

**Key metric**: times_before[12] gray-code prediction at a sensitive step.
- baseline: tb=4 (correct) on both texts
- A: tb=4 (fox only), tb=59 (time — misprediction)
- B: tb=4 (fox only), tb=59 (time — misprediction)
- C: tb=59 on both (misprediction everywhere)
- D: tb=59 on both (misprediction everywhere)
- E: tb=4 (correct) on both texts

**Notes**: No audio files logged for these runs — they were part of a systematic diagnostic comparison. Variant E selected as winner: smallest model with correct gray-code predictions on all tested phrases. This experiment was run pre-bugfix (before the autoregressive time feedback fix in 7e275ba). The gray-code mispredictions found here were later largely eliminated by that fix, making the variant-sweep findings less critical.

**User feedback**: Not individually evaluated. Variant E selected as winner.

---

## exp2-variant-sweep

**Date**: 2026-03-13
**Commit**: ec906e4
**Purpose**: Systematic comparison of all 6 quantization variants (baseline, A–E) across 6 phrases. Also ran F16 CPU as an additional reference.

**Parameters**:
- engine: candle
- device: metal (Var-E) / cpu (F16)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 5

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- woods: "There is a pleasure in the pathless woods"
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- call: "I had to call you up in the middle of the night"

**Variants**:
- Baseline Q4_0, Var-A through Var-E (all on Metal)
- F16 CPU (full precision reference)

**Notes**: Audio durations captured for Var-E and F16 only; no timing for other variants.

**User feedback**:
- "A B E are fine for 'fox', but all variants are equally bad for the 'time is money' test. All the version of the 'time' tests finish with the words 'dzouib'."
- "They *all* have parasite noise or voice."
- F16 time result: "Still has 'dzouib'"

---

## exp3-eos-fix

**Date**: 2026-03-13
**Commit**: 9db7c77
**Purpose**: EOS handling fix test. Also added Python reference comparison.

**Parameters**:
- engine: candle / python-bf16
- device: metal (Rust) / cpu (Python)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- woods: "There is a pleasure in the pathless woods"
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- call: "I had to call you up in the middle of the night"

**Variants**:
- Python BF16 CPU (reference)
- Q4_0 baseline (Metal)
- Var-E VV-F16 E-Q4 (Metal)
- F16 CPU

**Notes**: Python ref handles "smile" fine (6.28s) but Q4_0/varE produce 11s+ runaway on "smile". Timing data not captured.

**User feedback**: "smile is a bit garbled, lower quality than before. time and call both still have words (dzouig for time)"

---

## exp5-f32-baseline

**Date**: 2026-03-13
**Commit**: around 9db7c77
**Purpose**: F32 GGUF (full precision, ~6.5GB) to check whether quantization is causing the "dzouib" and trailing noise issues.

**Parameters**:
- engine: candle
- device: cpu (F32 too large for Metal GPU buffer)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- call: "I had to call you up in the middle of the night"

**Variants**: F32 GGUF (6.5G) — nothing quantized

**Notes**: Gen times only (no load/decode captured). Confirms noise and "zdouib" are not quantization artifacts.

**User feedback**: "They all include noise, and time still says 'zdouib'. Is this with the real full model, nothing is quantized? That's kinda good news, once we fix this, we'll be able to use the quantized models!"

---

## exp8-all-fixes

**Date**: 2026-03-13
**Commit**: 7e275ba
**Purpose**: Test combined fixes: autoregressive time feedback bug (fixed) + transition_steps=0 (retain all 5 voice prompt acoustic tokens instead of trimming them all away).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- call: "I had to call you up in the middle of the night"

**Variants**: Q4_0 baseline (Metal, transition_steps=0)

**Notes**: No timing data captured. Significant quality improvement over pre-fix runs — "dzouib" reduced to "azoulib" phantom.

**User feedback**: "exp8_q4_fox.wav sounds **great**. exp8_q4_time.wav sounds **great**, but... still has a phantom noise after attention. It's not 'zdouib' anymore, it's 'azoulib'? Let's commit!"

---

## exp9-eot-trim

**Date**: 2026-03-13
**Commit**: 4474519
**Purpose**: Trim the trailing EOT acoustic frame to remove the residual phantom noise ("azoulib") found in exp8.

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- call: "I had to call you up in the middle of the night"

**Variants**: Q4_0 baseline (Metal, trailing EOT frame trimmed)

**Notes**: Audio durations captured; gen/load/decode not recorded.

**User feedback**: "fox, time, smile are... Good! Weird voice, though, a bit of an Indian accent? call sounds like 'I had to call you up whaaa'"

---

## exp10-final-comparison

**Date**: 2026-03-13
**Commit**: 4474519
**Purpose**: Full comparison with all fixes applied — Q4_0 vs F32, across 8 test phrases. Establishes the post-fix quality baseline.

**Parameters**:
- Q4_0: candle, metal, seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech, transition_steps=0
- F32: candle, cpu, seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech, transition_steps=0

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- time: "Time is money, who can afford to pay attention?"
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- woods: "There is a pleasure in the pathless woods, There is a rapture on the lonely shore"
- universe: "I'll tell you one thing about the universe, though. The universe is a pretty big place."
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**:
- Q4_0 baseline (2.64G, Metal)
- F32 (6.5G, CPU)

**Notes**: Gen times captured for some F32 runs; load/decode not fully recorded. Q4_0 Metal gen times not recorded (only audio durations). F32 CPU gen time is very slow (20–63s per phrase). This run establishes which issues are model-level vs. implementation bugs.

**User feedback**: "call is cut early in both. tyger tyger starts really wrong, in both cases, tiiigger with a very strong Indian accent and even words that don't sound english. Apart from this, they all sound very good, in both cases! One weird thing, the wutang one, it's changing voices in the middle! Makes me think maybe we don't handle the voice setting very well."

---

## postfix-full-sweep

**Date**: 2026-03-14
**Commit**: 21d97e6
**Purpose**: Comprehensive post-bugfix benchmark across all model variants and 4 test phrases. Validates fixes for: autoregressive time feedback bug, trailing EOT frame trim, WASM EOT token.

**Parameters**:
- Rust runs: candle, metal, seed=42, noise_temp=0.9, flow_steps=10, voice=ljspeech, transition_steps=0
- Python run: python-bf16, cpu, acoustic_cfg_scale=1.6, transition_steps=5

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright" (note: shorter than the tyger text used in exp10 — "In the forests of the night" not included here)
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**:
- Python BF16 CPU
- F32 GGUF (6.5G)
- F16 GGUF (3.3G)
- Q4_0 baseline (2.6G)
- Var-B VV-F32 E-Q8 (2.5G)
- Var-A VV-F16 E-Q8 (1.9G)
- Var-E VV-F16 E-Q4 (1.8G)
- Mixed VV-Q8 E-Q8 (1.4G)
- Var-C VV-Q8 E-Q4 (1.3G)

**Notes**: All audio files in samples/bench_2026-03-13/ subdirectories.

**User feedback**:
- Python (all phrases): "voice is OK, output is complete, but quieter, more noise/wind"
- fox (all Rust variants): OK
- call (all Rust variants): "sounds good, just cut early — missing 'the middle of the night'"
- tyger (all Rust variants): universally bad, wrong phonemes
- wutang (all Rust variants): "sounds very good, voice switch mid-phrase"

---

## q8-llm-alignment

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Test Q8_0 LLM backbone with F16 VV head and Q4_0 embeddings to see if quantizing the LLM backbone further degrades output quality vs Var-E (F16 VV + Q4_0 embed).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0
- model file: tada-1b-llmq8-vvf16-eq4.gguf (2.24G)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**: Q8_0 LLM + F16 VV + Q4_0 Embed (2.24G)

**Notes**: Files in samples/bench_2026-03-13/llmq8_vvf16_eq4/

**User feedback**: Not individually evaluated (same perceptual quality as other Rust variants per user's earlier statement).

---

## burn-gpu-zeroshot-v2

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Burn/wgpu GPU benchmark with updated code. Zero-shot because the burn example does not support --voice. Intended to measure VibeVoice GPU acceleration potential.

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu
- noise_temp: 0.9
- flow_steps: 10
- voice: none (zero-shot — example limitation)
- model: Q4_0 baseline (2.6G)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**: Burn+candle wgpu+cpu Q4_0 baseline

**Notes**:
- Gen time breakdown (LLM / VibeVoice): fox 4.3s / 15.7s, call 11.6s / 65.2s, tyger 11.0s / 62.8s, wutang 3.6s / 12.8s
- call and tyger have no decode time logged (likely crash or skip)
- Files in samples/bench_2026-03-13/burn_gpu/

**User feedback**: "even worse, just noise and garbage" (expected — zero-shot)

---

## tyger-debug

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Investigate the tyger pronunciation issue — wrong phonemes on "Tyger". Vary transition_steps and voice conditioning to isolate cause.

**Parameters**:
- engine: candle
- seed: 42
- noise_temp: 0.9
- model: Q4_0 baseline (2.6G)
- voice: ljspeech (except novoice variant)

**Texts**:
- tyger: "Tyger Tyger, burning bright"

**Variants**:
- ts0 Metal: transition_steps=0, Metal
- ts5 Metal: transition_steps=5, Metal
- novoice Metal: no voice prompt, Metal
- ts0 CPU: transition_steps=0, CPU

**Notes**: No load/gen/decode timing captured for these runs.

**User feedback**:
- tyger_ts0: "shitty"
- tyger_ts5: "matches the python version, apart from this one being cut early"
- tyger_novoice: "nonsense 'aaaaaaha bnoeoeo'"
- tyger_cpu: "exactly equivalent to tyger_ts0"

---

## cross-variant-quality

**Date**: 2026-03-14
**Commit**: 83c08ab
**Purpose**: Verify that audio quality is consistent across all model sizes on a phrase known to be well-handled (not affected by the tyger/call known issues).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0

**Texts**:
- time: "Time is money, who can afford to pay attention?"

**Variants**:
- F32 GGUF (6.5G)
- Q4_0 baseline (2.6G)
- Var-E VV-F16 E-Q4 (1.8G)
- Mixed VV-Q8 E-Q8 (1.4G)
- Var-C VV-Q8 E-Q4 (1.3G)

**Notes**: No load/decode timing captured for these runs.

**User feedback**: "They all sound good"

---

## burn-voice-prompted

**Date**: 2026-03-14
**Commit**: f27c1d7
**Purpose**: First voice-prompted Burn/wgpu GPU test (newly added --voice support).

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu (LLM on wgpu GPU, VibeVoice on candle CPU)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0
- model: Q4_0 baseline (2.6G), file tada-1b-q4_0.gguf

**Texts**:
- time: "Time is money, who can afford to pay attention?"
- fox: "The quick brown fox jumps over the lazy dog."

**Variants**: Burn+candle wgpu+cpu Q4_0 baseline, voice-prompted

**Notes**: Gen time breakdown: llm=3.1–3.4s + vibe=11.0–11.5s. LLM per-step avg 129–137ms (GPU), VibeVoice per-step avg 457–459ms (CPU). Audio duration is significantly shorter than candle equivalent (0.62s vs 2.5s for time, 0.98s vs 2.7s for fox). Suspected cause: Burn GPU float precision divergence produces shorter time_before gray-code predictions than candle Metal, leading to shorter per-frame durations.

**User feedback**: Not yet evaluated.

---

## q4km-varc-benchmark

**Date**: 2026-03-14
**Commit**: 1d68952
**Purpose**: First benchmark of Q4_K_M LLM + Q8_0 VV + Q4_0 Embed model (Var-C style with better LLM quant).

**Parameters**:
- engine: candle
- device: metal
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0
- model file: tada-1b-mixed-llmq4_k-vvq8_0-eq4_0.gguf (Q4_K_M Var-C, 1.38G)

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- time: "Time is money, who can afford to pay attention?"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**: Q4_K_M Var-C (LLM Q4_K_M + VV Q8_0 + Embed Q4_0)

**Notes**: Sub-1x RTF on fox (0.84x), tyger (0.94x), time (0.92x), wutang (0.72x). Call truncated at 0.9s (known model issue). times_before for fox matches Q4_0 baseline almost exactly: [6,2,10,13,17,25,25,11,6,17,8,4,1] vs Q4_0's [6,2,10,12,17,25,25,11,6,17,8,4,1] — only one value differs (13 vs 12 at index 3).

**User feedback**: Not yet evaluated.

---

## burn-rope-fix

**Date**: 2026-03-14
**Commit**: e248a13
**Purpose**: First Burn/wgpu GPU voice-prompted benchmark after RoPE fix (split-half convention). The RoPE bug (interleaved vs split-half pairing) was the root cause of all Burn audio gibberish — step 2 hidden states diverged catastrophically (cosine similarity 0.005) because position embeddings paired wrong elements. Fix: change `apply_rotation` to split x into [x[0..d/2], x[d/2..d]] instead of interleaved pairs [(x[0],x[1]), (x[2],x[3])...].

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu (LLM on wgpu GPU, VibeVoice on candle CPU)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0
- model: Q4_0 baseline (2.6G), file tada-1b-q4_0.gguf

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- time: "Time is money, who can afford to pay attention?"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."

**Variants**: Burn+candle wgpu+cpu Q4_0 baseline, voice-prompted, RoPE split-half fix applied

**Notes**: LLM per-step avg 156–168ms (GPU), VibeVoice per-step avg 455–460ms (CPU). Fox times_before matches candle exactly: [6, 2, 10, 12, 17, 25, 25, 11, 6, 17, 8, 4, 1] — confirming the RoPE fix resolves the hidden-state divergence that previously caused gibberish output.

**User feedback**: Not yet evaluated.

---

## burn-varC

**Date**: 2026-03-14
**Commit**: e248a13 (RoPE fix) + gguf.rs fix for Q8_0/Q4_K dtype parsing
**Purpose**: Burn/wgpu GPU benchmark with Var-C model (1.3G, VV Q8_0 + Embed Q4_0). First Burn run with a small model — Q8_0 VV tensors are loaded by candle (not Burn), Q4_0 embeddings loaded by Burn. Also tested Mixed (1.4G) but it failed because Mixed has Q8_0 embeddings which Burn's EmbeddingStore can't handle (reads Q8_0 bytes as Q4_0 → garbage times_before [14,14,15,15...]).

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu (LLM on wgpu GPU, VibeVoice on candle CPU)
- seed: 42
- noise_temp: 0.9
- flow_steps: 10
- voice: ljspeech
- transition_steps: 0
- model: Var-C VV-Q8 E-Q4 (1.3G), file tada-1b-C-vvq8-eq4.gguf

**Texts**:
- fox: "The quick brown fox jumps over the lazy dog."
- call: "I had to call you up in the middle of the night"
- tyger: "Tyger Tyger, burning bright, In the forests of the night"
- time: "Time is money, who can afford to pay attention?"
- wutang: "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."
- smile: "Your smile is like a breath of spring, Your voice is soft like summer rain"
- woods: "There is a pleasure in the pathless woods, There is a rapture on the lonely shore"
- universe: "I'll tell you one thing about the universe, though. The universe is a pretty big place."

**Notes**:
- Per-step averages: LLM=134–140ms (GPU), VibeVoice=85–86ms (CPU, Q8_0)
- Fox times_before matches candle EXACTLY: [6, 2, 10, 12, 17, 25, 25, 11, 6, 17, 7, 4, 1]
- Key: VibeVoice Q8_0 runs at 85ms/step (vs 455ms with F32 VV) — 5x faster
- Mixed model (1.4G, Q8_0 embeddings) failed: Burn EmbeddingStore reads Q8_0 bytes as Q4_0 → garbage times_before [14,14,15,15...]. Only Var-C (Q4_0 embeddings) works with Burn.

**User feedback**: Not yet evaluated.

---

## simd128

**Date**: 2026-03-20
**Commit**: b5f720a
**Purpose**: Enable WASM SIMD128 for candle CPU ops (conv1d, matmul) to accelerate the DAC decoder, which was the bottleneck after Burn/wgpu GPU acceleration of the LLM.

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu
- flow_steps: 10
- CFG: 1.0
- voice: default (39 tokens)
- transition_steps: 5

**Chrome trace analysis** (23 steps, threshold 20ms):
- Total span: 31.2s
- Warmup (step 0): 3254ms
- Content steps: 19 steps, avg 90ms compute + 312ms gap = 402ms/step
- Decode: 5385ms (was 17.8s before SIMD — 3.3x improvement)

**User feedback**: "It does seem faster"

---

## tasks-max-512

**Date**: 2026-03-20
**Commit**: 13fbab4
**Purpose**: Increase GPU command batching from default 32 to 512 tasks per submission to reduce the per-step dispatch gaps observed in Chrome traces.

**Parameters**:
- engine: burn+candle
- device: wgpu+cpu
- voice: default (long prompt, 45 steps)

**Chrome trace**: Dispatch gaps dropped 6x (340ms → 55ms per step). Total: 53.1s (45 steps with long voice prompt).

**User feedback**: "it feels faster!"

---

## q8-vv-gpu

**Date**: 2026-03-20
**Commit**: 6c6011d
**Purpose**: Run Q8_0 VibeVoice on GPU via custom WGSL shader, instead of candle CPU, to eliminate the VV CPU bottleneck.

**Parameters**:
- engine: burn+candle
- device: wgpu+wgpu (both LLM and VV on GPU)
- model: Var-C VV-Q8 E-Q4

**Results**:
- Content steps: 551ms/step (vs 595ms CPU = 8% faster)
- Warmup: 10.6s (VV shader compilation on first run)
- Decode regressed to 17.8s (SIMD128 decoder not active in this build path)

**User feedback**: Not individually evaluated.

---

## vv-warmup

**Date**: 2026-03-20
**Commits**: abaad42 (initial), 4dd5a3f (device fix)
**Purpose**: Pre-compile VV Q8_0 WGSL shaders during model warmup to eliminate the ~13s first-generation compile stall. Warmup runs 10 ODE steps with CFG=1.6 to exercise all shader code paths and variants.

**Results**: First-gen VV compile stall reduced from ~13s to ~4.5s.

**User feedback**: Not individually evaluated.

---

## cfg-negcond-fix

**Date**: 2026-03-20
**Commits**: e10650f (initial fix), ea9da52 (dual KV cache)
**Purpose**: Fix the CFG negative condition implementation. Previous code used literal zero tensors as neg_cond; correct approach mirrors Python: run a second LLM forward pass with zero acoustic tokens to get the "unconditioned" hidden state, then use that as neg_cond. Implemented with dual independent KV caches for the positive and negative paths.

**Parameters**:
- voice: ex04_whisper
- CFG: 1.6
- model: Var-C

**Whisper RMS progression**:
- Literal zeros neg_cond: -12.4dB
- Single KV cache pop: -20.2dB
- Dual independent KV caches: -21.2dB
- Python reference: -34.5dB

**User feedback**: "now it IS whispering... sounds more like gollum than a woman, but it's progress!"

---

## sampling-fix

**Date**: 2026-03-21
**Commit**: 93eadb5
**Purpose**: Replace Gumbel-max token sampling with Python-matching approach: top_p=0.9 nucleus sampling + repetition_penalty=1.1 + multinomial sampling.

**Parameters**:
- engine: burn+candle
- device: wgpu+wgpu

**Results**: Same RMS for the test seed; should improve quality diversity across varied inputs.

**User feedback**: Not individually evaluated.

---

## neg-kv-all-steps

**Date**: 2026-03-21
**Commit**: 636a7d8
**Purpose**: Run neg CFG forward pass on ALL autoregressive steps (not just content steps) so the negative KV cache builds a complete history matching the positive path.

**Parameters**:
- voice: ex04_whisper
- CFG: 1.6

**Results**: Whisper RMS -22.1dB vs -21.2dB (previous) — minimal improvement. Gap from Python (-34.5dB) persists.

**User feedback**: Not individually evaluated.

---

## quality-all-variants

**Date**: 2026-03-21
**Commit**: 636a7d8
**Purpose**: Full quality comparison of all 11 GGUF quantization variants using Python-matching sampling parameters. Aimed to determine whether the ~12dB whisper RMS gap from Python is a quantization issue or a pipeline issue.

**Parameters**:
- noise_temp: 0.9
- text_temp: 0.6
- CFG: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- seed: 42
- transition_steps: 5

**Voices**: ex04_whisper, ex01_happy

**Text**: "I'll tell you one thing about the universe, though. The universe is a pretty big place."

**Variants tested**: python-bf16, F32, F16, Q4_0 baseline, Var-A (VV-F16 E-Q8), Var-B (VV-F32 E-Q8), Var-C (VV-Q8 E-Q4), Var-E (VV-F16 E-Q4), Mixed (VV-Q8 E-Q8), Q8 LLM + VV-F16, Q4K + VV-Q8

**Key finding**: Even F32 GGUF is 12dB louder than Python for whisper voice. All Rust variants cluster within 1dB of each other. The gap is a pipeline/CFG implementation difference, not a quantization artifact. Happy voice gap is only 3dB. Acoustic feature stats: Python raw std=1.10, Rust std=0.97 — similar scale, but the decoder magnifies the difference. "None of the quantizations can do whisper correctly" — the root cause is in the pipeline.

**User feedback**: "none of the quantizations can do whisper correctly"

## 2026-05-07 — voice-matrix-sweep — db4e68c — amazement

**Date**: 2026-05-07
**Commit**: db4e68c
**Purpose**: Automated voice sweep — amazement.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: amazement
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.2s total (load 0.8s, gen 6.8s, decode 2.6s)
- Audio duration: 3.3s
- RTF: 2.06x

**Output**: /tmp/tada_bench/amazement.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — db4e68c — ex01_default

**Date**: 2026-05-07
**Commit**: db4e68c
**Purpose**: Automated voice sweep — ex01_default.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_default
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.2s total (load 0.7s, gen 6.9s, decode 1.6s)
- Audio duration: 1.9s
- RTF: 3.63x

**Output**: /tmp/tada_bench/ex01_default.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — amusement

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — amusement.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: amusement
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.5s total (load 0.7s, gen 6.0s, decode 2.8s)
- Audio duration: 2.9s
- RTF: 2.07x

**Output**: /tmp/tada_bench/amusement.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ljspeech

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ljspeech.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ljspeech
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 6.6s total (load 0.7s, gen 3.7s, decode 2.2s)
- Audio duration: 2.6s
- RTF: 1.42x

**Output**: /tmp/tada_bench/ljspeech.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ljspeech_long

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ljspeech_long.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ljspeech_long
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.3s total (load 0.7s, gen 7.2s, decode 2.4s)
- Audio duration: 2.9s
- RTF: 2.48x

**Output**: /tmp/tada_bench/ljspeech_long.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — rickie

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — rickie.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: rickie
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 17.1s total (load 0.7s, gen 13.3s, decode 3.1s)
- Audio duration: 3.9s
- RTF: 3.41x

**Output**: /tmp/tada_bench/rickie.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — rickie_trimmed

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — rickie_trimmed.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: rickie_trimmed
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 16.8s total (load 0.9s, gen 13.1s, decode 2.8s)
- Audio duration: 3.6s
- RTF: 3.64x

**Output**: /tmp/tada_bench/rickie_trimmed.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex01_confused

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex01_confused.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_confused
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.5s total (load 0.8s, gen 5.5s, decode 4.2s)
- Audio duration: 5.6s
- RTF: 0.98x

**Output**: /tmp/tada_bench/ex01_confused.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex01_enunciated

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex01_enunciated.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_enunciated
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.3s total (load 0.7s, gen 5.5s, decode 4.1s)
- Audio duration: 5.4s
- RTF: 1.02x

**Output**: /tmp/tada_bench/ex01_enunciated.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex01_happy

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex01_happy.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_happy
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.9s total (load 0.7s, gen 6.2s, decode 3.0s)
- Audio duration: 3.9s
- RTF: 1.59x

**Output**: /tmp/tada_bench/ex01_happy.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex01_laughing

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex01_laughing.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_laughing
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.4s total (load 0.7s, gen 6.3s, decode 2.4s)
- Audio duration: 3.1s
- RTF: 2.03x

**Output**: /tmp/tada_bench/ex01_laughing.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex01_sad

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex01_sad.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_sad
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.8s total (load 0.7s, gen 6.8s, decode 3.3s)
- Audio duration: 3.7s
- RTF: 1.84x

**Output**: /tmp/tada_bench/ex01_sad.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex01_whisper

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex01_whisper.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex01_whisper
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.9s total (load 0.7s, gen 5.5s, decode 3.7s)
- Audio duration: 4.9s
- RTF: 1.12x

**Output**: /tmp/tada_bench/ex01_whisper.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_confused

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_confused.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_confused
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 11.0s total (load 0.7s, gen 6.2s, decode 4.1s)
- Audio duration: 5.4s
- RTF: 1.15x

**Output**: /tmp/tada_bench/ex02_confused.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_default

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_default.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_default
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.7s total (load 0.7s, gen 6.6s, decode 2.4s)
- Audio duration: 3.2s
- RTF: 2.06x

**Output**: /tmp/tada_bench/ex02_default.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_enunciated

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_enunciated.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_enunciated
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.0s total (load 0.7s, gen 6.1s, decode 3.2s)
- Audio duration: 4.2s
- RTF: 1.45x

**Output**: /tmp/tada_bench/ex02_enunciated.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_happy

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_happy.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_happy
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.7s total (load 0.7s, gen 6.6s, decode 2.4s)
- Audio duration: 3.2s
- RTF: 2.06x

**Output**: /tmp/tada_bench/ex02_happy.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_laughing

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_laughing.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_laughing
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.0s total (load 0.7s, gen 6.2s, decode 3.1s)
- Audio duration: 4.1s
- RTF: 1.51x

**Output**: /tmp/tada_bench/ex02_laughing.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_sad

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_sad.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_sad
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.6s total (load 0.7s, gen 6.2s, decode 2.7s)
- Audio duration: 3.6s
- RTF: 1.72x

**Output**: /tmp/tada_bench/ex02_sad.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex02_whisper

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex02_whisper.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex02_whisper
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.5s total (load 0.7s, gen 7.3s, decode 2.5s)
- Audio duration: 3.4s
- RTF: 2.15x

**Output**: /tmp/tada_bench/ex02_whisper.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_confused

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_confused.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_confused
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.3s total (load 0.7s, gen 5.7s, decode 2.9s)
- Audio duration: 3.9s
- RTF: 1.46x

**Output**: /tmp/tada_bench/ex03_confused.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_default

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_default.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_default
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.3s total (load 0.7s, gen 7.9s, decode 1.7s)
- Audio duration: 2.0s
- RTF: 3.95x

**Output**: /tmp/tada_bench/ex03_default.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_enunciated

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_enunciated.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_enunciated
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.4s total (load 0.7s, gen 5.7s, decode 3.0s)
- Audio duration: 3.3s
- RTF: 1.73x

**Output**: /tmp/tada_bench/ex03_enunciated.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_happy

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_happy.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_happy
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.4s total (load 0.7s, gen 7.9s, decode 1.8s)
- Audio duration: 2.2s
- RTF: 3.59x

**Output**: /tmp/tada_bench/ex03_happy.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_laughing

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_laughing.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_laughing
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.2s total (load 0.7s, gen 5.9s, decode 2.6s)
- Audio duration: 3.1s
- RTF: 1.90x

**Output**: /tmp/tada_bench/ex03_laughing.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_sad

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_sad.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_sad
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.8s total (load 0.7s, gen 6.7s, decode 2.4s)
- Audio duration: 3.1s
- RTF: 2.16x

**Output**: /tmp/tada_bench/ex03_sad.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex03_whisper

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex03_whisper.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex03_whisper
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.5s total (load 0.7s, gen 6.7s, decode 2.1s)
- Audio duration: 2.8s
- RTF: 2.39x

**Output**: /tmp/tada_bench/ex03_whisper.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 11 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex04_confused

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex04_confused.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex04_confused
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.4s total (load 0.7s, gen 6.9s, decode 2.8s)
- Audio duration: 3.5s
- RTF: 1.97x

**Output**: /tmp/tada_bench/ex04_confused.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex04_enunciated

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex04_enunciated.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex04_enunciated
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 8.9s total (load 0.7s, gen 5.4s, decode 2.8s)
- Audio duration: 3.4s
- RTF: 1.59x

**Output**: /tmp/tada_bench/ex04_enunciated.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex04_laughing

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex04_laughing.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex04_laughing
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 11.1s total (load 0.7s, gen 6.9s, decode 3.5s)
- Audio duration: 4.5s
- RTF: 1.53x

**Output**: /tmp/tada_bench/ex04_laughing.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex04_sad

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex04_sad.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex04_sad
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 10.8s total (load 0.7s, gen 7.6s, decode 2.5s)
- Audio duration: 3.1s
- RTF: 2.45x

**Output**: /tmp/tada_bench/ex04_sad.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 12 frames without hitting EOS — likely degraded model quality   ❌ 1 sanity check(s) failed — output is likely garbage 

---

## 2026-05-07 — voice-matrix-sweep — 290e602 — ex04_whisper

**Date**: 2026-05-07
**Commit**: 290e602
**Purpose**: Automated voice sweep — ex04_whisper.

**Parameters**:
- engine: candle
- device: metal
- model: Var-C VV-Q8 E-Q4 (1.3G)
- voice: ex04_whisper
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: 0.9
- temperature: 0.6
- cfg_scale: 1.6
- flow_steps: 10
- top_p: 0.9
- repetition_penalty: 1.1
- transition_steps: 5
- seed: 42

**Timings**:
- Wall-clock: 9.6s total (load 0.7s, gen 6.9s, decode 2.0s)
- Audio duration: 2.6s
- RTF: 2.65x

**Output**: /tmp/tada_bench/ex04_whisper.wav

**Notes**:   ⚠ Generation sanity checks FAILED:     - NO_EOS: model generated 10 frames without hitting EOS — likely degraded model quality   ⚠ Audio sanity checks FAILED: 

---

## 2026-05-08 — wasm-perf-instrument

**Date**: 2026-05-08
**Commit**: (see below)
**Purpose**: Instrument `web/tada-worker.js` to emit timing metrics; audit `tasks_max` source location and SIMD128 opcode presence in the production WASM binary.

**Worker instrumentation**:
- Added `performance.now()` brackets around the generation loop and `engine.decodeAudio()` call
- After decode, emits `self.postMessage({type: 'timing', gen_ms, decode_ms, audio_duration_ms, rtf})` and `console.log('[tada-timing]', JSON.stringify({...}))` so both the Playwright test (console scrape) and the page UI (postMessage) can capture timing

**tasks_max audit**:
- Source: `crates/tada-wasm/src/lib.rs:714`
- Current value: `tasks_max: 512`
- Mesh assumption (lib.rs:714) was correct

**SIMD128 audit**:
- Tool: `wasm-dis` (binaryen, via brew)
- File: `crates/tada-wasm/pkg/tada_wasm_bg.wasm` (13,710,066 bytes, built 2026-05-07)
- Opcode count (i32x4|f32x4|v128|i8x16|i16x8|f64x2): **189,650**
- Conclusion: SIMD128 **ON**

---

## Task 3 (SIMD fix conditional) — auto-closed

SIMD already on, no fix needed.

(Task 1's wasm-objdump audit found 189,650 SIMD opcodes; Task 3 was a conditional fix-if-off and is now [x] without code change.)

---

## wasm-baseline

**Date**: 2026-05-08
**Commit**: (see below)
**Purpose**: Capture first headless WASM RTF measurement for TADA. Establishes the current wasm performance floor for subsequent optimization tasks (CFG A/B, tasks_max sweep, etc.).

**Method**: Playwright headless Chromium with WebGPU (`--enable-unsafe-webgpu --use-angle=metal`). Cache stub injected into tada-worker.js to bypass the 1.3GB GGUF caching quota. Focused single-model script `scripts/measure_tada_baseline.mjs` — no Pocket TTS or KittenTTS to minimize memory pressure. Server POST `/timings` endpoint wrote result to `/tmp/wasm-timings.jsonl`.

**Parameters**:
- engine: burn+candle (wgpu LLM + candle CPU VV + candle CPU decoder)
- device: wgpu+cpu
- model: Var-C (VV-Q8_0, embed-Q4_0), SIMD128 on, tasks_max=512, cfg_scale=1.6
- size: 1.3G
- voice: ex01_default (matrix speaker ex01, default style)
- text: fox ("The quick brown fox jumps over the lazy dog.")
- seed: browser default

**Results** (from /tmp/wasm-timings.jsonl):
- gen_ms: 14406 (14.41s)
- decode_ms: 6358 (6.36s)
- audio_duration_ms: 3200 (3.20s)
- RTF: 6.49x (total wall = gen + decode = 20.77s, audio = 3.20s)

**Observations**:
- WebGPU functional in headless Chromium with --use-angle=metal on macOS.
- Model loaded successfully on first headless run (no crash).
- RTF 6.49x is well above real-time. Breakdown: gen 14.41s (69%), decode 6.36s (31%).
- Native Var-C metal reference: gen ~2.2s, decode ~2.1s, RTF ~0.80x (postfix-varC-fox row).
- WASM gen is ~6.5x slower than native candle metal; decode is ~3.0x slower.
- Decode bottleneck on CPU (candle SIMD) is significant — 6.36s for a 3.2s audio clip.

---

## 2026-05-08 — wasm-cfg-ab — 2e363de
- Method: Playwright headless TADA-only, fresh session each (cfg slider set before ex01 click)
- Run A (cfg=1.6): gen 14.52s, decode 6.34s, audio 3.20s, RTF 6.52x → tada-cfg16-fox.wav (153632 bytes)
- Run B (cfg=1.0): gen 8.70s, decode 4.16s, audio 2.04s, RTF 6.30x → tada-cfg10-fox.wav (97952 bytes)
- Speedup: cfg10_rtf / cfg16_rtf = 0.967 (-3.3% change)
- CFG hypothesis: CONFIRMED — CFG-off is 3.3% faster
- QUALITY: UNVERIFIED — pending user audition. Both files in /tmp/. User listens before any commit lands cfg=1.0 as production default.
