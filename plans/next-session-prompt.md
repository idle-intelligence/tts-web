# Next Session Handoff — TADA-1B TTS

## What This Is

Browser-based TTS engine for TADA-1B model.
Stack: Rust (candle + Burn ML frameworks) → WASM + WebGPU for browser deployment.

Repository: `/Users/tc/Code/idle-intelligence/tts-web`
Branch: `feat/burn-wgpu-llama`

---

## Session 2026-03-14 to 2026-03-20 — Major Progress

### Benchmarks & Quality

**Post-bugfix benchmark round** (9 GGUF variants × 4 phrases):
- All variants produce perceptually equivalent audio post-bugfix
- Mixed/Var-C (1.3G) hit sub-1x RTF on native Metal
- Q4_K_M Var-C (1.38G) added — near-identical quality to Q4_0

**Quality findings**:
- fox: OK across all variants
- call: sounds good, truncated early (model issue)
- tyger: universally bad pronunciation (model issue, not quantization)
- wutang: good with voice switch mid-phrase (weak voice conditioning)
- Python BF16 reference: "complete but quieter, more noise/wind"

**Voice prompts**:
- ljspeech (5 tokens) → too short for stable conditioning
- ljspeech_long (32 tokens) → much better with Python reference
- Rickie from SBCSAE (75 tokens) → natural conversational voice
- amazement/amusement from fb_ears → emotional voices
- The "voice prompt leaking" issue (hearing end of prompt text in output) needs investigation

### RoPE Bug Fix (THE breakthrough)

**Bug**: Burn's RoPE used interleaved pairs `(x[0],x[1]), (x[2],x[3])...` instead of Llama's split-half convention `(x[i], x[i+d/2])`.

**Impact**: Step 1 hidden states matched (cosine 0.9997) but step 2 completely diverged (cosine 0.005). All Burn GPU audio was gibberish.

**Fix**: One function change in `crates/tada-wasm/src/model/rope.rs` — split into first/second half instead of interleaved pairs.

**Verification**: After fix, step 2 cosine = 0.99971. F32 Burn vs F32 candle: cosine = 1.000000.

### Investigation Trail (for the write-up)

1. **Suspected precision issue** → tested dot() replacement in WGSL shader → no effect
2. **Suspected KV cache bug** → tested ring buffer vs concat → no effect
3. **Suspected tensor stride issue** → tested materialization → no effect
4. **F32 test proved**: Burn GPU operators are CORRECT for F32 (cosine 1.0 at step 1)
5. **F32 step 2 still diverged** → proved it's NOT a precision issue but a LOGIC bug
6. **Per-layer analysis**: cosine ~0.99999 at every layer, max_diff grows but RMSNorm normalizes
7. **Matmul sanity check**: cat+swap_dims+matmul correct for small tensors
8. **Debug dumps** revealed: divergence starts AFTER RoPE, not before
9. **RoPE convention identified**: interleaved vs split-half → ONE LINE FIX

### Quantization

- **Q4_K_M** quantization implemented in `quantize_tada.py` (k-quant super-blocks)
- **Q8_0 WGSL shader** written for VibeVoice GPU path
- Q8_0 LLM backbone tested: matches F16 on fox/call (exact sample count)
- Mixed/Var-C/Q4_K_M all produce good audio with correct voice conditioning

### WASM Deployment

**Memory optimizations**:
- Eliminated triple-copy of GGUF bytes (was 3× 1.38 GB → now 1× 1.38 GB)
- `load_gguf_no_llm` skips LLM in candle (saves ~620 MB)
- `load_gguf_no_llm_no_vv` skips VV when Burn handles it (saves ~350 MB more)
- Dummy LLM with no embed_tokens dequant (saves ~1 GB)
- F32Embedding stores data on CPU (avoids sync GPU readback panic on WASM)
- Token embedding uses Q4 CPU EmbeddingStore (no GPU readback)
- `into_data_async().await` for all GPU→CPU transfers

**WebGPU performance optimizations**:
- `tasks_max=512` (was 32 default) — batches entire LLM step into one queue.submit()
- GPU warmup: 5 LLM forward passes + 2 VV ODE steps to pre-compile all WGSL shaders
- VV skip during prompt phase — prompt acoustic frames are stripped, no need for VV
- Q8_0 WGSL shader for VibeVoice on GPU (551ms/step vs 595ms CPU — 8% faster)

**Performance evolution (traced)**:

| Config | Total | Content step | Decode | Notes |
|--------|-------|-------------|--------|-------|
| Original (no opts) | 40.1s | 486ms | 6.2s | 56 steps |
| VV skip + CPU warmup | 31.6s | 554ms | 9.7s | 25 steps |
| tasks_max=512 | 53.1s* | 540ms | 8.2s | *longer voice prompt |
| Q8_0 VV GPU | 50.6s | 551ms | 17.8s | Decode regressed |
| + VV warmup | TBD | TBD | TBD | Building |

*dispatch gap reduced 6x (340ms → 55ms per step)

### VV on GPU Experiment

**F32 VV on GPU**: 98.7s (3x WORSE) — Burn's generic F32 matmul on WebGPU is 20x slower than candle CPU Q8_0
**Q8_0 VV on GPU**: 551ms/step (8% faster than CPU) — custom Q8_0 WGSL shader
**Key lesson**: naive 1-thread-per-element shader works for LLM (50 dispatches/step) but struggles for VV (180 dispatches/step). Q8_0 is viable but not a dramatic win.

### Frontend

- Voice selector with 4 voices (LJSpeech, Amazement, Amusement, Rickie)
- Clicking voice populates demo text, user clicks Generate
- Parameter sliders: Noise (0.9), CFG (1.6), Steps (20) — model defaults
- Worker handles voice loading/switching
- Auto-populates first demo phrase on model ready

### Known Issues

- **Decode is the new bottleneck**: 17.8s in WASM (native: 1.5s)
- **First generation**: 10-16s warmup (shaders compiling). VV warmup added but untested.
- **Voice prompt leaking**: first word of voice text sometimes audible in output
- **Audio quality in browser**: untested with correct params + voice prompt

---

## Key Files

| File | Purpose |
|------|---------|
| `crates/tada-wasm/src/lib.rs` | WASM bindings, HybridTadaModel, generation loop |
| `crates/tada-wasm/src/model/rope.rs` | RoPE (FIXED: split-half convention) |
| `crates/tada-wasm/src/model/vibevoice.rs` | BurnVibeVoice (Q8_0 GPU flow matching) |
| `crates/tada-wasm/src/model/kv_cache.rs` | KV cache (concat-based) |
| `crates/tada-wasm/src/wgsl/shader_naive.wgsl` | Q4_0 matmul WGSL shader |
| `crates/tada-wasm/src/wgsl/shader_q8_0.wgsl` | Q8_0 matmul WGSL shader |
| `crates/tada-wasm/src/gguf.rs` | GGUF loader, Q4/Q8 tensors, embeddings |
| `crates/tada-core/src/flow_matching.rs` | ODE solver + CFG support |
| `crates/tada-core/src/tada_model.rs` | Candle model (load_gguf_no_llm_no_vv) |
| `scripts/quantize_tada.py` | Q4_0/Q8_0/Q4_K quantization |
| `scripts/analyze-trace.py` | Chrome DevTools trace analyzer |
| `scripts/precompute_voice.py` | Voice prompt precomputation |
| `web/tada-worker.js` | WASM worker (voice loading, generation loop) |
| `web/index.html` | Frontend (voice selector, parameter sliders) |
| `docs/results.md` | Benchmark results (pure data) |
| `docs/lab-notebook.md` | Experiment log (parameters + user feedback) |
| `plans/perf-plan.md` | Performance analysis + optimization roadmap |

---

## Model Files

```
Tokenizer:    /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json
Var-C (1.3G): /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-C-vvq8-eq4.gguf  ← default for web
Q4_K_M (1.4G): /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-mixed-llmq4_k-vvq8_0-eq4_0.gguf
Q4_0 (2.6G): /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf
F32 (6.5G):  DELETE — no longer needed
```

Symlink in repo root: `tada-1b-q4_0.gguf → tada-1b-C-vvq8-eq4.gguf`

---

## Build Commands

```bash
# WASM build
wasm-pack build crates/tada-wasm --target web --release -- --features wasm --no-default-features

# Native Burn example (voice-prompted)
cargo run --example tada_generate_burn -p tada-wasm --release -- \
  --model /Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-C-vvq8-eq4.gguf \
  --tokenizer /Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json \
  --voice voices/ljspeech_long.safetensors \
  --noise-temp 0.9 --transition-steps 5 --seed 42 --cfg-scale 1.6 \
  --output /tmp/test.wav --text "The quick brown fox jumps over the lazy dog."

# Dev server
node web/serve.mjs  # port 8081

# Analyze Chrome trace
python scripts/analyze-trace.py /path/to/Trace-*.json
```
