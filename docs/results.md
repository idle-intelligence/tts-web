# results.md

Pure benchmark data. No analysis. Each block is one experiment; use the ID to cross-reference lab-notebook.md.

Columns: ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File

---

## first-python-ref

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| first-python-ref-bf16-fox | python-bf16 | cpu | BF16 baseline | — | fox | ~10 | ~15 | ~1 | 5.58 | ~2.7x | bench1_python_bf16.wav |

---

## timing-v2

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| timing-v2-f16-cpu-fox | candle | cpu | F16 | — | fox | 7.4 | 19.0 | 2.4 | 3.68 | 5.17x | bench3_run2_rust_f16_cpu.wav |
| timing-v2-f16-metal-fox | candle | metal | F16 | — | fox | 4.5 | 19.9 | 3.2 | 3.68 | 5.42x | bench4_run2_rust_f16_metal.wav |
| timing-v2-q4-cpu-fox | candle | cpu | Q4_0 baseline | — | fox | 1.4 | 14.2 | 1.6 | 3.04 | 4.66x | bench5_run2_rust_q4_cpu.wav |
| timing-v2-q4-metal-fox | candle | metal | Q4_0 baseline | — | fox | 0.9 | 14.2 | 1.6 | 3.04 | 4.66x | bench6_run2_rust_q4_metal.wav |

---

## metal-discovery

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| metal-discovery-q4-metal-fox | candle | metal | Q4_0 baseline | — | fox | 2.6 | 6.8 | 2.5 | 3.04 | 2.25x | bench6_run3_rust_q4_metal.wav |

---

## burn-hybrid-zeroshot

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| burn-hybrid-zeroshot-candle-cpu-fox | candle | cpu | Q4_0 baseline | — | fox | 1.8 | 88.2 | 15.7 | 22.8 | 3.87x | bench7_run3_candle_q4_cpu.wav |
| burn-hybrid-zeroshot-burn-gpu-fox | burn+candle | wgpu+cpu | Q4_0 baseline | — | fox | 1.8 | 14.7 | 2.2 | 4.28 | 3.43x | bench7_run3_burn_q4_gpu.wav |

---

## bench8-noise-discovery

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| bench8-noise-q4-06-fox | candle | metal | Q4_0 baseline | 2.64G | fox | 3.7 | 6.9 | 2.6 | 3.18 | 2.18x | bench8_run1_rust_q4_metal.wav |
| bench8-noise-q4-09-fox | candle | metal | Q4_0 baseline | 2.64G | fox | 3.7 | 7.1 | 2.6 | 3.04 | 2.32x | bench8_run2_rust_q4_metal.wav |
| bench8-noise-mixv1-fox | candle | metal | Mixed v1 dec-Q4_0 | 1.48G | fox | 1.0 | 2.2 | 3.4 | 4.14 | 0.54x | bench8_run3_mixedv1_metal.wav |
| bench8-noise-mixv2-fox | candle | metal | Mixed v2 dec-Q8_0 | 1.52G | fox | 1.0 | 2.3 | 3.5 | 4.14 | 0.56x | bench8_run4_mixedv2_metal.wav |

---

## bench8-phrases

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| bench8-phrases-q4-tyger | candle | metal | Q4_0 baseline | 2.64G | tyger | — | — | — | — | — | bench8_run5a_q4_tyger.wav |
| bench8-phrases-mix-tyger | candle | metal | Mixed Q4+Q8 | 1.52G | tyger | — | — | — | — | — | bench8_run5b_mixed_tyger.wav |
| bench8-phrases-q4-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | — | — | bench8_run6a_q4_.wav |
| bench8-phrases-mix-time | candle | metal | Mixed Q4+Q8 | 1.52G | time | — | — | — | — | — | bench8_run6b_mixed_.wav |
| bench8-phrases-q4-universe | candle | metal | Q4_0 baseline | 2.64G | universe | — | — | — | — | — | bench8_run7a_q4_time.wav |

---

## variant-precision-sweep

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| variant-precision-baseline-fox | candle | metal | Q4_0 baseline | 2.64G | fox | — | — | — | 3.04 | — | — |
| variant-precision-baseline-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | 3.88 | — | — |
| variant-precision-A-fox | candle | metal | Var-A VV-F16 E-Q8 | 1.88G | fox | — | — | — | 3.04 | — | — |
| variant-precision-A-time | candle | metal | Var-A VV-F16 E-Q8 | 1.88G | time | — | — | — | 4.98 | — | — |
| variant-precision-B-fox | candle | metal | Var-B VV-F32 E-Q8 | 2.68G | fox | — | — | — | 3.04 | — | — |
| variant-precision-B-time | candle | metal | Var-B VV-F32 E-Q8 | 2.68G | time | — | — | — | 4.98 | — | — |
| variant-precision-C-fox | candle | metal | Var-C VV-Q8 E-Q4 | 1.38G | fox | — | — | — | 4.10 | — | — |
| variant-precision-C-time | candle | metal | Var-C VV-Q8 E-Q4 | 1.38G | time | — | — | — | 4.98 | — | — |
| variant-precision-D-fox | candle | metal | Var-D VV-Q8 E-Q8 | 1.52G | fox | — | — | — | 4.14 | — | — |
| variant-precision-D-time | candle | metal | Var-D VV-Q8 E-Q8 | 1.52G | time | — | — | — | 4.98 | — | — |
| variant-precision-E-fox | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | fox | — | — | — | 3.04 | — | — |
| variant-precision-E-time | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | time | — | — | — | 3.88 | — | — |

---

## exp2-variant-sweep

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| exp2-baseline-fox | candle | metal | Q4_0 baseline | 2.64G | fox | — | — | — | — | — | exp_baseline_fox.wav |
| exp2-baseline-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | — | — | exp_baseline_time.wav |
| exp2-varE-tyger | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | tyger | — | — | — | 3.00 | — | exp2_varE_tyger.wav |
| exp2-varE-woods | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | woods | — | — | — | 4.78 | — | exp2_varE_woods.wav |
| exp2-varE-smile | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | smile | — | — | — | 7.66 | — | exp2_varE_smile.wav |
| exp2-varE-call | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | call | — | — | — | 3.10 | — | exp2_varE_call.wav |
| exp2-varE-wutang | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | wutang | — | — | — | 8.92 | — | exp2_varE_wutang.wav |
| exp2-varE-universe | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | universe | — | — | — | 3.96 | — | exp2_varE_universe.wav |
| exp2-f16-fox | candle | cpu | F16 | 3.3G | fox | — | — | — | 3.68 | 4.88x | exp2_f16_fox.wav |
| exp2-f16-time | candle | cpu | F16 | 3.3G | time | — | — | — | 3.14 | 6.15x | exp2_f16_time.wav |
| exp2-f16-tyger | candle | cpu | F16 | 3.3G | tyger | — | — | — | 3.38 | 6.07x | exp2_f16_tyger.wav |
| exp2-f16-call | candle | cpu | F16 | 3.3G | call | — | — | — | 3.06 | 6.28x | exp2_f16_call.wav |

---

## exp3-eos-fix

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| exp3-pyref-fox | python-bf16 | cpu | BF16 baseline | — | fox | — | — | — | 5.58 | — | exp3_pyref_fox.wav |
| exp3-pyref-time | python-bf16 | cpu | BF16 baseline | — | time | — | — | — | 4.50 | — | exp3_pyref_time.wav |
| exp3-pyref-tyger | python-bf16 | cpu | BF16 baseline | — | tyger | — | — | — | 6.38 | — | exp3_pyref_tyger.wav |
| exp3-pyref-call | python-bf16 | cpu | BF16 baseline | — | call | — | — | — | 2.68 | — | exp3_pyref_call.wav |
| exp3-q4-fox | candle | metal | Q4_0 baseline | 2.64G | fox | — | — | — | 2.80 | — | exp3_q4_fox.wav |
| exp3-q4-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | 3.54 | — | exp3_q4_time.wav |
| exp3-q4-smile | candle | metal | Q4_0 baseline | 2.64G | smile | — | — | — | 11.20 | — | exp3_q4_smile.wav |
| exp3-varE-fox | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | fox | — | — | — | 2.80 | — | exp3_varE_fox.wav |
| exp3-varE-time | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | time | — | — | — | 4.64 | — | exp3_varE_time.wav |
| exp3-varE-call | candle | metal | Var-E VV-F16 E-Q4 | 1.75G | call | — | — | — | 2.62 | — | exp3_varE_call.wav |

---

## exp5-f32-baseline

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| exp5-f32-fox | candle | cpu | F32 | 6.5G | fox | — | 20.5 | — | 3.44 | 5.95x | exp5_f32_fox.wav |
| exp5-f32-time | candle | cpu | F32 | 6.5G | time | — | 21.4 | — | 3.12 | 6.84x | exp5_f32_time.wav |
| exp5-f32-smile | candle | cpu | F32 | 6.5G | smile | — | 24.1 | — | 4.28 | 5.63x | exp5_f32_smile.wav |
| exp5-f32-call | candle | cpu | F32 | 6.5G | call | — | 21.5 | — | 2.40 | 8.96x | exp5_f32_call.wav |

---

## exp8-all-fixes

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| exp8-q4-fox | candle | metal | Q4_0 baseline | 2.64G | fox | — | — | — | — | — | exp8_q4_fox.wav |
| exp8-q4-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | — | — | exp8_q4_time.wav |
| exp8-q4-smile | candle | metal | Q4_0 baseline | 2.64G | smile | — | — | — | — | — | exp8_q4_smile.wav |
| exp8-q4-call | candle | metal | Q4_0 baseline | 2.64G | call | — | — | — | — | — | exp8_q4_call.wav |

---

## exp9-eot-trim

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| exp9-q4-fox | candle | metal | Q4_0 baseline | 2.64G | fox | — | — | — | 2.74 | — | exp9_q4_fox.wav |
| exp9-q4-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | 2.34 | — | exp9_q4_time.wav |
| exp9-q4-smile | candle | metal | Q4_0 baseline | 2.64G | smile | — | — | — | 4.58 | — | exp9_q4_smile.wav |
| exp9-q4-call | candle | metal | Q4_0 baseline | 2.64G | call | — | — | — | 1.56 | — | exp9_q4_call.wav |

---

## exp10-final-comparison

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| exp10-q4-fox | candle | metal | Q4_0 baseline | 2.64G | fox | — | — | — | 2.74 | — | exp10_q4_fox.wav |
| exp10-q4-time | candle | metal | Q4_0 baseline | 2.64G | time | — | — | — | 2.34 | — | exp10_q4_time.wav |
| exp10-q4-smile | candle | metal | Q4_0 baseline | 2.64G | smile | — | — | — | 4.58 | — | exp10_q4_smile.wav |
| exp10-q4-call | candle | metal | Q4_0 baseline | 2.64G | call | — | — | — | 1.56 | — | exp10_q4_call.wav |
| exp10-q4-tyger | candle | metal | Q4_0 baseline | 2.64G | tyger | — | — | — | 2.54 | — | exp10_q4_tyger.wav |
| exp10-q4-woods | candle | metal | Q4_0 baseline | 2.64G | woods | — | — | — | 3.54 | — | exp10_q4_woods.wav |
| exp10-q4-universe | candle | metal | Q4_0 baseline | 2.64G | universe | — | — | — | 3.92 | — | exp10_q4_universe.wav |
| exp10-q4-wutang | candle | metal | Q4_0 baseline | 2.64G | wutang | — | — | — | 4.58 | — | exp10_q4_wutang.wav |
| exp10-f32-fox | candle | cpu | F32 | 6.5G | fox | — | 63.2 | — | 2.74 | 23.1x | exp10_f32_fox.wav |
| exp10-f32-time | candle | cpu | F32 | 6.5G | time | — | 58.3 | — | 2.50 | 23.3x | exp10_f32_time.wav |
| exp10-f32-smile | candle | cpu | F32 | 6.5G | smile | — | 60.0 | — | 4.64 | 12.9x | exp10_f32_smile.wav |
| exp10-f32-call | candle | cpu | F32 | 6.5G | call | — | 21.6 | — | 1.08 | 20.0x | exp10_f32_call.wav |
| exp10-f32-tyger | candle | cpu | F32 | 6.5G | tyger | — | 22.3 | — | 2.50 | 8.9x | exp10_f32_tyger.wav |
| exp10-f32-woods | candle | cpu | F32 | 6.5G | woods | — | — | — | 3.50 | — | exp10_f32_woods.wav |
| exp10-f32-universe | candle | cpu | F32 | 6.5G | universe | — | — | — | 4.28 | — | exp10_f32_universe.wav |
| exp10-f32-wutang | candle | cpu | F32 | 6.5G | wutang | — | — | — | 5.08 | — | exp10_f32_wutang.wav |

---

## postfix-full-sweep

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| postfix-python-fox | python-bf16 | cpu | BF16 baseline | — | fox | 2.0 | 4.68 | — | 5.58 | 0.84x | bench_2026-03-13/python_bf16/fox.wav |
| postfix-python-call | python-bf16 | cpu | BF16 baseline | — | call | 2.0 | 4.77 | — | 2.68 | 1.78x | bench_2026-03-13/python_bf16/call.wav |
| postfix-python-tyger | python-bf16 | cpu | BF16 baseline | — | tyger | 2.0 | 4.15 | — | 6.06 | 0.68x | bench_2026-03-13/python_bf16/tyger.wav |
| postfix-python-wutang | python-bf16 | cpu | BF16 baseline | — | wutang | 2.0 | 7.01 | — | 6.74 | 1.04x | bench_2026-03-13/python_bf16/wutang.wav |
| postfix-f32-fox | candle | metal | F32 GGUF | 6.5G | fox | 38.1 | 40.0 | 2.3 | 2.7 | 14.61x | bench_2026-03-13/f32/fox.wav |
| postfix-f32-call | candle | metal | F32 GGUF | 6.5G | call | 38.5 | 26.7 | 0.9 | 1.1 | 24.72x | bench_2026-03-13/f32/call.wav |
| postfix-f32-tyger | candle | metal | F32 GGUF | 6.5G | tyger | 37.4 | 29.4 | 1.6 | 2.1 | 14.29x | bench_2026-03-13/f32/tyger.wav |
| postfix-f32-wutang | candle | metal | F32 GGUF | 6.5G | wutang | 37.5 | 39.3 | 3.3 | 4.2 | 9.37x | bench_2026-03-13/f32/wutang.wav |
| postfix-f16-fox | candle | metal | F16 GGUF | 3.3G | fox | 10.6 | 19.1 | 2.2 | 2.7 | 6.97x | bench_2026-03-13/f16/fox.wav |
| postfix-f16-call | candle | metal | F16 GGUF | 3.3G | call | 8.0 | 17.1 | 1.0 | 1.1 | 15.87x | bench_2026-03-13/f16/call.wav |
| postfix-f16-tyger | candle | metal | F16 GGUF | 3.3G | tyger | 7.2 | 16.0 | 1.6 | 2.1 | 7.76x | bench_2026-03-13/f16/tyger.wav |
| postfix-f16-wutang | candle | metal | F16 GGUF | 3.3G | wutang | 6.6 | 20.5 | 3.4 | 4.2 | 4.87x | bench_2026-03-13/f16/wutang.wav |
| postfix-q4-fox | candle | metal | Q4_0 baseline | 2.6G | fox | 2.5 | 7.1 | 2.3 | 2.8 | 2.58x | bench_2026-03-13/q4_0_baseline/fox.wav |
| postfix-q4-call | candle | metal | Q4_0 baseline | 2.6G | call | 2.7 | 7.4 | 1.3 | 1.6 | 4.77x | bench_2026-03-13/q4_0_baseline/call.wav |
| postfix-q4-tyger | candle | metal | Q4_0 baseline | 2.6G | tyger | 2.7 | 6.4 | 1.5 | 1.8 | 3.53x | bench_2026-03-13/q4_0_baseline/tyger.wav |
| postfix-q4-wutang | candle | metal | Q4_0 baseline | 2.6G | wutang | 2.3 | 9.4 | 3.2 | 3.9 | 2.40x | bench_2026-03-13/q4_0_baseline/wutang.wav |
| postfix-varB-fox | candle | metal | Var-B VV-F32 E-Q8 | 2.5G | fox | 2.9 | 6.9 | 2.2 | 2.8 | 2.51x | bench_2026-03-13/var_b_vv_f32_e_q8/fox.wav |
| postfix-varB-call | candle | metal | Var-B VV-F32 E-Q8 | 2.5G | call | 3.0 | 7.5 | 1.3 | 1.6 | 4.83x | bench_2026-03-13/var_b_vv_f32_e_q8/call.wav |
| postfix-varB-tyger | candle | metal | Var-B VV-F32 E-Q8 | 2.5G | tyger | 2.8 | 6.3 | 1.6 | 1.9 | 3.33x | bench_2026-03-13/var_b_vv_f32_e_q8/tyger.wav |
| postfix-varB-wutang | candle | metal | Var-B VV-F32 E-Q8 | 2.5G | wutang | 2.7 | 9.6 | 3.3 | 4.2 | 2.29x | bench_2026-03-13/var_b_vv_f32_e_q8/wutang.wav |
| postfix-varA-fox | candle | metal | Var-A VV-F16 E-Q8 | 1.9G | fox | 1.4 | 6.9 | 2.3 | 2.8 | 2.49x | bench_2026-03-13/var_a_vv_f16_e_q8/fox.wav |
| postfix-varA-call | candle | metal | Var-A VV-F16 E-Q8 | 1.9G | call | 1.1 | 7.4 | 1.3 | 1.6 | 4.75x | bench_2026-03-13/var_a_vv_f16_e_q8/call.wav |
| postfix-varA-tyger | candle | metal | Var-A VV-F16 E-Q8 | 1.9G | tyger | 1.1 | 6.3 | 1.6 | 1.9 | 3.29x | bench_2026-03-13/var_a_vv_f16_e_q8/tyger.wav |
| postfix-varA-wutang | candle | metal | Var-A VV-F16 E-Q8 | 1.9G | wutang | 1.1 | 9.4 | 3.4 | 4.2 | 2.23x | bench_2026-03-13/var_a_vv_f16_e_q8/wutang.wav |
| postfix-varE-fox | candle | metal | Var-E VV-F16 E-Q4 | 1.8G | fox | 1.3 | 6.6 | 2.2 | 2.8 | 2.39x | bench_2026-03-13/var_e_vv_f16_e_q4/fox.wav |
| postfix-varE-call | candle | metal | Var-E VV-F16 E-Q4 | 1.8G | call | 1.0 | 7.1 | 1.3 | 1.6 | 4.54x | bench_2026-03-13/var_e_vv_f16_e_q4/call.wav |
| postfix-varE-tyger | candle | metal | Var-E VV-F16 E-Q4 | 1.8G | tyger | 1.0 | 5.9 | 1.5 | 1.8 | 3.26x | bench_2026-03-13/var_e_vv_f16_e_q4/tyger.wav |
| postfix-varE-wutang | candle | metal | Var-E VV-F16 E-Q4 | 1.8G | wutang | 1.0 | 9.3 | 3.2 | 3.9 | 2.37x | bench_2026-03-13/var_e_vv_f16_e_q4/wutang.wav |
| postfix-mixed-fox | candle | metal | Mixed VV-Q8 E-Q8 | 1.4G | fox | 1.0 | 2.3 | 2.3 | 2.7 | 0.84x | bench_2026-03-13/mixed_vv_q8_e_q8/fox.wav |
| postfix-mixed-call | candle | metal | Mixed VV-Q8 E-Q8 | 1.4G | call | 0.8 | 2.4 | 1.3 | 1.6 | 1.54x | bench_2026-03-13/mixed_vv_q8_e_q8/call.wav |
| postfix-mixed-tyger | candle | metal | Mixed VV-Q8 E-Q8 | 1.4G | tyger | 0.8 | 2.1 | 1.6 | 1.9 | 1.08x | bench_2026-03-13/mixed_vv_q8_e_q8/tyger.wav |
| postfix-mixed-wutang | candle | metal | Mixed VV-Q8 E-Q8 | 1.4G | wutang | 0.7 | 2.9 | 3.5 | 4.4 | 0.66x | bench_2026-03-13/mixed_vv_q8_e_q8/wutang.wav |
| postfix-varC-fox | candle | metal | Var-C VV-Q8 E-Q4 | 1.3G | fox | 0.9 | 2.2 | 2.1 | 2.7 | 0.80x | bench_2026-03-13/var_c_vv_q8_e_q4/fox.wav |
| postfix-varC-call | candle | metal | Var-C VV-Q8 E-Q4 | 1.3G | call | 0.7 | 2.4 | 1.3 | 1.6 | 1.52x | bench_2026-03-13/var_c_vv_q8_e_q4/call.wav |
| postfix-varC-tyger | candle | metal | Var-C VV-Q8 E-Q4 | 1.3G | tyger | 0.7 | 2.1 | 1.5 | 1.8 | 1.15x | bench_2026-03-13/var_c_vv_q8_e_q4/tyger.wav |
| postfix-varC-wutang | candle | metal | Var-C VV-Q8 E-Q4 | 1.3G | wutang | 0.7 | 2.9 | 3.3 | 3.9 | 0.75x | bench_2026-03-13/var_c_vv_q8_e_q4/wutang.wav |

---

## q8-llm-alignment

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| q8-llm-fox | candle | metal | Q8_0 LLM + F16 VV + Q4_0 E | 2.24G | fox | 2.4 | 7.1 | 2.2 | 2.7 | 2.57x | bench_2026-03-13/llmq8_vvf16_eq4/fox.wav |
| q8-llm-call | candle | metal | Q8_0 LLM + F16 VV + Q4_0 E | 2.24G | call | 2.0 | 7.4 | 0.9 | 1.1 | 6.89x | bench_2026-03-13/llmq8_vvf16_eq4/call.wav |
| q8-llm-tyger | candle | metal | Q8_0 LLM + F16 VV + Q4_0 E | 2.24G | tyger | 1.9 | 6.2 | 1.4 | 1.7 | 3.65x | bench_2026-03-13/llmq8_vvf16_eq4/tyger.wav |
| q8-llm-wutang | candle | metal | Q8_0 LLM + F16 VV + Q4_0 E | 2.24G | wutang | 1.9 | 9.5 | 3.1 | 4.0 | 2.35x | bench_2026-03-13/llmq8_vvf16_eq4/wutang.wav |

---

## burn-gpu-zeroshot-v2

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| burn-gpu-v2-fox | burn+candle | wgpu+cpu | Q4_0 baseline | 2.6G | fox | 2.5 | 20.7 | 1.6 | 2.86 | 7.23x | bench_2026-03-13/burn_gpu/fox.wav |
| burn-gpu-v2-call | burn+candle | wgpu+cpu | Q4_0 baseline | 2.6G | call | 2.4 | 80.6 | — | 6.06 | 13.30x | bench_2026-03-13/burn_gpu/call.wav |
| burn-gpu-v2-tyger | burn+candle | wgpu+cpu | Q4_0 baseline | 2.6G | tyger | 2.3 | 77.5 | — | 4.04 | 19.19x | bench_2026-03-13/burn_gpu/tyger.wav |
| burn-gpu-v2-wutang | burn+candle | wgpu+cpu | Q4_0 baseline | 2.6G | wutang | 2.3 | 16.7 | — | 2.48 | 6.72x | bench_2026-03-13/burn_gpu/wutang.wav |

---

## tyger-debug

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| tyger-debug-ts0-metal | candle | metal | Q4_0 baseline | 2.6G | tyger | — | — | — | 1.8 | — | tyger_ts0.wav |
| tyger-debug-ts5-metal | candle | metal | Q4_0 baseline | 2.6G | tyger | — | — | — | 1.2 | — | tyger_ts5.wav |
| tyger-debug-novoice-metal | candle | metal | Q4_0 baseline | 2.6G | tyger | — | — | — | 3.3 | — | tyger_novoice.wav |
| tyger-debug-ts0-cpu | candle | cpu | Q4_0 baseline | 2.6G | tyger | — | — | — | 1.8 | — | tyger_cpu.wav |

---

## cross-variant-quality

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| cross-variant-f32-time | candle | metal | F32 GGUF | 6.5G | time | — | 30.9 | — | 2.5 | 12.4x | time_f32.wav |
| cross-variant-q4-time | candle | metal | Q4_0 baseline | 2.6G | time | — | 7.1 | — | 2.5 | 2.9x | time_q4.wav |
| cross-variant-varE-time | candle | metal | Var-E VV-F16 E-Q4 | 1.8G | time | — | 6.7 | — | 2.5 | 2.7x | time_varE.wav |
| cross-variant-mixed-time | candle | metal | Mixed VV-Q8 E-Q8 | 1.4G | time | — | 2.3 | — | 2.5 | 0.9x | time_mixed.wav |
| cross-variant-varC-time | candle | metal | Var-C VV-Q8 E-Q4 | 1.3G | time | — | 2.3 | — | 2.5 | 0.9x | time_varC.wav |

---

## burn-voice-prompted

| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |
|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|
| burn-voice-burn-q4-time | burn+candle | wgpu+cpu | Q4_0 baseline | 2.6G | time | 2.9 | 14.9 | 0.4 | 0.62 | 24.11x | bench_2026-03-14/burn_voice/time.wav |
| burn-voice-burn-q4-fox | burn+candle | wgpu+cpu | Q4_0 baseline | 2.6G | fox | 2.9 | 14.1 | — | 0.98 | — | bench_2026-03-14/burn_voice/fox.wav |

Note: Gen time includes llm=3.1-3.4s + vibe=11.0-11.5s. LLM per-step avg 129-137ms (GPU), VibeVoice per-step avg 457-459ms (CPU).
