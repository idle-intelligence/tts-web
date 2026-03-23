#!/usr/bin/env python3
"""
compare_bench_audio.py

Compares WAV files across TADA-1B benchmark variants to determine whether
the autoregressive time feedback bugfix eliminated gray-code sensitivity to
embedding precision (Q8_0 vs Q4_0 embeddings).

In Run 5 (pre-bugfix), Q8_0-embedding variants produced ~4.1s fox audio vs
~3.0s for Q4_0-embedding variants, caused by gray-code mispredictions. Post-
bugfix all variants reportedly produce 2.7-2.8s. This script verifies that
and checks whether the waveforms are not just similar in length but actually
identical (i.e., same durations means same token sequence).
"""

import sys
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import soundfile as sf

BENCH_DIR = Path("/Users/tc/Code/idle-intelligence/tts-web/samples/bench_2026-03-13")

# Variant ordering and grouping
VARIANTS = [
    ("f32",                "f32"),
    ("f16",                "f16"),
    ("q4_0_baseline",      "q4_0_baseline"),
    ("var_a_vv_f16_e_q8",  "var_a (VV F16 + Embed Q8)"),
    ("var_b_vv_f32_e_q8",  "var_b (VV F32 + Embed Q8)"),
    ("var_c_vv_q8_e_q4",   "var_c (VV Q8 + Embed Q4)"),
    ("var_e_vv_f16_e_q4",  "var_e (VV F16 + Embed Q4)"),
    ("mixed_vv_q8_e_q8",   "mixed (VV Q8 + Embed Q8)"),
    ("python_bf16",        "python_bf16"),
]

# Grouping by embedding precision (the key variable for gray-code sensitivity)
Q8_EMBED_DIRS  = {"var_a_vv_f16_e_q8", "var_b_vv_f32_e_q8", "mixed_vv_q8_e_q8"}
Q4_EMBED_DIRS  = {"q4_0_baseline", "var_c_vv_q8_e_q4", "var_e_vv_f16_e_q4"}
F32_EQUIV_DIRS = {"f32", "f16", "python_bf16"}

PHRASES = ["fox", "call"]


def load_wav(path: Path):
    """Return (samples_float64, sample_rate) or (None, None) if missing."""
    if not path.exists():
        return None, None
    data, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono-mix if needed
    return data, sr


def duration_s(data, sr):
    return len(data) / sr if data is not None else None


def compare_pair(a_data, b_data):
    """Return dict of comparison metrics."""
    if a_data is None or b_data is None:
        return {"error": "one or both files missing"}

    result = {
        "len_a": len(a_data),
        "len_b": len(b_data),
        "same_length": len(a_data) == len(b_data),
    }

    # Correlation on common prefix length (works even if lengths differ)
    n = min(len(a_data), len(b_data))
    a_s, b_s = a_data[:n], b_data[:n]
    if np.std(a_s) > 1e-12 and np.std(b_s) > 1e-12:
        result["correlation"] = float(np.corrcoef(a_s, b_s)[0, 1])
    else:
        result["correlation"] = float("nan")

    if result["same_length"]:
        diff = a_data - b_data
        result["max_abs_diff"] = float(np.max(np.abs(diff)))
        result["rms_diff"] = float(np.sqrt(np.mean(diff ** 2)))
        result["identical"] = result["max_abs_diff"] == 0.0
    else:
        result["max_abs_diff"] = None
        result["rms_diff"] = None
        result["identical"] = False

    return result


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def fmt_dur(d):
    return f"{d:.3f}s" if d is not None else "MISSING"


def fmt_diff(v):
    if v is None:
        return "   N/A  "
    if v == 0.0:
        return "  0.000  (IDENTICAL)"
    return f"{v:.6f}"


def main():
    print("TADA-1B Benchmark Audio Comparison")
    print("Post-bugfix: autoregressive time feedback fix + trailing EOT trim")
    print(f"Bench dir: {BENCH_DIR}")

    for phrase in PHRASES:
        print_section(f"Phrase: {phrase}.wav")

        # Load all variants
        audio = {}
        for dir_name, label in VARIANTS:
            path = BENCH_DIR / dir_name / f"{phrase}.wav"
            data, sr = load_wav(path)
            audio[dir_name] = (data, sr, label)

        # --- Duration table ---
        print()
        print("  Durations:")
        print(f"  {'Variant':<35} {'Duration':>10}  {'Samples':>8}  {'SR':>6}")
        print(f"  {'-'*35} {'-'*10}  {'-'*8}  {'-'*6}")
        for dir_name, label in VARIANTS:
            data, sr, _ = audio[dir_name]
            if data is not None:
                print(f"  {label:<35} {fmt_dur(duration_s(data, sr)):>10}  {len(data):>8}  {sr:>6}")
            else:
                print(f"  {label:<35} {'MISSING':>10}")

        # --- Duration spread by embedding group ---
        print()
        print("  Duration spread by embedding precision group:")
        for group_name, group_dirs in [
            ("Q8_0 embeddings (pre-bugfix mispredictors)", Q8_EMBED_DIRS),
            ("Q4_0 embeddings (pre-bugfix correct)",       Q4_EMBED_DIRS),
            ("F32/F16/Python (reference)",                 F32_EQUIV_DIRS),
        ]:
            durs = []
            for dir_name in group_dirs:
                data, sr, _ = audio[dir_name]
                if data is not None:
                    durs.append(duration_s(data, sr))
            if durs:
                spread = max(durs) - min(durs)
                print(f"    {group_name}:")
                print(f"      durations: {', '.join(f'{d:.3f}s' for d in sorted(durs))}")
                print(f"      spread: {spread*1000:.1f} ms")

        # --- Full pairwise comparison matrix (max_abs_diff + correlation) ---
        print()
        print("  Pairwise comparison (all variants):")
        dir_names = [d for d, _ in VARIANTS]
        labels    = [l for _, l in VARIANTS]
        n = len(dir_names)

        # Header
        col_w = 8
        print(f"  {'':35}", end="")
        for l in labels:
            short = l[:col_w]
            print(f" {short:>{col_w}}", end="")
        print()
        print(f"  {'':-<35}", end="")
        for _ in labels:
            print(f" {'-'*col_w}", end="")
        print()

        for i, (dir_i, label_i) in enumerate(VARIANTS):
            data_i, sr_i, _ = audio[dir_i]
            print(f"  {label_i:<35}", end="")
            for j, (dir_j, label_j) in enumerate(VARIANTS):
                if j < i:
                    print(f" {'':>{col_w}}", end="")
                    continue
                data_j, sr_j, _ = audio[dir_j]
                if i == j:
                    print(f" {'---':>{col_w}}", end="")
                    continue
                metrics = compare_pair(data_i, data_j)
                if "error" in metrics:
                    cell = "ERR"
                elif metrics.get("identical"):
                    cell = "SAME"
                elif metrics["same_length"]:
                    rms = metrics["rms_diff"]
                    cell = f"r{rms:.4f}"
                else:
                    # Different lengths — show duration delta in ms
                    dur_i = len(data_i) / sr_i if data_i is not None else 0
                    dur_j = len(data_j) / sr_j if data_j is not None else 0
                    delta_ms = abs(dur_i - dur_j) * 1000
                    cell = f"Δ{delta_ms:.0f}ms"
                print(f" {cell:>{col_w}}", end="")
            print()

        # --- Key comparison: Q8_0 vs Q4_0 embedding groups ---
        print()
        print("  KEY: Q8_0-embed vs Q4_0-embed duration deltas (the gray-code sensitivity test):")
        q8_dirs = sorted(Q8_EMBED_DIRS)
        q4_dirs = sorted(Q4_EMBED_DIRS)
        for d8 in q8_dirs:
            data8, sr8, label8 = audio[d8]
            for d4 in q4_dirs:
                data4, sr4, label4 = audio[d4]
                if data8 is None or data4 is None:
                    continue
                dur8 = duration_s(data8, sr8)
                dur4 = duration_s(data4, sr4)
                delta_ms = (dur8 - dur4) * 1000
                metrics = compare_pair(data8, data4)
                corr = metrics.get("correlation", float("nan"))
                same = "SAME LENGTH" if metrics["same_length"] else f"Δ={delta_ms:+.0f}ms"
                print(f"    {label8:<35} vs {label4:<35} → {same}, corr={corr:.4f}")

        # --- Conclusion ---
        print()
        print("  CONCLUSION:")
        all_q8 = [audio[d] for d in Q8_EMBED_DIRS if audio[d][0] is not None]
        all_q4 = [audio[d] for d in Q4_EMBED_DIRS if audio[d][0] is not None]
        if all_q8 and all_q4:
            # Check if any Q8 vs Q4 pair has a duration delta > 200ms (Run 5 showed ~1100ms)
            max_delta_ms = 0.0
            for data8, sr8, _ in all_q8:
                for data4, sr4, _ in all_q4:
                    delta = abs(duration_s(data8, sr8) - duration_s(data4, sr4)) * 1000
                    max_delta_ms = max(max_delta_ms, delta)
            if max_delta_ms < 50:
                print(f"    Q8_0 vs Q4_0 embedding max duration delta: {max_delta_ms:.1f}ms (<50ms)")
                print("    => BUGFIX ELIMINATED gray-code sensitivity. Q8_0 and Q4_0 embeddings")
                print("       now produce the same durations. The autoregressive time feedback bug")
                print("       was amplifying embedding-precision errors into large timing divergences.")
            elif max_delta_ms < 200:
                print(f"    Q8_0 vs Q4_0 embedding max duration delta: {max_delta_ms:.1f}ms (<200ms)")
                print("    => Small residual difference remains, but much less than Run 5 (~1100ms).")
                print("       Bugfix substantially reduced gray-code sensitivity.")
            else:
                print(f"    Q8_0 vs Q4_0 embedding max duration delta: {max_delta_ms:.1f}ms (>=200ms)")
                print("    => WARNING: Large duration delta persists. Gray-code sensitivity may NOT")
                print("       be fully resolved by the bugfix alone.")

        # Check if all variants produce exactly the same waveform
        print()
        print("  Identical-waveform check (all Rust variants):")
        rust_dirs = [d for d, _ in VARIANTS if d != "python_bf16"]
        all_identical = True
        same_length_all = True
        ref_dir = rust_dirs[0]
        ref_data, ref_sr, ref_label = audio[ref_dir]
        for d in rust_dirs[1:]:
            data, sr, label = audio[d]
            if data is None or ref_data is None:
                continue
            m = compare_pair(ref_data, data)
            if not m["same_length"]:
                same_length_all = False
                all_identical = False
                dur_ref = duration_s(ref_data, ref_sr)
                dur_d   = duration_s(data, sr)
                print(f"    {ref_label} vs {label}: different length ({dur_ref:.3f}s vs {dur_d:.3f}s)")
            elif not m.get("identical", False):
                all_identical = False
                print(f"    {ref_label} vs {label}: same length, RMS diff={m['rms_diff']:.6f}, corr={m['correlation']:.6f}")

        if all_identical and same_length_all:
            print(f"    ALL Rust variants produce byte-identical audio for '{phrase}'.")
            print("    This confirms the output is fully deterministic and independent of")
            print("    quantization precision — all variants follow the same token sequence.")
        elif same_length_all and all_identical is False:
            print(f"    All Rust variants have the same length but differ slightly in waveform.")
        else:
            print(f"    Rust variants produce different-length audio — duration divergence remains.")


if __name__ == "__main__":
    main()
