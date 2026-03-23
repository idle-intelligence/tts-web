#!/usr/bin/env python3
"""Compare intermediate values from Python and Rust TADA generation.

Usage: scripts/venv/bin/python3 scripts/compare_tada_intermediates.py
"""

import json
import os
import struct
import numpy as np

PYTHON_NPZ = "/tmp/python_debug_time.npz"
RUST_DIR = "/tmp/rust_debug_time"
DIVERGENCE_THRESHOLD = 0.01


def load_f32_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


def load_rust_step(step_idx):
    d = os.path.join(RUST_DIR, f"step_{step_idx:03d}")
    if not os.path.isdir(d):
        return None
    with open(os.path.join(d, "scalars.json")) as f:
        scalars = json.load(f)
    result = {"scalars": scalars}
    for name in ("hidden_state", "flow_input", "flow_output", "acoustic", "time_bits"):
        path = os.path.join(d, f"{name}.bin")
        if os.path.exists(path):
            result[name] = load_f32_bin(path)
        else:
            result[name] = None
    return result


def load_python_data(npz):
    d = {}
    for key in npz.files:
        d[key] = npz[key]
    return d


def py_scalar(npz_data, key_prefix, step):
    k = f"{key_prefix}_step_{step}"
    if k in npz_data:
        v = npz_data[k]
        return float(v) if v.ndim == 0 else float(v.flat[0])
    return None


def py_arr(npz_data, key_prefix, step):
    k = f"{key_prefix}_step_{step}"
    if k in npz_data:
        return npz_data[k].astype(np.float32).ravel()
    return None


def max_abs_diff(a, b):
    if a is None or b is None:
        return None
    min_len = min(len(a), len(b))
    return float(np.max(np.abs(a[:min_len] - b[:min_len])))


def diff_stats(a, b, label):
    if a is None or b is None:
        return
    min_len = min(len(a), len(b))
    diff = np.abs(a[:min_len] - b[:min_len])
    print(f"  {label}: mean={diff.mean():.6f}  std={diff.std():.6f}  "
          f"min={diff.min():.6f}  max={diff.max():.6f}  "
          f"(shapes: py={len(a)}, rs={len(b)})")


def main():
    # --- Load data ---
    npz = np.load(PYTHON_NPZ)
    py = load_python_data(npz)

    n_rust = sum(1 for e in os.scandir(RUST_DIR) if e.is_dir())
    rust_steps = []
    for i in range(n_rust):
        s = load_rust_step(i)
        if s:
            rust_steps.append((i, s))

    n_python = sum(1 for k in py if k.startswith("input_token_step_"))
    print(f"Python steps: {n_python}   Rust steps: {len(rust_steps)}\n")

    # --- Token sequences ---
    print("=" * 70)
    print("TOKEN SEQUENCES")
    print("=" * 70)
    print(f"{'Step':>5}  {'Py token':>10}  {'Py t_before':>12}  {'Py t_after':>10}")
    py_tokens = []
    for i in range(n_python):
        tok = py_scalar(py, "input_token", i)
        tb = py_scalar(py, "time_before", i)
        ta = py_scalar(py, "time_after", i)
        py_tokens.append(int(tok) if tok is not None else None)
        print(f"  py[{i:02d}]  {int(tok) if tok is not None else 'N/A':>10}  "
              f"{tb if tb is not None else 'N/A':>12.1f}  "
              f"{ta if ta is not None else 'N/A':>10.1f}")

    print()
    print(f"{'Step':>5}  {'Rs token':>10}  {'Rs t_before':>12}  {'Rs t_after':>10}")
    rs_tokens = []
    for i, s in rust_steps:
        sc = s["scalars"]
        tok = sc.get("token_id")
        tb = sc.get("time_before")
        ta = sc.get("time_after")
        rs_tokens.append(tok)
        print(f"  rs[{i:02d}]  {tok if tok is not None else 'N/A':>10}  "
              f"{tb if tb is not None else 'N/A':>12.1f}  "
              f"{ta if ta is not None else 'N/A':>10.1f}")

    # --- Align by token ID ---
    print()
    print("=" * 70)
    print("STEP ALIGNMENT (matching token IDs)")
    print("=" * 70)

    # Find offset: try each possible shift
    best_matches = 0
    best_offset = 0
    for offset in range(-len(rust_steps), n_python):
        matches = 0
        for pi in range(n_python):
            ri = pi - offset
            if 0 <= ri < len(rust_steps):
                _, rs = rust_steps[ri]
                if py_tokens[pi] == rs["scalars"].get("token_id"):
                    matches += 1
        if matches > best_matches:
            best_matches = matches
            best_offset = offset

    print(f"Best alignment: rust_idx = python_idx - ({best_offset})  "
          f"({best_matches}/{min(n_python, len(rust_steps))} tokens match)")
    print(f"  i.e. Python step 0 corresponds to Rust step {-best_offset}")
    print()

    # Build aligned pairs: (py_step, rs_step)
    aligned = []
    for pi in range(n_python):
        ri = pi - best_offset
        if 0 <= ri < len(rust_steps):
            _, rs = rust_steps[ri]
            aligned.append((pi, ri, rs))

    # --- Time values comparison ---
    print("=" * 70)
    print("TIME VALUES COMPARISON (aligned steps)")
    print("=" * 70)
    print(f"{'py_i':>5}  {'rs_i':>5}  {'py_tok':>8}  {'rs_tok':>8}  "
          f"{'py_tb':>7}  {'rs_tb':>7}  {'py_ta':>7}  {'rs_ta':>7}  "
          f"{'tb_match':>8}  {'ta_match':>8}")
    for pi, ri, rs in aligned:
        sc = rs["scalars"]
        py_tok = py_tokens[pi]
        rs_tok = sc.get("token_id")
        py_tb = py_scalar(py, "time_before", pi)
        py_ta = py_scalar(py, "time_after", pi)
        rs_tb = sc.get("time_before")
        rs_ta = sc.get("time_after")
        tb_match = "OK" if py_tb == rs_tb else f"DIFF({py_tb}->{rs_tb})"
        ta_match = "OK" if py_ta == rs_ta else f"DIFF({py_ta}->{rs_ta})"
        print(f"  {pi:3d}    {ri:3d}    {py_tok if py_tok else 'N/A':>8}  "
              f"{rs_tok if rs_tok else 'N/A':>8}  "
              f"{py_tb if py_tb is not None else 'N/A':>7.1f}  "
              f"{rs_tb if rs_tb is not None else 'N/A':>7.1f}  "
              f"{py_ta if py_ta is not None else 'N/A':>7.1f}  "
              f"{rs_ta if rs_ta is not None else 'N/A':>7.1f}  "
              f"{tb_match:>8}  {ta_match:>8}")

    # --- Per-step tensor diffs ---
    print()
    print("=" * 70)
    print("TENSOR MAX-ABS-DIFF (aligned steps, threshold={})".format(DIVERGENCE_THRESHOLD))
    print("=" * 70)
    print(f"{'py_i':>5}  {'rs_i':>5}  {'hidden':>10}  {'flow_in':>10}  "
          f"{'flow_out':>10}  {'acoustic':>10}  {'time_bits':>10}  {'diverged':>8}")

    first_diverged_step = None
    first_diverged_info = None

    for pi, ri, rs in aligned:
        hs_diff = max_abs_diff(py_arr(py, "hidden_state", pi), rs["hidden_state"])
        fi_diff = max_abs_diff(py_arr(py, "flow_input", pi), rs["flow_input"])
        fo_diff = max_abs_diff(py_arr(py, "flow_output", pi), rs["flow_output"])
        ac_diff = max_abs_diff(py_arr(py, "acoustic", pi), rs["acoustic"])
        tb_diff = max_abs_diff(py_arr(py, "time_bits", pi), rs["time_bits"])

        diffs = {
            "hidden_state": hs_diff,
            "flow_input": fi_diff,
            "flow_output": fo_diff,
            "acoustic": ac_diff,
            "time_bits": tb_diff,
        }

        diverged_fields = [k for k, v in diffs.items() if v is not None and v > DIVERGENCE_THRESHOLD]
        diverged = bool(diverged_fields)

        def fmt(v):
            return f"{v:.5f}" if v is not None else "  N/A  "

        print(f"  {pi:3d}    {ri:3d}    {fmt(hs_diff):>10}  {fmt(fi_diff):>10}  "
              f"{fmt(fo_diff):>10}  {fmt(ac_diff):>10}  {fmt(tb_diff):>10}  "
              f"{'*** YES ***' if diverged else 'no':>8}")

        if diverged and first_diverged_step is None:
            first_diverged_step = (pi, ri)
            first_diverged_info = (diffs, diverged_fields, rs)

    # --- First divergence detailed analysis ---
    print()
    print("=" * 70)
    if first_diverged_step is None:
        print("NO DIVERGENCE FOUND (all diffs <= {})".format(DIVERGENCE_THRESHOLD))
    else:
        pi, ri = first_diverged_step
        diffs, diverged_fields, rs = first_diverged_info
        print(f"FIRST DIVERGENCE at Python step {pi} / Rust step {ri}")
        print(f"Diverged tensors: {', '.join(diverged_fields)}")
        print()
        print("Detailed diff stats for diverged tensors:")
        for field in ("hidden_state", "flow_input", "flow_output", "acoustic", "time_bits"):
            py_arr_data = py_arr(py, field, pi)
            rs_arr_data = rs[field]
            if py_arr_data is not None and rs_arr_data is not None:
                diff_stats(py_arr_data, rs_arr_data, field)
        print()
        print("Token at divergence point:")
        sc = rs["scalars"]
        print(f"  Python token={py_tokens[pi]}  Rust token={sc.get('token_id')}")
        print(f"  Python time_before={py_scalar(py, 'time_before', pi)}  "
              f"Rust time_before={sc.get('time_before')}")
        print(f"  Python time_after={py_scalar(py, 'time_after', pi)}  "
              f"Rust time_after={sc.get('time_after')}")

        # Check what diverged FIRST among the tensors (by field order in pipeline)
        field_order = ["hidden_state", "flow_input", "flow_output", "acoustic", "time_bits"]
        print()
        print("Which tensor diverges first (pipeline order):")
        for field in field_order:
            v = diffs.get(field)
            flag = "<-- FIRST DIVERGENCE" if field == diverged_fields[0] else ""
            mark = "DIVERGED" if v is not None and v > DIVERGENCE_THRESHOLD else "ok"
            print(f"  {field:20s}: max_diff={fmt(v):>10}  {mark}  {flag}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if first_diverged_step is None:
        print("Implementations agree within threshold on all aligned steps.")
    else:
        pi, ri = first_diverged_step
        print(f"First divergence: Python step {pi} / Rust step {ri}")
        print(f"Diverged fields at first step: {', '.join(first_diverged_info[1])}")
        # Check if step 0 diverges
        if pi == 0 or ri == 0:
            print("NOTE: Divergence starts at step 0 — likely a fundamental difference")
            print("      (model weights, quantization, or input encoding mismatch)")
        else:
            print(f"NOTE: Steps 0..{pi-1} agree — divergence accumulates from step {pi}")


if __name__ == "__main__":
    main()
