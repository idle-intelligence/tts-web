"""
Compare intermediate tensors between ONNX reference and Rust decoder.

Usage:
    /Users/tc/Code/idle-intelligence/tts-web-kitten/venv/bin/python \
        /Users/tc/Code/idle-intelligence/tts-web-kitten/scripts/compare_decoder_intermediates.py

Saves .npy files to /tmp/ for manual inspection.
"""

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
import os

ONNX_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_NPZ = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"

# Reference inputs matching the test phrase "Hello world"
IDS = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
SPEED = np.array([1.0], dtype=np.float32)
STYLE_VOICE = "expr-voice-2-m"
STYLE_ROW = 11  # len("Hello world") = 11


def stat_str(arr, name):
    flat = arr.flatten()
    first5 = flat[:5].tolist()
    return (f"  {name}: shape={arr.shape} "
            f"min={flat.min():.6f} max={flat.max():.6f} "
            f"mean={flat.mean():.6f} first5={[round(v,6) for v in first5]}")


def find_output(output_names, pattern):
    """Return all output names that contain `pattern` (case-insensitive)."""
    return [n for n in output_names if pattern.lower() in n.lower()]


def print_match(results, output_names, pattern, label, save_path=None):
    matches = find_output(output_names, pattern)
    if not matches:
        print(f"  [WARN] No output found for pattern: {pattern!r}")
        return None
    # Pick the last match (usually the most downstream node with that name)
    name = matches[-1]
    arr = results[name]
    print(f"\n--- {label} ---")
    print(f"  ONNX node output name: {name!r}")
    print(stat_str(arr, label))
    if save_path:
        np.save(save_path, arr)
        print(f"  Saved to {save_path}")
    return arr


def main():
    print("=" * 70)
    print("ONNX intermediate tensor extraction")
    print("=" * 70)

    # ── Load ONNX and expose ALL intermediates ──────────────────────────────
    print(f"\nLoading ONNX model from {ONNX_PATH}...")
    model = onnx.load(ONNX_PATH)

    # Collect ALL intermediate outputs from every node.
    # We use the value_info already in the model graph (which has type info),
    # plus infer any missing ones via ONNX shape inference.
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)

    # Build map from name → value_info (typed)
    value_info_map = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi
    for vi in model.graph.input:
        value_info_map[vi.name] = vi
    for vi in model.graph.output:
        value_info_map[vi.name] = vi

    existing_output_names = set(vi.name for vi in model.graph.output)
    existing_output_names.update(vi.name for vi in model.graph.input)
    existing_output_names.update(init.name for init in model.graph.initializer)

    added = 0
    added_no_type = 0
    for node in model.graph.node:
        for output_name in node.output:
            if not output_name or output_name in existing_output_names:
                continue
            if output_name in value_info_map:
                model.graph.output.append(value_info_map[output_name])
                added += 1
            else:
                # No type info available — skip
                added_no_type += 1
            existing_output_names.add(output_name)
    print(f"  Added {added} intermediate outputs ({added_no_type} skipped, no type info).")

    sess = ort.InferenceSession(model.SerializeToString())
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  Total outputs: {len(output_names)}")

    # ── Load inputs ─────────────────────────────────────────────────────────
    voices = np.load(VOICES_NPZ)
    available_voices = list(voices.keys())
    print(f"\nAvailable voices: {available_voices}")
    if STYLE_VOICE not in voices:
        print(f"[WARN] {STYLE_VOICE!r} not found, using first: {available_voices[0]!r}")
        style_key = available_voices[0]
    else:
        style_key = STYLE_VOICE
    style = voices[style_key][STYLE_ROW:STYLE_ROW + 1].astype(np.float32)
    print(f"Style: voice={style_key!r} row={STYLE_ROW} shape={style.shape}")

    print("\nRunning ONNX inference...")
    outputs = sess.run(None, {
        'input_ids': IDS,
        'style': style,
        'speed': SPEED,
    })
    results = dict(zip(output_names, outputs))
    print(f"  Got {len(results)} output tensors.")

    # ── Print all output names (for discovery) ──────────────────────────────
    print("\n--- All output names (first 60) ---")
    for n in output_names[:60]:
        arr = results[n]
        if hasattr(arr, 'shape'):
            print(f"  {n!r}: shape={arr.shape}")

    # ── Key checkpoint comparisons ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY CHECKPOINTS")
    print("=" * 70)

    # Print ALL outputs with their shapes for discovery
    print("\n--- All outputs with shapes (full list) ---")
    for n in output_names:
        arr = results[n]
        if hasattr(arr, 'shape') and len(arr.shape) >= 2:
            print(f"  {n!r}: {arr.shape}")

    # ── Find encoder output (shape ~[1, 256, T]) ────────────────────────────
    # Search for encode-related outputs
    print("\n\n=== ENCODER BLOCK OUTPUT ===")
    # Look for tensors with shape [1, 256, T]
    enc_candidates = [(n, results[n]) for n in output_names
                      if hasattr(results[n], 'shape') and
                      len(results[n].shape) == 3 and
                      results[n].shape[1] == 256]
    print(f"  Found {len(enc_candidates)} candidates with shape [1, 256, T]:")
    for n, arr in enc_candidates[:5]:
        print(stat_str(arr, n))
    if enc_candidates:
        # First one should be the encode block output
        enc_name, enc_arr = enc_candidates[0]
        print(f"\n  Using first candidate as ENCODE_OUT: {enc_name!r}")
        np.save("/tmp/onnx_encode_out.npy", enc_arr)
        print(f"  Saved to /tmp/onnx_encode_out.npy")

    # ── After decode blocks ─────────────────────────────────────────────────
    print("\n\n=== DECODE BLOCKS OUTPUT ===")
    dec_candidates_256 = [(n, results[n]) for n in output_names
                          if hasattr(results[n], 'shape') and
                          len(results[n].shape) == 3 and
                          results[n].shape[1] == 256]
    print(f"  All [1, 256, T] tensors (likely encode + decode block outputs):")
    for n, arr in dec_candidates_256:
        print(f"    {n!r}: {arr.shape}  first5={arr.flatten()[:5].tolist()}")

    # ── After ups.0 (shape ~[1, 128, T]) ───────────────────────────────────
    print("\n\n=== GENERATOR UPS.0 OUTPUT (~[1, 128, T]) ===")
    ups0_candidates = [(n, results[n]) for n in output_names
                       if hasattr(results[n], 'shape') and
                       len(results[n].shape) == 3 and
                       results[n].shape[1] == 128]
    print(f"  Found {len(ups0_candidates)} candidates with 128 channels:")
    for n, arr in ups0_candidates[:10]:
        print(stat_str(arr, n))
    if ups0_candidates:
        ups0_name, ups0_arr = ups0_candidates[0]
        np.save("/tmp/onnx_ups0_out.npy", ups0_arr)
        print(f"\n  Using first as UPS0_OUT: {ups0_name!r}, saved to /tmp/onnx_ups0_out.npy")

    # ── After conv_post (shape ~[1, 22, T]) ─────────────────────────────────
    print("\n\n=== GENERATOR CONV_POST OUTPUT (~[1, 22, T]) ===")
    post_candidates = [(n, results[n]) for n in output_names
                       if hasattr(results[n], 'shape') and
                       len(results[n].shape) == 3 and
                       results[n].shape[1] == 22]
    print(f"  Found {len(post_candidates)} candidates with 22 channels:")
    for n, arr in post_candidates:
        print(stat_str(arr, n))
    if post_candidates:
        post_name, post_arr = post_candidates[-1]  # last = most downstream
        np.save("/tmp/onnx_conv_post_out.npy", post_arr)
        print(f"\n  Using LAST as CONV_POST_OUT: {post_name!r}, saved to /tmp/onnx_conv_post_out.npy")

    # ── Harmonic source excitation (~[1, 1, T]) ──────────────────────────────
    print("\n\n=== HARMONIC SOURCE / AUDIO [1, 1, T] ===")
    audio_candidates = [(n, results[n]) for n in output_names
                        if hasattr(results[n], 'shape') and
                        len(results[n].shape) == 3 and
                        results[n].shape[1] == 1]
    print(f"  Found {len(audio_candidates)} candidates with 1 channel:")
    for n, arr in audio_candidates:
        print(stat_str(arr, n))

    # ── Final output ─────────────────────────────────────────────────────────
    print("\n\n=== FINAL OUTPUT (model.graph.output[0]) ===")
    # The true model outputs are first in output_names
    model_orig = onnx.load(ONNX_PATH)
    orig_output_names = [o.name for o in model_orig.graph.output]
    print(f"  Original model outputs: {orig_output_names}")
    for orig_name in orig_output_names:
        if orig_name in results:
            arr = results[orig_name]
            print(stat_str(arr, orig_name))
            np.save("/tmp/onnx_final_output.npy", arr)
            print(f"  Saved to /tmp/onnx_final_output.npy")

    # ── Detailed first-5 comparison table ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE — ONNX first5 values at each stage")
    print("(paste Rust first5 values alongside these)")
    print("=" * 70)

    stages = {}

    if enc_candidates:
        stages["ENCODE_OUT"] = enc_candidates[0][1]
    if dec_candidates_256:
        stages["DECODE_LAST (last [1,256,T])"] = dec_candidates_256[-1][1]
    if ups0_candidates:
        stages["UPS0_OUT (first [1,128,T])"] = ups0_candidates[0][1]
    if post_candidates:
        stages["CONV_POST_OUT (last [1,22,T])"] = post_candidates[-1][1]
    for n, arr in audio_candidates:
        stages[f"[1,1,T] {n}"] = arr
    for orig_name in orig_output_names:
        if orig_name in results:
            stages[f"FINAL: {orig_name}"] = results[orig_name]

    for label, arr in stages.items():
        flat = arr.flatten()
        first5 = [round(float(v), 6) for v in flat[:5]]
        print(f"\n  {label}")
        print(f"    shape={arr.shape}")
        print(f"    min={float(flat.min()):.6f} max={float(flat.max()):.6f} mean={float(flat.mean()):.6f}")
        print(f"    first5={first5}")

    # ── Also print 64-channel tensors (ups.1 output) ─────────────────────────
    print("\n\n=== GENERATOR UPS.1 OUTPUT (~[1, 64, T]) ===")
    ups1_candidates = [(n, results[n]) for n in output_names
                       if hasattr(results[n], 'shape') and
                       len(results[n].shape) == 3 and
                       results[n].shape[1] == 64]
    for n, arr in ups1_candidates[:5]:
        print(stat_str(arr, n))
    if ups1_candidates:
        ups1_name, ups1_arr = ups1_candidates[0]
        np.save("/tmp/onnx_ups1_out.npy", ups1_arr)
        print(f"  Using first as UPS1_OUT: {ups1_name!r}, saved to /tmp/onnx_ups1_out.npy")

    print("\n\nDone. Files saved to /tmp/onnx_*.npy")
    print("Now run Rust with debug output and compare first5 values above.")


if __name__ == "__main__":
    main()
