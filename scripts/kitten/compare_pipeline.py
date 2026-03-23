#!/usr/bin/env python3
"""Compare KittenTTS ONNX vs Rust pipeline stage by stage.

Runs ONNX on a fixed input, extracts all intermediate tensors, then
compares them against Rust debug output (saved via --debug-dir).

Usage:
  # Step 1: run Rust with debug dump
  cargo run --example kitten_generate -p kitten-core --release -- \
    --debug-dir /tmp/kitten_debug

  # Step 2: run this script
  python scripts/compare_pipeline.py [--debug-dir /tmp/kitten_debug]
"""

import argparse
import os
import struct
import subprocess
import sys
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

ONNX_MODEL  = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_NPZ  = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"

# Reference inputs matching the Rust integration tests
# "həlˈəʊ wˈɜːld" → IDs
REFERENCE_IDS   = [0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]
REFERENCE_VOICE = "expr-voice-2-m"
REFERENCE_IDX   = 11
REFERENCE_SPEED = 1.0

# ── ONNX helpers ─────────────────────────────────────────────────────────────

def run_onnx_with_intermediates(model_path, ids, style, speed):
    """Run the ONNX model and return every intermediate tensor by name."""
    import onnx
    import onnxruntime as ort
    from onnx import helper

    model = onnx.load(model_path)

    # Mark non-float outputs to skip
    skip_outputs = set()
    float_elem = onnx.TensorProto.FLOAT
    NON_FLOAT_OPS = {
        "SequenceEmpty", "SequenceInsert", "SequenceAt", "SplitToSequence",
        "ConcatFromSequence", "Loop", "Reshape", "Cast", "Shape",
        "Gather", "GatherElements", "Unsqueeze", "Squeeze",
        "NonZero", "TopK", "ArgMax", "ArgMin", "Where",
    }
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.HasField("sequence_type"):
            skip_outputs.add(vi.name)
        elif vi.type.HasField("tensor_type"):
            if vi.type.tensor_type.elem_type != float_elem:
                skip_outputs.add(vi.name)
    for node in model.graph.node:
        if node.op_type in NON_FLOAT_OPS:
            for o in node.output:
                skip_outputs.add(o)

    existing_outputs = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for output in node.output:
            if output and output not in existing_outputs and output not in skip_outputs:
                model.graph.output.extend(
                    [helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None)]
                )
                existing_outputs.add(output)

    sess = ort.InferenceSession(model.SerializeToString())
    outputs = sess.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "style": style,
        "speed": np.array([speed], dtype=np.float32),
    })
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outputs)), model


def find_stage(results, model, search_terms, shape_filter=None, last_only=True):
    """Find output tensors from nodes whose name contains any of search_terms."""
    hits = []
    for node in model.graph.node:
        name_lower = node.name.lower()
        if any(t in name_lower for t in search_terms):
            for out in node.output:
                if out and out in results:
                    arr = results[out]
                    if shape_filter is None or shape_filter(arr):
                        hits.append((node.name, out, arr))
    if last_only and hits:
        return [hits[-1]]
    return hits


# ── Rust debug file helpers ───────────────────────────────────────────────────

def load_rust_tensor(path):
    """Load a Rust debug tensor saved as raw f32 LE bytes.
    Filename convention: <name>_<d0>x<d1>x...xdN.bin
    Returns (ndarray, shape).
    """
    basename = os.path.basename(path)
    stem = basename.rsplit(".", 1)[0]  # strip .bin
    # Shape encoded in last part after final '_'
    # e.g. bert_output_1x16x128.bin → shape (1,16,128)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and "x" in parts[1]:
        shape = tuple(int(d) for d in parts[1].split("x"))
    else:
        shape = None

    with open(path, "rb") as f:
        raw = f.read()
    n_floats = len(raw) // 4
    data = struct.unpack(f"{n_floats}f", raw)
    arr = np.array(data, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def load_rust_debug_dir(debug_dir):
    """Load all .bin files from the debug directory."""
    if not os.path.isdir(debug_dir):
        return {}
    result = {}
    for fname in sorted(os.listdir(debug_dir)):
        if fname.endswith(".bin"):
            path = os.path.join(debug_dir, fname)
            stem = fname.rsplit("_", 1)[0]  # e.g. "bert_output"
            try:
                result[stem] = load_rust_tensor(path)
            except Exception as e:
                print(f"  [warn] could not load {fname}: {e}")
    return result


# ── Comparison helpers ────────────────────────────────────────────────────────

def compare(label, onnx_arr, rust_arr, n=10):
    """Print a side-by-side comparison of two arrays."""
    print(f"\n{'─'*70}")
    print(f"STAGE: {label}")

    onnx_arr = np.array(onnx_arr, dtype=np.float32)
    rust_arr = np.array(rust_arr, dtype=np.float32)

    print(f"  ONNX shape: {onnx_arr.shape}   Rust shape: {rust_arr.shape}")

    onnx_flat = onnx_arr.flatten()
    rust_flat = rust_arr.flatten()

    min_n = min(n, len(onnx_flat), len(rust_flat))
    onnx_first = onnx_flat[:min_n]
    rust_first = rust_flat[:min_n]

    print(f"  ONNX first{min_n}: {onnx_first.tolist()}")
    print(f"  Rust first{min_n}: {rust_first.tolist()}")

    if onnx_arr.shape == rust_arr.shape:
        diff = np.abs(onnx_arr - rust_arr)
        max_diff = diff.max()
        mean_diff = diff.mean()
        corr = float(np.corrcoef(onnx_flat, rust_flat)[0, 1]) if len(onnx_flat) > 1 else float("nan")
        print(f"  max_abs_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  correlation={corr:.6f}")
        if max_diff < 1e-4:
            print("  [OK] Tensors match (max_diff < 1e-4)")
        elif max_diff < 1e-2:
            print("  [~=] Close match (max_diff < 1e-2)")
        elif corr > 0.99:
            print(f"  [??] Different magnitudes but high correlation ({corr:.4f})")
        else:
            print(f"  [!!] MISMATCH (max_diff={max_diff:.4f}, corr={corr:.4f})")
    else:
        print("  [!!] SHAPE MISMATCH — cannot compute diff")


def show_onnx_only(label, arr, n=10):
    """Print an ONNX tensor when no Rust counterpart is available."""
    arr = np.array(arr, dtype=np.float32)
    flat = arr.flatten()
    print(f"\n{'─'*70}")
    print(f"STAGE: {label} (ONNX only — no Rust debug file)")
    print(f"  shape: {arr.shape}")
    print(f"  first{min(n, len(flat))}: {flat[:n].tolist()}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare KittenTTS ONNX vs Rust pipeline.")
    parser.add_argument("--debug-dir", default="/tmp/kitten_debug",
                        help="Directory containing Rust debug .bin files")
    parser.add_argument("--onnx-model", default=ONNX_MODEL)
    parser.add_argument("--voices-npz", default=VOICES_NPZ)
    parser.add_argument("--voice", default=REFERENCE_VOICE)
    parser.add_argument("--voice-idx", type=int, default=REFERENCE_IDX)
    parser.add_argument("--speed", type=float, default=REFERENCE_SPEED)
    args = parser.parse_args()

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnx and onnxruntime are required. Run: pip install onnx onnxruntime")
        sys.exit(1)

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("Loading voices...")
    voices = np.load(args.voices_npz)
    style = voices[args.voice][args.voice_idx:args.voice_idx + 1].astype(np.float32)
    print(f"  style shape: {style.shape}  first5: {style.flatten()[:5].tolist()}")

    ids = REFERENCE_IDS
    print(f"  input_ids ({len(ids)}): {ids}")

    # ── Run ONNX ─────────────────────────────────────────────────────────────
    print(f"\nRunning ONNX model: {args.onnx_model}")
    results, model = run_onnx_with_intermediates(args.onnx_model, ids, style, args.speed)
    print(f"  {len(results)} tensors captured")

    # ── Load Rust debug tensors ───────────────────────────────────────────────
    rust = load_rust_debug_dir(args.debug_dir)
    print(f"\nLoaded {len(rust)} Rust debug tensors from: {args.debug_dir}")
    if rust:
        print(f"  Keys: {sorted(rust.keys())}")
    else:
        print("  (no Rust debug files found — run with --debug-dir from kitten_generate)")

    print("\n\n" + "="*70)
    print("PIPELINE COMPARISON")
    print("="*70)

    # ── Stage 1: BERT output ─────────────────────────────────────────────────
    bert_hits = find_stage(results, model, ["bert", "albert"], shape_filter=lambda a: a.ndim == 3 and a.shape[2] == 128)
    if bert_hits:
        _, _, onnx_bert = bert_hits[-1]
        print(f"\n[ONNX] BERT output shape: {onnx_bert.shape}")
        print(f"  token0 first10: {onnx_bert[0, 0, :10].tolist()}")
        if "bert_output" in rust:
            compare("BERT output [1, seq, 128]", onnx_bert, rust["bert_output"])
        else:
            show_onnx_only("BERT output [1, seq, 128]", onnx_bert)

    # ── Stage 2: Text encoder LSTM output ────────────────────────────────────
    lstm_hits = find_stage(results, model, ["text_encoder", "texten", "text_enc"],
                           shape_filter=lambda a: a.ndim == 3 and a.shape[2] == 256)
    if lstm_hits:
        _, _, onnx_lstm = lstm_hits[-1]
        if "lstm_features" in rust:
            compare("Text encoder LSTM [1, seq, 256]", onnx_lstm, rust["lstm_features"])
        else:
            show_onnx_only("Text encoder LSTM [1, seq, 256]", onnx_lstm)

    # ── Stage 3: Text encoder CNN output ─────────────────────────────────────
    cnn_hits = find_stage(results, model, ["text_encoder", "texten"],
                          shape_filter=lambda a: a.ndim == 3 and a.shape[1] == 128 and a.shape[0] == 1)
    if cnn_hits:
        _, _, onnx_cnn = cnn_hits[-1]
        if "cnn_features" in rust:
            compare("Text encoder CNN [1, 128, seq]", onnx_cnn, rust["cnn_features"])
        else:
            show_onnx_only("Text encoder CNN [1, 128, seq]", onnx_cnn)

    # ── Stage 4: Durations ───────────────────────────────────────────────────
    # ONNX durations: look for integer tensors with shape [1, seq]
    dur_candidates = []
    for name, arr in results.items():
        if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == len(ids):
            # Check if it looks like rounded integers
            if np.allclose(arr, arr.round(), atol=0.01) and arr.min() >= 0:
                dur_candidates.append((name, arr))

    print(f"\n\n{'─'*70}")
    print("STAGE: Durations")
    if dur_candidates:
        _, onnx_durs = dur_candidates[0]
        onnx_dur_ints = onnx_durs.flatten().round().astype(int).tolist()
        print(f"  ONNX durations: {onnx_dur_ints}  total={sum(onnx_dur_ints)}")
        if "durations" in rust:
            rust_durs = rust["durations"].flatten().astype(int).tolist()
            print(f"  Rust durations: {rust_durs}  total={sum(rust_durs)}")
            diffs = [abs(a - b) for a, b in zip(onnx_dur_ints, rust_durs)]
            print(f"  per-token diffs: {diffs}  max_diff={max(diffs) if diffs else 0}")
    else:
        print("  [!] Could not identify duration tensor in ONNX outputs")
        if "durations" in rust:
            rust_durs = rust["durations"].flatten().astype(int).tolist()
            print(f"  Rust durations: {rust_durs}  total={sum(rust_durs)}")

    # ── Stage 5: Encode block output [1, 256, T] ─────────────────────────────
    encode_hits = find_stage(results, model, ["encode"],
                             shape_filter=lambda a: a.ndim == 3 and a.shape[1] == 256)
    if encode_hits:
        _, _, onnx_enc = encode_hits[-1]
        if "expanded_features" in rust:
            compare("Encode/expanded features [1, 256, T]", onnx_enc, rust["expanded_features"])
        else:
            show_onnx_only("Encode output [1, 256, T]", onnx_enc)

    # ── Stage 6: Shared LSTM output ──────────────────────────────────────────
    if "shared_lstm_out" in rust:
        rust_shared = rust["shared_lstm_out"]
        print(f"\n{'─'*70}")
        print(f"STAGE: Shared LSTM out (Rust only) [1, 128, T]")
        print(f"  shape={rust_shared.shape}  first10={rust_shared.flatten()[:10].tolist()}")

    # ── Stage 7: F0 and N amplitude ──────────────────────────────────────────
    for key in ("f0", "n_amp"):
        if key in rust:
            print(f"\n{'─'*70}")
            print(f"STAGE: {key} (Rust only) [1, 1, T]")
            arr = rust[key]
            print(f"  shape={arr.shape}  first10={arr.flatten()[:10].tolist()}")

    # ── Stage 8: Generator conv_post output ──────────────────────────────────
    post_hits = find_stage(results, model, ["conv_post"],
                           shape_filter=lambda a: a.ndim == 3 and a.shape[1] == 22)
    if post_hits:
        _, _, onnx_post = post_hits[-1]
        if "conv_post" in rust:
            compare("Generator conv_post [1, 22, T_stft]", onnx_post, rust["conv_post"])
        else:
            show_onnx_only("Generator conv_post [1, 22, T_stft]", onnx_post)

    # ── Stage 9: Final waveform ───────────────────────────────────────────────
    # Find graph's first output (the actual waveform before instrumentation)
    waveform_name = list(results.keys())[0]  # first output = original model output
    onnx_wave = results[waveform_name]
    if onnx_wave.ndim != 3 or onnx_wave.shape[1] != 1:
        # Search for [1, 1, N] tensor
        for name, arr in results.items():
            if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] > 5000:
                onnx_wave = arr
                break

    print(f"\n{'─'*70}")
    print("STAGE: Final waveform")
    print(f"  ONNX shape: {onnx_wave.shape}")
    print(f"  ONNX first20: {onnx_wave.flatten()[:20].tolist()}")
    if "waveform" in rust:
        rust_wave = rust["waveform"]
        print(f"  Rust shape: {rust_wave.shape}")
        print(f"  Rust first20: {rust_wave.flatten()[:20].tolist()}")
        # Align lengths for diff
        n = min(onnx_wave.size, rust_wave.size)
        diff = np.abs(onnx_wave.flatten()[:n] - rust_wave.flatten()[:n])
        print(f"  max_abs_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")
        corr = np.corrcoef(onnx_wave.flatten()[:n], rust_wave.flatten()[:n])[0, 1]
        print(f"  correlation={corr:.6f}")

    print("\n\nDone. To run with Rust debug output:")
    print("  cargo run --example kitten_generate -p kitten-core --release -- \\")
    print(f"    --debug-dir {args.debug_dir} --text 'Hello world'")
    print("  python scripts/compare_pipeline.py --debug-dir", args.debug_dir)


if __name__ == "__main__":
    main()
