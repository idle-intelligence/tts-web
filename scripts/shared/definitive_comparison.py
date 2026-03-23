#!/usr/bin/env python3
"""
Definitive comparison: trace ONNX graph by node names to get CORRECT intermediates,
then compare against Rust debug tensors from /tmp/kitten_debug/.
"""

import sys
import os
import subprocess
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
DEBUG_DIR = "/tmp/kitten_debug"
CARGO_MANIFEST = "/Users/tc/Code/idle-intelligence/tts-web-kitten"

# ── Step 1: Regenerate Rust intermediates ────────────────────────────────────
print("=" * 70)
print("STEP 1: Regenerating Rust intermediates via kitten_generate...")
print("=" * 70)

os.makedirs(DEBUG_DIR, exist_ok=True)
cmd = [
    "cargo", "run", "--example", "kitten_generate", "-p", "kitten-core", "--release", "--",
    "--model", "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-nano.safetensors",
    "--voices", "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-voices.safetensors",
    "--voice", "jasper",
    "--text", "Hello world",
    "--output", "/tmp/kitten_debug.wav",
    "--debug-dir", DEBUG_DIR,
]
result = subprocess.run(cmd, cwd=CARGO_MANIFEST, capture_output=True, text=True, timeout=300)
if result.returncode != 0:
    print("ERROR: cargo run failed!")
    print(result.stderr[-3000:])
    sys.exit(1)
print("Rust intermediates generated OK.")
print(result.stderr[-500:] if result.stderr else "")

# ── Step 2: Load ONNX model and trace node names ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Loading ONNX model and tracing node names...")
print("=" * 70)

model = onnx.load(MODEL_PATH)

# Trace specific node outputs
f0_output_name = None
n_output_name = None
f0_down_name = None
n_down_name = None
asr_name = None
cnn_embed_name = None
cnn_conv0_name = None
cnn_conv1_name = None
encode_out_name = None

for node in model.graph.node:
    # F0 prediction output (1ch F0 before F0_conv)
    if node.name == "/F0_proj/Conv" and node.op_type == "Conv":
        f0_output_name = node.output[0]
        print(f"  F0_proj/Conv output: {f0_output_name}")

    # N prediction output (before N_conv)
    if node.name == "/N_proj/Conv" and node.op_type == "Conv":
        n_output_name = node.output[0]
        print(f"  N_proj/Conv output: {n_output_name}")

    # F0_conv output (stride-2 downsample)
    if node.name == "/decoder/F0_conv/Conv" and node.op_type == "Conv":
        f0_down_name = node.output[0]
        print(f"  decoder/F0_conv/Conv output: {f0_down_name}")

    # N_conv output (stride-2 downsample)
    if node.name == "/decoder/N_conv/Conv" and node.op_type == "Conv":
        n_down_name = node.output[0]
        print(f"  decoder/N_conv/Conv output: {n_down_name}")

    # asr_res output (64ch CNN features after conv1x1)
    if node.name == "/decoder/asr_res/asr_res.0/Conv" and node.op_type == "Conv":
        asr_name = node.output[0]
        print(f"  decoder/asr_res/Conv output: {asr_name}")

    # CNN embedding (text_encoder embedding Gather)
    if node.name == "/text_encoder/embedding/Gather":
        cnn_embed_name = node.output[0]
        print(f"  text_encoder/embedding/Gather output: {cnn_embed_name}")

    # CNN conv0 LeakyRelu output
    if node.name == "/text_encoder/cnn.0/cnn.0.2/LeakyRelu":
        cnn_conv0_name = node.output[0]
        print(f"  text_encoder/cnn.0 LeakyRelu output: {cnn_conv0_name}")

    # CNN conv1 LeakyRelu output
    if node.name == "/text_encoder/cnn.1/cnn.0.2/LeakyRelu":
        cnn_conv1_name = node.output[0]
        print(f"  text_encoder/cnn.1 LeakyRelu output: {cnn_conv1_name}")

    # Encode block output (last node in encode block, after 1/sqrt2 multiply)
    if node.name == "/decoder/encode/Mul":
        encode_out_name = node.output[0]
        print(f"  decoder/encode/Mul output (encode block): {encode_out_name}")

# Collect only the outputs we need
needed = [x for x in [
    f0_output_name, n_output_name, f0_down_name, n_down_name,
    asr_name, cnn_embed_name, cnn_conv0_name, cnn_conv1_name, encode_out_name
] if x is not None]

missing = []
if f0_output_name is None:    missing.append("F0_proj/Conv")
if n_output_name is None:     missing.append("N_proj/Conv")
if f0_down_name is None:      missing.append("decoder/F0_conv/Conv")
if n_down_name is None:       missing.append("decoder/N_conv/Conv")
if asr_name is None:          missing.append("decoder/asr_res/Conv")
if cnn_embed_name is None:    missing.append("text_encoder/embedding/Gather")
if cnn_conv0_name is None:    missing.append("text_encoder/cnn.0/LeakyRelu")
if cnn_conv1_name is None:    missing.append("text_encoder/cnn.1/LeakyRelu")
if encode_out_name is None:   missing.append("decoder/encode/Mul")
if missing:
    print(f"\nWARNING: Could not find nodes: {missing}")

# ── Step 3: Add ONLY needed outputs, run inference ───────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Adding specific outputs to ONNX graph and running inference...")
print("=" * 70)

existing_outputs = {o.name for o in model.graph.output}
added = []
for name in needed:
    if name not in existing_outputs:
        model.graph.output.extend(
            [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)]
        )
        existing_outputs.add(name)
        added.append(name)

print(f"  Added {len(added)} intermediate outputs to graph.")

sess = ort.InferenceSession(model.SerializeToString())

KITTEN_VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten-voices.safetensors"
TEXT = "Hello world"

# Load jasper style from kitten-voices.safetensors using same logic as Rust:
# picks row index = min(len(text), 399), then unsqueeze -> [1, 256]
from safetensors import safe_open as st_open
st_voices = st_open(KITTEN_VOICES_PATH, framework="np")
print(f"  Voice keys: {list(st_voices.keys())}")
jasper_all = st_voices.get_tensor("jasper")  # [400, 256]
text_len = len(TEXT)
idx = min(text_len, jasper_all.shape[0] - 1)
style = jasper_all[idx:idx+1].astype(np.float32)  # [1, 256]
print(f"  Using jasper voice, text_len={text_len}, row_idx={idx}, style shape={style.shape}")

# "Hello world" tokenized — use same token IDs as Rust
# We need to check what Rust uses. Use a simple approach: look for bert token file or
# use a known mapping. For now use the same ids from the other scripts if available.
# Common approach: let's use hard-coded token ids for "Hello world" matching Rust.
# The text encoder uses its own phoneme IDs, not raw text — check voices.npz for clues.
# From inspect_kitten_onnx.py style: ids shape [1, 16] for "Hello world"
# Use the same ids that matched Rust in previous scripts:
ids = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

print(f"  Running ONNX inference with input_ids shape={ids.shape}, style shape={style.shape}...")
outputs = sess.run(None, {"input_ids": ids, "style": style, "speed": speed})
output_names = [o.name for o in sess.get_outputs()]
results = dict(zip(output_names, outputs))

print(f"  Got {len(results)} outputs.")

# ── Step 4: Load Rust debug tensors ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Loading Rust debug tensors from", DEBUG_DIR)
print("=" * 70)

def load_rust(filename):
    path = os.path.join(DEBUG_DIR, filename)
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.float32)
    return data

# Map filename → expected shape
rust_files = {
    "bert_output_1x16x128.bin":        (1, 16, 128),
    "cnn_features_1x128x16.bin":       (1, 128, 16),
    "durations_1x16.bin":              (1, 16),
    "expanded_features_1x256x50.bin":  (1, 256, 50),
    "f0_1x1x100.bin":                  (1, 1, 100),
    "lstm_features_1x16x256.bin":      (1, 16, 256),
    "n_amp_1x1x100.bin":               (1, 1, 100),
    "shared_lstm_out_1x128x50.bin":    (1, 128, 50),
    "waveform_1x1x29995.bin":          None,  # shape unknown ahead of time
}

rust_tensors = {}
for fname, expected_shape in rust_files.items():
    data = load_rust(fname)
    if data is None:
        print(f"  MISSING: {fname}")
        continue
    if expected_shape is not None:
        total = 1
        for d in expected_shape:
            total *= d
        if len(data) != total:
            print(f"  WARNING: {fname} has {len(data)} floats, expected {total}")
            rust_tensors[fname] = data
        else:
            rust_tensors[fname] = data.reshape(expected_shape)
    else:
        rust_tensors[fname] = data
    print(f"  Loaded {fname}: shape={rust_tensors[fname].shape}")

# ── Step 5: Compare each stage ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Comparison Table")
print("=" * 70)

def corr(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    if len(a) != len(b):
        return float('nan')
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def max_diff(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    if len(a) != len(b):
        return float('nan')
    return float(np.max(np.abs(a - b)))

def first5(arr):
    if arr is None:
        return "N/A"
    return [f"{x:.4f}" for x in arr.flatten()[:5]]

def fmt_shape(arr):
    if arr is None:
        return "N/A"
    return str(arr.shape)

def fmt_val(v):
    if v != v:  # NaN
        return "N/A"
    return f"{v:.4f}"

rows = []

def add_row(stage, onnx_name, rust_key, rust_arr_override=None):
    onnx_arr = results.get(onnx_name) if onnx_name else None
    rust_arr = rust_arr_override if rust_arr_override is not None else rust_tensors.get(rust_key)

    if onnx_arr is not None and rust_arr is not None:
        c = corr(onnx_arr, rust_arr)
        md = max_diff(onnx_arr, rust_arr)
    else:
        c = float('nan')
        md = float('nan')

    rows.append({
        "stage": stage,
        "onnx_shape": fmt_shape(onnx_arr),
        "rust_shape": fmt_shape(rust_arr),
        "corr": fmt_val(c),
        "max_diff": fmt_val(md),
        "onnx_first5": first5(onnx_arr) if onnx_arr is not None else ["N/A"],
        "rust_first5": first5(rust_arr) if rust_arr is not None else ["N/A"],
    })

# CNN embedding — ONNX: [1, T, 128] or [1, 128, T], Rust: no direct match
add_row("CNN embed", cnn_embed_name, None, None)

# CNN after conv0 — ONNX only
add_row("CNN after cnn.0 (LeakyRelu)", cnn_conv0_name, None, None)

# CNN after conv1 — Rust cnn_features is [1, 128, 16]
add_row("CNN after cnn.1 (LeakyRelu)", cnn_conv1_name, "cnn_features_1x128x16.bin")

# F0 pre-downsample
add_row("F0 (pre-downsamp)", f0_output_name, "f0_1x1x100.bin")

# N pre-downsample
add_row("N (pre-downsamp)", n_output_name, "n_amp_1x1x100.bin")

# asr_res (64ch CNN features after conv1x1 in decoder)
add_row("asr_res (decoder CNN)", asr_name, "expanded_features_1x256x50.bin")

# Encode block output
add_row("Encode block output", encode_out_name, "expanded_features_1x256x50.bin")

# Final waveform — ONNX output is named "waveform" and is 1D flat
onnx_waveform_name = "waveform"
onnx_waveform = results.get("waveform")
rust_waveform = rust_tensors.get("waveform_1x1x29995.bin")
add_row("Final waveform", onnx_waveform_name, "waveform_1x1x29995.bin", rust_waveform)

# Print table
col_widths = {
    "stage": 30, "onnx_shape": 18, "rust_shape": 18,
    "corr": 10, "max_diff": 10, "onnx_first5": 42, "rust_first5": 42
}
header = (
    f"{'Stage':<30} | {'ONNX shape':<18} | {'Rust shape':<18} | "
    f"{'Corr':<10} | {'MaxDiff':<10} | {'ONNX first5':<42} | {'Rust first5':<42}"
)
sep = "-" * len(header)
print(header)
print(sep)

for r in rows:
    o5 = " ".join(r["onnx_first5"])
    r5 = " ".join(r["rust_first5"])
    print(
        f"{r['stage']:<30} | {r['onnx_shape']:<18} | {r['rust_shape']:<18} | "
        f"{r['corr']:<10} | {r['max_diff']:<10} | {o5:<42} | {r5:<42}"
    )

# ── Step 6: Detailed printout for each stage ─────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Detailed values for each stage")
print("=" * 70)

def detail(label, onnx_name, rust_key, rust_arr_override=None):
    print(f"\n--- {label} ---")
    onnx_arr = results.get(onnx_name) if onnx_name else None
    rust_arr = rust_arr_override if rust_arr_override is not None else rust_tensors.get(rust_key)

    if onnx_arr is not None:
        print(f"  ONNX shape: {onnx_arr.shape}")
        print(f"  ONNX first10: {onnx_arr.flatten()[:10].tolist()}")
        print(f"  ONNX min/max: {onnx_arr.min():.4f} / {onnx_arr.max():.4f}")
    else:
        print(f"  ONNX: NOT FOUND (node name: {onnx_name})")

    if rust_arr is not None:
        print(f"  Rust shape: {rust_arr.shape}")
        print(f"  Rust first10: {rust_arr.flatten()[:10].tolist()}")
        print(f"  Rust min/max: {rust_arr.min():.4f} / {rust_arr.max():.4f}")
    else:
        print(f"  Rust: NOT FOUND ({rust_key})")

    if onnx_arr is not None and rust_arr is not None:
        c = corr(onnx_arr, rust_arr)
        md = max_diff(onnx_arr, rust_arr)
        print(f"  Correlation: {c:.6f}  MaxDiff: {md:.6f}")

detail("CNN embed", cnn_embed_name, None)
detail("CNN after cnn.0", cnn_conv0_name, None)
detail("CNN after cnn.1 (Rust: cnn_features)", cnn_conv1_name, "cnn_features_1x128x16.bin")
detail("F0 pre-downsamp (Rust: f0)", f0_output_name, "f0_1x1x100.bin")
detail("N pre-downsamp (Rust: n_amp)", n_output_name, "n_amp_1x1x100.bin")
detail("asr_res/decoder CNN (Rust: expanded_features)", asr_name, "expanded_features_1x256x50.bin")
detail("Encode block output (Rust: expanded_features)", encode_out_name, "expanded_features_1x256x50.bin")
detail("Final waveform", onnx_waveform_name, "waveform_1x1x29995.bin", rust_waveform)

# Extra: print CNN shape investigation note
print("\n" + "=" * 70)
print("STEP 7: Shape mismatch notes")
print("=" * 70)
cnn1_onnx = results.get(cnn_conv1_name)
cnn1_rust = rust_tensors.get("cnn_features_1x128x16.bin")
if cnn1_onnx is not None and cnn1_rust is not None:
    print(f"\n  CNN after cnn.1:")
    print(f"  ONNX shape: {cnn1_onnx.shape} (note: may be [1, T, C] = time-first)")
    print(f"  Rust shape: {cnn1_rust.shape} (note: [1, C, T] = channel-first)")
    # Try transposing ONNX to [1, C, T]
    if cnn1_onnx.ndim == 3:
        onnx_t = cnn1_onnx.transpose(0, 2, 1)  # [1, T, C] -> [1, C, T]
        c = corr(onnx_t, cnn1_rust)
        md = max_diff(onnx_t, cnn1_rust)
        print(f"  After transposing ONNX to {onnx_t.shape}: corr={c:.6f}  MaxDiff={md:.6f}")

# F0/N shape mismatch analysis
f0_onnx = results.get(f0_output_name)
f0_rust = rust_tensors.get("f0_1x1x100.bin")
if f0_onnx is not None and f0_rust is not None:
    print(f"\n  F0 shape investigation:")
    print(f"  ONNX: {f0_onnx.shape}, min={f0_onnx.min():.4f}, max={f0_onnx.max():.4f}")
    print(f"  Rust: {f0_rust.shape}, min={f0_rust.min():.4f}, max={f0_rust.max():.4f}")
    print(f"  ONNX values look different (small ~1.1 vs Rust large ~38-190)")
    print(f"  NOTE: Rust f0 is stored in Hz (raw F0 frequency), ONNX may be pre-sigmoid logits")

n_onnx = results.get(n_output_name)
n_rust = rust_tensors.get("n_amp_1x1x100.bin")
if n_onnx is not None and n_rust is not None:
    print(f"\n  N shape investigation:")
    print(f"  ONNX: {n_onnx.shape}, min={n_onnx.min():.4f}, max={n_onnx.max():.4f}")
    print(f"  Rust: {n_rust.shape}, min={n_rust.min():.4f}, max={n_rust.max():.4f}")
    print(f"  ONNX T=102, Rust T=100 → 2-frame difference (padding?)")

# Waveform details
print(f"\n  ONNX waveform: {results.get('waveform', np.array([])).shape}")
if 'waveform' in results:
    wf = results['waveform']
    print(f"  ONNX waveform first10: {wf.flatten()[:10].tolist()}")
    print(f"  ONNX waveform min/max: {wf.min():.4f} / {wf.max():.4f}")
if rust_waveform is not None:
    print(f"  Rust waveform first10: {rust_waveform.flatten()[:10].tolist()}")

print("\n" + "=" * 70)
print("Done.")
