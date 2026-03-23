#!/usr/bin/env python3
"""Extract ONNX intermediate tensors at decoder input/output points and save as .npy files.

Saves fixtures to /tmp/onnx_fixtures/ for feeding into the Rust decoder injection test.
"""

import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"
OUTPUT_DIR = "/tmp/onnx_fixtures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
ids = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
voices = np.load(VOICES_PATH)
style = voices["expr-voice-2-m"][11:12].astype(np.float32)   # [1, 256]
speed = np.array([1.0], dtype=np.float32)

print(f"Input IDs: {ids[0].tolist()}")
print(f"Style shape: {style.shape}")
print(f"Style[:, :5] = {style[0, :5].tolist()}")
print(f"Style[:, 128:133] = {style[0, 128:133].tolist()}")

# Save inputs
np.save(os.path.join(OUTPUT_DIR, "style.npy"), style)
np.save(os.path.join(OUTPUT_DIR, "input_ids.npy"), ids)
print(f"\nSaved style.npy and input_ids.npy")

# ── Load model and expose all intermediates ───────────────────────────────────
print("\nLoading ONNX model...", flush=True)
model = onnx.load(MODEL_PATH)

# Build skip set for non-float ops
skip_outputs: set = set()
float_elem = onnx.TensorProto.FLOAT
NON_FLOAT_OPS = {
    "SequenceEmpty", "SequenceInsert", "SequenceAt", "SplitToSequence",
    "ConcatFromSequence", "Loop", "Cast", "Shape", "Gather", "GatherElements",
    "Unsqueeze", "Squeeze", "NonZero", "TopK", "ArgMax", "ArgMin", "Where",
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

print("Running inference with all intermediates...", flush=True)
sess = ort.InferenceSession(model.SerializeToString())
raw_outputs = sess.run(None, {"input_ids": ids, "style": style, "speed": speed})
names = [o.name for o in sess.get_outputs()]
results = dict(zip(names, raw_outputs))
print(f"Total intermediate tensors captured: {len(results)}")

# ── Print all node output shapes for decoder-related nodes ───────────────────
print("\n=== NODE INVENTORY (decoder-related) ===")
decoder_keywords = ["decoder", "encode", "decode", "generator", "F0_conv", "N_conv", "asr_res"]
for node in model.graph.node:
    if any(kw.lower() in node.name.lower() for kw in decoder_keywords):
        for out in node.output:
            if out and out in results:
                arr = results[out]
                print(f"  op={node.op_type:20s}  name={node.name:60s}  out={out[:30]:30s}  shape={list(arr.shape)}")

# ── Helper ────────────────────────────────────────────────────────────────────
def save_and_show(label: str, arr: np.ndarray, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    np.save(path, arr)
    print(f"\n[SAVED] {label}")
    print(f"  file:  {filename}")
    print(f"  shape: {list(arr.shape)}")
    if arr.ndim == 3:
        print(f"  ch0 first10: {arr[0, 0, :10].tolist()}")
        if arr.shape[1] > 1:
            print(f"  ch1 first5:  {arr[0, 1, :5].tolist()}")
    elif arr.ndim == 2:
        print(f"  first10: {arr.flatten()[:10].tolist()}")
    elif arr.ndim == 1:
        print(f"  first20: {arr[:20].tolist()}")

def find_last_output_of_nodes(keyword: str, shape_filter=None):
    """Return (node_name, tensor_name, array) for last matching node that has a result."""
    found = None
    for node in model.graph.node:
        if keyword.lower() in node.name.lower():
            for out in node.output:
                if out and out in results:
                    arr = results[out]
                    if shape_filter is None or (arr.ndim == 3 and arr.shape[1] == shape_filter):
                        found = (node.name, out, arr)
    return found

# ── Save the style slices ─────────────────────────────────────────────────────
print("\n\n=== STYLE VECTORS ===")
style_half = style[:, 128:]  # [1, 128]
save_and_show("style full [1,256]", style, "style_full.npy")
save_and_show("style_half = style[:,128:] [1,128]", style_half, "style_half.npy")

# ── Find the 5 decoder inputs from the Rust decoder.forward() ────────────────
# shared_lstm_out [1,128,T], asr_features [1,128,T], f0 [1,1,2T], n_amp [1,1,2T]
# These come from the predictor. We want the exact tensors BEFORE decode enters.

print("\n\n=== DECODER INPUT TENSORS ===")

# The encoder block in the Rust decoder concatenates: [shared_lstm_out(128), f0_down(1), n_down(1)] = 130ch
# Find the Concat node output that is 130 channels feeding into encode block
enc_concat_result = None
for node in model.graph.node:
    if node.op_type == "Concat" and "encoder" not in node.name.lower():
        for out in node.output:
            if out and out in results:
                arr = results[out]
                if arr.ndim == 3 and arr.shape[1] == 130:
                    enc_concat_result = (node.name, out, arr)
                    break

if enc_concat_result:
    nname, oname, arr = enc_concat_result
    save_and_show(f"ENCODE_CONCAT_INPUT (130ch) from node {nname}", arr, "encode_input_130ch.npy")
else:
    print("\n[!] Could not find 130-channel concat (encode block input)")
    # Fallback: find by keyword
    r = find_last_output_of_nodes("/decoder/encode", 130)
    if r:
        save_and_show(f"ENCODE_BLOCK_INPUT fallback: {r[0]}", r[2], "encode_input_130ch.npy")

# Decode block input is 322ch concat: [h_256, asr_64, f0_1, n_1]
dec0_concat_result = None
for node in model.graph.node:
    if node.op_type == "Concat":
        for out in node.output:
            if out and out in results:
                arr = results[out]
                if arr.ndim == 3 and arr.shape[1] == 322:
                    # Take the first 322-ch concat (= decode.0 input)
                    if dec0_concat_result is None:
                        dec0_concat_result = (node.name, out, arr)

if dec0_concat_result:
    nname, oname, arr = dec0_concat_result
    save_and_show(f"DECODE0_CONCAT_INPUT (322ch) from node {nname}", arr, "decode0_input_322ch.npy")
else:
    print("\n[!] Could not find 322-channel concat (decode.0 input)")

# ── Find encode block output (256ch) ─────────────────────────────────────────
print("\n\n=== ENCODE BLOCK OUTPUT ===")
# The encode block output is the last 256ch tensor before decode.0's 322ch concat.
# Rust: encode block applies 1/√2 scaling via Mul or Div.
# Look for last encode-related 256ch output.
enc_out = None
for node in model.graph.node:
    if "encode" in node.name.lower() and "decoder" in node.name.lower():
        for out in node.output:
            if out and out in results:
                arr = results[out]
                if arr.ndim == 3 and arr.shape[1] == 256:
                    enc_out = (node.name, out, arr)

if enc_out:
    nname, oname, arr = enc_out
    save_and_show(f"ENCODE_OUTPUT (256ch) from {nname}", arr, "encode_output_256ch.npy")
else:
    print("\n[!] Could not find encode block output (256ch). Searching by shape...")
    # Try mul/div nodes (√2 scaling)
    for node in model.graph.node:
        if node.op_type in ("Mul", "Div") and "encode" in node.name.lower():
            for out in node.output:
                if out and out in results:
                    arr = results[out]
                    if arr.ndim == 3 and arr.shape[1] == 256:
                        enc_out = (node.name, out, arr)
    if enc_out:
        nname, oname, arr = enc_out
        save_and_show(f"ENCODE_OUTPUT fallback (256ch) from {nname}", arr, "encode_output_256ch.npy")
    else:
        print("  [FAIL] Could not find encode output")

# ── Find decode block outputs ─────────────────────────────────────────────────
print("\n\n=== DECODE BLOCK OUTPUTS ===")
# Rust applies 1/√2 at end of each decode block (Mul node).
for i in range(4):
    kw = f"decode.{i}"
    # Find last 256ch output from decode.{i} nodes
    block_out = None
    for node in model.graph.node:
        if kw in node.name:
            for out in node.output:
                if out and out in results:
                    arr = results[out]
                    if arr.ndim == 3 and arr.shape[1] == 256:
                        block_out = (node.name, out, arr)
    if block_out:
        nname, oname, arr = block_out
        save_and_show(f"DECODE_{i}_OUTPUT (256ch) from {nname}", arr, f"decode{i}_output_256ch.npy")
    else:
        print(f"\n[!] Could not find decode.{i} output (256ch)")

# ── Generator input (after decode.3 output) ───────────────────────────────────
print("\n\n=== GENERATOR INPUT ===")
# Generator input = decode.3 output = [1,256,2T]
# Already saved as decode3_output_256ch.npy above.
# Now also look for the generator ups.0 ConvTranspose output.
gen_ups0 = None
for node in model.graph.node:
    if "ups.0" in node.name or "ups_0" in node.name:
        for out in node.output:
            if out and out in results:
                arr = results[out]
                if arr.ndim == 3 and arr.shape[1] == 128:
                    gen_ups0 = (node.name, out, arr)

if gen_ups0:
    nname, oname, arr = gen_ups0
    save_and_show(f"GENERATOR_UPS0_OUTPUT (128ch) from {nname}", arr, "gen_ups0_output_128ch.npy")
else:
    print("[!] Could not find generator ups.0 output (128ch)")

# ── conv_post output ──────────────────────────────────────────────────────────
print("\n\n=== CONV_POST OUTPUT ===")
conv_post_out = None
for node in model.graph.node:
    if "conv_post" in node.name.lower():
        for out in node.output:
            if out and out in results:
                arr = results[out]
                if arr.ndim == 3 and arr.shape[1] == 22:
                    conv_post_out = (node.name, out, arr)

if conv_post_out:
    nname, oname, arr = conv_post_out
    save_and_show(f"CONV_POST_OUTPUT (22ch) from {nname}", arr, "conv_post_output_22ch.npy")
    print(f"  ch11 first5: {arr[0, 11, :5].tolist()}")
else:
    print("[!] Could not find conv_post output (22ch)")

# ── Final waveform ────────────────────────────────────────────────────────────
print("\n\n=== FINAL WAVEFORM ===")
# Run original model to get clean output
sess_orig = ort.InferenceSession(MODEL_PATH)
orig_results = sess_orig.run(None, {"input_ids": ids, "style": style, "speed": speed})
for i, out_info in enumerate(sess_orig.get_outputs()):
    arr = orig_results[i]
    print(f"  output[{i}] {out_info.name}: shape={list(arr.shape)}")
    if hasattr(arr, "flatten"):
        print(f"    first20: {arr.flatten()[:20].tolist()}")
    if hasattr(arr, "shape") and arr.ndim == 3 and arr.shape[1] == 1:
        save_and_show(f"WAVEFORM output[{i}]", arr, "waveform_onnx.npy")

# Also find waveform before tanh (largest 1-channel tensor)
waveform_before_tanh = None
for name, arr in results.items():
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] > 5000:
        if waveform_before_tanh is None or arr.shape[2] > waveform_before_tanh[1].shape[2]:
            waveform_before_tanh = (name, arr)

if waveform_before_tanh:
    nname, arr = waveform_before_tanh
    save_and_show(f"WAVEFORM_BEFORE_TANH (largest 1ch) from {nname}", arr, "waveform_before_tanh.npy")

# ── Find the decoder's individual input components ────────────────────────────
print("\n\n=== DECODER COMPONENT TENSORS ===")
# F0_conv input/output
for node in model.graph.node:
    if "F0_conv" in node.name or "f0_conv" in node.name.lower():
        for out in node.output:
            if out and out in results:
                arr = results[out]
                print(f"  F0_conv node={node.name} out={out} shape={list(arr.shape)}")
                if arr.ndim == 3 and arr.shape[1] == 1:
                    save_and_show(f"F0_DOWN (1ch) from {node.name}", arr, "f0_down_1ch.npy")

# N_conv input/output
for node in model.graph.node:
    if "N_conv" in node.name or "n_conv" in node.name.lower():
        for out in node.output:
            if out and out in results:
                arr = results[out]
                print(f"  N_conv node={node.name} out={out} shape={list(arr.shape)}")
                if arr.ndim == 3 and arr.shape[1] == 1:
                    save_and_show(f"N_DOWN (1ch) from {node.name}", arr, "n_down_1ch.npy")

# asr_res output (64ch)
for node in model.graph.node:
    if "asr_res" in node.name.lower():
        for out in node.output:
            if out and out in results:
                arr = results[out]
                print(f"  asr_res node={node.name} out={out} shape={list(arr.shape)}")
                if arr.ndim == 3 and arr.shape[1] == 64:
                    save_and_show(f"ASR_RES (64ch) from {node.name}", arr, "asr_res_64ch.npy")

# shared_lstm_out: 128ch feeding into encoder
# In Rust it comes from Predictor.forward() as shared_lstm_out [1,128,T]
# In ONNX: look for the shared LSTM / predictor LSTM output that is 128ch
for node in model.graph.node:
    if node.op_type == "LSTM" and "shared" in node.name.lower():
        print(f"\n  Shared LSTM node: {node.name}  outputs={list(node.output)}")
        for oi, out_name in enumerate(node.output):
            if out_name and out_name in results:
                arr = results[out_name]
                print(f"    output[{oi}] {out_name}: shape={list(arr.shape)}")

# Also find 128ch tensors feeding into the 130ch concat
print("\n  Searching for 128ch tensors in decoder path...")
for node in model.graph.node:
    if node.op_type == "Concat":
        for out in node.output:
            if out and out in results:
                arr = results[out]
                if arr.ndim == 3 and arr.shape[1] == 130:
                    # Found 130ch concat — trace its inputs
                    print(f"  Found 130ch Concat: {node.name}")
                    for inp in node.input:
                        if inp in results:
                            inp_arr = results[inp]
                            print(f"    input {inp}: shape={list(inp_arr.shape)}")
                            if inp_arr.ndim == 3 and inp_arr.shape[1] == 128:
                                save_and_show(f"SHARED_LSTM_OUT (128ch) input to encode", inp_arr, "shared_lstm_out_128ch.npy")

# ── Print summary of what was saved ──────────────────────────────────────────
print("\n\n=== SAVED FIXTURES ===")
saved = sorted(os.listdir(OUTPUT_DIR))
for fname in saved:
    path = os.path.join(OUTPUT_DIR, fname)
    arr = np.load(path)
    print(f"  {fname:50s} shape={list(arr.shape)}")

print(f"\nAll fixtures saved to {OUTPUT_DIR}")
print("Done.")
