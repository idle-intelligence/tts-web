#!/usr/bin/env python3
"""Extract and print key intermediate outputs from the ONNX decoder."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"

# ── Load & instrument model ──────────────────────────────────────────────────
print("Loading ONNX model...", flush=True)
model = onnx.load(MODEL_PATH)

# Build type map from value_info — mark non-float outputs to skip
skip_outputs = set()
float_elem = onnx.TensorProto.FLOAT  # 1

for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
    if vi.type.HasField("sequence_type"):
        # Sequence types are never float tensors we want
        skip_outputs.add(vi.name)
    elif vi.type.HasField("tensor_type"):
        if vi.type.tensor_type.elem_type != float_elem:
            skip_outputs.add(vi.name)

# Also skip outputs from known non-float ops
NON_FLOAT_OPS = {"SequenceEmpty", "SequenceInsert", "SequenceAt", "SplitToSequence",
                 "ConcatFromSequence", "Loop", "Reshape", "Cast", "Shape",
                 "Gather", "GatherElements", "Unsqueeze", "Squeeze",
                 "NonZero", "TopK", "ArgMax", "ArgMin", "Where"}
for node in model.graph.node:
    if node.op_type in NON_FLOAT_OPS:
        for o in node.output:
            skip_outputs.add(o)

# Add only float-typed intermediate outputs
existing_outputs = {o.name for o in model.graph.output}
for node in model.graph.node:
    for output in node.output:
        if output and output not in existing_outputs and output not in skip_outputs:
            model.graph.output.extend(
                [helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None)]
            )
            existing_outputs.add(output)

print("Running inference...", flush=True)
sess = ort.InferenceSession(model.SerializeToString())

voices = np.load(VOICES_PATH)
ids = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
style = voices["expr-voice-2-m"][11:12].astype(np.float32)
speed = np.array([1.0], dtype=np.float32)

outputs = sess.run(None, {"input_ids": ids, "style": style, "speed": speed})
names = [o.name for o in sess.get_outputs()]
results = dict(zip(names, outputs))

print(f"\nTotal intermediate tensors: {len(results)}")
print("=" * 70)

# ── Helper ────────────────────────────────────────────────────────────────────
def show(label, arr, n_ch0=10, n_ch11=5):
    """Print shape + first N values of ch0 (and optionally ch11)."""
    arr = np.array(arr)
    print(f"\n{'─'*60}")
    print(f"[CMP] {label}")
    print(f"  shape: {arr.shape}")
    if arr.ndim >= 3:
        flat0 = arr[0, 0, :].flatten()
        print(f"  ch0 first{n_ch0}: {flat0[:n_ch0].tolist()}")
        if n_ch11 and arr.shape[1] > 11:
            flat11 = arr[0, 11, :].flatten()
            print(f"  ch11 first{n_ch11}: {flat11[:n_ch11].tolist()}")
    elif arr.ndim == 2:
        print(f"  first10: {arr.flatten()[:10].tolist()}")
    elif arr.ndim == 1:
        print(f"  first20: {arr[:20].tolist()}")
    else:
        print(f"  values: {arr.flatten()[:20].tolist()}")


# ── Print every node for debug ────────────────────────────────────────────────
print("\n\n=== NODE INVENTORY ===")
for node in model.graph.node:
    out_names = [o for o in node.output if o]
    if out_names:
        first_out = out_names[0]
        if first_out in results:
            shape = results[first_out].shape
        else:
            shape = "?"
        print(f"  op={node.op_type:20s}  name={node.name:50s}  outputs={out_names}  shape={shape}")


# ── Stage 1: ENCODE block output ─────────────────────────────────────────────
print("\n\n=== STAGE 1: ENCODE OUTPUT ===")
# The encode block output feeds into decode.0.
# Look for outputs with shape [1, 256, T] that appear after encode-related nodes.
encode_candidates = []
for name, arr in results.items():
    if arr.ndim == 3 and arr.shape[1] == 256:
        encode_candidates.append((name, arr))

# Show nodes with "encode" in name
encode_nodes = [n for n in model.graph.node if "encode" in n.name.lower()]
print(f"Nodes containing 'encode': {len(encode_nodes)}")
for n in encode_nodes[-5:]:  # last few encode nodes
    for out in n.output:
        if out and out in results:
            arr = results[out]
            print(f"  node={n.name}  out={out}  shape={arr.shape}")

# Find the last encode-related output with shape [1, 256, T]
last_encode_out = None
for n in model.graph.node:
    if "encode" in n.name.lower():
        for out in n.output:
            if out and out in results and results[out].ndim == 3 and results[out].shape[1] == 256:
                last_encode_out = (out, results[out])

if last_encode_out:
    show("ENCODE_OUT (last encode node → 256ch)", last_encode_out[1], n_ch0=10)
else:
    print("  [!] Could not find encode output with shape [1,256,T]")

# ── Stage 2: DECODE blocks ────────────────────────────────────────────────────
print("\n\n=== STAGE 2: DECODE BLOCK OUTPUTS ===")
decode_nodes_by_block = {0: [], 1: [], 2: [], 3: []}
for n in model.graph.node:
    for i in range(4):
        if f"decode.{i}" in n.name or f"decode_{i}" in n.name:
            decode_nodes_by_block[i].append(n)

for i in range(4):
    nodes = decode_nodes_by_block[i]
    print(f"\n  decode.{i}: {len(nodes)} nodes")
    if nodes:
        last_node = nodes[-1]
        for out in last_node.output:
            if out and out in results:
                arr = results[out]
                show(f"DECODE_{i}_OUT (last node={last_node.name})", arr, n_ch0=5)

# ── Stage 3: Generator ups.0 ─────────────────────────────────────────────────
print("\n\n=== STAGE 3: GENERATOR UPS.0 ===")
ups0_nodes = [n for n in model.graph.node if "ups.0" in n.name or "ups_0" in n.name]
print(f"ups.0 nodes: {len(ups0_nodes)}")
for n in ups0_nodes:
    for out in n.output:
        if out and out in results:
            arr = results[out]
            print(f"  node={n.name}  op={n.op_type}  out={out}  shape={arr.shape}")
            if n.op_type == "ConvTranspose":
                show("GENERATOR UPS.0 ConvTranspose output", arr, n_ch0=5)

# ── Stage 4: After resblocks averaging ────────────────────────────────────────
print("\n\n=== STAGE 4: AFTER RESBLOCKS AVERAGING ===")
# Look for Div nodes (the /2 averaging) in the generator
div_nodes = [n for n in model.graph.node if n.op_type == "Div" and "generator" in n.name.lower()]
print(f"Div nodes in generator: {len(div_nodes)}")
for n in div_nodes[:4]:
    for out in n.output:
        if out and out in results:
            arr = results[out]
            show(f"After Div (resblock avg?) node={n.name}", arr, n_ch0=5)

# ── Stage 5: conv_post ────────────────────────────────────────────────────────
print("\n\n=== STAGE 5: GENERATOR CONV_POST ===")
post_nodes = [n for n in model.graph.node if "conv_post" in n.name.lower()]
print(f"conv_post nodes: {len(post_nodes)}")
for n in post_nodes:
    for out in n.output:
        if out and out in results:
            arr = results[out]
            show("CONV_POST output (22ch)", arr, n_ch0=5, n_ch11=5)

# ── Stage 6: Final waveform ───────────────────────────────────────────────────
print("\n\n=== STAGE 6: FINAL WAVEFORM ===")
# The final output is the waveform — find it by looking for the graph's original outputs
# or the last node's output
orig_outputs = [o.name for o in sess.get_outputs()[:1]]  # first real output before instrumentation
print(f"First graph output name: {orig_outputs}")

# The actual waveform is likely a 3D tensor [1, 1, N]
waveform_candidates = []
for name, arr in results.items():
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] > 1000:
        waveform_candidates.append((name, arr))

waveform_candidates.sort(key=lambda x: x[1].shape[2], reverse=True)
if waveform_candidates:
    name, arr = waveform_candidates[0]
    print(f"\n  Largest [1,1,N] tensor: name={name}  shape={arr.shape}")
    print(f"  first20: {arr.flatten()[:20].tolist()}")

# Also print via the session's named outputs (original model outputs)
print("\n  Session output names (original model):")
for i, out_info in enumerate(sess.get_outputs()[:5]):
    out_name = out_info.name
    if out_name in results:
        arr = results[out_name]
        print(f"    [{i}] {out_name}: shape={arr.shape}")
        if arr.ndim >= 1:
            print(f"         first20: {arr.flatten()[:20].tolist()}")

print("\n\nDone.")
