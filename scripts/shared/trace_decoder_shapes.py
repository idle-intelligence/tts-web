#!/usr/bin/env python3
"""
Trace intermediate tensor shapes through the KittenTTS decoder.
Enumerates all Conv/ConvTranspose/Concat/Resize nodes in the ONNX graph,
then runs inference with all decoder-related intermediate outputs captured.
"""

import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, TensorProto
import copy

ONNX_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"

# ─── Reference inputs ────────────────────────────────────────────────────────
input_ids = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
voices = np.load(VOICES_PATH)
style = voices['expr-voice-2-m'][11:12].astype(np.float32)   # shape [1, 256]
speed = np.array([1.0], dtype=np.float32)

print("=== Input shapes ===")
print(f"  input_ids: {input_ids.shape}")
print(f"  style:     {style.shape}")
print(f"  speed:     {speed.shape}")
print()

# ─── Load model ──────────────────────────────────────────────────────────────
model = onnx.load(ONNX_PATH)
graph = model.graph

# ─── PART 1: enumerate every node with key attributes ─────────────────────────
print("=== ONNX graph node enumeration (Conv/ConvTranspose/Concat/Resize/Unsqueeze/Transpose/Gather) ===")
TARGET_OPS = {"Conv", "ConvTranspose", "Concat", "Resize", "Unsqueeze", "Transpose",
              "Gather", "Gemm", "MatMul", "Squeeze", "Shape", "Slice", "Pad"}

# Build value_info lookup for shape info
shape_map = {}
for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
    t = vi.type.tensor_type
    if t.HasField("elem_type"):
        dims = []
        for d in t.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else d.dim_param if d.dim_param else "?")
        shape_map[vi.name] = dims

# Build initializer name set
init_names = {init.name for init in graph.initializer}

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            elif a.type == onnx.AttributeProto.FLOAT:
                return a.f
            elif a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            elif a.type == onnx.AttributeProto.FLOATS:
                return list(a.floats)
            elif a.type == onnx.AttributeProto.STRING:
                return a.s.decode()
            elif a.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(a.t).tolist()
    return default

for i, node in enumerate(graph.node):
    if node.op_type not in TARGET_OPS:
        continue

    name = node.name or f"node_{i}"
    inputs_nonconst = [inp for inp in node.input if inp and inp not in init_names]
    outputs = list(node.output)

    in_shapes = [shape_map.get(inp, ["?"]) for inp in node.input if inp]
    out_shapes = [shape_map.get(out, ["?"]) for out in node.output if out]

    print(f"[{i:4d}] {node.op_type:18s}  name={name}")
    for j, inp in enumerate(node.input):
        if not inp:
            continue
        marker = " (init)" if inp in init_names else ""
        print(f"       in[{j}] {inp[:60]}{marker}  shape={shape_map.get(inp, '?')}")
    for j, out in enumerate(node.output):
        if not out:
            continue
        print(f"       out[{j}] {out[:60]}  shape={shape_map.get(out, '?')}")

    # Op-specific attributes
    if node.op_type in ("Conv", "ConvTranspose"):
        print(f"       kernel_shape={get_attr(node,'kernel_shape')}  strides={get_attr(node,'strides')}  "
              f"pads={get_attr(node,'pads')}  dilations={get_attr(node,'dilations')}  "
              f"group={get_attr(node,'group',1)}")
        if node.op_type == "ConvTranspose":
            print(f"       output_padding={get_attr(node,'output_padding')}  output_shape={get_attr(node,'output_shape')}")
    elif node.op_type == "Concat":
        print(f"       axis={get_attr(node,'axis')}")
    elif node.op_type == "Resize":
        print(f"       mode={get_attr(node,'mode')}  coordinate_transformation_mode={get_attr(node,'coordinate_transformation_mode')}")
        # scales/sizes are often inputs, not attributes
        if len(node.input) > 2 and node.input[2]:
            # try to get from initializer
            for init in graph.initializer:
                if init.name == node.input[2]:
                    arr = numpy_helper.to_array(init)
                    print(f"       scales (from init)={arr.tolist()}")
        if len(node.input) > 3 and node.input[3]:
            for init in graph.initializer:
                if init.name == node.input[3]:
                    arr = numpy_helper.to_array(init)
                    print(f"       sizes (from init)={arr.tolist()}")
    print()

# ─── PART 2: run inference capturing ALL intermediate outputs ─────────────────
print()
print("=== Runtime inference: capturing all intermediate outputs ===")

# Add every value_info as an output so ORT can return it
augmented_model = copy.deepcopy(model)
existing_outputs = {o.name for o in augmented_model.graph.output}

added = 0
for vi in augmented_model.graph.value_info:
    if vi.name not in existing_outputs:
        augmented_model.graph.output.append(vi)
        added += 1

print(f"Added {added} intermediate outputs to graph")

# Run with ORT
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(augmented_model.SerializeToString(), sess_opts=sess_opts)

feed = {
    "input_ids": input_ids,
    "style": style,
    "speed": speed,
}

outputs = sess.run(None, feed)
output_names = [o.name for o in sess.get_outputs()]

print(f"\nTotal outputs captured: {len(outputs)}")
print()

# Print name → shape for ALL outputs, sorted by name for readability
name_shape = list(zip(output_names, [o.shape for o in outputs]))

print("=== All captured tensor shapes ===")
for name, shape in sorted(name_shape, key=lambda x: x[0]):
    print(f"  {name[:80]:80s}  {shape}")

# ─── PART 3: targeted summary ─────────────────────────────────────────────────
print()
print("=== TARGETED SUMMARY ===")

# Find the original model outputs
orig_out_names = [o.name for o in model.graph.output]
print(f"Original model outputs: {orig_out_names}")
for name, shape in name_shape:
    if name in orig_out_names:
        print(f"  {name}: {shape}")

print()
print("--- Searching for large-channel tensors (ch>=128) ---")
for name, shape in name_shape:
    if len(shape) == 3:  # [batch, ch, time]
        if shape[1] >= 128:
            print(f"  {name[:70]:70s}  {shape}")

print()
print("--- ConvTranspose outputs (upsampling candidates) ---")
for i, node in enumerate(graph.node):
    if node.op_type == "ConvTranspose":
        for out in node.output:
            for name, shape in name_shape:
                if name == out:
                    strides = get_attr(node, 'strides')
                    kernel = get_attr(node, 'kernel_shape')
                    print(f"  node[{i}] {node.name or ''} stride={strides} kernel={kernel}  out={out[:50]}  shape={shape}")

print()
print("--- Resize node outputs ---")
for i, node in enumerate(graph.node):
    if node.op_type == "Resize":
        for out in node.output:
            for name, shape in name_shape:
                if name == out:
                    print(f"  node[{i}] {node.name or ''}  out={out[:50]}  shape={shape}")

print()
print("--- Concat node outputs ---")
for i, node in enumerate(graph.node):
    if node.op_type == "Concat":
        for out in node.output:
            for name, shape in name_shape:
                if name == out:
                    axis = get_attr(node, 'axis')
                    in_shapes_rt = []
                    for inp in node.input:
                        for n2, s2 in name_shape:
                            if n2 == inp:
                                in_shapes_rt.append(s2)
                    print(f"  node[{i}] axis={axis}  inputs={in_shapes_rt}  out={out[:50]}  shape={shape}")

print()
print("Done.")
