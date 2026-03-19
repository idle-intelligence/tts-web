#!/usr/bin/env python3
"""
Focused decoder shape tracer for KittenTTS.
Filters to decoder/generator nodes only, runs inference, reports shapes.
"""

import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, TensorProto
import copy

ONNX_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"

# Reference inputs
input_ids = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
voices = np.load(VOICES_PATH)
style = voices['expr-voice-2-m'][11:12].astype(np.float32)
speed = np.array([1.0], dtype=np.float32)

print("=== Input shapes ===")
print(f"  input_ids: {input_ids.shape}  values: {input_ids[0].tolist()}")
print(f"  style:     {style.shape}")
print(f"  speed:     {speed.shape}")
print()

model = onnx.load(ONNX_PATH)
graph = model.graph

# Build shape map
shape_map = {}
for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
    t = vi.type.tensor_type
    if t.HasField("elem_type"):
        dims = []
        for d in t.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param if d.dim_param else "?"))
        shape_map[vi.name] = dims

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

# ─── PART 1: Enumerate DECODER nodes only ────────────────────────────────────
print("=== Decoder/Generator node enumeration (Conv/ConvTranspose/Concat/Resize) ===")
TARGET_OPS = {"Conv", "ConvTranspose", "Concat", "Resize"}

for i, node in enumerate(graph.node):
    if node.op_type not in TARGET_OPS:
        continue
    # Filter to decoder nodes only
    node_id = node.name or ""
    inputs_str = " ".join(node.input)
    if "decoder" not in node_id and "decoder" not in inputs_str and "decoder" not in " ".join(node.output):
        continue

    name = node.name or f"node_{i}"
    print(f"\n[{i:4d}] {node.op_type:18s}  name={name}")
    for j, inp in enumerate(node.input):
        if not inp:
            continue
        marker = " (init)" if inp in init_names else ""
        shp = shape_map.get(inp, "?")
        # For initializers, get actual shape from graph
        if inp in init_names:
            for init in graph.initializer:
                if init.name == inp:
                    shp = list(init.dims)
                    break
        print(f"       in[{j}] {inp[:70]}{marker}  shape={shp}")
    for j, out in enumerate(node.output):
        if not out:
            continue
        print(f"       out[{j}] {out[:70]}  shape={shape_map.get(out,'?')}")

    if node.op_type in ("Conv", "ConvTranspose"):
        print(f"       kernel_shape={get_attr(node,'kernel_shape')}  strides={get_attr(node,'strides')}  "
              f"pads={get_attr(node,'pads')}  dilations={get_attr(node,'dilations')}  group={get_attr(node,'group',1)}")
        if node.op_type == "ConvTranspose":
            print(f"       output_padding={get_attr(node,'output_padding')}  output_shape={get_attr(node,'output_shape')}")
        # Get weight shape from initializer
        if len(node.input) > 1 and node.input[1] in init_names:
            for init in graph.initializer:
                if init.name == node.input[1]:
                    print(f"       weight_dims={list(init.dims)}")
                    break
    elif node.op_type == "Concat":
        print(f"       axis={get_attr(node,'axis')}")
    elif node.op_type == "Resize":
        mode = get_attr(node, 'mode')
        ctm = get_attr(node, 'coordinate_transformation_mode')
        print(f"       mode={mode}  coordinate_transformation_mode={ctm}")
        if len(node.input) > 2 and node.input[2]:
            for init in graph.initializer:
                if init.name == node.input[2]:
                    print(f"       scales={numpy_helper.to_array(init).tolist()}")
        if len(node.input) > 3 and node.input[3]:
            for init in graph.initializer:
                if init.name == node.input[3]:
                    print(f"       sizes={numpy_helper.to_array(init).tolist()}")

# ─── PART 2: Runtime inference ────────────────────────────────────────────────
print()
print()
print("=" * 70)
print("=== RUNTIME INFERENCE ===")
print("=" * 70)

augmented_model = copy.deepcopy(model)
existing_outputs = {o.name for o in augmented_model.graph.output}
for vi in augmented_model.graph.value_info:
    if vi.name not in existing_outputs:
        augmented_model.graph.output.append(vi)

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(augmented_model.SerializeToString(), sess_opts=sess_opts)

feed = {"input_ids": input_ids, "style": style, "speed": speed}
print("Running inference...")
outputs = sess.run(None, feed)
output_names = [o.name for o in sess.get_outputs()]
ns = dict(zip(output_names, outputs))
print(f"Captured {len(outputs)} tensors.\n")

# Only show decoder-related tensors
print("=== Decoder tensor shapes (runtime) ===")
dec_tensors = {k: v for k, v in ns.items() if "decoder" in k}
for k in sorted(dec_tensors.keys()):
    print(f"  {k[:80]:80s}  {ns[k].shape}")

print()
print("=== Original model outputs ===")
orig_out_names = [o.name for o in model.graph.output]
for n in orig_out_names:
    if n in ns:
        print(f"  {n}: shape={ns[n].shape}  min={ns[n].min():.4f}  max={ns[n].max():.4f}")

print()
print("=== Key decoder tensors by shape ===")
# Group by channel count for 3D tensors
print("\n3D tensors [batch, ch, time] in decoder:")
for k in sorted(dec_tensors.keys()):
    v = ns[k]
    if v.ndim == 3:
        print(f"  ch={v.shape[1]:4d}  t={v.shape[2]:6d}  {k[:70]}")

print("\n2D/1D tensors in decoder:")
for k in sorted(dec_tensors.keys()):
    v = ns[k]
    if v.ndim < 3:
        print(f"  shape={v.shape}  {k[:70]}")

# ─── PART 3: trace specific flow ─────────────────────────────────────────────
print()
print("=" * 70)
print("=== ARCHITECTURE TRACE ===")
print("=" * 70)

# Find ConvTranspose outputs in decoder at runtime
print("\nConvTranspose decoder outputs:")
for i, node in enumerate(graph.node):
    if node.op_type != "ConvTranspose":
        continue
    if "decoder" not in (node.name or "") and all("decoder" not in o for o in node.output):
        continue
    for out in node.output:
        if out in ns:
            strides = get_attr(node, 'strides')
            kernel = get_attr(node, 'kernel_shape')
            wdims = None
            if len(node.input) > 1 and node.input[1] in init_names:
                for init in graph.initializer:
                    if init.name == node.input[1]:
                        wdims = list(init.dims)
            print(f"  [{i:4d}] stride={strides} kernel={kernel} weight={wdims}  shape={ns[out].shape}  name={out[:50]}")

print("\nConcat decoder outputs:")
for i, node in enumerate(graph.node):
    if node.op_type != "Concat":
        continue
    node_id = node.name or ""
    if "decoder" not in node_id and all("decoder" not in o for o in node.output):
        continue
    for out in node.output:
        if out in ns:
            axis = get_attr(node, 'axis')
            in_shapes = []
            for inp in node.input:
                if inp in ns:
                    in_shapes.append(ns[inp].shape)
                elif inp in init_names:
                    for init in graph.initializer:
                        if init.name == inp:
                            in_shapes.append(tuple(init.dims))
            print(f"  [{i:4d}] axis={axis}  inputs={in_shapes}  out_shape={ns[out].shape}")
            print(f"         out_name={out[:60]}")

print("\nResize decoder outputs:")
for i, node in enumerate(graph.node):
    if node.op_type != "Resize":
        continue
    if "decoder" not in (node.name or "") and all("decoder" not in o for o in node.output):
        continue
    for out in node.output:
        if out in ns:
            mode = get_attr(node, 'mode')
            inp_shape = ns[node.input[0]].shape if node.input[0] in ns else "?"
            scales = None
            if len(node.input) > 2 and node.input[2]:
                for init in graph.initializer:
                    if init.name == node.input[2]:
                        scales = numpy_helper.to_array(init).tolist()
            print(f"  [{i:4d}] mode={mode} scales={scales}  in={inp_shape}  out={ns[out].shape}")
            print(f"         out_name={out[:60]}")

print("\nDone.")
