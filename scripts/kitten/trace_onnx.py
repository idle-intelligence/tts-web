#!/usr/bin/env python3
"""Trace the KittenTTS nano ONNX model data flow."""

import onnx
from onnx import numpy_helper
import numpy as np
from collections import defaultdict, OrderedDict

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"

model = onnx.load(MODEL_PATH)
graph = model.graph

print("=" * 80)
print("MODEL INPUTS:")
for inp in graph.input:
    shape = [d.dim_value if d.dim_value != 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
    print(f"  {inp.name}: {dtype} {shape}")

print("\nMODEL OUTPUTS:")
for out in graph.output:
    shape = [d.dim_value if d.dim_value != 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
    dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
    print(f"  {out.name}: {dtype} {shape}")

# Build initializer map
print("\n" + "=" * 80)
print(f"INITIALIZERS: {len(graph.initializer)} total")
init_map = {}
init_shapes = {}
for init in graph.initializer:
    arr = numpy_helper.to_array(init)
    init_map[init.name] = arr
    init_shapes[init.name] = list(arr.shape)

# Build value_info map for intermediate tensors
value_info_map = {}
for vi in graph.value_info:
    shape = [d.dim_value if d.dim_value != 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
    dtype = onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
    value_info_map[vi.name] = (dtype, shape)

# Build producer map
producer_map = {}  # tensor_name -> node
for node in graph.node:
    for out in node.output:
        if out:
            producer_map[out] = node

# Build consumer map
consumer_map = defaultdict(list)  # tensor_name -> [nodes]
for node in graph.node:
    for inp in node.input:
        if inp:
            consumer_map[inp].append(node)

def get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode()
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
            elif attr.type == onnx.AttributeProto.GRAPH:
                return "<subgraph>"
            elif attr.type == onnx.AttributeProto.GRAPHS:
                return "<subgraphs>"
    return default

def tensor_desc(name):
    if not name:
        return "(empty)"
    if name in init_shapes:
        return f"[init {init_shapes[name]}]"
    if name in value_info_map:
        dtype, shape = value_info_map[name]
        return f"[{dtype} {shape}]"
    # Check model inputs
    for inp in graph.input:
        if inp.name == name:
            return "[model_input]"
    return "[?]"

def get_module(node_name):
    if not node_name:
        return "unknown"
    parts = node_name.strip("/").split("/")
    if len(parts) >= 2:
        return "/" + parts[0] + "/"
    return "/"

# Group nodes by module
module_nodes = defaultdict(list)
for i, node in enumerate(graph.node):
    mod = get_module(node.name)
    module_nodes[mod].append((i, node))

print("\n" + "=" * 80)
print("MODULES FOUND:")
for mod, nodes in sorted(module_nodes.items()):
    print(f"  {mod}: {len(nodes)} nodes")

print("\n" + "=" * 80)
print("COMPLETE EXECUTION ORDER (all nodes):")
print("=" * 80)

for i, node in enumerate(graph.node):
    op = node.op_type
    name = node.name or f"node_{i}"

    inputs_desc = []
    for inp in node.input:
        if not inp:
            inputs_desc.append("(empty)")
        elif inp in init_shapes:
            inputs_desc.append(f"{inp} [init {init_shapes[inp]}]")
        else:
            inputs_desc.append(f"{inp} {tensor_desc(inp)}")

    outputs_desc = []
    for out in node.output:
        if not out:
            outputs_desc.append("(empty)")
        else:
            outputs_desc.append(f"{out} {tensor_desc(out)}")

    print(f"\n[{i:04d}] {name}")
    print(f"  op_type: {op}")
    print(f"  inputs:  {inputs_desc}")
    print(f"  outputs: {outputs_desc}")

    # Key attributes by op type
    if op == "Conv":
        ks = get_attr(node, "kernel_shape")
        st = get_attr(node, "strides")
        pa = get_attr(node, "pads")
        di = get_attr(node, "dilations")
        gr = get_attr(node, "group", 1)
        print(f"  attrs: kernel={ks} strides={st} pads={pa} dilations={di} group={gr}")
        # Weight shape
        if len(node.input) > 1 and node.input[1] in init_shapes:
            ws = init_shapes[node.input[1]]
            print(f"  weight_shape: {ws}")
    elif op == "LSTM":
        hs = get_attr(node, "hidden_size")
        di = get_attr(node, "direction", "forward")
        print(f"  attrs: hidden_size={hs} direction={di}")
        # Infer input_size from W weight
        if len(node.input) > 1 and node.input[1] in init_shapes:
            ws = init_shapes[node.input[1]]  # [num_directions, 4*hidden, input_size]
            print(f"  W_shape: {ws} => input_size={ws[2] if len(ws)==3 else '?'}")
        if len(node.input) > 2 and node.input[2] in init_shapes:
            rs = init_shapes[node.input[2]]  # [num_directions, 4*hidden, hidden]
            print(f"  R_shape: {rs}")
    elif op in ("Gemm", "MatMul"):
        # Try to get weight shape
        for idx in [1, 2]:
            if idx < len(node.input) and node.input[idx] in init_shapes:
                ws = init_shapes[node.input[idx]]
                print(f"  weight[{idx}]_shape: {ws}")
        if op == "Gemm":
            transA = get_attr(node, "transA", 0)
            transB = get_attr(node, "transB", 0)
            print(f"  attrs: transA={transA} transB={transB}")
    elif op == "Concat":
        ax = get_attr(node, "axis")
        print(f"  attrs: axis={ax}")
    elif op == "Reshape":
        if len(node.input) > 1 and node.input[1] in init_map:
            print(f"  shape_val: {init_map[node.input[1]].tolist()}")
    elif op == "Transpose":
        perm = get_attr(node, "perm")
        print(f"  attrs: perm={perm}")
    elif op == "Slice":
        # Slice inputs: data, starts, ends, axes, steps
        for si, sname in enumerate(["starts", "ends", "axes", "steps"]):
            if si+1 < len(node.input) and node.input[si+1] in init_map:
                print(f"  {sname}: {init_map[node.input[si+1]].tolist()}")
    elif op == "Gather":
        ax = get_attr(node, "axis", 0)
        print(f"  attrs: axis={ax}")
    elif op == "Unsqueeze":
        axes = get_attr(node, "axes")
        if axes is None and len(node.input) > 1 and node.input[1] in init_map:
            axes = init_map[node.input[1]].tolist()
        print(f"  attrs: axes={axes}")
    elif op == "Squeeze":
        axes = get_attr(node, "axes")
        print(f"  attrs: axes={axes}")
    elif op in ("Add", "Mul", "Sub", "Div"):
        pass  # no special attrs
    elif op == "LayerNormalization":
        ax = get_attr(node, "axis", -1)
        eps = get_attr(node, "epsilon", 1e-5)
        print(f"  attrs: axis={ax} epsilon={eps}")
    elif op == "InstanceNormalization":
        eps = get_attr(node, "epsilon", 1e-5)
        print(f"  attrs: epsilon={eps}")
    elif op == "BatchNormalization":
        eps = get_attr(node, "epsilon", 1e-5)
        print(f"  attrs: epsilon={eps}")
    elif op == "Relu" or op == "Tanh" or op == "Sigmoid" or op == "Softmax":
        pass
    elif op == "Pad":
        mode = get_attr(node, "mode", "constant")
        print(f"  attrs: mode={mode}")
        if len(node.input) > 1 and node.input[1] in init_map:
            print(f"  pads_val: {init_map[node.input[1]].tolist()}")
    elif op == "ConstantOfShape":
        val = get_attr(node, "value")
        print(f"  attrs: value={val}")
    elif op == "Constant":
        val_attr = None
        for attr in node.attribute:
            if attr.name == "value":
                t = attr.t
                arr = numpy_helper.to_array(t)
                val_attr = arr.tolist()
        print(f"  value: {val_attr}")

print("\n" + "=" * 80)
print("STYLE INPUT CONSUMERS (direct and through reshape/slice):")
print("=" * 80)

def trace_consumers(tensor_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return
    visited.add(tensor_name)
    for node in consumer_map[tensor_name]:
        print(f"{'  ' * depth}{tensor_name} -> [{node.op_type}] {node.name}")
        for out in node.output:
            if out and out not in visited:
                trace_consumers(out, depth + 1, visited)

print("\n--- style ---")
trace_consumers("style")

print("\n--- speed ---")
trace_consumers("speed")

print("\n--- input_ids ---")
trace_consumers("input_ids")

print("\n" + "=" * 80)
print("KEY DATA FLOW PATHS:")
print("=" * 80)

# Find nodes by module prefix
def nodes_in_module(prefix):
    return [(i, n) for i, n in enumerate(graph.node) if n.name.startswith(prefix)]

print("\n--- BERT module ---")
bert_nodes = nodes_in_module("/bert/")
print(f"  Total: {len(bert_nodes)} nodes")
if bert_nodes:
    first_inputs = set()
    last_outputs = set()
    for i, n in bert_nodes:
        for inp in n.input:
            if inp and inp not in init_map:
                # Check if produced by bert
                if inp not in producer_map or not producer_map[inp].name.startswith("/bert/"):
                    first_inputs.add(inp)
        for out in n.output:
            if out and out not in consumer_map or all(
                not c.name.startswith("/bert/") for c in consumer_map.get(out, [])
            ):
                # Outputs consumed outside bert
                for c in consumer_map.get(out, []):
                    if not c.name.startswith("/bert/"):
                        last_outputs.add(out)
    print(f"  First inputs (from outside): {list(first_inputs)[:10]}")
    print(f"  Outputs to outside: {list(last_outputs)[:10]}")

print("\n--- text_encoder module ---")
te_nodes = nodes_in_module("/text_encoder/")
print(f"  Total: {len(te_nodes)} nodes")

print("\n--- predictor module ---")
pred_nodes = nodes_in_module("/predictor/")
print(f"  Total: {len(pred_nodes)} nodes")

print("\n--- decoder module ---")
dec_nodes = nodes_in_module("/decoder/")
print(f"  Total: {len(dec_nodes)} nodes")

print("\n--- shared module ---")
shared_nodes = nodes_in_module("/shared/")
print(f"  Total: {len(shared_nodes)} nodes")

print("\n" + "=" * 80)
print("LSTM NODES SUMMARY:")
print("=" * 80)
for i, node in enumerate(graph.node):
    if node.op_type == "LSTM":
        hs = get_attr(node, "hidden_size")
        di = get_attr(node, "direction", "forward")
        ws = init_shapes.get(node.input[1], "?") if len(node.input) > 1 else "?"
        rs = init_shapes.get(node.input[2], "?") if len(node.input) > 2 else "?"
        input_size = ws[2] if isinstance(ws, list) and len(ws) == 3 else "?"
        print(f"  [{i:04d}] {node.name}: hidden={hs} dir={di} input_size={input_size}")
        print(f"         inputs={list(node.input)} outputs={list(node.output)}")

print("\n" + "=" * 80)
print("CONV NODES SUMMARY:")
print("=" * 80)
for i, node in enumerate(graph.node):
    if node.op_type == "Conv":
        ks = get_attr(node, "kernel_shape")
        st = get_attr(node, "strides")
        gr = get_attr(node, "group", 1)
        ws = init_shapes.get(node.input[1], "?") if len(node.input) > 1 else "?"
        print(f"  [{i:04d}] {node.name}: kernel={ks} stride={st} group={gr} weight={ws}")

print("\n" + "=" * 80)
print("LINEAR (Gemm/MatMul) NODES SUMMARY:")
print("=" * 80)
for i, node in enumerate(graph.node):
    if node.op_type in ("Gemm", "MatMul"):
        shapes = []
        for idx in range(len(node.input)):
            if node.input[idx] in init_shapes:
                shapes.append(f"inp[{idx}]={init_shapes[node.input[idx]]}")
        print(f"  [{i:04d}] {node.name} ({node.op_type}): {' '.join(shapes)}")

print("\nDone.")
