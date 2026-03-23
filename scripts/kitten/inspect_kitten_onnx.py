#!/usr/bin/env python3
"""
Comprehensive ONNX inspection script for Kitten TTS models.
Usage: python inspect_kitten_onnx.py <model.onnx> [voices.npz]
"""

import sys
import os
import argparse
import math
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

# dtype map
DTYPE_MAP = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.FLOAT16: "float16",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.INT8: "int8",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.UINT8: "uint8",
    TensorProto.UINT16: "uint16",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.BOOL: "bool",
    TensorProto.STRING: "string",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
}

def dtype_name(elem_type):
    return DTYPE_MAP.get(elem_type, f"unknown({elem_type})")

def shape_str(shape_proto):
    if shape_proto is None:
        return "[]"
    dims = []
    for d in shape_proto.dim:
        if d.HasField("dim_param"):
            dims.append(d.dim_param)
        elif d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"

def tensor_shape_str(dims):
    return "[" + ", ".join(str(d) for d in dims) + "]"

def num_params(dims):
    n = 1
    for d in dims:
        n *= d
    return n

def module_prefix(name):
    parts = name.split(".")
    # Return up to 2 levels
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

def attr_to_str(attr):
    from onnx import AttributeProto
    t = attr.type
    if t == AttributeProto.FLOAT:
        return f"{attr.f:.6g}"
    elif t == AttributeProto.INT:
        return str(attr.i)
    elif t == AttributeProto.STRING:
        return repr(attr.s.decode("utf-8", errors="replace"))
    elif t == AttributeProto.TENSOR:
        arr = numpy_helper.to_array(attr.t)
        return f"Tensor{list(arr.shape)} dtype={arr.dtype}"
    elif t == AttributeProto.FLOATS:
        vals = list(attr.floats)
        if len(vals) <= 8:
            return str([f"{v:.4g}" for v in vals])
        return f"[{len(vals)} floats, first: {vals[:4]}]"
    elif t == AttributeProto.INTS:
        vals = list(attr.ints)
        if len(vals) <= 16:
            return str(vals)
        return f"[{len(vals)} ints, first: {vals[:8]}]"
    elif t == AttributeProto.GRAPH:
        return "<subgraph>"
    else:
        return f"<attr type {t}>"

def inspect_model(onnx_path, voices_path=None, model_label="Model"):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  INSPECTING: {model_label}")
    print(f"  Path: {onnx_path}")
    print(f"{sep}\n")

    model = onnx.load(onnx_path)
    graph = model.graph

    # --- Opset ---
    print("## OPSET VERSIONS")
    for op in model.opset_import:
        domain = op.domain if op.domain else "ai.onnx (default)"
        print(f"  {domain}: version {op.version}")
    print()

    # --- Inputs ---
    print("## GRAPH INPUTS")
    for inp in graph.input:
        t = inp.type.tensor_type
        print(f"  {inp.name}")
        print(f"    dtype: {dtype_name(t.elem_type)}")
        print(f"    shape: {shape_str(t.shape)}")
    print()

    # --- Outputs ---
    print("## GRAPH OUTPUTS")
    for out in graph.output:
        t = out.type.tensor_type
        print(f"  {out.name}")
        print(f"    dtype: {dtype_name(t.elem_type)}")
        print(f"    shape: {shape_str(t.shape)}")
    print()

    # --- Initializers ---
    print("## INITIALIZERS (weights/biases)")
    print(f"  Total count: {len(graph.initializer)}")
    print()

    total_params = 0
    module_params = defaultdict(int)
    module_tensors = defaultdict(list)

    init_by_name = {}
    for init in graph.initializer:
        shape = list(init.dims)
        n = num_params(shape)
        total_params += n
        prefix = module_prefix(init.name)
        module_params[prefix] += n
        module_tensors[prefix].append((init.name, shape, dtype_name(init.data_type)))
        init_by_name[init.name] = (shape, dtype_name(init.data_type))

    # Print grouped by prefix
    for prefix in sorted(module_tensors.keys()):
        tensors = module_tensors[prefix]
        prefix_total = module_params[prefix]
        print(f"  [{prefix}]  ({prefix_total:,} params)")
        for (name, shape, dt) in tensors:
            n = num_params(shape)
            print(f"    {name}")
            print(f"      shape={tensor_shape_str(shape)}  dtype={dt}  params={n:,}")
        print()

    print(f"  TOTAL INITIALIZER PARAMS: {total_params:,}")
    print()

    # --- Nodes ---
    print("## ALL GRAPH NODES")
    print(f"  Total node count: {len(graph.node)}")
    print()

    op_counts = defaultdict(int)
    for i, node in enumerate(graph.node):
        op_counts[node.op_type] += 1
        name = node.name or f"<anon_{i}>"
        print(f"  [{i:04d}] op={node.op_type}  name={name}")
        print(f"    inputs:  {list(node.input)}")
        print(f"    outputs: {list(node.output)}")
        if node.attribute:
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = attr_to_str(attr)
            print(f"    attrs:   {attrs}")
        print()

    # --- Op type summary ---
    print("## OP TYPE COUNTS")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
    print()

    # --- Module param summary ---
    print("## PARAM COUNT BY MODULE PREFIX")
    sorted_modules = sorted(module_params.items(), key=lambda x: -x[1])
    for prefix, count in sorted_modules:
        pct = 100.0 * count / total_params if total_params else 0
        print(f"  {prefix}: {count:,} ({pct:.1f}%)")
    print(f"\n  GRAND TOTAL: {total_params:,} params")
    print()

    # --- Voices ---
    if voices_path and os.path.exists(voices_path):
        print("## VOICES.NPZ")
        data = np.load(voices_path)
        for key in sorted(data.files):
            arr = data[key]
            print(f"  {key}: shape={list(arr.shape)}  dtype={arr.dtype}  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
        print()
    elif voices_path:
        print(f"## VOICES.NPZ: NOT FOUND at {voices_path}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect Kitten TTS ONNX models")
    parser.add_argument("onnx_path", help="Path to .onnx file")
    parser.add_argument("--voices", help="Path to voices.npz", default=None)
    parser.add_argument("--label", help="Label for this model", default=None)
    args = parser.parse_args()

    label = args.label or os.path.basename(args.onnx_path)
    voices = args.voices
    if voices is None:
        # auto-detect voices.npz next to onnx
        candidate = os.path.join(os.path.dirname(args.onnx_path), "voices.npz")
        if os.path.exists(candidate):
            voices = candidate

    inspect_model(args.onnx_path, voices, label)


if __name__ == "__main__":
    main()
