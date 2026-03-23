#!/usr/bin/env python3
"""
Trace KittenTTS ONNX intermediate values to identify divergence from candle.

Uses exact same inputs as the candle test:
  input_ids: [0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]
  style:     voices.npz['expr-voice-2-m'][11:12]
  speed:     [1.0]
"""

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"

# ---------------------------------------------------------------------------
# Load voices
# ---------------------------------------------------------------------------
voices = dict(np.load(VOICES_PATH))
print("Voices keys:", list(voices.keys()))
for k, v in voices.items():
    print(f"  {k}: {v.shape} {v.dtype}")

# Style: expr-voice-2-m row 11
style = voices["expr-voice-2-m"][11:12].astype(np.float32)  # [1, 256]
print(f"\nStyle shape: {style.shape}, dtype: {style.dtype}")
print(f"  style[:, :5]   (first half start): {style[0, :5].tolist()}")
print(f"  style[:, 128:133] (second half start): {style[0, 128:133].tolist()}")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
input_ids = np.array([[0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

print(f"\nInput IDs: {input_ids[0].tolist()}")
print(f"Speed: {speed.tolist()}")

# ---------------------------------------------------------------------------
# Add ALL intermediate nodes as outputs
# ---------------------------------------------------------------------------
print("\nLoading ONNX model and adding intermediate outputs...")
model = onnx.load(MODEL_PATH)
graph = model.graph

# Collect all intermediate value_info names
intermediate_names = set()
for vi in graph.value_info:
    intermediate_names.add(vi.name)

# Also add existing outputs
for out in graph.output:
    intermediate_names.add(out.name)

# Add all intermediate tensors as outputs
existing_output_names = {out.name for out in graph.output}
for vi in graph.value_info:
    if vi.name not in existing_output_names:
        graph.output.append(vi)

print(f"Total outputs (including intermediates): {len(graph.output)}")

# ---------------------------------------------------------------------------
# Run inference with all intermediate outputs
# ---------------------------------------------------------------------------
import io
model_bytes = model.SerializeToString()
sess_options = ort.SessionOptions()
sess = ort.InferenceSession(model_bytes, sess_options=sess_options)

print("Running ONNX inference...")
output_names = [out.name for out in sess.get_outputs()]
results = sess.run(
    None,
    {
        "input_ids": input_ids,
        "style": style,
        "speed": speed,
    },
)

# Build name→value dict
result_map = dict(zip(output_names, results))
print(f"Got {len(result_map)} output tensors")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def show(name, arr, max_vals=10):
    """Print shape and first few values of an array."""
    if arr is None:
        print(f"  {name}: NOT FOUND")
        return
    flat = arr.flatten()
    vals = flat[:max_vals].tolist()
    print(f"  {name}: shape={list(arr.shape)} vals={[f'{v:.6f}' for v in vals]}")

def find_tensor(name):
    return result_map.get(name)

# ---------------------------------------------------------------------------
# Print all LSTM node names so we can find the right output names
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("LSTM NODES AND OUTPUT NAMES:")
for node in graph.node:
    if node.op_type == "LSTM":
        print(f"  name={node.name!r}  outputs={list(node.output)}")

# ---------------------------------------------------------------------------
# Print all node names (for finding bert embedding output)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("LAYER NORM NODES:")
for node in graph.node:
    if "LayerNorm" in node.op_type or "layer_norm" in node.name.lower():
        print(f"  name={node.name!r}  op={node.op_type}  outputs={list(node.output)}")

# ---------------------------------------------------------------------------
# Print all Gemm/MatMul nodes related to bert_encoder
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("BERT_ENCODER (Gemm/MatMul) NODES:")
for node in graph.node:
    if "bert_encoder" in node.name:
        print(f"  name={node.name!r}  op={node.op_type}  outputs={list(node.output)}")

# ---------------------------------------------------------------------------
# Now print key intermediate values
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("KEY INTERMEDIATE VALUES:")
print("="*70)

# 1. Style slices
print("\n[1] STYLE SLICES:")
print(f"  style[:, :5]    = {style[0, :5].tolist()}")
print(f"  style[:, 128:133] = {style[0, 128:133].tolist()}")

# 2. BERT embedding output (LayerNorm after word+type+pos embeddings)
# Find the LayerNorm node under /bert/embeddings/
print("\n[2] BERT EMBEDDING LAYERNORM OUTPUT (token 0 and 1):")
bert_emb_ln_candidates = []
for node in graph.node:
    if "bert" in node.name and "embeddings" in node.name and "LayerNorm" in node.name:
        for out in node.output:
            if out:
                bert_emb_ln_candidates.append((node.name, out))
print(f"  Candidates: {bert_emb_ln_candidates}")
for cname, cout in bert_emb_ln_candidates:
    arr = find_tensor(cout)
    if arr is not None:
        print(f"  {cname} -> {cout}: shape={list(arr.shape)}")
        print(f"    token 0, first 5: {arr[0, 0, :5].tolist()}")
        print(f"    token 1, first 5: {arr[0, 1, :5].tolist()}")

# 3. BERT encoder (embedding_hidden_mapping_in) output — first transformer input
print("\n[3] EMBEDDING HIDDEN MAPPING IN (bert encoder input):")
ehmi_candidates = []
for node in graph.node:
    if "embedding_hidden_mapping_in" in node.name:
        for out in node.output:
            if out:
                ehmi_candidates.append((node.name, out))
print(f"  Candidates: {ehmi_candidates}")
for cname, cout in ehmi_candidates:
    arr = find_tensor(cout)
    if arr is not None:
        print(f"  {cname} -> {cout}: shape={list(arr.shape)}")
        print(f"    token 0, first 5: {arr[0, 0, :5].tolist()}")
        print(f"    token 1, first 5: {arr[0, 1, :5].tolist()}")

# 4. BERT final output (after bert_encoder linear)
print("\n[4] BERT FINAL OUTPUT (after bert_encoder linear):")
bert_enc_candidates = []
for node in graph.node:
    if node.name == "/bert_encoder/Gemm" or (
        "bert_encoder" in node.name and node.op_type in ("Gemm", "MatMul", "Add")
    ):
        for out in node.output:
            if out:
                bert_enc_candidates.append((node.name, node.op_type, out))
print(f"  Candidates: {bert_enc_candidates}")
for cname, cop, cout in bert_enc_candidates:
    arr = find_tensor(cout)
    if arr is not None:
        print(f"  {cname} ({cop}) -> {cout}: shape={list(arr.shape)}")
        print(f"    token 0, first 5: {arr[0, 0, :5].tolist()}")
        if arr.shape[1] > 1:
            print(f"    token 1, first 5: {arr[0, 1, :5].tolist()}")

# Look specifically for bert_encoder output feeding into text_encoder
# Find what bert produces that feeds into text_encoder
print("\n[4b] Tracing bert_encoder output:")
bert_enc_outputs = {}
for node in graph.node:
    if "bert_encoder" in node.name:
        for out in node.output:
            if out:
                arr = find_tensor(out)
                if arr is not None and len(arr.shape) >= 2:
                    bert_enc_outputs[out] = arr
                    print(f"  {node.name} -> {out}: shape={list(arr.shape)}")
                    if len(arr.shape) == 3:
                        print(f"    token 0, first 5: {arr[0, 0, :5].tolist()}")
                        print(f"    token 1, first 5: {arr[0, 1, :5].tolist()}")

# 5. text_encoder LSTM outputs
print("\n[5] TEXT ENCODER LSTM OUTPUTS:")
for node in graph.node:
    if node.op_type == "LSTM" and "text_encoder" in node.name:
        print(f"  LSTM node: {node.name}  outputs={list(node.output)}")
        # ONNX LSTM output: [0]=Y [1]=Y_h [2]=Y_c
        # Y shape: [seq, num_dir, batch, hidden] (ONNX layout)
        for oi, out_name in enumerate(node.output):
            if out_name:
                arr = find_tensor(out_name)
                if arr is not None:
                    print(f"    output[{oi}] {out_name}: shape={list(arr.shape)}")
                    if len(arr.shape) >= 3:
                        # Y is [seq, num_dir, hidden] or [seq, 1, batch, hidden] in some layouts
                        flat = arr.flatten()[:5]
                        print(f"    first 5 flat: {flat.tolist()}")

# 6. Duration LSTM output
print("\n[6] DURATION LSTM OUTPUT (/lstm/LSTM):")
for node in graph.node:
    if node.op_type == "LSTM" and node.name in ("/lstm/LSTM", "/predictor/lstm/LSTM"):
        print(f"  LSTM node: {node.name}  outputs={list(node.output)}")
        for oi, out_name in enumerate(node.output):
            if out_name:
                arr = find_tensor(out_name)
                if arr is not None:
                    print(f"    output[{oi}] {out_name}: shape={list(arr.shape)}")
                    flat = arr.flatten()[:5]
                    print(f"    first 5 flat: {flat.tolist()}")

# 7. Duration logits (before sigmoid)
print("\n[7] DURATION LOGITS (before sigmoid):")
# Find duration_proj linear layer output
dur_logit_candidates = []
for node in graph.node:
    if "duration_proj" in node.name and node.op_type in ("Gemm", "MatMul", "Add"):
        for out in node.output:
            if out:
                dur_logit_candidates.append((node.name, node.op_type, out))
print(f"  Candidates: {dur_logit_candidates}")
for cname, cop, cout in dur_logit_candidates:
    arr = find_tensor(cout)
    if arr is not None:
        print(f"  {cname} ({cop}) -> {cout}: shape={list(arr.shape)}")
        if len(arr.shape) == 3:
            print(f"    token 0, first 10: {arr[0, 0, :10].tolist()}")
        elif len(arr.shape) == 2:
            print(f"    token 0, first 10: {arr[0, :10].tolist()}")

# 8. Duration sigmoid values (sum per token)
print("\n[8] DURATION SIGMOID (sum per token = raw durations before div/round):")
dur_sigmoid_candidates = []
for node in graph.node:
    if "duration" in node.name.lower() and node.op_type == "Sigmoid":
        for out in node.output:
            if out:
                dur_sigmoid_candidates.append((node.name, out))
# Also look for Sigmoid node consuming duration_proj output
for node in graph.node:
    if node.op_type == "Sigmoid":
        # check if any input comes from duration_proj
        for inp in node.input:
            if "duration" in inp.lower():
                for out in node.output:
                    if out:
                        dur_sigmoid_candidates.append((node.name, out))
print(f"  Candidates: {dur_sigmoid_candidates}")
for cname, cout in dur_sigmoid_candidates:
    arr = find_tensor(cout)
    if arr is not None:
        print(f"  {cname} -> {cout}: shape={list(arr.shape)}")
        if len(arr.shape) == 3:
            sums = arr[0].sum(axis=-1)
            print(f"    sum per token (raw durations): {sums.tolist()}")

# 9. Final durations
print("\n[9] FINAL DURATIONS (output):")
# The model outputs durations as one of the outputs
dur_output = result_map.get("durations")
if dur_output is None:
    # Try to find it among outputs
    for k, v in result_map.items():
        if "dur" in k.lower() and len(v.shape) <= 2:
            print(f"  Found candidate: {k}: {v}")
    # Also check the standard outputs
    for i, out in enumerate(sess.get_outputs()):
        print(f"  Output[{i}]: name={out.name!r}")
else:
    print(f"  durations: {dur_output}")

# ---------------------------------------------------------------------------
# Now let's find all nodes related to the duration computation
# so we can trace the full pipeline
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("DURATION PIPELINE (all nodes mentioning 'dur' or 'lstm'):")
for node in graph.node:
    if "dur" in node.name.lower() or (
        node.op_type == "LSTM" and "/lstm/" in node.name
    ):
        print(f"  [{node.op_type}] {node.name}  in={list(node.input)[:3]}  out={list(node.output)}")

# ---------------------------------------------------------------------------
# Print the very first model output (waveform) and second (durations)
# to confirm what the original outputs look like
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("MODEL OUTPUTS (original):")
orig_out_names = [out.name for out in model.graph.output][:5]
print(f"  First 5 output names: {orig_out_names}")

# Reload original model to get the actual outputs only
sess_orig = ort.InferenceSession(MODEL_PATH)
orig_outputs = sess_orig.get_outputs()
print(f"  Original model output count: {len(orig_outputs)}")
for o in orig_outputs:
    print(f"    {o.name}: shape_hint={o.shape}")

orig_results = sess_orig.run(
    None,
    {
        "input_ids": input_ids,
        "style": style,
        "speed": speed,
    },
)
for i, (out, val) in enumerate(zip(orig_outputs, orig_results)):
    print(f"  [{i}] {out.name}: shape={list(val.shape) if hasattr(val, 'shape') else type(val)}")
    if hasattr(val, 'shape') and val.size <= 50:
        print(f"       values: {val.tolist()}")
    elif hasattr(val, 'shape'):
        print(f"       first 10: {val.flatten()[:10].tolist()}")

print("\nDone.")
