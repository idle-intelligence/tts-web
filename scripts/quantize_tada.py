#!/usr/bin/env python3
"""Convert TADA-1B safetensors (BF16) to GGUF format with mixed precision.

Usage:
    python3 scripts/quantize_tada.py [--input PATH] [--output PATH] [--format mixed|q4_0|f16|f32]
                                     [--vv-type f32|f16|q8_0|q4_0] [--embed-type f32|f16|q8_0|q4_0]

Requires numpy for fast vectorized quantization.

Mixed precision strategy (default):
  - Llama backbone (model.layers.*) → Q4_0  (GPU, validated)
  - Embeddings (model.embed_tokens) → Q8_0  (biggest single win, configurable via --embed-type)
  - VibeVoice (prediction_head.*)   → Q8_0  (near-lossless for flow matching, configurable via --vv-type)
  - Decoder attention               → Q8_0  (Q4_0 destroys decoder quality)
  - Decoder convs/Snake1d           → F32   (small + precision-sensitive)
  - Norms, adapters, small tensors  → F32
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

# GGUF tensor types
GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1
GGUF_TENSOR_Q4_0 = 2
GGUF_TENSOR_Q8_0 = 8

Q4_0_BLOCK_SIZE = 32  # values per block
Q4_0_BYTES_PER_BLOCK = 18  # 2 (f16 scale) + 16 (packed nibbles)

Q8_0_BLOCK_SIZE = 32  # values per block
Q8_0_BYTES_PER_BLOCK = 34  # 2 (f16 scale) + 32 (int8 values)

# ---------------------------------------------------------------------------
# BF16 → F32 conversion (numpy vectorized)
# ---------------------------------------------------------------------------

def bf16_to_f32_numpy(raw: bytes) -> np.ndarray:
    """Convert raw BF16 bytes to F32 numpy array.

    BF16 is the upper 16 bits of F32 — we pad the lower 16 bits with zeros.
    """
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bytes(values: np.ndarray) -> bytes:
    """Pack F32 numpy array to raw bytes."""
    return values.astype(np.float32).tobytes()


def f32_to_f16_bytes(values: np.ndarray) -> bytes:
    """Pack F32 numpy array to F16 raw bytes."""
    return values.astype(np.float16).tobytes()


# ---------------------------------------------------------------------------
# Q4_0 quantization
# ---------------------------------------------------------------------------

def quantize_q4_0(values: np.ndarray) -> bytes:
    """Quantize F32 numpy array to Q4_0 format (vectorized).

    Input length must be a multiple of 32.
    Returns packed Q4_0 bytes: each block is 2 bytes (f16 scale) + 16 bytes (nibbles).
    """
    n = len(values)
    assert n % Q4_0_BLOCK_SIZE == 0, f"Length {n} not multiple of {Q4_0_BLOCK_SIZE}"
    n_blocks = n // Q4_0_BLOCK_SIZE

    # Reshape to [n_blocks, 32]
    blocks = values.reshape(n_blocks, Q4_0_BLOCK_SIZE).astype(np.float32)

    # Scale per block: d = max(|v|) / 7
    amax = np.abs(blocks).max(axis=1)  # [n_blocks]
    d = amax / 7.0  # [n_blocks]

    # Quantize: q = clamp(round(v / d) + 8, 0, 15)
    # Avoid division by zero
    inv_d = np.where(d != 0.0, 1.0 / d, 0.0)[:, np.newaxis]  # [n_blocks, 1]
    q = np.clip(np.round(blocks * inv_d) + 8, 0, 15).astype(np.uint8)  # [n_blocks, 32]

    # Pack nibbles: GGML Q4_0 convention — low nibble = first 16 values,
    # high nibble = second 16 values.
    # byte[j] = q[j] | (q[j+16] << 4)  for j = 0..15
    q_lo = q[:, :16]   # [n_blocks, 16] — first half of block
    q_hi = q[:, 16:]   # [n_blocks, 16] — second half of block
    packed = (q_lo | (q_hi << 4)).astype(np.uint8)  # [n_blocks, 16]

    # Convert scales to F16
    scales_f16 = d.astype(np.float16)  # [n_blocks]

    # Build output: interleave [scale_f16 (2 bytes), packed (16 bytes)] per block
    out = np.empty((n_blocks, Q4_0_BYTES_PER_BLOCK), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:] = packed

    return out.tobytes()


# ---------------------------------------------------------------------------
# Q8_0 quantization
# ---------------------------------------------------------------------------

def quantize_q8_0(values: np.ndarray) -> bytes:
    """Quantize F32 numpy array to Q8_0 format (vectorized).

    Input length must be a multiple of 32.
    Returns packed Q8_0 bytes: each block is 2 bytes (f16 scale) + 32 bytes (int8 values).
    """
    n = len(values)
    assert n % Q8_0_BLOCK_SIZE == 0, f"Length {n} not multiple of {Q8_0_BLOCK_SIZE}"
    n_blocks = n // Q8_0_BLOCK_SIZE

    blocks = values.reshape(n_blocks, Q8_0_BLOCK_SIZE).astype(np.float32)

    # Scale per block: d = max(|v|) / 127
    amax = np.abs(blocks).max(axis=1)  # [n_blocks]
    d = amax / 127.0  # [n_blocks]

    # Quantize: q = clamp(round(v / d), -128, 127)
    inv_d = np.where(d != 0.0, 1.0 / d, 0.0)[:, np.newaxis]
    q = np.clip(np.round(blocks * inv_d), -128, 127).astype(np.int8)  # [n_blocks, 32]

    # Convert scales to F16
    scales_f16 = d.astype(np.float16)

    # Build output: [scale_f16 (2 bytes), int8 values (32 bytes)] per block = 34 bytes
    out = np.empty((n_blocks, Q8_0_BYTES_PER_BLOCK), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:] = q.view(np.uint8)

    return out.tobytes()


# ---------------------------------------------------------------------------
# GGUF writer helpers
# ---------------------------------------------------------------------------

def write_gguf_string(f, s: str):
    """Write a GGUF string (uint64 length + utf8 bytes, no null terminator)."""
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def write_gguf_metadata_kv_string(f, key: str, value: str):
    """Write a GGUF metadata key-value pair (string type)."""
    write_gguf_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_STRING))
    write_gguf_string(f, value)


def write_gguf_metadata_kv_uint32(f, key: str, value: int):
    """Write a GGUF metadata key-value pair (uint32 type)."""
    write_gguf_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', value))


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

def should_skip(name: str) -> bool:
    """Return True for tensors we skip entirely."""
    return '_precomputed_mask' in name or '.rope_freqs' in name


def should_quantize_q4_0(name: str, shape: list[int], n_elements: int) -> bool:
    """Return True if this tensor should be quantized to Q4_0.

    Q4_0 targets: 2D weight tensors >= 256 elements from:
    - model.layers.* (Llama backbone weights)
    - prediction_head.* weights (not norms)
    - _decoder.local_attention_decoder.layers.*.self_attn.{qkv,out_proj}.weight
    - _decoder.local_attention_decoder.layers.*.ffn.{0,3}.weight
    """
    # Must be 2D and >= 256 elements
    if len(shape) != 2 or n_elements < 256:
        return False

    # GGUF dims are reversed by candle on load, so candle's "last dim" is
    # our first dim.  Both dims must be divisible by the Q4_0 block size.
    if any(d % Q4_0_BLOCK_SIZE != 0 for d in shape):
        return False

    # Must be a .weight tensor
    if not name.endswith('.weight'):
        return False

    # Llama backbone (including embed_tokens)
    if name.startswith('model.'):
        # Skip layernorm weights
        if 'layernorm' in name or 'norm.weight' in name:
            return False
        return True

    # Prediction head — keep F32! Q4_0 degrades flow matching quality
    # (produces DC offset and noise in decoded audio)
    # if name.startswith('prediction_head.'):
    #     if '.norm.weight' in name:
    #         return False
    #     if '.t_embedder.' in name:
    #         return False
    #     return True

    # Decoder local attention — keep F32! Q4_0 destroys decoder quality
    # (the decoder operates on final acoustic features and is very sensitive
    # to quantization noise — produces DC offset and noise output with Q4_0)

    return False


def get_quant_type(name: str, shape: list[int]) -> str:
    """Determine quantization type for a tensor.

    Returns 'q4_0', 'q8_0', or 'f32'.

    Strategy:
      - Llama backbone (model.layers.*) → Q4_0 (GPU, validated)
      - Embeddings (model.embed_tokens) → Q8_0 (biggest single win)
      - VibeVoice (prediction_head.*)   → Q8_0 (near-lossless for flow matching)
      - Decoder attention               → Q8_0 (Q4_0 destroys decoder quality)
      - Decoder convs/Snake1d           → F32  (small + precision-sensitive)
      - Norms, adapters, small tensors  → F32
    """
    # Must be 2D weight tensor with dims divisible by 32
    if not name.endswith('.weight') or len(shape) != 2:
        return 'f32'
    if any(d % 32 != 0 for d in shape):
        return 'f32'  # e.g. 528-dim VibeVoice projections
    if 'layernorm' in name or 'norm.weight' in name:
        return 'f32'

    # Llama backbone layers → Q4_0 (GPU)
    if name.startswith('model.layers.'):
        return 'q4_0'

    # Embeddings → Q8_0
    if name == 'model.embed_tokens.weight':
        return 'q8_0'

    # VibeVoice prediction head → Q8_0
    if name.startswith('prediction_head.'):
        return 'q8_0'

    # Decoder local attention → Q8_0 (Q4_0 destroys decoder quality)
    if '_decoder.' in name and 'local_attention_decoder' in name:
        return 'q8_0'

    # Everything else (decoder convs, Snake1d, adapters) → F32
    return 'f32'


def get_quant_type_custom(name: str, shape: list[int],
                          vv_type: str = 'q8_0', embed_type: str = 'q8_0') -> str:
    """Determine quantization type for a tensor with configurable VV and embed types.

    Works like get_quant_type but allows overriding the VibeVoice (prediction_head.*)
    and embedding (model.embed_tokens.weight) quantization types.

    vv_type / embed_type: 'f32', 'f16', 'q8_0', or 'q4_0'

    Strategy:
      - Llama backbone (model.layers.*) → Q4_0  (always, not overridable)
      - Embeddings (model.embed_tokens) → embed_type
      - VibeVoice (prediction_head.*)   → vv_type
      - Decoder attention               → Q8_0  (Q4_0 destroys decoder quality)
      - Decoder convs/Snake1d           → F32   (small + precision-sensitive)
      - Norms, adapters, small tensors  → F32
    """
    # Must be 2D weight tensor with dims divisible by 32 for any quantization
    if not name.endswith('.weight') or len(shape) != 2:
        return 'f32'
    if 'layernorm' in name or 'norm.weight' in name:
        return 'f32'

    # Llama backbone layers → Q4_0 (always)
    if name.startswith('model.layers.'):
        if any(d % 32 != 0 for d in shape):
            return 'f32'
        return 'q4_0'

    # Embeddings → embed_type (fall back to f32 if dims not divisible by 32)
    if name == 'model.embed_tokens.weight':
        if embed_type in ('q4_0', 'q8_0') and any(d % 32 != 0 for d in shape):
            return 'f32'
        return embed_type

    # VibeVoice prediction head → vv_type (fall back to f32 if dims not divisible by 32)
    if name.startswith('prediction_head.'):
        if vv_type in ('q4_0', 'q8_0') and any(d % 32 != 0 for d in shape):
            return 'f32'
        return vv_type

    # Decoder local attention → Q8_0 (Q4_0 destroys decoder quality)
    if '_decoder.' in name and 'local_attention_decoder' in name:
        if any(d % 32 != 0 for d in shape):
            return 'f32'
        return 'q8_0'

    # Everything else (decoder convs, Snake1d, adapters) → F32
    return 'f32'


# ---------------------------------------------------------------------------
# Weight norm fusion
# ---------------------------------------------------------------------------

def fuse_weight_norm(g_values: np.ndarray, g_shape: list[int],
                     v_values: np.ndarray, v_shape: list[int]) -> tuple[np.ndarray, list[int]]:
    """Fuse weight_norm parametrization: w = g * v / ||v|| (vectorized).

    g (original0): shape [C_out, 1, 1] — magnitude per output channel
    v (original1): shape [C_out, C_in, K] — direction

    ||v|| is computed per output channel (over C_in * K dimensions).
    Result shape = v_shape.
    """
    c_out = g_shape[0]
    elements_per_channel = len(v_values) // c_out

    g = np.array(g_values, dtype=np.float32)[:c_out]
    v = np.array(v_values, dtype=np.float32).reshape(c_out, elements_per_channel)

    # Compute per-channel norms
    norms = np.linalg.norm(v, axis=1)  # [c_out]
    norms = np.where(norms > 0.0, norms, 1.0)

    # w = g * v / ||v||
    scale = g / norms  # [c_out]
    result = (v * scale[:, np.newaxis]).reshape(-1)

    return result, v_shape


# ---------------------------------------------------------------------------
# Safetensors reader
# ---------------------------------------------------------------------------

def parse_safetensors(path: str):
    """Parse safetensors file, return (header_dict, data_offset, file_handle)."""
    f = open(path, 'rb')
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr_raw = f.read(hdr_len)
    hdr = json.loads(hdr_raw)
    data_offset = 8 + hdr_len
    return hdr, data_offset, f


def read_tensor_bf16(f, data_offset: int, info: dict) -> np.ndarray:
    """Read a BF16 tensor and return as F32 numpy array."""
    offsets = info['data_offsets']
    start = data_offset + offsets[0]
    end = data_offset + offsets[1]
    f.seek(start)
    raw = f.read(end - start)
    return bf16_to_f32_numpy(raw)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

GGUF_ALIGNMENT = 32


def align_offset(offset: int, alignment: int = GGUF_ALIGNMENT) -> int:
    """Round up offset to next alignment boundary."""
    return (offset + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_tada_codec_decoder() -> str | None:
    """Find the cached HumeAI/tada-codec decoder safetensors file."""
    import glob
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--HumeAI--tada-codec/snapshots/*/decoder/model.safetensors"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description='Convert TADA-1B safetensors to GGUF')
    parser.add_argument('--input', type=str,
                        default='/Users/tc/Code/idle-intelligence/hf/tada-1b/model.safetensors',
                        help='Input safetensors file')
    parser.add_argument('--decoder', type=str, default=None,
                        help='Decoder safetensors from HumeAI/tada-codec (auto-detected from HF cache)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output GGUF file (default: tada-1b-{format}.gguf in input dir)')
    parser.add_argument('--format', type=str, default='mixed', choices=['mixed', 'q4_0', 'f16', 'f32'],
                        help='Output format: mixed (Q4_0+Q8_0+F32), q4_0 (legacy), f16, f32')
    parser.add_argument('--vv-type', type=str, default='q8_0', choices=['f32', 'f16', 'q8_0', 'q4_0'],
                        help='Quantization type for VibeVoice (prediction_head.*) in mixed mode (default: q8_0)')
    parser.add_argument('--embed-type', type=str, default='q8_0', choices=['f32', 'f16', 'q8_0', 'q4_0'],
                        help='Quantization type for embeddings (model.embed_tokens.weight) in mixed mode (default: q8_0)')
    args = parser.parse_args()

    input_path = args.input
    output_format = args.format
    if args.output:
        output_path = args.output
    elif output_format == 'mixed':
        output_path = os.path.join(
            os.path.dirname(input_path),
            f'tada-1b-mixed-vv{args.vv_type}-e{args.embed_type}.gguf'
        )
    else:
        output_path = os.path.join(os.path.dirname(input_path), f'tada-1b-{output_format}.gguf')

    # Find decoder weights (from tada-codec, NOT from main model!)
    # The main model safetensors has _decoder.* weights but they are stale/untrained.
    # Python loads the real decoder from HumeAI/tada-codec separately.
    decoder_path = args.decoder or find_tada_codec_decoder()
    if decoder_path is None:
        print("ERROR: Cannot find HumeAI/tada-codec decoder weights.")
        print("  Run this Python first to cache them:")
        print('    python3 -c "from tada.modules.decoder import Decoder; Decoder.from_pretrained(\\"HumeAI/tada-codec\\", subfolder=\\"decoder\\")"')
        sys.exit(1)

    input_size = os.path.getsize(input_path)
    print(f"Input:   {input_path} ({input_size / 1e9:.2f} GB)")
    print(f"Decoder: {decoder_path}")
    print(f"Output:  {output_path}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Parse safetensors headers and plan tensors
    # -----------------------------------------------------------------------
    hdr, data_offset, sf = parse_safetensors(input_path)
    all_tensors = {k: v for k, v in hdr.items() if k != '__metadata__'}

    # Parse decoder safetensors (from tada-codec)
    dec_hdr, dec_data_offset, dec_sf = parse_safetensors(decoder_path)
    dec_tensors = {k: v for k, v in dec_hdr.items() if k != '__metadata__'}
    print(f"Decoder has {len(dec_tensors)} tensors")

    # Merge decoder tensors into all_tensors, replacing _decoder.* entries.
    # Decoder tensors from tada-codec use names without the "_decoder." prefix.
    # We add them with the "_decoder." prefix and mark their source.
    decoder_tensor_names = set()
    for dec_name in sorted(dec_tensors.keys()):
        full_name = f"_decoder.{dec_name}"
        decoder_tensor_names.add(full_name)

    # Remove stale _decoder.* entries from main safetensors
    main_decoder_count = sum(1 for n in all_tensors if n.startswith('_decoder.'))
    all_tensors = {k: v for k, v in all_tensors.items() if not k.startswith('_decoder.')}
    print(f"Replaced {main_decoder_count} stale _decoder.* tensors from main model with {len(dec_tensors)} from tada-codec")

    # Add decoder tensors with _decoder. prefix
    for dec_name, info in dec_tensors.items():
        full_name = f"_decoder.{dec_name}"
        all_tensors[full_name] = info

    # Identify weight_norm pairs (now from both main and decoder tensors)
    orig0_names = sorted([n for n in all_tensors if '.parametrizations.weight.original0' in n])
    wn_pairs = {}  # base_name -> (orig0_name, orig1_name)
    wn_consumed = set()  # names consumed by weight_norm fusion

    for o0 in orig0_names:
        o1 = o0.replace('original0', 'original1')
        assert o1 in all_tensors, f"Missing weight_norm partner: {o1}"
        # Fused name: strip .parametrizations.weight.original0 → .weight
        base = o0.replace('.parametrizations.weight.original0', '.weight')
        wn_pairs[base] = (o0, o1)
        wn_consumed.add(o0)
        wn_consumed.add(o1)

    print(f"Found {len(wn_pairs)} weight_norm pairs to fuse", flush=True)

    # Build output tensor list: (name, shape, quant_type)
    output_plan = []  # list of (name, shape, 'q4_0'|'q8_0'|'f32')

    skipped = 0
    for name in sorted(all_tensors.keys()):
        if should_skip(name):
            skipped += 1
            continue
        if name in wn_consumed:
            continue  # handled as part of fused pairs
        info = all_tensors[name]
        shape = info['shape']
        n_elements = 1
        for s in shape:
            n_elements *= s
        if output_format == 'mixed':
            qt = get_quant_type_custom(name, shape, args.vv_type, args.embed_type)
        elif output_format == 'q4_0':
            # Legacy: only Llama backbone to Q4_0
            qt = 'q4_0' if should_quantize_q4_0(name, shape, n_elements) else 'f32'
        elif output_format == 'f16':
            qt = 'f16'
        else:
            qt = 'f32'
        output_plan.append((name, shape, qt))

    # Add fused weight_norm tensors
    for base_name in sorted(wn_pairs.keys()):
        o0_name, o1_name = wn_pairs[base_name]
        shape = all_tensors[o1_name]['shape']  # fused shape = v shape
        n_elements = 1
        for s in shape:
            n_elements *= s
        qt = 'f16' if output_format == 'f16' else 'f32'  # decoder conv weights never quantized
        output_plan.append((base_name, shape, qt))

    # Sort by name for deterministic output
    output_plan.sort(key=lambda x: x[0])

    n_q4 = sum(1 for _, _, qt in output_plan if qt == 'q4_0')
    n_q8 = sum(1 for _, _, qt in output_plan if qt == 'q8_0')
    n_f16 = sum(1 for _, _, qt in output_plan if qt == 'f16')
    n_f32 = sum(1 for _, _, qt in output_plan if qt == 'f32')
    print(f"Skipping {skipped} tensors (precomputed masks, rope_freqs)")
    print(f"Output tensors: {len(output_plan)} ({n_q4} Q4_0, {n_q8} Q8_0, {n_f16} F16, {n_f32} F32)")
    print(flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Read, transform, quantize, and collect tensor data
    # -----------------------------------------------------------------------
    tensor_data = []  # list of (name, shape, gguf_type, raw_bytes)
    total_f32_params = 0
    total_f16_params = 0
    total_q4_params = 0
    total_q8_params = 0

    t0 = time.time()

    for idx, (name, shape, quant_type) in enumerate(output_plan):
        n_elements = 1
        for s in shape:
            n_elements *= s

        # Read tensor data — use decoder file for _decoder.* tensors
        def _read_bf16(tensor_name):
            """Read tensor from correct source file."""
            if tensor_name in decoder_tensor_names:
                return read_tensor_bf16(dec_sf, dec_data_offset, all_tensors[tensor_name])
            return read_tensor_bf16(sf, data_offset, all_tensors[tensor_name])

        if name in wn_pairs:
            # This is a fused weight_norm tensor
            o0_name, o1_name = wn_pairs[name]
            print(f"  [{idx + 1}/{len(output_plan)}] Fusing weight_norm: {name} {shape}")
            g_values = _read_bf16(o0_name)
            v_values = _read_bf16(o1_name)
            g_shape = all_tensors[o0_name]['shape']
            v_shape = all_tensors[o1_name]['shape']
            values, shape = fuse_weight_norm(g_values, g_shape, v_values, v_shape)
            n_elements = len(values)
        else:
            values = _read_bf16(name)

        if quant_type == 'q4_0':
            assert n_elements % Q4_0_BLOCK_SIZE == 0, \
                f"{name}: {n_elements} elements not divisible by {Q4_0_BLOCK_SIZE}"
            raw = quantize_q4_0(values)
            gguf_type = GGUF_TENSOR_Q4_0
            total_q4_params += n_elements
            label = "Q4_0"
        elif quant_type == 'q8_0':
            assert n_elements % Q8_0_BLOCK_SIZE == 0, \
                f"{name}: {n_elements} elements not divisible by {Q8_0_BLOCK_SIZE}"
            raw = quantize_q8_0(values)
            gguf_type = GGUF_TENSOR_Q8_0
            total_q8_params += n_elements
            label = "Q8_0"
        elif quant_type == 'f16':
            raw = f32_to_f16_bytes(values)
            gguf_type = GGUF_TENSOR_F16
            total_f16_params += n_elements
            label = "F16"
        else:
            raw = f32_to_bytes(values)
            gguf_type = GGUF_TENSOR_F32
            total_f32_params += n_elements
            label = "F32"

        tensor_data.append((name, shape, gguf_type, raw))

        if (idx + 1) % 20 == 0 or idx == len(output_plan) - 1:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(output_plan)}] {name}: {shape} → {label} "
                  f"({len(raw)} bytes) [{elapsed:.1f}s]", flush=True)

    sf.close()
    dec_sf.close()

    print()
    print(f"Processed {len(tensor_data)} tensors in {time.time() - t0:.1f}s")
    print(f"  Q4_0 params: {total_q4_params:,}")
    print(f"  Q8_0 params: {total_q8_params:,}")
    print(f"  F16 params:  {total_f16_params:,}")
    print(f"  F32 params:  {total_f32_params:,}")

    # -----------------------------------------------------------------------
    # Phase 3: Write GGUF file
    # -----------------------------------------------------------------------
    print()
    print("Writing GGUF file...")

    # Metadata KVs we'll write
    if output_format == 'mixed':
        quant_str = (
            f"Q4_0+vv{args.vv_type.upper()}+e{args.embed_type.upper()}+Q8_0+F32"
        )
    else:
        quant_str = output_format.upper()
    metadata = [
        ("general.architecture", "tada", "string"),
        ("general.name", "TADA-1B", "string"),
        ("general.quantization", quant_str, "string"),
    ]
    n_metadata = len(metadata)
    n_tensors = len(tensor_data)

    with open(output_path, 'wb') as f:
        # --- GGUF Header ---
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_metadata))

        # --- Metadata KVs ---
        for key, value, vtype in metadata:
            if vtype == "string":
                write_gguf_metadata_kv_string(f, key, value)

        # --- Tensor infos ---
        # We need to compute offsets relative to the start of tensor data.
        # First, figure out where tensor data starts (after all tensor info).
        # Tensor info: name_string + n_dims(uint32) + dims(uint64 each) + type(uint32) + offset(uint64)

        # Pre-compute tensor info sizes to find data start
        tensor_info_sizes = []
        for name, shape, gguf_type, raw in tensor_data:
            name_bytes = name.encode('utf-8')
            size = 8 + len(name_bytes)  # string (uint64 len + bytes)
            size += 4  # n_dims
            size += 8 * len(shape)  # dims
            size += 4  # type
            size += 8  # offset
            tensor_info_sizes.append(size)

        # Current position after header + metadata
        # We'll write tensor infos next, then align, then tensor data
        header_end_pos = f.tell() + sum(tensor_info_sizes)
        data_start = align_offset(header_end_pos)

        # Compute aligned offsets for each tensor's data
        tensor_offsets = []
        current_data_offset = 0
        for _, _, _, raw in tensor_data:
            tensor_offsets.append(current_data_offset)
            current_data_offset += len(raw)
            # Align next tensor
            current_data_offset = align_offset(current_data_offset)

        # Write tensor infos
        # NOTE: GGUF stores dims in reversed order (column-major).
        # Candle's reader reverses them back on load.  So we must write
        # dims in reversed order so that candle sees the original shape.
        for i, (name, shape, gguf_type, raw) in enumerate(tensor_data):
            write_gguf_string(f, name)
            n_dims = len(shape)
            f.write(struct.pack('<I', n_dims))
            for dim in reversed(shape):
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', gguf_type))
            f.write(struct.pack('<Q', tensor_offsets[i]))

        # Pad to alignment before tensor data
        current_pos = f.tell()
        if current_pos < data_start:
            f.write(b'\x00' * (data_start - current_pos))

        # --- Tensor data ---
        for i, (name, shape, gguf_type, raw) in enumerate(tensor_data):
            f.write(raw)
            # Pad to alignment
            current_pos = f.tell() - data_start
            aligned = align_offset(current_pos)
            if aligned > current_pos:
                f.write(b'\x00' * (aligned - current_pos))

        output_size = f.tell()

    print()
    print("=" * 60)
    print(f"Input size:      {input_size:>14,} bytes ({input_size / 1e9:.2f} GB)")
    print(f"Output size:     {output_size:>14,} bytes ({output_size / 1e9:.2f} GB)")
    print(f"Compression:     {input_size / output_size:.2f}x")
    print(f"Output file:     {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
