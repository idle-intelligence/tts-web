#!/usr/bin/env python3
"""
Convert pocket-tts safetensors (BF16) to GGUF with Q8_0 quantization.

Reads the HuggingFace model (kyutai/pocket-tts-without-voice-cloning), applies
HF→internal name remapping, then writes a GGUF file with:
  - Q8_0 quantized tensors for large linear/projection weights
  - F32 tensors for everything else (norms, biases, embeddings, SEANet convs…)
  - Model metadata (architecture, sample_rate, frame_rate)

The Q8_0 block format (blocks of 32 elements):
  - 2 bytes: f16 scale (half-precision)
  - 32 bytes: 32 × int8 quantized values
  - Total: 34 bytes per block

Usage:
  # From HuggingFace model ID
  python quantize_to_gguf.py --model-id kyutai/pocket-tts-without-voice-cloning -o model.gguf

  # From local safetensors file
  python quantize_to_gguf.py --input model.safetensors -o model.gguf

  # With tensor-level validation
  python quantize_to_gguf.py --input model.safetensors -o model.gguf --validate
"""

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

try:
    from gguf import GGUFWriter
    import gguf
except ImportError:
    print("ERROR: gguf package is required.", file=sys.stderr)
    print("  pip install gguf", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# HF → internal name remapping
# ---------------------------------------------------------------------------

_SKIP_ENCODER = False  # Set by --no-encoder flag


def remap_key(name: str) -> str | None:
    """Map HF tensor name to internal model name. Returns None to skip."""
    if any(s in name for s in [
        "flow.w_s_t",
        "quantizer.vq",
        "quantizer.logvar_proj",
        "learnt_padding",
    ]):
        return None
    # Skip encoder components (not needed for TTS-only GGUF).
    # HF names use both `mimi.encoder.` and `mimi.model.encoder.` patterns.
    if _SKIP_ENCODER and any(s in name for s in [
        "mimi.encoder.",
        "mimi.encoder_transformer.",
        "mimi.downsample.",
        "mimi.model.encoder.",
        "mimi.model.encoder_transformer.",
        "mimi.model.downsample.",
    ]):
        return None

    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    )
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    )
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.")
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.")
    name = name.replace("mimi.model.", "mimi.")
    return name


# ---------------------------------------------------------------------------
# Quantization decision logic
# ---------------------------------------------------------------------------

_SEANET_CONV_RE = re.compile(r"(encoder|decoder)\.model\.\d+\.")


def should_quantize(name: str, tensor: torch.Tensor) -> tuple[bool, str]:
    """Return (quantize, reason) for a single tensor.

    NOTE: `name` must be the REMAPPED (internal) name, not the HF name.
    """
    if tensor.ndim < 2:
        return False, "1-D tensor (norm/bias/buffer)"

    if tensor.numel() < 1024:
        return False, f"too small ({tensor.numel()} elements)"

    if "embed" in name:
        return False, "embedding table"

    if any(p in name for p in (".norm", "_norm", "_ln.", "layer_norm")):
        return False, "normalization weight"

    if "layer_scale" in name:
        return False, "layer scale parameter"

    if "bias" in name:
        return False, "bias vector"

    if _SEANET_CONV_RE.search(name):
        return False, "Mimi SEANet conv/deconv weight"

    if "quantizer" in name:
        return False, "Mimi quantizer weight"

    if "downsample" in name or "upsample" in name:
        return False, "resampling conv weight"

    if not (name.endswith(".weight") or name.endswith("_weight")):
        return False, "not a weight tensor"

    return True, "quantizable linear/projection weight"


# ---------------------------------------------------------------------------
# Q8_0 quantization
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32  # Q8_0 block size
BYTES_PER_BLOCK = 34  # 2 (f16 scale) + 32 (int8 values)


def quantize_q8_0(tensor: torch.Tensor) -> tuple[bytes, list[int]]:
    """Quantize a tensor to Q8_0 block format.

    The tensor is flattened, then split into blocks of 32 elements.
    Each block is independently quantized:
      scale = max(|block|) / 127
      quant = round(block / scale).clamp(-128, 127)

    Returns raw bytes in Q8_0 layout and the original tensor shape.
    """
    flat = tensor.float().flatten()
    n = flat.numel()

    # Pad to multiple of BLOCK_SIZE if needed.
    remainder = n % BLOCK_SIZE
    if remainder != 0:
        pad = BLOCK_SIZE - remainder
        flat = torch.cat([flat, torch.zeros(pad)])
        n = flat.numel()

    n_blocks = n // BLOCK_SIZE
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)

    # Per-block scale: max|val| / 127
    block_max = blocks.abs().amax(dim=1)
    scales = block_max / 127.0
    # Guard against all-zero blocks.
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    # Quantize
    quantized = (blocks / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    # Pack into Q8_0 binary layout: [f16_scale, 32×int8] per block
    scales_f16 = scales.half()

    buf = bytearray(n_blocks * BYTES_PER_BLOCK)
    for i in range(n_blocks):
        offset = i * BYTES_PER_BLOCK
        # f16 scale (2 bytes, little-endian)
        buf[offset:offset + 2] = struct.pack("<e", scales_f16[i].item())
        # 32 int8 values
        buf[offset + 2:offset + BYTES_PER_BLOCK] = quantized[i].numpy().tobytes()

    return bytes(buf), list(tensor.shape)


def dequantize_q8_0(raw: bytes, shape: list[int]) -> torch.Tensor:
    """Dequantize Q8_0 raw bytes back to F32 tensor for validation."""
    numel = 1
    for d in shape:
        numel *= d

    n_padded = numel
    remainder = n_padded % BLOCK_SIZE
    if remainder != 0:
        n_padded += BLOCK_SIZE - remainder

    n_blocks = n_padded // BLOCK_SIZE
    flat = torch.zeros(n_padded, dtype=torch.float32)

    for i in range(n_blocks):
        offset = i * BYTES_PER_BLOCK
        scale = struct.unpack("<e", raw[offset:offset + 2])[0]
        quants = np.frombuffer(raw[offset + 2:offset + BYTES_PER_BLOCK], dtype=np.int8)
        flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = torch.from_numpy(
            quants.astype(np.float32)
        ) * scale

    return flat[:numel].reshape(shape)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tensors(
    original_tensors: dict[str, torch.Tensor],
    q8_data: dict[str, tuple[bytes, list[int]]],
) -> None:
    """Compare original vs round-tripped (dequantized) Q8_0 weights.

    Reports per-tensor max absolute error, RMSE, and SQNR.
    """
    print("\n=== Validation ===\n")

    worst_sqnr = float("inf")
    worst_name = ""
    warn_count = 0

    for name, (raw, shape) in sorted(q8_data.items()):
        orig = original_tensors[name].float()
        deq = dequantize_q8_0(raw, shape)

        diff = orig - deq
        max_err = diff.abs().max().item()
        rmse = diff.pow(2).mean().sqrt().item()
        signal_power = orig.pow(2).mean().item()
        noise_power = diff.pow(2).mean().item()
        sqnr = 10 * torch.log10(
            torch.tensor(signal_power / max(noise_power, 1e-30))
        ).item()

        if sqnr < worst_sqnr:
            worst_sqnr = sqnr
            worst_name = name

        if sqnr < 37:
            status = "WARN"
            warn_count += 1
        else:
            status = "OK  "

        print(
            f"  [{status}] {name:70s}  "
            f"max_err={max_err:.6f}  rmse={rmse:.6f}  sqnr={sqnr:.1f} dB"
        )

        # Simulated matmul for 2-D weights
        if orig.ndim == 2:
            x = torch.randn(4, orig.shape[1])
            y_orig = x @ orig.T
            y_quant = x @ deq.T
            matmul_err = (y_orig - y_quant).abs().max().item()
            matmul_rmse = (y_orig - y_quant).pow(2).mean().sqrt().item()
            print(
                f"         matmul check: max_err={matmul_err:.6f}  "
                f"rmse={matmul_rmse:.6f}"
            )

    print(f"\n  Worst SQNR: {worst_sqnr:.1f} dB  ({worst_name})")
    if warn_count == 0:
        print("  All layers above 37 dB SQNR threshold — quantization quality is good.")
    else:
        print(f"  {warn_count} layer(s) below 37 dB — review for audio quality impact.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pocket-tts safetensors to GGUF with Q8_0 quantization"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID (e.g. kyutai/pocket-tts-without-voice-cloning)",
    )
    group.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to local safetensors file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for GGUF file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run tensor-level validation after quantization",
    )
    parser.add_argument(
        "--no-encoder",
        action="store_true",
        help="Strip mimi encoder/encoder_transformer/downsample (TTS only needs decoder)",
    )
    args = parser.parse_args()

    global _SKIP_ENCODER
    _SKIP_ENCODER = args.no_encoder

    # ---- Resolve input path ----
    if args.model_id:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: huggingface_hub is required for --model-id.", file=sys.stderr)
            print("  pip install huggingface_hub", file=sys.stderr)
            sys.exit(1)
        print(f"Downloading safetensors from {args.model_id}...")
        for filename in ("tts_b6369a24.safetensors", "model.safetensors"):
            try:
                input_path = Path(
                    hf_hub_download(repo_id=args.model_id, filename=filename)
                )
                break
            except Exception:
                continue
        else:
            print(f"ERROR: no safetensors file found in {args.model_id}", file=sys.stderr)
            sys.exit(1)
        print(f"  Downloaded to {input_path}")
    else:
        input_path = args.input
        if not input_path.exists():
            print(f"ERROR: {input_path} not found", file=sys.stderr)
            sys.exit(1)

    # ---- Load tensors ----
    print(f"\nLoading {input_path}...")
    hf_tensors: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            hf_tensors[name] = f.get_tensor(name)

    total_params = sum(t.numel() for t in hf_tensors.values())
    total_original_bytes = sum(t.numel() * t.element_size() for t in hf_tensors.values())
    dominant_dtype = max(
        set(t.dtype for t in hf_tensors.values()),
        key=lambda d: sum(t.numel() for t in hf_tensors.values() if t.dtype == d),
    )
    print(
        f"  {len(hf_tensors)} tensors, {total_params:,} parameters "
        f"({total_original_bytes / 1e6:.1f} MB, dtype={dominant_dtype})"
    )

    # ---- Remap keys ----
    print("\nRemapping HF tensor names to internal names...\n")
    tensors: dict[str, torch.Tensor] = {}
    skipped_remap = 0
    for hf_name, tensor in hf_tensors.items():
        internal_name = remap_key(hf_name)
        if internal_name is None:
            print(f"  [SKIP] {hf_name:70s}  (filtered by remap)")
            skipped_remap += 1
            continue
        if internal_name != hf_name:
            print(f"  [MAP ] {hf_name}")
            print(f"       → {internal_name}")
        tensors[internal_name] = tensor

    print(f"\n  {len(tensors)} tensors after remapping ({skipped_remap} skipped)")

    # ---- Classify & quantize ----
    print("\nClassifying and quantizing tensors:\n")

    # For GGUF writing and optional validation
    q8_raw: dict[str, tuple[bytes, list[int]]] = {}  # name → (raw_bytes, shape)
    f32_tensors: dict[str, np.ndarray] = {}  # name → numpy array
    original_for_validation: dict[str, torch.Tensor] = {}  # name → original tensor

    quantized_params = 0
    kept_params = 0
    skipped_reasons: dict[str, list[str]] = {}

    for name in sorted(tensors.keys()):
        tensor = tensors[name]
        do_quant, reason = should_quantize(name, tensor)

        shape_str = "x".join(str(d) for d in tensor.shape)

        if do_quant:
            raw_bytes, shape = quantize_q8_0(tensor)
            q8_raw[name] = (raw_bytes, shape)
            if args.validate:
                original_for_validation[name] = tensor
            quantized_params += tensor.numel()
            tag = "Q8_0 "
        else:
            f32_tensors[name] = tensor.float().numpy()
            kept_params += tensor.numel()
            tag = "F32  "
            skipped_reasons.setdefault(reason, []).append(name)

        print(f"  [{tag}] {name:70s}  {shape_str:>20s}  {reason}")

    # ---- Write GGUF ----
    print(f"\nWriting GGUF to {args.output}...")

    writer = GGUFWriter(str(args.output), "pocket-tts")

    # Metadata
    writer.add_uint32("pocket-tts.sample_rate", 24000)
    writer.add_uint32("pocket-tts.frame_rate", 12)

    # Add Q8_0 tensors
    # gguf library expects ndarray shaped so last dim is a multiple of
    # Q8_0 type_size (34 bytes).  quant_shape_from_byte_shape then
    # converts byte-shape -> logical element shape in the metadata.
    n_q8 = 0
    for name, (raw_bytes, shape) in sorted(q8_raw.items()):
        data = np.frombuffer(raw_bytes, dtype=np.uint8)
        # Reshape: [..., ceil(cols/32)*34] so the library can infer logical shape
        n_last_blocks = (shape[-1] + BLOCK_SIZE - 1) // BLOCK_SIZE
        byte_shape = list(shape[:-1]) + [n_last_blocks * BYTES_PER_BLOCK]
        data = data.reshape(byte_shape)
        writer.add_tensor(
            name,
            data,
            raw_dtype=gguf.GGMLQuantizationType.Q8_0,
        )
        n_q8 += 1

    # Add F32 tensors
    n_f32 = 0
    for name, arr in sorted(f32_tensors.items()):
        writer.add_tensor(name, arr)
        n_f32 += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ---- Statistics ----
    output_bytes = args.output.stat().st_size

    q8_data_bytes = sum(len(raw) for raw, _ in q8_raw.values())
    f32_data_bytes = sum(arr.nbytes for arr in f32_tensors.values())

    print(f"\n=== Summary ===")
    print(f"  Tensors in GGUF:   {n_q8 + n_f32:>12d}  ({n_q8} Q8_0 + {n_f32} F32)")
    print(f"  Quantized params:  {quantized_params:>12,}  ({quantized_params / total_params * 100:.1f}%)")
    print(f"  Kept params (F32): {kept_params:>12,}  ({kept_params / total_params * 100:.1f}%)")
    print(f"  Original size:     {total_original_bytes / 1e6:>12.1f} MB  ({dominant_dtype})")
    print(f"  Q8_0 data:         {q8_data_bytes / 1e6:>12.1f} MB")
    print(f"  F32 data:          {f32_data_bytes / 1e6:>12.1f} MB")
    print(f"  GGUF file size:    {output_bytes / 1e6:>12.1f} MB")
    print(f"  Reduction:         {(1 - output_bytes / total_original_bytes) * 100:>11.1f}%")

    print(f"\n  Layers kept (F32) by category:")
    for reason, names in sorted(skipped_reasons.items()):
        n_params = sum(tensors[n].numel() for n in names)
        print(f"    {reason:40s}  {len(names):3d} tensors  {n_params:>10,} params")

    # ---- Validate ----
    if args.validate:
        validate_tensors(original_for_validation, q8_raw)

    print("\nDone.")


if __name__ == "__main__":
    main()
