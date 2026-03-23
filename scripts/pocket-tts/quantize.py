#!/usr/bin/env python3
"""
Offline INT8 weight quantization for pocket-tts safetensors.

The HF model (kyutai/pocket-tts-without-voice-cloning) stores weights in BF16.
This script applies per-channel weight-only INT8 quantization to large linear
and projection weight matrices while keeping sensitive layers in their original
dtype (BF16):

  - Embedding tables (integer lookup — no accumulation to average out error)
  - LayerNorm / RMSNorm / LayerScale (tiny 1-D vectors, normalization-critical)
  - All bias vectors (tiny, additive — quantization not worthwhile)
  - Mimi SEANet encoder/decoder conv weights (directly shape the audio waveform;
    even small errors produce audible clicks/tonal artifacts)
  - Mimi quantizer output projection (latent-space bottleneck, 32→512 dim)
  - Resampling conv weights (frame-rate conversion, small and sensitive)

Outputs a new safetensors file containing:
  - INT8 quantized weights (same tensor name, dtype=int8)
  - Per-channel BF16 scale factors (tensor name = original + "_scale")
  - All skipped tensors as-is in their original dtype (BF16)

Usage:
  # From HuggingFace model ID
  python quantize.py --model-id kyutai/pocket-tts-without-voice-cloning -o model_int8.safetensors

  # From local safetensors file
  python quantize.py --input model.safetensors -o model_int8.safetensors

  # With tensor-level validation
  python quantize.py --input model.safetensors -o model_int8.safetensors --validate
"""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Quantization decision logic
# ---------------------------------------------------------------------------

# Patterns that identify Mimi SEANet convolutional layers (encoder & decoder).
# These exist both in the raw HF naming (`mimi.model.encoder.model.N.…`) and
# after the Rust remap (`mimi.encoder.model.N.…`).
_SEANET_CONV_RE = re.compile(r"(encoder|decoder)\.model\.\d+\.")


def should_quantize(name: str, tensor: torch.Tensor) -> tuple[bool, str]:
    """Return (quantize, reason) for a single tensor."""

    # Only 2-D+ weight matrices benefit from quantization.
    if tensor.ndim < 2:
        return False, "1-D tensor (norm/bias/buffer)"

    # Skip tiny tensors where INT8 overhead outweighs savings.
    if tensor.numel() < 1024:
        return False, f"too small ({tensor.numel()} elements)"

    # --- Name-based exclusions ---

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

    # Must look like a weight tensor.
    if not (name.endswith(".weight") or name.endswith("_weight")):
        return False, "not a weight tensor"

    return True, "quantizable linear/projection weight"


# ---------------------------------------------------------------------------
# Per-channel INT8 quantization
# ---------------------------------------------------------------------------

def quantize_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8 quantization.

    For a 2-D weight [out, in] the scale is computed per row (output channel).
    For a 3-D conv weight [out, in/groups, kernel] the scale is per out-channel.

    Returns
    -------
    quantized : torch.int8    — same shape as input
    scale     : torch.bfloat16 — shape [out_channels]
    """
    original_shape = weight.shape
    flat = weight.reshape(weight.shape[0], -1).float()

    # Symmetric range: scale = max|w| / 127
    channel_max = flat.abs().amax(dim=1)
    scale = channel_max / 127.0
    # Guard against all-zero channels.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    quantized = (flat / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape(original_shape), scale.bfloat16()


def dequantize_per_channel(
    quantized: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct weight from INT8 + per-channel BF16 scale (all in bf16)."""
    flat = quantized.reshape(quantized.shape[0], -1).to(torch.bfloat16)
    return (flat * scale.unsqueeze(1)).reshape(quantized.shape).float()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    original_path: Path,
    quantized_path: Path,
) -> None:
    """Compare original vs round-tripped (dequantized) weights.

    Reports per-tensor max absolute error, RMSE, and signal-to-quantization-
    noise ratio (SQNR).  Also runs a simulated matmul through each quantized
    layer with random input to check the compute-path error.
    """
    print("\n=== Validation ===\n")

    orig_tensors: dict[str, torch.Tensor] = {}
    with safe_open(original_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            orig_tensors[name] = f.get_tensor(name)

    quant_tensors: dict[str, torch.Tensor] = {}
    with safe_open(quantized_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            quant_tensors[name] = f.get_tensor(name)

    worst_sqnr = float("inf")
    worst_name = ""

    for name, orig in orig_tensors.items():
        if name not in quant_tensors:
            # This tensor was quantized — reconstruct from INT8 + scale.
            if name + "_scale" not in quant_tensors:
                print(f"  WARNING: {name} missing from quantized file")
                continue
            qw = quant_tensors[name.replace(name, name)]
            # The quantized tensor is stored under the SAME name but as int8.
            # If the name exists in quant_tensors it might be int8.
        # Check if the stored tensor is int8 (quantized).
        if name in quant_tensors and quant_tensors[name].dtype == torch.int8:
            scale_name = f"{name}_scale"
            if scale_name not in quant_tensors:
                print(f"  WARNING: scale tensor missing for {name}")
                continue
            qw = quant_tensors[name]
            scale = quant_tensors[scale_name]
            deq = dequantize_per_channel(qw, scale)

            diff = (orig.float() - deq.float())
            max_err = diff.abs().max().item()
            rmse = diff.pow(2).mean().sqrt().item()
            signal_power = orig.float().pow(2).mean().item()
            noise_power = diff.pow(2).mean().item()
            sqnr = 10 * torch.log10(
                torch.tensor(signal_power / max(noise_power, 1e-30))
            ).item()

            if sqnr < worst_sqnr:
                worst_sqnr = sqnr
                worst_name = name

            status = "OK" if sqnr > 30 else "WARN" if sqnr > 20 else "BAD"
            print(
                f"  [{status}] {name}: max_err={max_err:.6f}  "
                f"rmse={rmse:.6f}  sqnr={sqnr:.1f} dB"
            )

            # Simulated matmul validation (random input through this layer).
            if orig.ndim == 2:
                x = torch.randn(4, orig.shape[1])
                y_orig = x @ orig.float().T
                y_quant = x @ deq.T
                matmul_err = (y_orig - y_quant).abs().max().item()
                matmul_rmse = (y_orig - y_quant).pow(2).mean().sqrt().item()
                print(
                    f"         matmul check: max_err={matmul_err:.6f}  "
                    f"rmse={matmul_rmse:.6f}"
                )
        else:
            # Tensor kept as-is — should be bit-identical.
            if name in quant_tensors:
                if not torch.equal(orig, quant_tensors[name]):
                    print(f"  WARNING: {name} kept tensor not bit-identical!")
                # else: silently OK

    print(f"\n  Worst SQNR: {worst_sqnr:.1f} dB  ({worst_name})")
    if worst_sqnr > 30:
        print("  All layers above 30 dB SQNR threshold — quantization quality is good.")
    elif worst_sqnr > 20:
        print("  Some layers between 20-30 dB — acceptable but monitor audio quality.")
    else:
        print("  WARNING: Some layers below 20 dB — may affect audio quality.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="INT8 weight quantization for pocket-tts safetensors"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID (e.g. kyutai/pocket-tts-mini-v0.1-en)",
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
        help="Output path for quantized safetensors",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run tensor-level validation after quantization",
    )
    args = parser.parse_args()

    # ---- Resolve input path ----
    if args.model_id:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: huggingface_hub is required for --model-id.", file=sys.stderr)
            print("  pip install huggingface_hub", file=sys.stderr)
            sys.exit(1)
        print(f"Downloading safetensors from {args.model_id}...")
        # Try common safetensors filenames used by different HF repos.
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
    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            tensors[name] = f.get_tensor(name)

    total_params = sum(t.numel() for t in tensors.values())
    total_original_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    dominant_dtype = max(
        set(t.dtype for t in tensors.values()),
        key=lambda d: sum(t.numel() for t in tensors.values() if t.dtype == d),
    )
    print(f"  {len(tensors)} tensors, {total_params:,} parameters ({total_original_bytes / 1e6:.1f} MB, dtype={dominant_dtype})")

    # ---- Classify & quantize ----
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    quantized_params = 0
    kept_params = 0
    skipped_reasons: dict[str, list[str]] = {}

    print("\nClassifying tensors:\n")
    for name, tensor in tensors.items():
        do_quant, reason = should_quantize(name, tensor)

        if do_quant:
            qw, scale = quantize_per_channel(tensor)
            output[name] = qw
            output[f"{name}_scale"] = scale
            quantized_params += tensor.numel()
            tag = "QUANT"
        else:
            output[name] = tensor
            kept_params += tensor.numel()
            tag = "KEEP "
            skipped_reasons.setdefault(reason, []).append(name)

        shape_str = "x".join(str(d) for d in tensor.shape)
        print(f"  [{tag}] {name:70s}  {shape_str:>20s}  {reason}")

    # ---- Save ----
    print(f"\nSaving to {args.output}...")
    save_file(output, args.output)

    # ---- Statistics ----
    # Compute actual output file size from what was saved.
    output_bytes = args.output.stat().st_size
    quantized_bytes = quantized_params * 1  # INT8 = 1 byte
    scale_bytes = sum(
        t.numel() * t.element_size() for n, t in output.items() if n.endswith("_scale")
    )
    kept_bytes = sum(
        t.numel() * t.element_size() for n, t in output.items()
        if not n.endswith("_scale") and output[n].dtype != torch.int8
    )

    print(f"\n=== Summary ===")
    print(f"  Quantized params:  {quantized_params:>12,}  ({quantized_params / total_params * 100:.1f}%)")
    print(f"  Kept params:       {kept_params:>12,}  ({kept_params / total_params * 100:.1f}%)")
    print(f"  Original size:     {total_original_bytes / 1e6:>12.1f} MB  ({dominant_dtype})")
    print(f"  Quantized size:    {output_bytes / 1e6:>12.1f} MB  (actual file)")
    print(f"  Reduction:         {(1 - output_bytes / total_original_bytes) * 100:>11.1f}%")

    print(f"\n  Layers kept by category:")
    for reason, names in sorted(skipped_reasons.items()):
        n_params = sum(tensors[n].numel() for n in names)
        print(f"    {reason:40s}  {len(names):3d} tensors  {n_params:>10,} params")

    # ---- Validate ----
    if args.validate:
        validate(input_path, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
