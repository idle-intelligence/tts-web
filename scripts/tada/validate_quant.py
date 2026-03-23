#!/usr/bin/env python3
"""
Validate INT8 dequantization correctness against the original BF16 model.

Checks:
1. For each INT8 tensor: dequantize and compare to original (max abs error, RMSE, SQNR)
2. Key remapping matches between dequantize.rs and quantize.py output
3. Key completeness after remapping — all expected model prefixes are present
4. Non-quantized tensors are bit-identical to original

Usage:
  python scripts/validate_quant.py \
    --original /tmp/tts_original.safetensors \
    --quantized /Users/tc/Code/idle-intelligence/tts-web/model_int8.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Key remapping — must mirror dequantize.rs exactly
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = [
    "flow.w_s_t",
    "quantizer.vq",
    "quantizer.logvar_proj",
    "learnt_padding",
]

_REPLACEMENTS = [
    (
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    ),
    (
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    ),
    ("flow_lm.backbone.", "flow_lm.transformer."),
    ("flow_lm.flow.", "flow_lm.flow_net."),
    ("mimi.model.", "mimi."),
]

# Expected key prefixes after remapping (model must have at least one key per prefix)
_EXPECTED_PREFIXES = [
    "flow_lm.conditioner.",
    "flow_lm.transformer.",
    "flow_lm.flow_net.",
    "flow_lm.input_linear.",
    "flow_lm.out_norm.",
    "flow_lm.out_eos.",
    "mimi.encoder.",
    "mimi.decoder.",
    "mimi.encoder_transformer.",
    "mimi.decoder_transformer.",
    "mimi.quantizer.",
    "mimi.downsample.",
    "mimi.upsample.",
]

# Scalar/embedding keys (no prefix check needed, check individually)
_EXPECTED_SCALAR_KEYS = [
    "flow_lm.emb_std",
    "flow_lm.emb_mean",
    "flow_lm.bos_emb",
    "flow_lm.speaker_proj_weight",
]


def remap_key(name: str) -> str | None:
    """Python mirror of dequantize.rs remap_key()."""
    for pattern in _SKIP_PATTERNS:
        if pattern in name:
            return None

    for old, new in _REPLACEMENTS:
        if old in name:
            name = name.replace(old, new)

    return name


# ---------------------------------------------------------------------------
# Dequantization — mirrors dequantize.rs exactly
# ---------------------------------------------------------------------------

def dequantize_per_channel(i8_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mirror of dequantize.rs: bf16(i8_val) * scale[channel], per output channel.

    i8_weight: int8 tensor of arbitrary shape, first dim = out_channels
    scale: bfloat16 tensor of shape [out_channels]
    Returns: float32 tensor of same shape as i8_weight
    """
    original_shape = i8_weight.shape
    out_channels = original_shape[0]

    # Cast i8 → bfloat16 first (matches Rust: bf16::from_f32(q as f32) which
    # goes i8 → f32 → bf16; equivalent to i8 → bf16 cast)
    flat = i8_weight.reshape(out_channels, -1).to(torch.bfloat16)
    result = flat * scale.to(torch.bfloat16).unsqueeze(1)
    return result.reshape(original_shape).to(torch.float32)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(orig: torch.Tensor, deq: torch.Tensor) -> dict:
    orig_f = orig.float()
    deq_f = deq.float()
    diff = orig_f - deq_f
    max_err = diff.abs().max().item()
    rmse = diff.pow(2).mean().sqrt().item()
    signal_power = orig_f.pow(2).mean().item()
    noise_power = diff.pow(2).mean().item()
    if noise_power == 0.0:
        # Perfect reconstruction (could be zero weights, or lossless kept tensor)
        sqnr = float("inf")
    else:
        sqnr = 10 * torch.log10(torch.tensor(signal_power / max(noise_power, 1e-30))).item()
    return {"max_err": max_err, "rmse": rmse, "sqnr": sqnr}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate INT8 dequantization")
    parser.add_argument("--original", type=Path, required=True, help="Original BF16 safetensors")
    parser.add_argument("--quantized", type=Path, required=True, help="Quantized INT8 safetensors")
    parser.add_argument(
        "--sqnr-threshold", type=float, default=30.0,
        help="SQNR threshold in dB below which to flag a tensor (default: 30)",
    )
    args = parser.parse_args()

    if not args.original.exists():
        print(f"ERROR: original model not found: {args.original}", file=sys.stderr)
        sys.exit(1)
    if not args.quantized.exists():
        print(f"ERROR: quantized model not found: {args.quantized}", file=sys.stderr)
        sys.exit(1)

    print(f"Original:  {args.original}  ({args.original.stat().st_size / 1e6:.1f} MB)")
    print(f"Quantized: {args.quantized}  ({args.quantized.stat().st_size / 1e6:.1f} MB)")

    # ---- Load both models ----
    print("\nLoading original model...")
    orig_tensors: dict[str, torch.Tensor] = load_file(str(args.original), device="cpu")
    print(f"  {len(orig_tensors)} tensors")

    print("Loading quantized model...")
    quant_tensors: dict[str, torch.Tensor] = load_file(str(args.quantized), device="cpu")
    print(f"  {len(quant_tensors)} tensors (includes _scale tensors)")

    scale_suffix = "_scale"
    int8_names = [n for n, t in quant_tensors.items() if t.dtype == torch.int8]
    kept_names = [n for n, t in quant_tensors.items()
                  if t.dtype != torch.int8 and not n.endswith(scale_suffix)]

    print(f"  {len(int8_names)} INT8 quantized tensors")
    print(f"  {len(kept_names)} kept (non-quantized) tensors")

    # ---- Section 1: Dequantization accuracy ----
    print("\n" + "=" * 72)
    print("SECTION 1: Dequantization Accuracy")
    print("=" * 72)

    worst_sqnr = float("inf")
    worst_name = ""
    bad_tensors = []
    warn_tensors = []

    for name in sorted(int8_names):
        if name not in orig_tensors:
            print(f"  [MISS] {name}: not in original model!")
            bad_tensors.append(name)
            continue

        scale_name = f"{name}{scale_suffix}"
        if scale_name not in quant_tensors:
            print(f"  [ERR ] {name}: scale tensor missing!")
            bad_tensors.append(name)
            continue

        i8_w = quant_tensors[name]
        scale = quant_tensors[scale_name]
        orig = orig_tensors[name]

        deq = dequantize_per_channel(i8_w, scale)
        stats = compute_stats(orig, deq)
        sqnr = stats["sqnr"]

        if sqnr == float("inf"):
            status = "OK  "
        elif sqnr < 20:
            status = "BAD "
            bad_tensors.append(name)
        elif sqnr < args.sqnr_threshold:
            status = "WARN"
            warn_tensors.append(name)
        else:
            status = "OK  "

        finite_sqnr = sqnr if sqnr != float("inf") else 999.0
        if finite_sqnr < worst_sqnr:
            worst_sqnr = finite_sqnr
            worst_name = name

        short_name = name if len(name) <= 62 else "..." + name[-59:]
        print(
            f"  [{status}] {short_name:62s}  max_err={stats['max_err']:.6f}  "
            f"rmse={stats['rmse']:.7f}  sqnr={sqnr:6.1f} dB"
        )

    print(f"\n  Worst SQNR: {worst_sqnr:.1f} dB  ({worst_name})")
    if worst_sqnr >= args.sqnr_threshold:
        print(f"  All INT8 tensors >= {args.sqnr_threshold:.0f} dB SQNR — dequantization accuracy GOOD.")
    elif worst_sqnr >= 20:
        print(f"  Some tensors below {args.sqnr_threshold:.0f} dB SQNR — monitor quality.")
    else:
        print("  WARNING: Some tensors below 20 dB SQNR — likely to cause audio artifacts!")

    # ---- Section 2: Key remapping check ----
    print("\n" + "=" * 72)
    print("SECTION 2: Key Remapping Check")
    print("=" * 72)

    orig_keys = set(orig_tensors.keys())
    remap_errors = 0

    # Build the set of remapped keys expected from the original model
    expected_remapped: dict[str, str] = {}
    for name in orig_keys:
        remapped = remap_key(name)
        if remapped is not None:
            expected_remapped[name] = remapped

    # Build the set of remapped keys produced by the quantized file
    produced_remapped: set[str] = set()
    for name in quant_tensors:
        if name.endswith(scale_suffix):
            continue
        remapped = remap_key(name)
        if remapped is not None:
            produced_remapped.add(remapped)

    missing_from_quant = [
        (orig_name, remapped)
        for orig_name, remapped in expected_remapped.items()
        if remapped not in produced_remapped
    ]

    if missing_from_quant:
        print(f"  MISSING {len(missing_from_quant)} KEYS (in original, absent from quantized after remap):")
        for orig_name, remapped in sorted(missing_from_quant):
            print(f"    {orig_name}  ->  {remapped}")
        remap_errors += len(missing_from_quant)
    else:
        print(f"  All {len(expected_remapped)} original keys accounted for in quantized model.")

    # ---- Section 3: Key completeness ----
    print("\n" + "=" * 72)
    print("SECTION 3: Key Completeness After Remapping")
    print("=" * 72)

    all_remapped = produced_remapped  # already computed above

    completeness_ok = True
    for prefix in _EXPECTED_PREFIXES:
        matching = [k for k in all_remapped if k.startswith(prefix)]
        if matching:
            print(f"  [OK] {prefix}  ({len(matching)} keys)")
        else:
            print(f"  [MISSING] {prefix}  — NO keys found!")
            completeness_ok = False

    for key in _EXPECTED_SCALAR_KEYS:
        if key in all_remapped:
            print(f"  [OK] {key}")
        else:
            print(f"  [MISSING] {key}  — key not found!")
            completeness_ok = False

    # ---- Section 4: Bit-identical check for non-quantized tensors ----
    print("\n" + "=" * 72)
    print("SECTION 4: Non-Quantized Tensor Fidelity")
    print("=" * 72)

    mismatch_kept = []
    checked_kept = 0
    for name in sorted(kept_names):
        if name not in orig_tensors:
            continue
        orig = orig_tensors[name]
        kept = quant_tensors[name]
        checked_kept += 1
        if not torch.equal(orig, kept):
            mismatch_kept.append(name)
            print(f"  [DIFF] {name}: not bit-identical to original!")

    if not mismatch_kept:
        print(f"  All {checked_kept} kept (non-quantized) tensors are bit-identical to original.")
    else:
        print(f"  {len(mismatch_kept)} kept tensors differ from original!")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    ok = True
    if bad_tensors:
        print(f"  FAIL: {len(bad_tensors)} tensors with SQNR < 20 dB: {bad_tensors[:5]}")
        ok = False
    if warn_tensors:
        print(f"  WARN: {len(warn_tensors)} tensors with SQNR < {args.sqnr_threshold:.0f} dB")
    if remap_errors:
        print(f"  FAIL: {remap_errors} key remapping errors")
        ok = False
    if not completeness_ok:
        print("  FAIL: Some expected key prefixes are missing after remapping")
        ok = False
    if mismatch_kept:
        print(f"  FAIL: {len(mismatch_kept)} kept tensors differ from original")
        ok = False

    if ok and not warn_tensors:
        print("  ALL CHECKS PASSED — dequantization is correct.")
        sys.exit(0)
    elif ok:
        print("  PASSED WITH WARNINGS — check SQNR values above.")
        sys.exit(0)
    else:
        print("  VALIDATION FAILED — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
