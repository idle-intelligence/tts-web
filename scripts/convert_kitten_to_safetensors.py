#!/usr/bin/env python3
"""Convert KittenTTS nano ONNX model weights to safetensors format.

Source: kitten_tts_nano_v0_8.onnx
Output:
  kitten-nano.safetensors  -- all model weights
  kitten-voices.safetensors -- voice style vectors (friendly names)
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

HF_DIR = Path("/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8")
ONNX_PATH = HF_DIR / "kitten_tts_nano_v0_8.onnx"
VOICES_PATH = HF_DIR / "voices.npz"
CONFIG_PATH = HF_DIR / "config.json"
OUT_MODEL = HF_DIR / "kitten-nano.safetensors"
OUT_VOICES = HF_DIR / "kitten-voices.safetensors"


# ---------------------------------------------------------------------------
# LSTM tensors: ONNX uses anonymous initializer names. Map them to logical
# names based on the LSTM node's position in the graph.
# ONNX LSTM input order: X, W, R, B, seq_lens, h0, c0
# ---------------------------------------------------------------------------
LSTM_TENSOR_MAP = {
    # /text_encoder/lstm/LSTM  (predictor.text_encoder base lstm)
    "onnx::LSTM_5651": "predictor.text_encoder.lstm.B",
    "onnx::LSTM_5652": "predictor.text_encoder.lstm.W",
    "onnx::LSTM_5653": "predictor.text_encoder.lstm.R",
    # /text_encoder/lstms.0/LSTM
    "onnx::LSTM_5871": "predictor.text_encoder.lstms.0.B",
    "onnx::LSTM_5872": "predictor.text_encoder.lstms.0.W",
    "onnx::LSTM_5873": "predictor.text_encoder.lstms.0.R",
    # /text_encoder/lstms.2/LSTM
    "onnx::LSTM_5921": "predictor.text_encoder.lstms.2.B",
    "onnx::LSTM_5922": "predictor.text_encoder.lstms.2.W",
    "onnx::LSTM_5923": "predictor.text_encoder.lstms.2.R",
    # /lstm/LSTM  (predictor duration lstm)
    "onnx::LSTM_5970": "predictor.lstm.B",
    "onnx::LSTM_5971": "predictor.lstm.W",
    "onnx::LSTM_5972": "predictor.lstm.R",
    # /shared/LSTM  (shared lstm)
    "onnx::LSTM_6019": "shared.lstm.B",
    "onnx::LSTM_6020": "shared.lstm.W",
    "onnx::LSTM_6021": "shared.lstm.R",
}

# ---------------------------------------------------------------------------
# MatMul tensors: ONNX anonymous weights for linear layers that have no
# explicit parameter name. Identified by tracing which MatMul node uses them.
# ---------------------------------------------------------------------------
MATMUL_TENSOR_MAP = {
    # /bert/encoder/embedding_hidden_mapping_in/MatMul  (128, 768)
    "onnx::MatMul_5661": "bert.encoder.embedding_hidden_mapping_in.weight",
    # attention Q/K/V/dense weights  (768, 768)
    "onnx::MatMul_5662": "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
    "onnx::MatMul_5665": "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
    "onnx::MatMul_5668": "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
    "onnx::MatMul_5672": "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
    # FFN weights
    "onnx::MatMul_5673": "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",       # (768, 2048)
    "onnx::MatMul_5674": "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight", # (2048, 768)
    # /bert_encoder/MatMul  — projects BERT hidden → model hidden  (768, 128)
    "onnx::MatMul_5818": "bert_encoder.weight",
    # duration projection linear weight  (128, 50)
    "onnx::MatMul_5973": "predictor.duration_proj.linear_layer.weight",
    # F0 source linear weight  (9, 1)
    "onnx::MatMul_6116": "decoder.generator.m_source.l_linear.weight",
}

# Initializer names to skip (intermediate constants, optimizer artefacts, etc.)
# We skip anything that isn't a model parameter.
SKIP_PREFIXES = (
    "/",             # computed constants stored as initializers (e.g. /decoder/generator/...)
    "const_transpose_optimizer",
    "onnx::Range_",
    "onnx::Reshape_",
    "onnx::Expand_",
    "onnx::Slice_",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rename_kmodel(name: str) -> str:
    """Strip the top-level `kmodel.` prefix."""
    if name.startswith("kmodel."):
        return name[len("kmodel."):]
    return name


def convert_model(onnx_sha256: str) -> dict[str, np.ndarray]:
    print(f"Loading {ONNX_PATH} …")
    model = onnx.load(str(ONNX_PATH))

    tensors: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for init in model.graph.initializer:
        name = init.name
        arr = numpy_helper.to_array(init).copy()

        # 1. LSTM anonymous weights
        if name in LSTM_TENSOR_MAP:
            key = LSTM_TENSOR_MAP[name]
            tensors[key] = arr
            continue

        # 2. MatMul anonymous weights
        # ONNX MatMul stores weights as [in_features, out_features].
        # Candle Linear expects [out_features, in_features] (computes x @ W^T).
        # Transpose 2D weight matrices so they match candle's convention.
        if name in MATMUL_TENSOR_MAP:
            key = MATMUL_TENSOR_MAP[name]
            if arr.ndim == 2:
                arr = arr.T
            tensors[key] = arr
            continue

        # 3. Skip computed/optimizer constants
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            skipped.append(name)
            continue

        # 4. Normal named parameters — strip kmodel. prefix
        key = rename_kmodel(name)
        tensors[key] = arr

    print(f"  Skipped {len(skipped)} computed-constant initializers.")
    return tensors


def convert_voices() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    print(f"Loading {CONFIG_PATH} …")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # voice_aliases: {"Bella": "expr-voice-2-f", ...}
    aliases: dict[str, str] = config["voice_aliases"]
    # Invert: internal_name → friendly_name (lowercase)
    internal_to_friendly: dict[str, str] = {
        v: k.lower() for k, v in aliases.items()
    }

    print(f"Loading {VOICES_PATH} …")
    npz = np.load(str(VOICES_PATH))

    voices: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}  # internal → friendly (for metadata)

    for internal_name in npz.files:
        friendly = internal_to_friendly.get(internal_name)
        if friendly is None:
            print(f"  WARNING: no alias for voice '{internal_name}', using raw name")
            friendly = internal_name
        arr = npz[internal_name].copy()
        voices[friendly] = arr
        mapping[internal_name] = friendly
        print(f"  {internal_name:20s} → {friendly:10s}  shape={arr.shape}  dtype={arr.dtype}")

    return voices, mapping


def main():
    # --- SHA256 of source ONNX ---
    print("Computing SHA256 of ONNX file …")
    onnx_sha256 = sha256_file(ONNX_PATH)
    print(f"  SHA256: {onnx_sha256}")

    # --- Convert model weights ---
    tensors = convert_model(onnx_sha256)

    print(f"\nSaving {OUT_MODEL} …")
    metadata = {
        "__kitten_onnx_sha256__": onnx_sha256,
        "source": str(ONNX_PATH.name),
        "version": "0.8",
    }
    save_file(tensors, str(OUT_MODEL), metadata=metadata)
    print(f"  Saved {len(tensors)} tensors.")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("MODEL WEIGHT SUMMARY")
    print("=" * 80)
    total_params = 0
    for key in sorted(tensors.keys()):
        arr = tensors[key]
        n = arr.size
        total_params += n
        print(f"  {key:<80s}  {str(arr.shape):<20s}  {arr.dtype}")
    print(f"\nTotal tensors: {len(tensors)}")
    print(f"Total parameters: {total_params:,}")

    # --- Convert voices ---
    print("\n" + "=" * 80)
    print("VOICE CONVERSION")
    print("=" * 80)
    voices, mapping = convert_voices()

    print(f"\nSaving {OUT_VOICES} …")
    # Store internal→friendly mapping as metadata
    voices_metadata = {
        "__kitten_onnx_sha256__": onnx_sha256,
        "__voice_mapping__": json.dumps(mapping),
    }
    save_file(voices, str(OUT_VOICES), metadata=voices_metadata)
    print(f"  Saved {len(voices)} voices.")

    print("\nDone.")
    print(f"  Model: {OUT_MODEL}  ({OUT_MODEL.stat().st_size / 1e6:.1f} MB)")
    print(f"  Voices: {OUT_VOICES}  ({OUT_VOICES.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
