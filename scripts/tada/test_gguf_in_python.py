#!/Users/tc/Code/idle-intelligence/tts-web/scripts/venv/bin/python3
"""Test GGUF quantized TADA-1B model by loading in Python.

Parses the GGUF file, dequantizes Q4_0 tensors back to F32, builds a PyTorch
state_dict, loads into the TADA reference model, and generates audio.

This validates that:
1. GGUF tensor names match HF format
2. Q4_0 quantization/dequantization is correct
3. Weight norm fusion was done properly
4. The model produces reasonable audio from GGUF weights

Usage:
    scripts/venv/bin/python3 scripts/test_gguf_in_python.py
"""

import json
import os
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REFS_TADA = os.path.join(REPO_ROOT, "refs", "tada")
if REFS_TADA not in sys.path:
    sys.path.insert(0, REFS_TADA)

# ---------------------------------------------------------------------------
# GGUF constants (must match quantize_tada.py)
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1
GGUF_TENSOR_Q4_0 = 2

Q4_0_BLOCK_SIZE = 32
Q4_0_BYTES_PER_BLOCK = 18  # 2 (f16 scale) + 16 (packed nibbles)

# GGUF metadata value types
GGUF_META_TYPES = {
    0: ("uint8", "<B", 1),
    1: ("int8", "<b", 1),
    2: ("uint16", "<H", 2),
    3: ("int16", "<h", 2),
    4: ("uint32", "<I", 4),
    5: ("int32", "<i", 4),
    6: ("float32", "<f", 4),
    7: ("bool", "<?", 1),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("uint64", "<Q", 8),
    11: ("int64", "<q", 8),
    12: ("float64", "<d", 8),
}


# ---------------------------------------------------------------------------
# GGUF parser
# ---------------------------------------------------------------------------

def read_gguf_string(f):
    """Read a GGUF string: uint64 length + utf8 bytes."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_gguf_metadata_value(f):
    """Read a single GGUF metadata value, return (type_id, value)."""
    type_id = struct.unpack('<I', f.read(4))[0]
    if type_id == 8:  # string
        return type_id, read_gguf_string(f)
    elif type_id == 9:  # array
        elem_type = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        values = []
        for _ in range(n):
            if elem_type == 8:
                values.append(read_gguf_string(f))
            else:
                fmt = GGUF_META_TYPES[elem_type][1]
                size = GGUF_META_TYPES[elem_type][2]
                values.append(struct.unpack(fmt, f.read(size))[0])
        return type_id, values
    else:
        fmt = GGUF_META_TYPES[type_id][1]
        size = GGUF_META_TYPES[type_id][2]
        return type_id, struct.unpack(fmt, f.read(size))[0]


def align_offset(offset, alignment=GGUF_ALIGNMENT):
    return (offset + alignment - 1) // alignment * alignment


def parse_gguf(path):
    """Parse GGUF file. Returns (metadata_dict, tensor_infos, data_start_offset).

    tensor_infos: list of (name, shape, type_id, data_offset_relative)
    """
    with open(path, 'rb') as f:
        # Header
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Bad magic: {magic:#x}"
        version = struct.unpack('<I', f.read(4))[0]
        assert version == GGUF_VERSION, f"Unsupported version: {version}"
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_metadata = struct.unpack('<Q', f.read(8))[0]

        # Metadata
        metadata = {}
        for _ in range(n_metadata):
            key = read_gguf_string(f)
            type_id, value = read_gguf_metadata_value(f)
            metadata[key] = value

        # Tensor infos
        tensor_infos = []
        for _ in range(n_tensors):
            name = read_gguf_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            # Dims are stored reversed in GGUF (candle reverses on load)
            dims_reversed = []
            for _ in range(n_dims):
                dims_reversed.append(struct.unpack('<Q', f.read(8))[0])
            # Reverse back to get original shape
            shape = list(reversed(dims_reversed))
            tensor_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((name, shape, tensor_type, offset))

        # Data starts at aligned position after all tensor infos
        data_start = align_offset(f.tell())

    return metadata, tensor_infos, data_start


# ---------------------------------------------------------------------------
# Q4_0 dequantization
# ---------------------------------------------------------------------------

def dequantize_q4_0(raw_bytes, n_elements):
    """Dequantize Q4_0 bytes to F32 numpy array."""
    n_blocks = n_elements // Q4_0_BLOCK_SIZE
    assert len(raw_bytes) == n_blocks * Q4_0_BYTES_PER_BLOCK, \
        f"Expected {n_blocks * Q4_0_BYTES_PER_BLOCK} bytes, got {len(raw_bytes)}"

    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, Q4_0_BYTES_PER_BLOCK)

    # Extract scales (first 2 bytes of each block, as f16)
    scales_raw = data[:, :2].copy()  # copy to ensure contiguous
    scales = scales_raw.view(np.float16).astype(np.float32).flatten()  # [n_blocks]

    # Extract packed nibbles (bytes 2-17, 16 bytes per block)
    packed = data[:, 2:]  # [n_blocks, 16]

    # Unpack nibbles: low nibble first, then high nibble
    low_nibbles = (packed & 0x0F).astype(np.float32)   # [n_blocks, 16]
    high_nibbles = (packed >> 4).astype(np.float32)     # [n_blocks, 16]

    # First half = low nibbles (elements 0-15), second half = high nibbles (elements 16-31)
    values = np.empty((n_blocks, Q4_0_BLOCK_SIZE), dtype=np.float32)
    values[:, :16] = low_nibbles
    values[:, 16:] = high_nibbles

    # Dequantize: value = (nibble - 8) * scale
    values = (values - 8.0) * scales[:, np.newaxis]

    return values.reshape(-1)[:n_elements]


# ---------------------------------------------------------------------------
# Load GGUF into PyTorch state dict
# ---------------------------------------------------------------------------

def gguf_to_state_dict(gguf_path):
    """Load GGUF file and return a PyTorch state dict (all F32)."""
    import torch

    print(f"Parsing GGUF: {gguf_path}")
    metadata, tensor_infos, data_start = parse_gguf(gguf_path)

    print(f"  Metadata: {metadata}")
    print(f"  {len(tensor_infos)} tensors, data starts at offset {data_start}")

    state_dict = {}
    n_q4 = 0
    n_f32 = 0
    file_size = os.path.getsize(gguf_path)

    with open(gguf_path, 'rb') as f:
        for i, (name, shape, tensor_type, offset) in enumerate(tensor_infos):
            n_elements = 1
            for d in shape:
                n_elements *= d

            abs_offset = data_start + offset

            if tensor_type == GGUF_TENSOR_Q4_0:
                n_bytes = (n_elements // Q4_0_BLOCK_SIZE) * Q4_0_BYTES_PER_BLOCK
                f.seek(abs_offset)
                raw = f.read(n_bytes)
                values = dequantize_q4_0(raw, n_elements)
                n_q4 += 1
            elif tensor_type == GGUF_TENSOR_F32:
                n_bytes = n_elements * 4
                f.seek(abs_offset)
                raw = f.read(n_bytes)
                values = np.frombuffer(raw, dtype=np.float32).copy()
                n_f32 += 1
            elif tensor_type == GGUF_TENSOR_F16:
                n_bytes = n_elements * 2
                f.seek(abs_offset)
                raw = f.read(n_bytes)
                values = np.frombuffer(raw, dtype=np.float16).astype(np.float32).copy()
                n_f32 += 1
            else:
                print(f"  WARNING: Unknown tensor type {tensor_type} for {name}, skipping")
                continue

            tensor = torch.from_numpy(values.reshape(shape))
            state_dict[name] = tensor

            if (i + 1) % 50 == 0 or i == len(tensor_infos) - 1:
                print(f"  [{i+1}/{len(tensor_infos)}] loaded {name} {shape} "
                      f"({'Q4_0' if tensor_type == GGUF_TENSOR_Q4_0 else 'F32'})")

    print(f"\nLoaded {len(state_dict)} tensors ({n_q4} Q4_0, {n_f32} F32)")
    return state_dict


# ---------------------------------------------------------------------------
# Validate tensor names against HF model
# ---------------------------------------------------------------------------

def validate_names(state_dict, hf_model_path):
    """Compare GGUF tensor names against HF safetensors tensor names."""
    import json as _json

    sf_path = os.path.join(hf_model_path, "model.safetensors")
    if not os.path.exists(sf_path):
        print(f"WARNING: {sf_path} not found, skipping name validation")
        return True

    # Parse safetensors header to get tensor names
    with open(sf_path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = _json.loads(f.read(hdr_len))

    hf_names = set(k for k in hdr.keys() if k != '__metadata__')

    # Names that quantize_tada.py skips
    skipped_patterns = ['_precomputed_mask', '.rope_freqs']
    # Names consumed by weight_norm fusion (parametrizations.weight.original0/1)
    wn_patterns = ['.parametrizations.weight.original0', '.parametrizations.weight.original1']

    expected_hf_names = set()
    wn_bases = set()
    for name in hf_names:
        if any(p in name for p in skipped_patterns):
            continue
        if any(p in name for p in wn_patterns):
            # Weight norm pair -> fused as .weight
            base = name.replace('.parametrizations.weight.original0', '.weight')
            base = base.replace('.parametrizations.weight.original1', '.weight')
            wn_bases.add(base)
            continue
        expected_hf_names.add(name)

    expected_hf_names.update(wn_bases)
    gguf_names = set(state_dict.keys())

    missing = expected_hf_names - gguf_names
    extra = gguf_names - expected_hf_names

    if missing:
        print(f"\nMISSING from GGUF ({len(missing)} tensors):")
        for n in sorted(missing)[:20]:
            print(f"  - {n}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if extra:
        print(f"\nEXTRA in GGUF ({len(extra)} tensors):")
        for n in sorted(extra)[:20]:
            print(f"  - {n}")
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more")

    if not missing and not extra:
        print(f"\nName validation PASSED: all {len(gguf_names)} tensor names match")
        return True
    else:
        print(f"\nName validation FAILED: {len(missing)} missing, {len(extra)} extra")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    GGUF_PATH = os.environ.get("GGUF_PATH", "/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-q4_0.gguf")
    HF_MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/tada-1b"
    VOICE_PATH = os.path.join(REPO_ROOT, "voices", "ljspeech.safetensors")
    OUTPUT_PATH = os.path.join(REPO_ROOT, "samples", "python_from_gguf_ljspeech.wav")
    TEXT = "Hello world"
    SEED = 42

    # ------------------------------------------------------------------
    # Step 1: Parse GGUF and build state dict
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Parse GGUF and dequantize")
    print("=" * 60)
    t0 = time.time()
    state_dict = gguf_to_state_dict(GGUF_PATH)
    print(f"Parsed in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 2: Validate tensor names
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Validate tensor names against HF model")
    print("=" * 60)
    names_ok = validate_names(state_dict, HF_MODEL_PATH)

    if not names_ok:
        print("\nStopping: fix tensor name mismatches before running inference.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Load into TADA model and generate audio
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Load GGUF weights into TADA model")
    print("=" * 60)

    import torch

    torch.manual_seed(SEED)

    # Monkey-patch tokenizer (avoid gated Llama repo)
    def _ensure_llama_tokenizer_local():
        local_path = "/tmp/llama-3.2-1b-tokenizer"
        if not os.path.exists(os.path.join(local_path, "tokenizer.json")):
            print("[gguf-test] Downloading Llama-3.2-1B tokenizer ...", file=sys.stderr)
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
            tok.save_pretrained(local_path)
        return local_path

    _LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()

    import tada.modules.aligner as _aligner_mod
    _aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH

    # Monkey-patch transformers API (same as tada_reference_generate.py)
    from transformers import GenerationMixin
    from transformers.cache_utils import DynamicCache

    _orig_prepare_generation_config = GenerationMixin._prepare_generation_config

    def _shim_prepare_generation_config(self, generation_config, *args, **kwargs):
        result = _orig_prepare_generation_config(self, generation_config, **kwargs)
        return result

    GenerationMixin._prepare_generation_config = _shim_prepare_generation_config

    _orig_prepare_cache = GenerationMixin._prepare_cache_for_generation

    def _shim_prepare_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, *args, **kwargs):
        if "past_key_values" not in model_kwargs or model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = DynamicCache()

    GenerationMixin._prepare_cache_for_generation = _shim_prepare_cache

    # Monkey-patch forward_one_step dtype
    import tada.modules.tada as _tada_mod

    _orig_forward_one_step = _tada_mod.TadaForCausalLM.forward_one_step

    def _patched_forward_one_step(self, *args, acoustic_features=None, **kwargs):
        if acoustic_features is not None:
            model_dtype = next(self.parameters()).dtype
            acoustic_features = acoustic_features.to(dtype=model_dtype)
        return _orig_forward_one_step(self, *args, acoustic_features=acoustic_features, **kwargs)

    _tada_mod.TadaForCausalLM.forward_one_step = _patched_forward_one_step

    # Load model architecture (with original weights, then replace)
    from tada.modules.tada import TadaForCausalLM, InferenceOptions
    from tada.modules.encoder import EncoderOutput

    print("[gguf-test] Loading TadaForCausalLM architecture ...", file=sys.stderr)
    t0 = time.time()
    model = TadaForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=torch.float32,  # F32 since our dequantized weights are F32
        local_files_only=False,
    )

    # Remove weight_norm parametrizations so we can load fused weights directly
    print("[gguf-test] Removing weight_norm parametrizations ...", file=sys.stderr)
    removed_wn = 0
    for module_name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
            removed_wn += 1
    print(f"  Removed {removed_wn} weight_norm parametrizations", file=sys.stderr)

    # Replace weights with GGUF-dequantized weights
    print("[gguf-test] Replacing weights with GGUF-dequantized values ...", file=sys.stderr)
    model_state = model.state_dict()

    replaced = 0
    mismatched_shape = 0
    missing_in_model = 0

    for name, gguf_tensor in state_dict.items():
        if name in model_state:
            if model_state[name].shape == gguf_tensor.shape:
                model_state[name].copy_(gguf_tensor)
                replaced += 1
            else:
                print(f"  SHAPE MISMATCH: {name}: model={model_state[name].shape} vs gguf={gguf_tensor.shape}")
                mismatched_shape += 1
        else:
            missing_in_model += 1

    if missing_in_model > 0 or mismatched_shape > 0:
        print(f"  Direct copy: {replaced} replaced, {mismatched_shape} shape mismatches, "
              f"{missing_in_model} not in model state dict")
        result = model.load_state_dict(state_dict, strict=False)
        print(f"  load_state_dict: missing_keys={len(result.missing_keys)}, "
              f"unexpected_keys={len(result.unexpected_keys)}")
        if result.missing_keys:
            print("  Missing keys (first 10):")
            for k in sorted(result.missing_keys)[:10]:
                print(f"    - {k}")
        if result.unexpected_keys:
            print("  Unexpected keys (first 10):")
            for k in sorted(result.unexpected_keys)[:10]:
                print(f"    - {k}")
    else:
        print(f"  Replaced all {replaced} tensors successfully")

    model.eval()

    t_load = time.time() - t0
    print(f"[gguf-test] Model loaded in {t_load:.1f}s", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 4: Build voice prompt and generate
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: Generate audio")
    print("=" * 60)

    device = "cpu"

    # Load voice prompt
    from safetensors.torch import load_file as st_load
    tensors = st_load(VOICE_PATH, device=device)

    voice_meta_path = os.path.splitext(VOICE_PATH)[0] + ".json"
    with open(voice_meta_path) as f:
        voice_meta = json.load(f)
    voice_text = voice_meta.get("text", "")

    token_values = tensors["token_values"].unsqueeze(0).to(device)
    token_positions = tensors["token_positions"].unsqueeze(0).to(device)
    token_masks = tensors["token_masks"].unsqueeze(0).to(device)
    audio_len = tensors["audio_len"].unsqueeze(0).to(device)

    from tada.modules.aligner import AlignerConfig
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(AlignerConfig.tokenizer_name)
    text_token_ids = tokenizer.encode(voice_text, add_special_tokens=False)
    text_tokens_len = torch.tensor([len(text_token_ids)], dtype=torch.long, device=device)
    text_tokens = torch.tensor([text_token_ids], dtype=torch.long, device=device)

    prompt = EncoderOutput(
        audio=torch.zeros(1, int(audio_len.item()), device=device),
        audio_len=audio_len,
        text=[voice_text],
        text_tokens=text_tokens,
        text_tokens_len=text_tokens_len,
        token_positions=token_positions,
        token_values=token_values,
        token_masks=token_masks,
        sample_rate=voice_meta.get("sample_rate", 24000),
    )

    opts = InferenceOptions(
        noise_temperature=0.9,
        text_temperature=0.6,
        num_flow_matching_steps=10,
    )

    print(f"[gguf-test] Generating: {TEXT!r}", file=sys.stderr)
    print(f"[gguf-test] Voice: ljspeech, prompt text: {voice_text!r}", file=sys.stderr)
    t1 = time.time()

    with torch.no_grad():
        output = model.generate(
            prompt=prompt,
            text=TEXT,
            num_transition_steps=5,
            num_extra_steps=0,
            inference_options=opts,
        )

    t_gen = time.time() - t1
    print(f"[gguf-test] Generation finished in {t_gen:.2f}s", file=sys.stderr)

    if output.audio is None or len(output.audio) == 0 or output.audio[0] is None:
        print("[gguf-test] ERROR: No audio generated!", file=sys.stderr)
        sys.exit(1)

    wav = output.audio[0].cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    duration = wav.shape[-1] / 24000
    rtf = t_gen / max(duration, 1e-6)
    print(f"[gguf-test] Audio: {duration:.2f}s | RTF: {rtf:.3f}x", file=sys.stderr)

    # Save WAV
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    wav_np = wav.float().squeeze(0).numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.T
    sf.write(OUTPUT_PATH, wav_np, samplerate=24000, subtype="PCM_16")
    print(f"[gguf-test] Saved to {OUTPUT_PATH}", file=sys.stderr)
    print(f"\nDone! Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
