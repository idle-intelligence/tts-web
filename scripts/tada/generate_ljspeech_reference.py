#!/usr/bin/env python3
"""Generate TADA audio with ljspeech voice prompt and save intermediates.

Loads the BF16 model in float32, uses pre-computed ljspeech voice prompt,
generates "The quick brown fox jumps over the lazy dog.", and saves:
  - samples/python_bf16_ljspeech_foxdog.wav
  - samples/python_ljspeech_intermediates.bin (stripped acoustics + time_before)
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
# Config
# ---------------------------------------------------------------------------
HF_MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/tada-1b"
TEXT = "The quick brown fox jumps over the lazy dog."
VOICE_NAME = "ljspeech"
OUTPUT_WAV = os.path.join(REPO_ROOT, "samples", "python_bf16_ljspeech_foxdog.wav")
OUTPUT_BIN = os.path.join(REPO_ROOT, "samples", "python_ljspeech_intermediates.bin")
SEED = 42
NOISE_TEMPERATURE = 0.9
TEXT_TEMPERATURE = 0.6
FLOW_STEPS = 10
NUM_TRANSITION_STEPS = 5
NUM_EXTRA_STEPS = 0  # voice-prompted mode uses transition, not extra steps

# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------
import torch

torch.manual_seed(SEED)


def _ensure_llama_tokenizer_local():
    local_path = "/tmp/llama-3.2-1b-tokenizer"
    if not os.path.exists(os.path.join(local_path, "tokenizer.json")):
        print("[gen] Downloading Llama-3.2-1B tokenizer ...", file=sys.stderr)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tok.save_pretrained(local_path)
    return local_path


_LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()

import tada.modules.aligner as _aligner_mod
_aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH

from tada.modules.encoder import EncoderOutput
from tada.modules.tada import TadaForCausalLM, InferenceOptions

# ---------------------------------------------------------------------------
# Monkey-patches for transformers API compat
# ---------------------------------------------------------------------------
from transformers import GenerationMixin
from transformers.cache_utils import DynamicCache

_orig_prepare = GenerationMixin._prepare_generation_config
def _shim_prepare(self, generation_config, *args, **kwargs):
    return _orig_prepare(self, generation_config, **kwargs)
GenerationMixin._prepare_generation_config = _shim_prepare

_orig_cache = GenerationMixin._prepare_cache_for_generation
def _shim_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, *args, **kwargs):
    if "past_key_values" not in model_kwargs or model_kwargs.get("past_key_values") is None:
        model_kwargs["past_key_values"] = DynamicCache()
GenerationMixin._prepare_cache_for_generation = _shim_cache

import tada.modules.tada as _tada_mod
_orig_fos = _tada_mod.TadaForCausalLM.forward_one_step
def _patched_fos(self, *args, acoustic_features=None, **kwargs):
    if acoustic_features is not None:
        model_dtype = next(self.parameters()).dtype
        acoustic_features = acoustic_features.to(dtype=model_dtype)
    return _orig_fos(self, *args, acoustic_features=acoustic_features, **kwargs)
_tada_mod.TadaForCausalLM.forward_one_step = _patched_fos


def main():
    device = "cpu"
    print(f"[gen] Device: {device}")
    print(f"[gen] Text: {TEXT!r}")
    print(f"[gen] Voice: {VOICE_NAME}")
    print(f"[gen] Seed: {SEED}")
    print(f"[gen] noise_temperature={NOISE_TEMPERATURE}, text_temperature={TEXT_TEMPERATURE}, flow_steps={FLOW_STEPS}")

    # -----------------------------------------------------------------------
    # Load model (BF16 weights -> float32)
    # -----------------------------------------------------------------------
    print("[gen] Loading TadaForCausalLM (float32) ...")
    t0 = time.time()
    model = TadaForCausalLM.from_pretrained(HF_MODEL_PATH, torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    print(f"[gen] Model loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Load voice prompt from safetensors
    # -----------------------------------------------------------------------
    voice_path = os.path.join(REPO_ROOT, "voices", f"{VOICE_NAME}.safetensors")
    meta_path = os.path.join(REPO_ROOT, "voices", f"{VOICE_NAME}.json")

    print(f"[gen] Loading voice prompt from {voice_path}")
    from safetensors.torch import load_file as st_load

    tensors = st_load(voice_path, device=device)

    with open(meta_path) as f:
        meta = json.load(f)
    voice_text = meta.get("text", "")
    sample_rate = meta.get("sample_rate", 24000)

    token_values = tensors["token_values"].unsqueeze(0).to(device)       # (1, T, D)
    token_positions = tensors["token_positions"].unsqueeze(0).to(device)  # (1, T)
    token_masks = tensors["token_masks"].unsqueeze(0).to(device)          # (1, L)
    audio_len = tensors["audio_len"].unsqueeze(0).to(device)              # (1,)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_LLAMA_TOKENIZER_PATH)
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
        sample_rate=sample_rate,
    )
    print(f"[gen] Voice: text={voice_text!r}, token_values={token_values.shape}, audio_len={audio_len.item():.0f}")

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    opts = InferenceOptions(
        noise_temperature=NOISE_TEMPERATURE,
        text_temperature=TEXT_TEMPERATURE,
        num_flow_matching_steps=FLOW_STEPS,
    )

    print("[gen] Generating ...")
    t1 = time.time()
    with torch.no_grad():
        output = model.generate(
            prompt=prompt,
            text=TEXT,
            num_transition_steps=NUM_TRANSITION_STEPS,
            num_extra_steps=NUM_EXTRA_STEPS,
            inference_options=opts,
        )
    t_gen = time.time() - t1
    print(f"[gen] Generation: {t_gen:.1f}s")

    if output.audio is None or len(output.audio) == 0 or output.audio[0] is None:
        print("[gen] ERROR: No audio generated!")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Audio stats
    # -----------------------------------------------------------------------
    wav = output.audio[0].cpu().float().squeeze()
    if wav.ndim == 2:
        wav = wav[0]
    wav_np = wav.numpy()
    duration = len(wav_np) / 24000
    peak = np.max(np.abs(wav_np))
    dc_offset = np.mean(wav_np)
    print(f"[gen] Audio duration: {duration:.2f}s")
    print(f"[gen] Peak: {peak:.4f}")
    print(f"[gen] DC offset: {dc_offset:.6f}")

    # -----------------------------------------------------------------------
    # Extract stripped intermediates (replicate generate() stripping logic)
    #
    # In generate():
    #   prompt_acoustic_features is padded with prefix_len zeros and then
    #   trimmed by num_transition_steps. Its final shape[1] = num_prompt_tokens.
    #   encoded = acoustic_features[..., num_prompt_tokens + num_transition_steps - 1:, :]
    #   time_before = outputs.time_before[..., num_prompt_tokens + num_transition_steps - 1:]
    #
    # But GenerationOutput.time_before is already the stripped version.
    # GenerationOutput.acoustic_features is the full denormalized version.
    # We need to strip acoustic_features the same way.
    # -----------------------------------------------------------------------
    full_acoustics = output.acoustic_features  # (1, total_steps, 512) denormalized
    time_before_stripped = output.time_before   # (1, stripped_len) already stripped

    # The stripped time_before length tells us how many frames were generated
    # (after stripping prompt). The acoustic_features needs to be stripped to match.
    # stripped starts at: total - stripped_len
    stripped_len = time_before_stripped.shape[-1]
    encoded = full_acoustics[0, -stripped_len:, :]  # (stripped_len, 512)
    time_before = time_before_stripped[0]           # (stripped_len,)

    num_frames = encoded.shape[0]
    num_times = time_before.shape[0]
    print(f"[gen] Acoustic frames after stripping: {num_frames}")
    print(f"[gen] time_before values: {time_before.tolist()}")

    # -----------------------------------------------------------------------
    # Save intermediates binary
    # Format: u32 num_frames, u32 num_times,
    #         f32[num_frames * 512] acoustics (denormalized),
    #         u32[num_times] time_before
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_BIN)), exist_ok=True)
    acoustics_np = encoded.cpu().float().numpy()
    time_before_np = time_before.cpu().numpy().astype(np.uint32)

    with open(OUTPUT_BIN, "wb") as f:
        f.write(struct.pack("<I", num_frames))
        f.write(struct.pack("<I", num_times))
        f.write(acoustics_np.tobytes())
        f.write(time_before_np.tobytes())
    print(f"[gen] Saved intermediates to {OUTPUT_BIN}")
    print(f"[gen]   acoustics: {num_frames} x 512 f32 = {num_frames * 512 * 4} bytes")
    print(f"[gen]   time_before: {num_times} u32 = {num_times * 4} bytes")

    # -----------------------------------------------------------------------
    # Save WAV
    # -----------------------------------------------------------------------
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_WAV)), exist_ok=True)
    sf.write(OUTPUT_WAV, wav_np, samplerate=24000, subtype="PCM_16")
    print(f"[gen] Saved audio to {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
