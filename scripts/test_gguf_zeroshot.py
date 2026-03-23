#!/usr/bin/env python3
"""Quick test: load GGUF weights into Python TADA, run zero-shot generation.

This validates GGUF weights work with the correct generation flow.
"""
import json
import os
import struct
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REFS_TADA = os.path.join(REPO_ROOT, "refs", "tada")
if REFS_TADA not in sys.path:
    sys.path.insert(0, REFS_TADA)

# Reuse the GGUF loader from test_gguf_in_python.py
sys.path.insert(0, SCRIPT_DIR)
from test_gguf_in_python import gguf_to_state_dict

GGUF_PATH = os.environ.get("GGUF_PATH", "/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-f16.gguf")
HF_MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/tada-1b"
TEXT = "The quick brown fox jumps over the lazy dog."
OUTPUT_PATH = os.path.join(REPO_ROOT, "samples", "python_gguf_zeroshot.wav")
FLOW_STEPS = 10
SEED = 42

def main():
    import torch
    torch.manual_seed(SEED)

    # Step 1: Parse GGUF
    print(f"Loading GGUF: {GGUF_PATH}")
    state_dict = gguf_to_state_dict(GGUF_PATH)

    # Step 2: Load model and replace weights
    def _ensure_llama_tokenizer_local():
        local_path = "/tmp/llama-3.2-1b-tokenizer"
        if not os.path.exists(os.path.join(local_path, "tokenizer.json")):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
            tok.save_pretrained(local_path)
        return local_path

    _LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()
    import tada.modules.aligner as _aligner_mod
    _aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH

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

    from tada.modules.tada import TadaForCausalLM, InferenceOptions
    from tada.modules.encoder import EncoderOutput

    print("Loading model...")
    model = TadaForCausalLM.from_pretrained(HF_MODEL_PATH, torch_dtype=torch.float32)

    # Remove weight_norm
    removed = 0
    for _, module in model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
            removed += 1
    print(f"Removed {removed} weight_norm parametrizations")

    # Replace weights
    model_state = model.state_dict()
    replaced = 0
    for name, tensor in state_dict.items():
        if name in model_state and model_state[name].shape == tensor.shape:
            model_state[name].copy_(tensor)
            replaced += 1
    print(f"Replaced {replaced}/{len(state_dict)} tensors")
    model.eval()

    # Step 3: Generate zero-shot
    prompt = EncoderOutput.empty("cpu")
    opts = InferenceOptions(
        noise_temperature=0.9,
        text_temperature=0.6,
        num_flow_matching_steps=FLOW_STEPS,
    )

    print(f"Generating zero-shot: {TEXT!r}")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            prompt=prompt,
            text=TEXT,
            num_transition_steps=0,
            num_extra_steps=50,
            inference_options=opts,
        )
    elapsed = time.time() - t0
    print(f"Generation: {elapsed:.1f}s")

    if output.audio is None or len(output.audio) == 0 or output.audio[0] is None:
        print("ERROR: No audio generated!")
        sys.exit(1)

    wav = output.audio[0].cpu().float().squeeze(0).numpy()
    if wav.ndim == 2:
        wav = wav.T
    duration = len(wav) / 24000
    print(f"Audio: {duration:.2f}s, peak={np.max(np.abs(wav)):.4f}, "
          f"dc={np.mean(wav):.6f}, flatness={np.mean(np.abs(np.diff(wav)) < 1e-6):.4f}")

    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    sf.write(OUTPUT_PATH, wav, samplerate=24000, subtype="PCM_16")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
