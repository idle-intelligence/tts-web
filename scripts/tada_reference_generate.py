#!/Users/tc/Code/idle-intelligence/tts-web/scripts/venv/bin/python3
"""Generate audio using the TADA-1B reference implementation (BF16, full precision).

Used to produce a ground-truth WAV file for quality comparison against our Rust Q4_0 implementation.

Usage:
    ./scripts/tada_reference_generate.py --text "Hello world" --output /tmp/test_tada_python.wav
"""

import argparse
import json
import sys
import time
import os

# ---------------------------------------------------------------------------
# Path setup — allow importing from refs/tada
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REFS_TADA = os.path.join(REPO_ROOT, "refs", "tada")
if REFS_TADA not in sys.path:
    sys.path.insert(0, REFS_TADA)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Monkey-patch AlignerConfig to use locally-cached tokenizer instead of the
# gated meta-llama/Llama-3.2-1B repo (which requires HF access approval).
# The unsloth mirror of the same tokenizer is publicly accessible.
# We pre-download to /tmp/llama-3.2-1b-tokenizer on first use.
# ---------------------------------------------------------------------------
def _ensure_llama_tokenizer_local():
    import os
    local_path = "/tmp/llama-3.2-1b-tokenizer"
    if not os.path.exists(os.path.join(local_path, "tokenizer.json")):
        print("[tada-ref] Downloading Llama-3.2-1B tokenizer from unsloth mirror ...", file=sys.stderr)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tok.save_pretrained(local_path)
        print(f"[tada-ref] Tokenizer saved to {local_path}", file=sys.stderr)
    return local_path

_LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()

# Patch AlignerConfig before anything imports it
import tada.modules.aligner as _aligner_mod
_aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH  # type: ignore[assignment]

from tada.modules.encoder import EncoderOutput
from tada.modules.tada import TadaForCausalLM, InferenceOptions

# ---------------------------------------------------------------------------
# Monkey-patch transformers GenerationMixin API changes
#
# TADA was written against an older transformers where:
#   _prepare_generation_config(gen_config, use_model_defaults: bool) -> (config, kwargs)
#   _prepare_cache_for_generation(gen_config, model_kwargs, assistant_model, batch_size, max_cache_length) -> None
#
# In transformers >= 4.47 these signatures changed.  We shim the old API.
# ---------------------------------------------------------------------------
from transformers import GenerationMixin
from transformers.cache_utils import DynamicCache

_orig_prepare_generation_config = GenerationMixin._prepare_generation_config

def _shim_prepare_generation_config(self, generation_config, *args, **kwargs):
    """Accept old positional bool arg and drop it; delegate to new API."""
    # Old call: _prepare_generation_config(generation_config, True)
    # New call: _prepare_generation_config(generation_config)
    result = _orig_prepare_generation_config(self, generation_config, **kwargs)
    # New API returns (config, kwargs); if old code expects that, we're fine.
    return result

GenerationMixin._prepare_generation_config = _shim_prepare_generation_config  # type: ignore[method-assign]

_orig_prepare_cache = GenerationMixin._prepare_cache_for_generation

def _shim_prepare_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, *args, **kwargs):
    """Old call: (gen_config, model_kwargs, assistant_model, batch_size, max_length)
    New call: (gen_config, model_kwargs, assistant_model, batch_size, max_length, device)
    Just initialise DynamicCache in model_kwargs directly — that's all TADA needs.
    """
    if "past_key_values" not in model_kwargs or model_kwargs.get("past_key_values") is None:
        model_kwargs["past_key_values"] = DynamicCache()

GenerationMixin._prepare_cache_for_generation = _shim_prepare_cache  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Monkey-patch TadaForCausalLM.forward_one_step to auto-cast acoustic_features
# to the model's dtype.  TADA creates acoustic_features as float32 zeros but
# acoustic_proj weight is bfloat16 when loaded with torch_dtype=torch.bfloat16.
# ---------------------------------------------------------------------------
import tada.modules.tada as _tada_mod

_orig_forward_one_step = _tada_mod.TadaForCausalLM.forward_one_step

def _patched_forward_one_step(self, *args, acoustic_features=None, **kwargs):
    if acoustic_features is not None:
        model_dtype = next(self.parameters()).dtype
        acoustic_features = acoustic_features.to(dtype=model_dtype)
    return _orig_forward_one_step(self, *args, acoustic_features=acoustic_features, **kwargs)

_tada_mod.TadaForCausalLM.forward_one_step = _patched_forward_one_step  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TADA-1B reference audio generation")
    parser.add_argument("--text", default="Hello world", help="Text to synthesize")
    parser.add_argument("--output", default="/tmp/test_tada_python.wav", help="Output WAV path")
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Noise temperature for flow matching (default: 0.9, matches Rust)"
    )
    parser.add_argument(
        "--noise-temp", type=float, default=0.6,
        help="Text token sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--flow-steps", type=int, default=10,
        help="Number of flow matching steps (default: 10, matches Rust)"
    )
    parser.add_argument(
        "--model", default="/Users/tc/Code/idle-intelligence/hf/tada-1b",
        help="Path to local TADA-1B checkpoint"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "Name of a pre-computed voice prompt (e.g. 'ljspeech') or a path to a "
            ".safetensors file.  Named voices are looked up in the voices/ directory "
            "next to this script's repo root.  If omitted, zero-shot generation is used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # Device selection: prefer CUDA > CPU > MPS.
    # MPS is disabled by default: the BF16 LLM inference hits MPS dtype assertion
    # errors in MPSNDArrayMatrixMultiplication on current macOS + torch versions.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[tada-ref] Device: {device}", file=sys.stderr)
    print(f"[tada-ref] Text: {args.text!r}", file=sys.stderr)
    print(f"[tada-ref] Output: {args.output}", file=sys.stderr)
    print(f"[tada-ref] Seed: {args.seed}", file=sys.stderr)
    print(f"[tada-ref] noise_temperature={args.temperature}, text_temperature={args.noise_temp}, flow_steps={args.flow_steps}", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------------------
    print("[tada-ref] Loading TadaForCausalLM ...", file=sys.stderr)
    t0 = time.time()

    # Note: from_pretrained() also loads the encoder (tada-codec) from HuggingFace
    # and will download it on first run (~350MB). Subsequent runs use the local cache.
    model = TadaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=False,  # allow downloading tada-codec if not cached
    )
    model.eval()
    model.to(device)

    t_load = time.time() - t0
    print(f"[tada-ref] Model loaded in {t_load:.1f}s", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Build voice prompt
    # ---------------------------------------------------------------------------
    num_transition_steps = 0
    num_extra_steps = 50

    if args.voice is not None:
        # Resolve path: named voice → voices/<name>.safetensors
        voice_path = args.voice
        if not voice_path.endswith(".safetensors"):
            voice_path = os.path.join(REPO_ROOT, "voices", args.voice + ".safetensors")
        meta_path = os.path.splitext(voice_path)[0] + ".json"

        print(f"[tada-ref] Loading voice prompt from {voice_path}", file=sys.stderr)

        try:
            from safetensors.torch import load_file as st_load
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "safetensors"], check=True)
            from safetensors.torch import load_file as st_load

        tensors = st_load(voice_path, device=device)

        # Load companion metadata
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            voice_text = meta.get("text", "")
            sample_rate = meta.get("sample_rate", 24000)
        else:
            voice_text = ""
            sample_rate = 24000

        # Re-add batch dimension (saved without batch dim)
        token_values    = tensors["token_values"].unsqueeze(0).to(device)       # (1, T, D)
        token_positions = tensors["token_positions"].unsqueeze(0).to(device)    # (1, T)
        token_masks     = tensors["token_masks"].unsqueeze(0).to(device)        # (1, L)
        audio_len       = tensors["audio_len"].unsqueeze(0).to(device)          # (1,)

        # Tokenize the voice text to get text_tokens_len (used by generate() to split prompt vs. target)
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
            sample_rate=sample_rate,
        )
        # With a voice prompt, use transition + no extra steps
        num_transition_steps = 5
        num_extra_steps = 0
        print(
            f"[tada-ref] Voice prompt: text={voice_text!r}, "
            f"token_values={token_values.shape}, audio_len={audio_len.item():.0f} samples",
            file=sys.stderr,
        )
    else:
        prompt = EncoderOutput.empty(device)
        print("[tada-ref] Zero-shot mode (no voice prompt)", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Inference options — match what our Rust code uses
    # ---------------------------------------------------------------------------
    opts = InferenceOptions(
        noise_temperature=args.temperature,
        text_temperature=args.noise_temp,
        num_flow_matching_steps=args.flow_steps,
        # Keep other defaults (acoustic_cfg_scale=1.6, time_schedule="logsnr", etc.)
        # These match the model defaults used at inference time
    )

    # ---------------------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------------------
    print("[tada-ref] Generating ...", file=sys.stderr)
    t1 = time.time()

    with torch.no_grad():
        output = model.generate(
            prompt=prompt,
            text=args.text,
            num_transition_steps=num_transition_steps,
            num_extra_steps=num_extra_steps,
            inference_options=opts,
        )

    t_gen = time.time() - t1
    print(f"[tada-ref] Generation finished in {t_gen:.2f}s", file=sys.stderr)

    if output.audio is None or len(output.audio) == 0 or output.audio[0] is None:
        print("[tada-ref] ERROR: No audio generated", file=sys.stderr)
        sys.exit(1)

    wav = output.audio[0].cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (1, samples)

    duration = wav.shape[-1] / 24000
    rtf = t_gen / max(duration, 1e-6)
    print(f"[tada-ref] Audio duration: {duration:.2f}s  |  RTF: {rtf:.3f}x (lower=faster)", file=sys.stderr)

    if output.llm_time is not None:
        llm_ms = float(output.llm_time)
        diff_ms = float(output.diffusion_time)
        print(f"[tada-ref] Avg LLM step: {llm_ms:.1f}ms  |  Avg diffusion step: {diff_ms:.1f}ms", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Save WAV (use soundfile as torchaudio.save requires torchcodec in newer
    # torchaudio versions which is not installed in this venv)
    # ---------------------------------------------------------------------------
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    wav_np = wav.float().squeeze(0).numpy()  # (samples,) or (channels, samples)
    if wav_np.ndim == 2:
        wav_np = wav_np.T  # soundfile expects (samples, channels)
    sf.write(args.output, wav_np, samplerate=24000, subtype="PCM_16")
    print(f"[tada-ref] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
