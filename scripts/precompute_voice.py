#!/Users/tc/Code/idle-intelligence/tts-web/scripts/venv/bin/python3
"""Encode a reference audio clip with the TADA encoder and save the voice prompt as safetensors.

The saved tensors (token_values, token_positions, token_masks, audio_len, sample_rate) can be
loaded later to condition TADA generation on a particular speaker voice.

Usage:
    scripts/venv/bin/python scripts/precompute_voice.py \
        --audio  refs/tada/tada/samples/ljspeech.wav \
        --text   "in being comparatively modern." \
        --output voices/ljspeech.safetensors

The script also writes a companion <output>.json with the text and sample_rate metadata.
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup
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

# ---------------------------------------------------------------------------
# Monkey-patch AlignerConfig to use locally-cached tokenizer (same as
# tada_reference_generate.py — avoids needing access to the gated
# meta-llama/Llama-3.2-1B HuggingFace repo).
# ---------------------------------------------------------------------------
def _ensure_llama_tokenizer_local():
    local_path = "/tmp/llama-3.2-1b-tokenizer"
    if not os.path.exists(os.path.join(local_path, "tokenizer.json")):
        print("[precompute-voice] Downloading Llama-3.2-1B tokenizer from unsloth mirror ...", file=sys.stderr)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tok.save_pretrained(local_path)
        print(f"[precompute-voice] Tokenizer saved to {local_path}", file=sys.stderr)
    return local_path

_LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()

import tada.modules.aligner as _aligner_mod
_aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH  # type: ignore[assignment]

from tada.modules.encoder import Encoder, EncoderOutput


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute TADA voice prompt from reference audio")
    parser.add_argument(
        "--audio",
        default=os.path.join(REPO_ROOT, "refs", "tada", "tada", "samples", "ljspeech.wav"),
        help="Path to reference audio WAV file",
    )
    parser.add_argument(
        "--text",
        default="in being comparatively modern.",
        help="Transcript of the reference audio clip",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "voices", "ljspeech.safetensors"),
        help="Path to write the output .safetensors file",
    )
    parser.add_argument(
        "--model",
        default="HumeAI/tada-codec",
        help="HuggingFace repo or local path for tada-codec encoder weights",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Apply encoder noise sampling (default: off, use deterministic mean for stable voice prompt)",
    )
    return parser.parse_args()


def load_audio(path: str, target_sr: int = 24000) -> tuple[torch.Tensor, int]:
    """Load audio file and resample to target_sr. Returns (waveform, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    # Convert stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)  # (samples,)

    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


def main():
    args = parse_args()

    print(f"[precompute-voice] Audio:  {args.audio}", file=sys.stderr)
    print(f"[precompute-voice] Text:   {args.text!r}", file=sys.stderr)
    print(f"[precompute-voice] Output: {args.output}", file=sys.stderr)

    # Device: CPU to avoid MPS dtype issues with BF16 models
    device = "cpu"
    print(f"[precompute-voice] Device: {device}", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Load encoder
    # ---------------------------------------------------------------------------
    print("[precompute-voice] Loading Encoder ...", file=sys.stderr)
    t0 = time.time()
    encoder = Encoder.from_pretrained(args.model, subfolder="encoder")
    encoder.eval()
    encoder.to(device)
    print(f"[precompute-voice] Encoder loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------------------------
    wav, sr = load_audio(args.audio, target_sr=24000)
    print(f"[precompute-voice] Audio: {len(wav)/sr:.2f}s @ {sr}Hz", file=sys.stderr)

    # Encoder expects float32 (it will cast internally as needed)
    wav_batch = wav.unsqueeze(0).to(device)  # (1, samples)
    audio_length = torch.tensor([wav_batch.shape[-1]], dtype=torch.float32, device=device)

    # ---------------------------------------------------------------------------
    # Encode
    # ---------------------------------------------------------------------------
    print("[precompute-voice] Encoding ...", file=sys.stderr)
    t1 = time.time()
    with torch.no_grad():
        enc_out: EncoderOutput = encoder(
            audio=wav_batch,
            text=[args.text],
            audio_length=audio_length,
            sample_rate=sr,
            sample=args.sample,
        )
    print(f"[precompute-voice] Encoded in {time.time() - t1:.1f}s", file=sys.stderr)

    print(f"[precompute-voice] token_values:    {enc_out.token_values.shape}", file=sys.stderr)
    print(f"[precompute-voice] token_positions: {enc_out.token_positions.shape}", file=sys.stderr)
    print(f"[precompute-voice] token_masks:     {enc_out.token_masks.shape}", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Save safetensors
    # ---------------------------------------------------------------------------
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("[precompute-voice] safetensors not found, installing ...", file=sys.stderr)
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "safetensors"],
            check=True,
            capture_output=True,
        )
        from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    tensors = {
        "token_values":    enc_out.token_values.squeeze(0).float().contiguous(),   # (T, D)
        "token_positions": enc_out.token_positions.squeeze(0).contiguous(),         # (T,)
        "token_masks":     enc_out.token_masks.squeeze(0).contiguous(),             # (L,) encoded seq len
        "audio_len":       enc_out.audio_len.squeeze(0).float().contiguous(),       # scalar
    }
    save_file(tensors, args.output)
    print(f"[precompute-voice] Saved safetensors → {args.output}", file=sys.stderr)

    # ---------------------------------------------------------------------------
    # Save companion JSON
    # ---------------------------------------------------------------------------
    meta_path = os.path.splitext(args.output)[0] + ".json"
    meta = {
        "text": args.text,
        "sample_rate": sr,
        "audio_file": os.path.abspath(args.audio),
        "token_values_shape": list(enc_out.token_values.squeeze(0).shape),
        "token_positions_shape": list(enc_out.token_positions.squeeze(0).shape),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[precompute-voice] Saved metadata   → {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
