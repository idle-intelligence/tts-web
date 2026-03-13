"""Decode Rust acoustic intermediates using Python's TADA decoder.

Loads samples/rust_intermediates.bin (denormalized acoustics + time_before),
runs through HumeAI/tada-codec decoder, saves WAV.
"""

import struct
import sys
import os
import numpy as np
import torch
import soundfile as sf

# Add refs/tada to path so we can import the tada package
sys.path.insert(0, "/Users/tc/Code/idle-intelligence/tts-web/refs/tada")

from tada.modules.decoder import Decoder, DecoderConfig
from safetensors.torch import load_file
import glob as globmod


def main():
    bin_path = "/Users/tc/Code/idle-intelligence/tts-web/samples/rust_intermediates.bin"
    out_path = "/Users/tc/Code/idle-intelligence/tts-web/samples/python_decoded_rust_intermediates.wav"

    # 1. Load binary intermediates
    ACOUSTIC_DIM = 512
    with open(bin_path, "rb") as f:
        num_frames = struct.unpack("<I", f.read(4))[0]
        num_times = struct.unpack("<I", f.read(4))[0]
        print(f"num_frames={num_frames}, num_times={num_times}")

        acoustic_data = np.frombuffer(f.read(num_frames * ACOUSTIC_DIM * 4), dtype=np.float32).copy()
        acoustic = acoustic_data.reshape(num_frames, ACOUSTIC_DIM)
        print(f"acoustic shape: {acoustic.shape}, range: [{acoustic.min():.4f}, {acoustic.max():.4f}]")
        print(f"acoustic mean: {acoustic.mean():.4f}, std: {acoustic.std():.4f}")

        time_before = np.frombuffer(f.read(num_times * 4), dtype=np.uint32).copy().astype(np.int64)
        print(f"time_before ({len(time_before)} values): {time_before.tolist()}")

    # Convert to tensors
    encoded = torch.from_numpy(acoustic).float()  # [T, 512]
    time_before_t = torch.from_numpy(time_before).long()

    # 2. Load decoder
    print("\nLoading TADA decoder from HumeAI/tada-codec...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--HumeAI--tada-codec")
    snapshot_dirs = globmod.glob(os.path.join(cache_dir, "snapshots", "*"))
    snapshot_dir = snapshot_dirs[0]
    decoder_dir = os.path.join(snapshot_dir, "decoder")

    config = DecoderConfig()
    decoder = Decoder(config)
    state_dict = load_file(os.path.join(decoder_dir, "model.safetensors"))
    decoder.load_state_dict(state_dict)
    decoder = decoder.float()
    decoder.eval()
    print(f"Decoder loaded. Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # 3. Decode using _decode_wav logic
    with torch.no_grad():
        time_before_t = time_before_t[: encoded.shape[0] + 1]

        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            n_silence = max(0, int(time_before_t[pos].item()) - 1)
            if n_silence > 0:
                encoded_expanded.append(
                    torch.zeros(n_silence, encoded.shape[-1], dtype=encoded.dtype)
                )
            encoded_expanded.append(encoded[pos].unsqueeze(0))

        # Trailing silence
        encoded_expanded.append(
            torch.zeros(int(time_before_t[-1].item()), encoded.shape[-1], dtype=encoded.dtype)
        )

        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)  # [1, T_expanded, 512]
        print(f"\nExpanded shape: {encoded_expanded.shape}")

        # Build token mask
        token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()
        print(f"Token masks sum: {token_masks.sum().item()} / {token_masks.shape[1]}")

        # Run decoder
        print("Running decoder forward...")
        wav = decoder(encoded_expanded, token_masks)
        print(f"Raw decoder output shape: {wav.shape}")

        # Strip leading silence
        strip_samples = int(24000 * time_before_t[0].item() / 50)
        wav = wav[..., strip_samples:]
        print(f"After stripping {strip_samples} leading samples: {wav.shape}")

        audio = wav.squeeze().cpu().numpy()

    # 4. Save WAV
    sf.write(out_path, audio, 24000, subtype="PCM_16")
    print(f"\nSaved to {out_path}")

    # 5. Print stats
    duration = len(audio) / 24000
    peak = np.max(np.abs(audio))
    dc_offset = np.mean(audio)
    rms = np.sqrt(np.mean(audio ** 2))

    fft = np.fft.rfft(audio)
    power = np.abs(fft) ** 2
    power = power[power > 0]
    log_mean = np.mean(np.log(power))
    geom_mean = np.exp(log_mean)
    arith_mean = np.mean(power)
    spectral_flatness = geom_mean / arith_mean if arith_mean > 0 else 0

    print(f"\nAudio stats:")
    print(f"  Duration: {duration:.3f}s ({len(audio)} samples)")
    print(f"  Peak: {peak:.6f}")
    print(f"  RMS: {rms:.6f}")
    print(f"  DC offset: {dc_offset:.6f}")
    print(f"  Spectral flatness: {spectral_flatness:.6f}")
    if spectral_flatness > 0.4:
        print("  WARNING: High spectral flatness suggests noise, not speech")
    elif spectral_flatness < 0.2:
        print("  OK: Low spectral flatness consistent with speech")
    else:
        print("  BORDERLINE: Spectral flatness in ambiguous range")


if __name__ == "__main__":
    main()
