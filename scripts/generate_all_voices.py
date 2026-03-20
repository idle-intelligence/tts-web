#!/usr/bin/env python3
"""Generate all speaker × style combinations from Expresso + Python reference audio.

1. Download one clip per (speaker, style) from Expresso parquet
2. Precompute voice prompts
3. Generate audio with Python reference for each

Usage: source .venv/bin/activate && python scripts/generate_all_voices.py
"""

import json
import os
import subprocess
import sys
import struct

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(REPO_ROOT, "samples", "voice_matrix")
VOICES_DIR = os.path.join(REPO_ROOT, "voices", "matrix")
WAV_DIR = os.path.join(REPO_ROOT, "refs", "tada", "tada", "samples", "matrix")

STYLES = ["default", "confused", "enunciated", "happy", "laughing", "sad", "whisper"]
SPEAKERS = ["ex01", "ex02", "ex03", "ex04"]
TEST_TEXT = "I'll tell you one thing about the universe, though. The universe is a pretty big place."

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)


def decode_audio_from_parquet_bytes(audio_bytes):
    """Decode audio bytes from parquet. Try soundfile first, fall back to raw."""
    import io
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        return data, sr
    except Exception:
        return None, None


def find_best_clip(speaker, style):
    """Find a clip with 8-15s duration for this speaker+style."""
    for shard in range(12):
        path = hf_hub_download(
            "ylacombe/expresso",
            f"read/train-{shard:05d}-of-00012.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(path, columns=["speaker_id", "style", "text", "audio"])

        for i in range(len(table)):
            spk = table["speaker_id"][i].as_py()
            sty = table["style"][i].as_py()
            if spk != speaker or sty != style:
                continue

            text = table["text"][i].as_py()
            audio_col = table["audio"][i].as_py()

            # audio_col is a dict with 'bytes', 'path', etc.
            audio_bytes = audio_col.get("bytes") if isinstance(audio_col, dict) else None
            if not audio_bytes:
                continue

            data, sr = decode_audio_from_parquet_bytes(audio_bytes)
            if data is None:
                continue

            dur = len(data) / sr
            words = len(text.split()) if text else 0

            if 8.0 <= dur <= 20.0 and words >= 15:
                return data, sr, text, dur

    return None, None, None, None


def main():
    results = []

    # Step 1: Download clips and precompute voice prompts
    print("=" * 60)
    print("Step 1: Download clips + precompute voice prompts")
    print("=" * 60)

    for speaker in SPEAKERS:
        for style in STYLES:
            voice_name = f"{speaker}_{style}"
            voice_path = os.path.join(VOICES_DIR, f"{voice_name}.safetensors")
            wav_path = os.path.join(WAV_DIR, f"{voice_name}.wav")

            if os.path.exists(voice_path):
                print(f"  {voice_name}: already exists, skipping")
                results.append({"name": voice_name, "speaker": speaker, "style": style, "status": "cached"})
                continue

            print(f"  {voice_name}: searching...", end="", flush=True)
            data, sr, text, dur = find_best_clip(speaker, style)

            if data is None:
                print(f" NOT FOUND (no clip 8-20s)")
                results.append({"name": voice_name, "speaker": speaker, "style": style, "status": "not_found"})
                continue

            print(f" found {dur:.1f}s, {len(text.split())}w", flush=True)

            # Save WAV
            sf.write(wav_path, data, sr)

            # Precompute voice prompt
            cmd = [
                sys.executable, "scripts/precompute_voice.py",
                "--audio", wav_path,
                "--text", text,
                "--output", voice_path,
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                print(f"    precompute FAILED: {proc.stderr[-200:]}")
                results.append({"name": voice_name, "speaker": speaker, "style": style, "status": "precompute_failed"})
                continue

            results.append({"name": voice_name, "speaker": speaker, "style": style, "status": "ok"})

    # Step 2: Generate audio with Python reference
    print("\n" + "=" * 60)
    print("Step 2: Generate audio with Python reference")
    print("=" * 60)

    for r in results:
        if r["status"] not in ("ok", "cached"):
            continue

        voice_name = r["name"]
        voice_path = os.path.join(VOICES_DIR, f"{voice_name}.safetensors")
        output_path = os.path.join(SAMPLES_DIR, f"{voice_name}.wav")

        if os.path.exists(output_path):
            print(f"  {voice_name}: audio exists, skipping")
            continue

        print(f"  {voice_name}: generating...", end="", flush=True)
        cmd = [
            sys.executable, "scripts/tada_reference_generate.py",
            "--text", TEST_TEXT,
            "--output", output_path,
            "--voice", voice_path,
            "--seed", "42",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f" FAILED")
            continue

        # Extract duration from output
        for line in proc.stderr.split("\n"):
            if "Audio duration" in line:
                print(f" {line.strip().split(']')[-1].strip()}")
                break
        else:
            print(f" done")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    print(f"{ok}/{len(results)} voice prompts ready")
    print(f"Audio samples in: {SAMPLES_DIR}/")
    print(f"Voice prompts in: {VOICES_DIR}/")


if __name__ == "__main__":
    main()
