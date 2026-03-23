#!/usr/bin/env python3
"""
KittenTTS ONNX reference audio generator for validation.
Generates audio using the KittenTTS Nano v0.8 ONNX model.
"""

import json
import os
import struct
import subprocess
import sys
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/kitten_tts_nano_v0_8.onnx"
VOICES_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/voices.npz"
CONFIG_PATH = "/Users/tc/Code/idle-intelligence/hf/kitten-tts-nano-0.8/config.json"
SAMPLES_DIR = "/Users/tc/Code/idle-intelligence/tts-web-kitten/samples"
SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Symbol table (178 symbols, matching KittenTTS TextCleaner)
# Index 0 = "$" (padding/boundary), Index 10 = ";" (used as end token)
# ---------------------------------------------------------------------------

IPA_CHARS = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʓʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

SYMBOLS = "$" + ";:,.!?¡¿—…\"«»\u201c\u201d " + \
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + \
          "abcdefghijklmnopqrstuvwxyz" + \
          IPA_CHARS

SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def print_symbol_table():
    print(f"\n=== Phoneme Symbol Table ({len(SYMBOLS)} symbols) ===")
    for i, s in enumerate(SYMBOLS):
        print(f"  [{i:3d}] {repr(s)}")


# ---------------------------------------------------------------------------
# Phonemization via espeak-ng
# ---------------------------------------------------------------------------

def check_espeak():
    result = subprocess.run(["which", "espeak-ng"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: espeak-ng is not installed.")
        print("Install with: brew install espeak-ng")
        sys.exit(1)
    return result.stdout.strip()


def phonemize(text: str) -> str:
    """Run espeak-ng and return IPA string."""
    result = subprocess.run(
        ["espeak-ng", "--ipa", "-q", text],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {result.stderr}")
    ipa = result.stdout.strip()
    print(f"  espeak IPA output: {repr(ipa)}")
    return ipa


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(ipa: str) -> list[int]:
    """Convert IPA string to token IDs with boundary tokens."""
    tokens = []
    for ch in ipa:
        if ch in SYMBOL_TO_ID:
            tokens.append(SYMBOL_TO_ID[ch])
        # skip unmapped characters silently
    # Add boundary tokens: [0, ...tokens..., 10, 0]
    result = [0] + tokens + [10, 0]
    return result


# ---------------------------------------------------------------------------
# Style vector selection
# ---------------------------------------------------------------------------

def get_style(voices: dict, voice_key: str, text: str) -> np.ndarray:
    """Select style vector by text length (clamped to [0, 399])."""
    matrix = voices[voice_key]  # shape (400, 256)
    idx = min(len(text), 399)
    style = matrix[idx : idx + 1]  # shape (1, 256)
    return style


# ---------------------------------------------------------------------------
# WAV writing (float32, 32-bit IEEE)
# ---------------------------------------------------------------------------

def write_wav_float32(path: str, audio: np.ndarray, sample_rate: int):
    """Write float32 PCM WAV without scipy dependency."""
    audio = audio.astype(np.float32)
    num_samples = len(audio)
    num_channels = 1
    bits_per_sample = 32
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk (3 = IEEE_FLOAT)
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 3))          # PCM IEEE float
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio.tobytes())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(
    sess: ort.InferenceSession,
    voices: dict,
    voice_key: str,
    text: str,
    ipa: str,
    speed: float,
    output_path: str,
    label: str,
):
    print(f"\n--- {label} ---")
    print(f"  text: {repr(text)}")
    print(f"  voice key: {voice_key}")

    token_ids = tokenize(ipa)
    print(f"  token IDs ({len(token_ids)}): {token_ids}")

    style = get_style(voices, voice_key, text)
    print(f"  style vector shape: {style.shape}, idx={min(len(text), 399)}")

    input_ids = np.array([token_ids], dtype=np.int64)  # (1, seq_len)
    speed_arr = np.array([speed], dtype=np.float32)

    outputs = sess.run(
        None,
        {
            "input_ids": input_ids,
            "style": style,
            "speed": speed_arr,
        },
    )

    waveform = outputs[0]  # shape (num_samples,)
    durations = outputs[1]

    print(f"  raw output length: {len(waveform)} samples")
    print(f"  durations: {durations}")

    # Trim last 5000 samples
    if len(waveform) > 5000:
        waveform = waveform[:-5000]

    duration_sec = len(waveform) / SAMPLE_RATE
    print(f"  trimmed output length: {len(waveform)} samples ({duration_sec:.3f}s)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_wav_float32(output_path, waveform, SAMPLE_RATE)
    print(f"  saved: {output_path}")

    return waveform


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    espeak_path = check_espeak()
    print(f"espeak-ng: {espeak_path}")

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    voice_aliases = config["voice_aliases"]
    speed_priors = config.get("speed_priors", {})
    print(f"\nVoice aliases: {voice_aliases}")
    print(f"Speed priors: {speed_priors}")

    # Load voices
    voices = dict(np.load(VOICES_PATH))
    print(f"Voices keys: {list(voices.keys())}")
    for k, v in voices.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    # Load ONNX model
    print(f"\nLoading ONNX model from {MODEL_PATH} ...")
    sess = ort.InferenceSession(MODEL_PATH)
    print("Model loaded.")

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # --- Generation 1: "Hello world" with Jasper ---
    text1 = "Hello world"
    voice_name1 = "Jasper"
    voice_key1 = voice_aliases[voice_name1]
    speed1 = 1.0

    print(f"\nPhonemizing: {repr(text1)}")
    ipa1 = phonemize(text1)

    generate(
        sess, voices, voice_key1, text1, ipa1, speed1,
        f"{SAMPLES_DIR}/kitten-reference-hello.wav",
        f"{voice_name1} / {voice_key1}",
    )

    # --- Generation 2: "The quick brown fox..." with Bella ---
    text2 = "The quick brown fox jumps over the lazy dog."
    voice_name2 = "Bella"
    voice_key2 = voice_aliases[voice_name2]
    speed2 = 1.0

    print(f"\nPhonemizing: {repr(text2)}")
    ipa2 = phonemize(text2)

    generate(
        sess, voices, voice_key2, text2, ipa2, speed2,
        f"{SAMPLES_DIR}/kitten-reference-fox.wav",
        f"{voice_name2} / {voice_key2}",
    )

    # Print full symbol table
    print_symbol_table()
    print(f"\nTotal symbols: {len(SYMBOLS)}")


if __name__ == "__main__":
    main()
