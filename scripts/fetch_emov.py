#!/usr/bin/env python3
"""Fetch EmoV-DB clips for angry, disgusted, sleepy (male preferred) and save concatenated WAVs."""
import re
import sys
import numpy as np
import soundfile as sf
from datasets import load_dataset

REPO_ROOT = "/Users/tc/Code/idle-intelligence/tts-web"
OUT_DIR = f"{REPO_ROOT}/refs/tada/tada/samples"
TARGETS = ["angry", "disgusted", "sleepy"]
SILENCE_SEC = 0.3
TARGET_SR = 24000

def resample(audio_array, from_sr, to_sr):
    if from_sr == to_sr:
        return audio_array
    import torchaudio, torch
    wav = torch.from_numpy(audio_array).unsqueeze(0)
    wav = torchaudio.functional.resample(wav, from_sr, to_sr)
    return wav.squeeze(0).numpy()

print("[emov] Loading dataset (streaming)...", flush=True)
ds = load_dataset("CLAPv2/EmoV_DB", split="train", streaming=True)

clips_by_emo = {t: {"male": [], "female": []} for t in TARGETS}
found_enough = set()

for i, row in enumerate(ds):
    if len(found_enough) == len(TARGETS):
        break
    if i > 20000:
        break

    text = row.get("text", "")
    m = re.search(r'in (?:a |an )(\w+) voice', text, re.IGNORECASE)
    if not m:
        continue
    emo = m.group(1).lower()
    if emo not in TARGETS:
        continue

    text_lower = text.lower()
    is_female = "female" in text_lower
    is_male = "male" in text_lower and not is_female
    gender = "male" if is_male else "female"

    spoken_m = re.search(r'"([^"]+)"', text)
    if not spoken_m:
        continue
    spoken = spoken_m.group(1)

    audio_len = row.get("audio_len", 0)
    if audio_len < 2.0:
        continue

    bucket = clips_by_emo[emo][gender]
    # prefer male, cap at 4 clips per gender
    if len(bucket) >= 4:
        continue

    audio = row["audio"]
    arr = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if sr != TARGET_SR:
        arr = resample(arr, sr, TARGET_SR)
    bucket.append({"text": spoken, "audio": arr, "sr": TARGET_SR})
    print(f"[emov] {emo}/{gender}: got clip {len(bucket)} ({audio_len:.1f}s) | {spoken[:60]}", flush=True)

    # Check if we have enough for each target
    for t in TARGETS:
        male_clips = clips_by_emo[t]["male"]
        female_clips = clips_by_emo[t]["female"]
        total = len(male_clips) + len(female_clips)
        if total >= 3 and t not in found_enough:
            found_enough.add(t)

    if i % 500 == 0:
        print(f"[emov] Scanned {i} rows, found_enough={found_enough}", flush=True)

print(f"\n[emov] Collection complete. Summary:", flush=True)
for emo in TARGETS:
    m = len(clips_by_emo[emo]["male"])
    f = len(clips_by_emo[emo]["female"])
    print(f"  {emo}: {m} male, {f} female clips", flush=True)

# Concatenate and save
silence = np.zeros(int(TARGET_SR * SILENCE_SEC), dtype=np.float32)
results = {}

for emo in TARGETS:
    # prefer male, fall back to female
    clips = clips_by_emo[emo]["male"]
    if len(clips) < 2:
        clips = clips + clips_by_emo[emo]["female"]
    if len(clips) == 0:
        print(f"[emov] WARNING: no clips for {emo}, skipping", flush=True)
        continue

    clips = clips[:3]  # max 3
    parts = []
    texts = []
    for j, c in enumerate(clips):
        parts.append(c["audio"])
        texts.append(c["text"])
        if j < len(clips) - 1:
            parts.append(silence)

    combined = np.concatenate(parts)
    combined_text = " ".join(texts)
    duration = len(combined) / TARGET_SR

    out_wav = f"{OUT_DIR}/emov_{emo}.wav"
    sf.write(out_wav, combined, TARGET_SR)
    print(f"[emov] Saved {out_wav} ({duration:.1f}s)", flush=True)
    print(f"[emov] Text: {combined_text}", flush=True)
    results[emo] = {"wav": out_wav, "text": combined_text}

# Write results summary for next step
import json
with open(f"{OUT_DIR}/emov_manifest.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[emov] Manifest written to {OUT_DIR}/emov_manifest.json", flush=True)
