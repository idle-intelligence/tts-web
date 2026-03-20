#!/usr/bin/env python3
"""
Download curated Expresso dataset clips and save as WAV for voice prompt precompute.

Target styles (one clip each, 10-15s, 30-50 words):
  default, happy, sad, angry, whisper, narration, confused, laughing

Run from repo root with venv active:
    python scripts/download_expresso_clips.py
"""

import os
import sys
import json
import soundfile as sf
import numpy as np
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(REPO_ROOT, "refs", "tada", "tada", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

TARGET_STYLES = {
    "default":   {"gender": "female", "style": "default"},
    "happy":     {"gender": "female", "style": "happy"},
    "sad":       {"gender": "male",   "style": "sad"},
    "angry":     {"gender": "male",   "style": "angry"},
    "whisper":   {"gender": "female", "style": "whisper"},
    "narration": {"gender": "male",   "style": "narration"},
    "confused":  {"gender": "female", "style": "confused"},
    "laughing":  {"gender": "female", "style": "laughing"},
}

# Expresso speaker IDs: ex01 (female), ex02 (male), ex03 (female), ex04 (male)
FEMALE_SPEAKERS = {"ex01", "ex03"}
MALE_SPEAKERS   = {"ex02", "ex04"}

def word_count(text):
    return len(text.split())

def duration_sec(audio_dict):
    arr = audio_dict["array"]
    sr  = audio_dict["sampling_rate"]
    return len(arr) / sr

def safe_get(ds, idx):
    """Fetch a single dataset row, returning None if audio decoding fails."""
    try:
        return ds[idx]
    except Exception:
        return None

def score_row(speaker, dur, words, target_gender):
    gender_ok = (target_gender == "female" and speaker in FEMALE_SPEAKERS) or \
                (target_gender == "male"   and speaker in MALE_SPEAKERS)
    if not gender_ok:
        return None
    if not (10.0 <= dur <= 15.0):
        return None
    if not (30 <= words <= 55):
        return None
    dur_score  = 1.0 - abs(dur   - 12.0) / 3.0
    word_score = 1.0 - abs(words - 40)   / 15.0
    return dur_score + word_score

def main():
    print("Loading Expresso dataset (read config) ...", flush=True)
    ds = load_dataset("ylacombe/expresso", "read", split="train")
    print(f"  Total clips: {len(ds)}", flush=True)

    # Read metadata columns (no audio decode) to find candidate indices
    styles_col   = ds["style"]
    speakers_col = ds["speaker_id"]
    texts_col    = ds["text"]
    styles_in_ds = set(styles_col)
    print(f"  Styles found: {sorted(styles_in_ds)}", flush=True)

    # Build index: style -> list of (idx, speaker, word_count)
    from collections import defaultdict
    style_index = defaultdict(list)
    for idx, (sty, spk, txt) in enumerate(zip(styles_col, speakers_col, texts_col)):
        style_index[sty].append((idx, spk, word_count(txt), txt))

    selections = {}  # style_name -> (idx, clip)

    for voice_name, spec in TARGET_STYLES.items():
        target_style  = spec["style"]
        target_gender = spec["gender"]

        best_score = -1
        best_idx   = None

        candidates = style_index.get(target_style, [])
        # Sort by word count closeness to 40 to try best candidates first
        candidates_sorted = sorted(candidates, key=lambda c: abs(c[2] - 40))

        for idx, speaker, words, text in candidates_sorted:
            # Quick filter on word count before fetching audio
            if not (25 <= words <= 60):
                continue
            gender_ok = (target_gender == "female" and speaker in FEMALE_SPEAKERS) or \
                        (target_gender == "male"   and speaker in MALE_SPEAKERS)
            if not gender_ok:
                continue

            clip = safe_get(ds, idx)
            if clip is None:
                print(f"    [skip] idx={idx} audio decode failed", flush=True)
                continue

            dur = duration_sec(clip["audio"])
            score = score_row(speaker, dur, words, target_gender)
            if score is not None and score > best_score:
                best_score = score
                best_idx   = idx
                # Keep fetching to find the best, but stop after a good score
                if score > 1.5:
                    break

        if best_idx is None:
            print(f"[WARN] No clip found for {voice_name} — widening to 8-18s")
            for idx, speaker, words, text in candidates_sorted:
                if not (20 <= words <= 70):
                    continue
                gender_ok = (target_gender == "female" and speaker in FEMALE_SPEAKERS) or \
                            (target_gender == "male"   and speaker in MALE_SPEAKERS)
                if not gender_ok:
                    continue
                clip = safe_get(ds, idx)
                if clip is None:
                    continue
                dur = duration_sec(clip["audio"])
                if 8.0 <= dur <= 18.0:
                    score = 2.0 - abs(dur - 12.0) / 5.0
                    if score > best_score:
                        best_score = score
                        best_idx   = idx

        if best_idx is None:
            print(f"[ERROR] Completely failed to find clip for {voice_name}")
            continue

        best_clip = ds[best_idx]
        selections[voice_name] = (best_idx, best_clip)
        dur   = duration_sec(best_clip["audio"])
        words = word_count(best_clip["text"])
        print(f"  {voice_name:12s} idx={best_idx:5d}  speaker={best_clip.get('speaker_id','?')}  "
              f"dur={dur:.1f}s  words={words:3d}  score={best_score:.3f}")
        print(f"               text: {best_clip['text'][:80]!r}")

    print("\nSaving WAV clips ...", flush=True)
    manifest = {}
    for voice_name, (idx, clip) in selections.items():
        wav_path = os.path.join(SAMPLES_DIR, f"expresso_{voice_name}.wav")
        audio    = clip["audio"]
        arr      = np.array(audio["array"], dtype=np.float32)
        sr       = audio["sampling_rate"]
        sf.write(wav_path, arr, sr)
        dur = len(arr) / sr
        print(f"  Saved {wav_path}  ({dur:.1f}s @ {sr}Hz)")
        manifest[voice_name] = {
            "wav":     wav_path,
            "text":    clip["text"],
            "speaker": clip.get("speaker_id", ""),
            "style":   clip.get("style", ""),
            "dur":     round(dur, 2),
            "words":   word_count(clip["text"]),
        }

    manifest_path = os.path.join(SAMPLES_DIR, "expresso_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")
    print("\nDone. Next: run precompute_voice.py for each clip.")

if __name__ == "__main__":
    main()
