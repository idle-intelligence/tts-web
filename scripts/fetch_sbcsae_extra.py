#!/usr/bin/env python3
"""Fetch 2-3 diverse SBCSAE speakers by parsing speaker-tagged transcripts.

SBCSAE format: each row is a full conversation.
transcript field: <SPK:ID_NAME><ST:t>text<ET:t>...
We parse segments per speaker, find runs >=10s, prefer male speakers.
We already have rickie (female, conversational). Looking for:
 - Male speaker with a substantial monologue-ish segment
 - Any speaker with notably different style
"""
import sys
import re
import numpy as np
import soundfile as sf
from datasets import load_dataset
import json

REPO_ROOT = "/Users/tc/Code/idle-intelligence/tts-web"
OUT_DIR = f"{REPO_ROOT}/refs/tada/tada/samples"
TARGET_SR = 24000
MIN_SEGMENT_DURATION = 8.0  # seconds of consecutive speech
TARGET_VOICES = 3  # how many new voices to extract

# Known female names from SBCSAE to help gender guessing
# (not exhaustive, just common ones in the corpus)
LIKELY_FEMALE = {"rebecca", "rickie", "alice", "carol", "mary", "linda",
                 "susan", "betty", "helen", "sandra", "dorothy", "beverly",
                 "patricia", "barbara", "ruth", "shirley", "jean", "joan",
                 "edna", "virginia"}
SKIP_SPEAKERS = {"rickie"}  # already have this one


def parse_segments(transcript: str):
    """Parse <SPK:ID_NAME><ST:float>text<ET:float> into list of dicts."""
    # Pattern: speaker tag, start time, text, end time
    pattern = re.compile(
        r'<SPK:([^>]+)><ST:([\d.]+)>(.*?)<ET:([\d.]+)>'
    )
    segs = []
    for m in pattern.finditer(transcript):
        spk_id = m.group(1)
        start = float(m.group(2))
        text = m.group(3).strip()
        end = float(m.group(4))
        if text and end > start:
            segs.append({"speaker": spk_id, "start": start, "end": end, "text": text})
    return segs


def guess_gender(speaker_id: str) -> str:
    """Guess gender from speaker name embedded in ID like '0023_REBECCA'."""
    parts = speaker_id.split("_")
    name = parts[-1].lower() if parts else ""
    if name in LIKELY_FEMALE:
        return "female"
    return "unknown"  # assume male unless we know female


def merge_consecutive_segments(segs, speaker_id, gap_threshold=2.0):
    """Merge consecutive segments from same speaker into runs."""
    speaker_segs = [s for s in segs if s["speaker"] == speaker_id]
    if not speaker_segs:
        return []

    runs = []
    current = dict(speaker_segs[0])
    current["texts"] = [current["text"]]

    for seg in speaker_segs[1:]:
        if seg["start"] - current["end"] <= gap_threshold:
            current["end"] = seg["end"]
            current["texts"].append(seg["text"])
        else:
            runs.append(current)
            current = dict(seg)
            current["texts"] = [current["text"]]
    runs.append(current)

    for r in runs:
        r["duration"] = r["end"] - r["start"]
        r["full_text"] = " ".join(r["texts"])
    return runs


def resample(audio_array, from_sr, to_sr):
    if from_sr == to_sr:
        return audio_array
    import torchaudio, torch
    wav = torch.from_numpy(np.array(audio_array, dtype=np.float32)).unsqueeze(0)
    wav = torchaudio.functional.resample(wav, from_sr, to_sr)
    return wav.squeeze(0).numpy()


print("[sbcsae] Loading dataset (streaming)...", flush=True)
ds = load_dataset("dklement/SBCSAE", split="test", streaming=True)

# Collect candidates: (speaker_id, conversation_id, best_run)
candidates = []  # list of (priority, speaker_name, conv_id, run, gender)

for conv_idx, row in enumerate(ds):
    conv_id = row.get("id", f"conv{conv_idx}")
    transcript = row.get("transcript", "")
    if not transcript:
        continue

    segs = parse_segments(transcript)
    if not segs:
        continue

    # Find all speakers in this conversation
    speakers = list({s["speaker"] for s in segs})

    for spk in speakers:
        name_part = spk.split("_")[-1].lower()
        if name_part in SKIP_SPEAKERS:
            continue
        if any(c["speaker_name"] == name_part for c in candidates):
            continue  # already have this speaker

        gender = guess_gender(spk)
        runs = merge_consecutive_segments(segs, spk)

        # Find best run (longest >= MIN_SEGMENT_DURATION)
        good_runs = [r for r in runs if r["duration"] >= MIN_SEGMENT_DURATION]
        if not good_runs:
            continue

        best_run = max(good_runs, key=lambda r: r["duration"])
        # Cap at 18 seconds
        if best_run["duration"] > 18.0:
            best_run["duration"] = 18.0  # will trim audio later

        # Priority: male=0, unknown=1, female=2
        priority = 0 if gender != "female" else 2

        candidates.append({
            "priority": priority,
            "speaker_name": name_part,
            "speaker_id": spk,
            "conv_id": conv_id,
            "run": best_run,
            "gender": gender,
            "row_idx": conv_idx,
        })
        print(f"[sbcsae] Candidate: spk={spk!r} conv={conv_id} dur={best_run['duration']:.1f}s "
              f"gender={gender} text={best_run['full_text'][:60]!r}", flush=True)

    if conv_idx % 5 == 0:
        print(f"[sbcsae] Scanned {conv_idx+1} conversations, {len(candidates)} candidates", flush=True)

    if len(candidates) >= 20:
        # Have enough candidates, stop scanning
        break

print(f"\n[sbcsae] Total candidates: {len(candidates)}", flush=True)

# Sort by priority (male first), then duration
candidates.sort(key=lambda c: (c["priority"], -c["run"]["duration"]))
print("[sbcsae] Top candidates:", flush=True)
for c in candidates[:6]:
    print(f"  {c['speaker_name']} ({c['gender']}) dur={c['run']['duration']:.1f}s conv={c['conv_id']}", flush=True)

# Pick top TARGET_VOICES distinct speakers
chosen = candidates[:TARGET_VOICES]
if not chosen:
    print("[sbcsae] ERROR: no candidates found!", flush=True)
    sys.exit(1)

# Now we need to actually load audio for the chosen speakers.
# Re-scan the dataset to find the right rows.
needed = {c["conv_id"]: c for c in chosen}
results = {}

print(f"\n[sbcsae] Loading audio for {len(needed)} conversations...", flush=True)
ds2 = load_dataset("dklement/SBCSAE", split="test", streaming=True)

for row in ds2:
    conv_id = row.get("id", "")
    if conv_id not in needed:
        continue

    c = needed[conv_id]
    run = c["run"]

    # Load audio (AudioDecoder — decode by slicing)
    audio_decoder = row["audio"]
    # Some HF datasets return an AudioDecoder object; try decoding
    try:
        decoded = audio_decoder.decode()  # returns dict with array+sr
        arr = np.array(decoded["array"], dtype=np.float32)
        sr = decoded["sampling_rate"]
    except AttributeError:
        # May already be decoded
        arr = np.array(audio_decoder["array"], dtype=np.float32)
        sr = audio_decoder["sampling_rate"]

    # Slice to the run's time range
    start_sample = int(run["start"] * sr)
    end_sample = int(min(run["start"] + min(run["duration"], 18.0), run["end"]) * sr)
    clip = arr[start_sample:end_sample]

    if sr != TARGET_SR:
        clip = resample(clip, sr, TARGET_SR)

    # Use actual run text (limited to what fits in the clipped duration)
    text = run["full_text"]
    # Trim text heuristically if too long (keep proportional to duration)
    words = text.split()
    approx_words = int(len(words) * min(run["duration"], 18.0) / run["duration"])
    text = " ".join(words[:approx_words])

    speaker_name = c["speaker_name"]
    out_wav = f"{OUT_DIR}/sbcsae_{speaker_name}.wav"
    sf.write(out_wav, clip, TARGET_SR)
    actual_dur = len(clip) / TARGET_SR
    print(f"[sbcsae] Saved {out_wav} ({actual_dur:.1f}s) text={text[:80]!r}", flush=True)

    results[speaker_name] = {
        "wav": out_wav,
        "text": text,
        "speaker_id": c["speaker_id"],
        "gender": c["gender"],
        "conv_id": conv_id,
        "duration": actual_dur,
    }

    needed.pop(conv_id)
    if not needed:
        break

if needed:
    print(f"[sbcsae] WARNING: could not find audio for: {list(needed.keys())}", flush=True)

manifest_path = f"{OUT_DIR}/sbcsae_extra_manifest.json"
# Save without numpy arrays
save_results = {k: {kk: vv for kk, vv in v.items() if kk != "audio"} for k, v in results.items()}
with open(manifest_path, "w") as f:
    json.dump(save_results, f, indent=2)
print(f"\n[sbcsae] Manifest written to {manifest_path}", flush=True)
for name, info in results.items():
    print(f"  {name}: {info['duration']:.1f}s | {info['text'][:80]}", flush=True)
