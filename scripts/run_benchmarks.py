#!/usr/bin/env python3
"""TADA-1B inference benchmark runner across quantization variants and test phrases."""

import argparse
import json
import os
import re
import subprocess
from datetime import date
from pathlib import Path

PHRASES = [
    ("fox", "The quick brown fox jumps over the lazy dog."),
    ("call", "I had to call you up in the middle of the night"),
    ("tyger", "Tyger Tyger, burning bright"),
    ("wutang", "Cash rules everything around me, dollar dollar bill y'all, you need to diversify your bonds."),
]

RUST_VARIANTS = [
    ("F32", "tada-1b-f32.gguf", "6.5G"),
    ("F16", "tada-1b-f16.gguf", "3.3G"),
    ("Q4_0 (baseline)", "tada-1b-q4_0.gguf", "2.6G"),
    ("Var-B (VV-F32 E-Q8)", "tada-1b-B-vvf32-eq8.gguf", "2.5G"),
    ("Var-A (VV-F16 E-Q8)", "tada-1b-A-vvf16-eq8.gguf", "1.9G"),
    ("Var-E (VV-F16 E-Q4)", "tada-1b-E-vvf16-eq4.gguf", "1.8G"),
    ("Mixed (VV-Q8 E-Q8)", "tada-1b-mixed.gguf", "1.4G"),
    ("Var-C (VV-Q8 E-Q4)", "tada-1b-C-vvq8-eq4.gguf", "1.3G"),
]

MODEL_BASE = "/Users/tc/Code/idle-intelligence/hf/tada-1b/"
TOKENIZER = "/Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json"
WORK_DIR = "/Users/tc/Code/idle-intelligence/tts-web"
OUTPUT_JSON = f"/Users/tc/Code/idle-intelligence/tts-web/docs/benchmark_run_{date.today()}.json"
WAV_BASE = "/tmp/tada_bench"

RUST_COMMON_ARGS = [
    "--voice", "voices/ljspeech.safetensors",
    "--noise-temp", "0.9",
    "--transition-steps", "0",
    "--seed", "42",
]

RUST_CMD_PREFIX = [
    "cargo", "run", "--example", "tada_generate",
    "-p", "tada-core", "--release", "--features", "metal", "--",
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_rust_output(stderr: str) -> dict:
    result = {}
    patterns = {
        "load_s": r"load:\s+([\d.]+)s",
        "gen_s": r"generation:\s+([\d.]+)s",
        "decode_s": r"decode:\s+([\d.]+)s",
        "audio_s": r"audio:\s+([\d.]+)s",
        "rtf": r"RTF \(gen\):\s+([\d.]+)x",
    }
    for key, pat in patterns.items():
        m = re.search(pat, stderr)
        result[key] = float(m.group(1)) if m else None
    return result


def parse_python_output(stderr: str) -> dict:
    result = {"load_s": None, "gen_s": None, "decode_s": None, "audio_s": None, "rtf": None}
    m = re.search(r"Model loaded in ([\d.]+)s", stderr)
    if m:
        result["load_s"] = float(m.group(1))
    m = re.search(r"Generation finished in ([\d.]+)s", stderr)
    if m:
        result["gen_s"] = float(m.group(1))
    m = re.search(r"Audio duration:\s+([\d.]+)s", stderr)
    if m:
        result["audio_s"] = float(m.group(1))
    m = re.search(r"RTF:\s+([\d.]+)x", stderr)
    if m:
        result["rtf"] = float(m.group(1))
    return result


def run_rust(variant_name: str, model_file: str, model_size: str, phrase_id: str, phrase_text: str) -> dict:
    slug = slugify(variant_name)
    wav_dir = Path(WAV_BASE) / slug
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_file = str(wav_dir / f"{phrase_id}.wav")

    cmd = RUST_CMD_PREFIX + [
        "--model", os.path.join(MODEL_BASE, model_file),
        "--tokenizer", TOKENIZER,
        "--text", phrase_text,
        "--output", wav_file,
    ] + RUST_COMMON_ARGS

    print(f"  Running: {variant_name} / {phrase_id} ...", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        success = proc.returncode == 0
        stderr = proc.stderr
        if not success:
            print(f"    FAILED (exit {proc.returncode})", flush=True)
            return {
                "variant": variant_name,
                "model_size": model_size,
                "phrase_id": phrase_id,
                "phrase_text": phrase_text,
                "load_s": None,
                "gen_s": None,
                "decode_s": None,
                "audio_s": None,
                "rtf": None,
                "wav_file": wav_file,
                "success": False,
                "error": proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:],
            }
        timings = parse_rust_output(stderr)
        print(f"    OK — RTF: {timings.get('rtf')}x, gen: {timings.get('gen_s')}s", flush=True)
        return {
            "variant": variant_name,
            "model_size": model_size,
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            **timings,
            "wav_file": wav_file,
            "success": True,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", flush=True)
        return {
            "variant": variant_name,
            "model_size": model_size,
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            "load_s": None,
            "gen_s": None,
            "decode_s": None,
            "audio_s": None,
            "rtf": None,
            "wav_file": wav_file,
            "success": False,
            "error": "timeout after 300s",
        }


def run_python(phrase_id: str, phrase_text: str) -> dict:
    slug = slugify("Python (BF16)")
    wav_dir = Path(WAV_BASE) / slug
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_file = str(wav_dir / f"{phrase_id}.wav")

    cmd = (
        f"source .venv/bin/activate && "
        f"python scripts/tada_reference_generate.py "
        f"--text {json.dumps(phrase_text)} "
        f"--output {json.dumps(wav_file)} "
        f"--voice ljspeech "
        f"--seed 42"
    )

    print(f"  Running: Python (BF16) / {phrase_id} ...", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=300,
            shell=True,
            executable="/bin/zsh",
        )
        success = proc.returncode == 0
        stderr = proc.stderr
        if not success:
            print(f"    FAILED (exit {proc.returncode})", flush=True)
            return {
                "variant": "Python (BF16)",
                "model_size": "N/A",
                "phrase_id": phrase_id,
                "phrase_text": phrase_text,
                "load_s": None,
                "gen_s": None,
                "decode_s": None,
                "audio_s": None,
                "rtf": None,
                "wav_file": wav_file,
                "success": False,
                "error": (proc.stderr + proc.stdout)[-2000:],
            }
        timings = parse_python_output(stderr)
        print(f"    OK — RTF: {timings.get('rtf')}x, gen: {timings.get('gen_s')}s", flush=True)
        return {
            "variant": "Python (BF16)",
            "model_size": "N/A",
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            **timings,
            "wav_file": wav_file,
            "success": True,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", flush=True)
        return {
            "variant": "Python (BF16)",
            "model_size": "N/A",
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            "load_s": None,
            "gen_s": None,
            "decode_s": None,
            "audio_s": None,
            "rtf": None,
            "wav_file": wav_file,
            "success": False,
            "error": "timeout after 300s",
        }


def main():
    parser = argparse.ArgumentParser(description="TADA-1B benchmark runner")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python reference runs")
    parser.add_argument("--only", type=str, default=None, help="Run only variant matching this substring")
    args = parser.parse_args()

    results = []

    # Python reference runs first
    if not args.skip_python:
        if args.only is None or args.only.lower() in "python (bf16)":
            print("\n=== Python (BF16) reference ===", flush=True)
            for phrase_id, phrase_text in PHRASES:
                result = run_python(phrase_id, phrase_text)
                results.append(result)

    # Rust variants
    for variant_name, model_file, model_size in RUST_VARIANTS:
        if args.only is not None and args.only.lower() not in variant_name.lower():
            continue
        print(f"\n=== {variant_name} ({model_size}) ===", flush=True)
        for phrase_id, phrase_text in PHRASES:
            result = run_rust(variant_name, model_file, model_size, phrase_id, phrase_text)
            results.append(result)

    # Save results (append to existing if present)
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    # Remove old entries for variants we just ran, keep the rest
    ran_variants = {r["variant"] for r in results}
    existing = [r for r in existing if r["variant"] not in ran_variants]
    combined = existing + results
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}", flush=True)

    # Summary
    total = len(results)
    success = sum(1 for r in results if r["success"])
    print(f"Done: {success}/{total} succeeded.", flush=True)


if __name__ == "__main__":
    main()
