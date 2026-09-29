#!/usr/bin/env python3
"""Generate deterministic parity fixtures against the official `pocket-tts`
PyTorch package, one per language, for `crates/tts-core/tests/official_parity.rs`.

temp=0.0 makes the flow-matching sampler's noise tensor identically zero in
both the official model and this port (std = temp**0.5 = 0 on both sides), so
the two should walk the same ODE integration deterministically.

IMPORTANT — voice-embeddings snapshot pinning:
`pocket_tts.utils.utils.get_predefined_voice()` resolves a named voice (e.g.
"estelle") to a hardcoded commit of
`kyutai/pocket-tts-without-voice-cloning`'s `languages/<lang>/embeddings/`
that can be older than the snapshot this repo's own quantization pipeline
fetched (see `docs/runs/2026-09-29-multilingual.md`'s source-checkpoints
table). Those two commits are NOT guaranteed byte-identical for a given
voice file — confirmed for French/estelle: the officially-catalogued
commit's cached KV state has 168 valid positions, this repo's freshly-fetched
snapshot has 154. Comparing a fixture generated from one against a Rust run
fed the other silently compares two different voice conditionings, not two
implementations of the same model, and produces a divergence that looks like
a porting bug (it doesn't cancel out at layer 0: it grows through every
self-attention layer that attends over the voice-conditioning prefix).
This script pins the voice (and model/tokenizer) fetch to the exact same
Hub snapshot commit this repo's quantization step used, overriding the
package's own catalog resolution, so the fixture and the Rust run being
tested against it are guaranteed to load byte-identical voice data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.utils import utils as pocket_utils

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "crates/tts-core/tests/fixtures/parity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Must match the "Source checkpoints" table in
# docs/runs/2026-09-29-multilingual.md (the snapshot this repo's
# quantization pipeline fetched for each language).
SNAPSHOT_COMMIT = "069025daa1d6a8a9640bd52581e2a2252f8bede4"

CASES = {
    "english": ("english", "alba", "Hello, this is a test of the text to speech system."),
    "french": ("french", "estelle", "Bonjour, ceci est un test du système de synthèse vocale."),
    "german": ("german", "juergen", "Hallo, dies ist ein Test des Text-zu-Sprache-Systems."),
    "spanish": ("spanish", "lola", "Hola, esta es una prueba del sistema de texto a voz."),
    "portuguese": ("portuguese", "rafael", "Olá, este é um teste do sistema de texto para voz."),
    "italian": ("italian", "giovanni", "Ciao, questo è un test del sistema da testo a voce."),
}

orig_get_predefined_voice = pocket_utils.get_predefined_voice


def pinned_get_predefined_voice(language: str, name: str) -> str:
    return (
        f"hf://kyutai/pocket-tts-without-voice-cloning/"
        f"languages/{language}/embeddings/{name}.safetensors@{SNAPSHOT_COMMIT}"
    )


def frame_stats(chunk: torch.Tensor) -> dict:
    arr = chunk.flatten().to(torch.float32).numpy()
    return {
        "len": int(arr.shape[0]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "rms": float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
    }


def main():
    only = sys.argv[1:] or list(CASES.keys())

    # Pin voice resolution to the snapshot this repo's quantization used, both
    # at the module level (module.tts_model imported `get_predefined_voice`
    # by name) and where `pocket_utils` is called directly.
    from pocket_tts.models import tts_model as tts_model_mod
    pocket_utils.get_predefined_voice = pinned_get_predefined_voice
    tts_model_mod.get_predefined_voice = pinned_get_predefined_voice

    for name in only:
        lang, voice, text = CASES[name]
        print(f"=== {name} (lang={lang} voice={voice}, voice snapshot={SNAPSHOT_COMMIT}) ===", flush=True)
        model = TTSModel.load_model(language=lang, temp=0.0)
        tokens = model.flow_lm.tokenizer.encode(text) if hasattr(model.flow_lm, "tokenizer") else None
        state = model.get_state_for_audio_prompt(voice)

        frames = []
        total_samples = 0
        for chunk in model.generate_audio_stream(model_state=state, text_to_generate=text):
            frames.append(frame_stats(chunk))
            total_samples += chunk.flatten().shape[0]

        result = {
            "language": lang,
            "voice": voice,
            "voice_snapshot": SNAPSHOT_COMMIT,
            "text": text,
            "temperature": 0.0,
            "sample_rate": model.sample_rate,
            "num_frames": len(frames),
            "total_samples": total_samples,
            "frames": frames,
        }
        if tokens is not None:
            result["token_ids"] = list(int(t) for t in (tokens if not hasattr(tokens, "tolist") else tokens.tolist()))

        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  wrote {out_path} ({len(frames)} frames, {total_samples} samples)")


if __name__ == "__main__":
    main()
