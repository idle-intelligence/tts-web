#!/usr/bin/env python3
"""
Generate golden SentencePiece tokenization fixtures for the Rust port.

Reads each .model file in crates/tts-core/tests/fixtures/tokenizers/, encodes
a per-language corpus of test strings (plus a shared set of cross-language
edge cases) with the reference `sentencepiece` library, and writes the ids +
string pieces to crates/tts-core/tests/fixtures/golden.json.

Also records the normalizer/trainer flags (model_type, byte_fallback,
normalizer name, precompiled_charsmap size, add_dummy_prefix,
remove_extra_whitespaces, escape_whitespaces) per language, so that if a
future re-export of these .model files silently changes one of those flags,
comparing against this file will fail loudly instead of the Rust port just
producing subtly wrong tokens.

Usage:
  python3 scripts/pocket-tts/gen_tokenizer_fixtures.py

Regenerate whenever the .model fixtures in
crates/tts-core/tests/fixtures/tokenizers/ change.
"""

import json
import sys
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as spm_pb2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "crates" / "tts-core" / "tests" / "fixtures"
TOKENIZERS_DIR = FIXTURES_DIR / "tokenizers"
OUTPUT_PATH = FIXTURES_DIR / "golden.json"

LANGUAGES = ["english", "french_24l", "german", "spanish", "portuguese", "italian"]

# Expected trainer/normalizer flags, established by prior inspection of all
# six .model files. If any of these ever changes, we want to know.
EXPECTED_FLAGS = {
    "vocab_size": 4000,
    "model_type": 1,  # UNIGRAM
    "byte_fallback": True,
    "normalizer_name": "identity",
    "precompiled_charsmap_len": 0,
    "add_dummy_prefix": True,
    "remove_extra_whitespaces": False,
    "escape_whitespaces": True,
}

# Shared edge cases, run against every language's tokenizer.
SHARED_CASES = [
    ("empty_string", ""),
    ("single_space", " "),
    ("three_spaces", "   "),
    ("eight_leading_spaces_hello", "        Hello world."),
    ("hello_world_no_prefix", "Hello world."),
    ("real_tab_char", "a\tb"),
    ("real_newline_char", "a\nb"),
    ("emoji_non_bmp", "café 🚀"),
    ("rare_byte_fallback", "œuf ☃ ﬁ"),
    ("mixed_diacritics_currency", "Dr. Müller a payé 42,50 € à l'hôtel."),
    ("digits_float", "3.14159"),
    ("digits_date", "2026-07-26"),
    ("punctuation_run", "a,b;c!d?e"),
    ("repeated_adjacent_spaces", "a  b   c"),
    ("leading_whitespace", " leading"),
    ("trailing_whitespace", "trailing "),
]

# The same French sentence, deliberately also run through the English
# tokenizer below (a known cross-language divergence we want pinned).
CROSS_LANGUAGE_FRENCH_SENTENCE = "L'eau qu'il a bue était glacée, dit-elle « n'est-ce pas ? »"

# Per-language natural sentences exercising characteristic diacritics.
LANGUAGE_SENTENCES = {
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "It was the best of times, it was the worst of times.",
    ],
    "french_24l": [
        CROSS_LANGUAGE_FRENCH_SENTENCE,
        "L'eau est très froide à cette heure-ci, n'est-ce pas ?",
        "Le cœur a ses raisons que la raison ne connaît point.",
    ],
    "german": [
        "Die Straßenverkehrszulassungsordnung regelt vieles.",
        "Können Sie mir bitte einen Kaffee mit Milch geben?",
        "Der Fußballweltmeisterschaftsendspieltermin steht fest.",
    ],
    "spanish": [
        "¿Cómo estás? ¡Muy bien, gracias!",
        "El niño pequeño comió mañana en el jardín.",
        "La señora García vive cerca del río.",
    ],
    "portuguese": [
        "Não sei se ele vai à praça amanhã.",
        "O pão quente é ótimo com manteiga.",
        "As crianças cantam canções na educação infantil.",
    ],
    "italian": [
        "È bello passeggiare per il centro città.",
        "Gli spaghetti sono più buoni con l'olio d'oliva.",
        "Che tempo fa oggi? Fa freddo e piove.",
    ],
}


def check_flags(model_path: Path) -> dict:
    """Parse the raw ModelProto and verify the expected trainer/normalizer flags."""
    proto = spm_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())

    actual = {
        "vocab_size": len(proto.pieces),
        "model_type": int(proto.trainer_spec.model_type),
        "byte_fallback": bool(proto.trainer_spec.byte_fallback),
        "normalizer_name": proto.normalizer_spec.name,
        "precompiled_charsmap_len": len(proto.normalizer_spec.precompiled_charsmap),
        "add_dummy_prefix": bool(proto.normalizer_spec.add_dummy_prefix),
        "remove_extra_whitespaces": bool(proto.normalizer_spec.remove_extra_whitespaces),
        "escape_whitespaces": bool(proto.normalizer_spec.escape_whitespaces),
    }

    for key, expected_value in EXPECTED_FLAGS.items():
        if actual[key] != expected_value:
            print(
                f"ERROR: {model_path.name}: flag '{key}' = {actual[key]!r}, "
                f"expected {expected_value!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    return actual


def encode_case(sp: spm.SentencePieceProcessor, text: str) -> dict:
    ids = sp.encode(text, out_type=int)
    pieces = sp.encode(text, out_type=str)
    return {"text": text, "ids": ids, "pieces": pieces}


def main() -> None:
    result = {}

    for lang in LANGUAGES:
        model_path = TOKENIZERS_DIR / f"{lang}.model"
        if not model_path.exists():
            print(f"ERROR: missing model file {model_path}", file=sys.stderr)
            sys.exit(1)

        flags = check_flags(model_path)

        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))

        cases = []
        for i, sentence in enumerate(LANGUAGE_SENTENCES[lang]):
            cases.append({"name": f"sentence_{i}", **encode_case(sp, sentence)})
        for name, text in SHARED_CASES:
            cases.append({"name": name, **encode_case(sp, text)})

        result[lang] = {"flags": flags, "cases": cases}

    # Cross-language case: run the French sentence through the English tokenizer.
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(str(TOKENIZERS_DIR / "english.model"))
    result["english"]["cases"].append({
        "name": "cross_language_french_sentence_via_english_tokenizer",
        **encode_case(sp_en, CROSS_LANGUAGE_FRENCH_SENTENCE),
    })

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    total_cases = sum(len(v["cases"]) for v in result.values())
    print(f"Wrote {OUTPUT_PATH} ({total_cases} cases across {len(LANGUAGES)} languages)")


if __name__ == "__main__":
    main()
