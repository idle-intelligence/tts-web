#!/usr/bin/env bash
set -euo pipefail

# benchmark_voices.sh — idempotent TADA voice sweep
#
# Usage:
#   benchmark_voices.sh                          # all 32 voices
#   benchmark_voices.sh <voice.safetensors> ...  # specific voices only
#
# Skips any voice that already has a numeric RTF in results.md.
# Appends one row to docs/tada/results.md and one block to docs/tada/lab-notebook.md per voice.
# On failure: logs FAIL row + reproducer block, continues.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_MD="$REPO_ROOT/docs/tada/results.md"
NOTEBOOK_MD="$REPO_ROOT/docs/tada/lab-notebook.md"
OUTPUT_DIR="/tmp/tada_bench"
mkdir -p "$OUTPUT_DIR"

MODEL="/Users/tc/Code/idle-intelligence/hf/tada-1b/tada-1b-C-vvq8-eq4.gguf"
TOKENIZER="/Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B/tokenizer.json"
TEXT="The quick brown fox jumps over the lazy dog."
MODEL_LABEL="Var-C VV-Q8 E-Q4"
MODEL_SIZE="1.3G"
ENGINE="candle"
DEVICE="metal"

# Default params
NOISE_TEMP=0.9
TEMPERATURE=0.6
TRANSITION_STEPS=5
SEED=42
CFG_SCALE=1.6
FLOW_STEPS=10
TOP_P=0.9
REP_PENALTY=1.1

COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
TODAY="$(date +%Y-%m-%d)"

# ── Voice list ──────────────────────────────────────────────────────────────

if [[ $# -gt 0 ]]; then
    VOICE_FILES=("$@")
else
    # 6 base voices (tracked)
    BASE_VOICES=(
        "$REPO_ROOT/voices/amazement.safetensors"
        "$REPO_ROOT/voices/amusement.safetensors"
        "$REPO_ROOT/voices/ljspeech.safetensors"
        "$REPO_ROOT/voices/ljspeech_long.safetensors"
        "$REPO_ROOT/voices/rickie.safetensors"
        "$REPO_ROOT/voices/rickie_trimmed.safetensors"
    )
    # 26 matrix voices (local-only, untracked)
    mapfile -t MATRIX_VOICES < <(ls "$REPO_ROOT/voices/matrix/"*.safetensors 2>/dev/null || true)
    VOICE_FILES=("${BASE_VOICES[@]}" "${MATRIX_VOICES[@]}")
fi

# ── Idempotency check ───────────────────────────────────────────────────────

already_benched() {
    local voice_name="$1"
    # Row exists AND RTF column (field 11) has a numeric value (e.g. 0.84x)
    if grep -q "voice-sweep.*${voice_name}" "$RESULTS_MD" 2>/dev/null; then
        local rtf
        rtf=$(grep "voice-sweep.*${voice_name}" "$RESULTS_MD" | head -1 | awk -F'|' '{gsub(/ /,"",$12); print $12}')
        if [[ "$rtf" =~ ^[0-9]+\.[0-9]+x$ ]]; then
            return 0
        fi
    fi
    return 1
}

# ── Audio duration via ffprobe or python wave ───────────────────────────────

audio_duration() {
    local wav="$1"
    if command -v ffprobe &>/dev/null; then
        ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$wav" 2>/dev/null
    else
        python3 -c "import wave; f=wave.open('$wav'); print(f.getnframes()/f.getframerate()); f.close()" 2>/dev/null
    fi
}

# ── Results.md helpers ───────────────────────────────────────────────────────

ensure_section() {
    local section="voice-matrix-sweep"
    if ! grep -q "^## ${section}$" "$RESULTS_MD" 2>/dev/null; then
        printf '\n## %s\n\n| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |\n|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|\n' \
            "$section" >> "$RESULTS_MD"
    fi
}

append_results_row() {
    local id="$1" load="$2" gen="$3" decode="$4" audio="$5" rtf="$6" file="$7"
    local section="voice-matrix-sweep"
    local row="| ${id} | ${ENGINE} | ${DEVICE} | ${MODEL_LABEL} | ${MODEL_SIZE} | fox | ${load} | ${gen} | ${decode} | ${audio} | ${rtf} | ${file} |"
    # Append row under the section table (before next --- or EOF)
    # Use python to insert after the last row of the section
    python3 - "$RESULTS_MD" "$section" "$row" <<'PYEOF'
import sys
path, section, row = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    lines = f.readlines()
in_section = False
last_row_idx = -1
for i, line in enumerate(lines):
    if line.strip() == f'## {section}':
        in_section = True
    elif in_section and line.startswith('|') and not line.startswith('|-'):
        last_row_idx = i
    elif in_section and line.strip() == '---':
        break
if last_row_idx >= 0:
    lines.insert(last_row_idx + 1, row + '\n')
else:
    lines.append(row + '\n')
with open(path, 'w') as f:
    f.writelines(lines)
PYEOF
}

# ── Lab-notebook helpers ─────────────────────────────────────────────────────

append_notebook_block() {
    local voice_name="$1" load="$2" gen="$3" decode="$4" audio="$5" rtf="$6" notes="$7"
    local total
    total=$(python3 -c "print(f'{float(\"${load}\") + float(\"${gen}\") + float(\"${decode}\"):.1f}')" 2>/dev/null || echo "—")
    cat >> "$NOTEBOOK_MD" <<BLOCK

## ${TODAY} — voice-matrix-sweep — ${COMMIT_SHA} — ${voice_name}

**Date**: ${TODAY}
**Commit**: ${COMMIT_SHA}
**Purpose**: Automated voice sweep — ${voice_name}.

**Parameters**:
- engine: ${ENGINE}
- device: ${DEVICE}
- model: ${MODEL_LABEL} (${MODEL_SIZE})
- voice: ${voice_name}
- text: "The quick brown fox jumps over the lazy dog."
- noise_temp: ${NOISE_TEMP}
- temperature: ${TEMPERATURE}
- cfg_scale: ${CFG_SCALE}
- flow_steps: ${FLOW_STEPS}
- top_p: ${TOP_P}
- repetition_penalty: ${REP_PENALTY}
- transition_steps: ${TRANSITION_STEPS}
- seed: ${SEED}

**Timings**:
- Wall-clock: ${total}s total (load ${load}s, gen ${gen}s, decode ${decode}s)
- Audio duration: ${audio}s
- RTF: ${rtf}

**Output**: ${OUTPUT_DIR}/${voice_name}.wav

**Notes**: ${notes}

---
BLOCK
}

append_notebook_fail() {
    local voice_name="$1" cmd="$2" errfile="$3"
    cat >> "$NOTEBOOK_MD" <<BLOCK

## ${TODAY} — voice-matrix-sweep — ${COMMIT_SHA} — ${voice_name} — FAIL

**Date**: ${TODAY}
**Commit**: ${COMMIT_SHA}
**Purpose**: Automated voice sweep — ${voice_name} — FAILED.

**Reproducer**:
\`\`\`bash
${cmd}
\`\`\`

**Error log**: ${errfile}

---
BLOCK
}

# ── Main loop ────────────────────────────────────────────────────────────────

ensure_section

total_voices=${#VOICE_FILES[@]}
skipped=0
succeeded=0
failed=0

for voice_path in "${VOICE_FILES[@]}"; do
    voice_name="$(basename "$voice_path" .safetensors)"
    out_wav="$OUTPUT_DIR/${voice_name}.wav"

    if already_benched "$voice_name"; then
        echo "[skip] $voice_name — already has RTF in results.md"
        ((skipped++)) || true
        continue
    fi

    echo ""
    echo "=== Benchmarking: $voice_name ==="

    CMD=(
        cargo run --example tada_generate -p tada-core --release --features metal --
        --model "$MODEL"
        --tokenizer "$TOKENIZER"
        --voice "$voice_path"
        --output "$out_wav"
        --text "$TEXT"
        --noise-temp "$NOISE_TEMP"
        --temperature "$TEMPERATURE"
        --transition-steps "$TRANSITION_STEPS"
        --seed "$SEED"
        --cfg-scale "$CFG_SCALE"
        --flow-steps "$FLOW_STEPS"
        --top-p "$TOP_P"
        --repetition-penalty "$REP_PENALTY"
    )

    ERR_FILE="$OUTPUT_DIR/${voice_name}.err"

    set +e
    STDERR_OUT=$( "${CMD[@]}" 2>&1 )
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "$STDERR_OUT" > "$ERR_FILE"
        echo "[FAIL] $voice_name — exit $EXIT_CODE — see $ERR_FILE"
        # Append FAIL row to results.md
        ensure_section
        append_results_row "voice-sweep-${voice_name}" "—" "—" "—" "—" "FAIL" "${voice_name}.wav"
        append_notebook_fail "$voice_name" "${CMD[*]}" "$ERR_FILE"
        ((failed++)) || true
        continue
    fi

    # Parse timings from stderr output
    t_load=$(echo "$STDERR_OUT"  | grep -E "^  load:" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "—")
    t_gen=$(echo "$STDERR_OUT"   | grep -E "^  generation:" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "—")
    t_decode=$(echo "$STDERR_OUT"| grep -E "^  decode:" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "—")
    t_audio=$(echo "$STDERR_OUT" | grep -E "^  audio:" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "—")

    # Compute RTF = gen / audio (matches tada_generate's own formula)
    if [[ "$t_gen" != "—" && "$t_audio" != "—" ]]; then
        rtf=$(python3 -c "g=float('$t_gen'); a=float('$t_audio'); print(f'{g/a:.2f}x') if a>0 else print('—')" 2>/dev/null || echo "—")
    else
        rtf="—"
    fi

    # Collect any notable notes (sanity check warnings, NO_EOS, etc.)
    notes=$(echo "$STDERR_OUT" | grep -E "⚠|WARNING|NO_EOS|sanity check|FAILED" | head -3 | tr '\n' ' ' || echo "")
    [[ -z "$notes" ]] && notes="OK"

    echo "[done] $voice_name — load=${t_load}s gen=${t_gen}s decode=${t_decode}s audio=${t_audio}s RTF=${rtf}"

    append_results_row "voice-sweep-${voice_name}" "$t_load" "$t_gen" "$t_decode" "$t_audio" "$rtf" "${voice_name}.wav"
    append_notebook_block "$voice_name" "$t_load" "$t_gen" "$t_decode" "$t_audio" "$rtf" "$notes"

    ((succeeded++)) || true
done

echo ""
echo "=== Sweep complete: $succeeded succeeded, $failed failed, $skipped skipped (of $total_voices total) ==="
