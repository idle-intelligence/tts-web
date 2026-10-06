#!/usr/bin/env bash
# Builds tts-wasm and kitten-wasm and assembles the deployed site into _site/.
#
#   crates/tts-wasm/pkg      pocket-tts voice (default demo model)
#   crates/kitten-wasm/pkg   KittenTTS nano voice (second model option)
#
# then rewrites the `?v=` build tag to ENGINE_BUILD on every loading URL
# (web/worker.js, web/kitten-worker.js) and checks neither built wasm carries
# a local build path.
#
# Requires wasm-pack, wasm-bindgen-cli (version matching Cargo.lock's
# wasm-bindgen exactly: `cargo install wasm-bindgen-cli --version <ver>
# --locked`).
#
# Usage: ENGINE_BUILD=<tag> scripts/build.sh
# ENGINE_BUILD defaults to "dev" for local builds; CI passes the commit sha.
# It is required on every real deploy - a rebuild with no tag bump keeps
# browsers running the cached module.
set -euo pipefail

ENGINE_BUILD="${ENGINE_BUILD:-dev}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"

echo "==> Building tts-wasm for ENGINE_BUILD=$ENGINE_BUILD"
RUSTFLAGS="--remap-path-prefix=$HOME=/home --remap-path-prefix=$CARGO_HOME_DIR=/cargo" \
  wasm-pack build crates/tts-wasm --target web --release

echo "==> Building kitten-wasm (wasm feature) for ENGINE_BUILD=$ENGINE_BUILD"
RUSTFLAGS="--remap-path-prefix=$HOME=/home --remap-path-prefix=$CARGO_HOME_DIR=/cargo" \
  wasm-pack build crates/kitten-wasm --target web --release --no-default-features --features wasm

TTS_WASM="crates/tts-wasm/pkg/tts_wasm_bg.wasm"
KITTEN_WASM="crates/kitten-wasm/pkg/kitten_wasm_bg.wasm"

# --- Local-path / user-name leak check on built wasm outputs ---
for WASM in "$TTS_WASM" "$KITTEN_WASM"; do
    LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
    USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
    if [ -n "$LEAKS$USER_HITS" ]; then
        echo "error: $WASM contains local paths or the user name:" >&2
        printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
        exit 1
    fi
done
echo "==> no local paths in built wasm"

# --- Assemble the deployed site into _site/ ---
echo "==> Assembling _site"
rm -rf _site
mkdir -p _site/pkg _site/kitten-pkg _site/web
cp crates/tts-wasm/pkg/tts_wasm.js crates/tts-wasm/pkg/tts_wasm_bg.wasm crates/tts-wasm/pkg/package.json _site/pkg/
cp crates/kitten-wasm/pkg/kitten_wasm.js crates/kitten-wasm/pkg/kitten_wasm_bg.wasm crates/kitten-wasm/pkg/package.json _site/kitten-pkg/
cp web/index.html web/worker.js web/kitten-worker.js web/tts-client.js web/audio-worklet.js \
   web/apple-touch-icon.png web/favicon.ico _site/web/

# --- Rewrite the ?v= build tag to ENGINE_BUILD on every loading URL ---
echo "==> Rewriting ENGINE_BUILD tag to $ENGINE_BUILD"
sed -i.bak "s/const ENGINE_BUILD = \"[^\"]*\";/const ENGINE_BUILD = \"$ENGINE_BUILD\";/" \
  _site/web/worker.js _site/web/kitten-worker.js
rm -f _site/web/worker.js.bak _site/web/kitten-worker.js.bak

COUNT="$(grep -c "const ENGINE_BUILD = \"$ENGINE_BUILD\";" _site/web/worker.js _site/web/kitten-worker.js | awk -F: '{s+=$2} END {print s}')"
if [ "$COUNT" -ne 2 ]; then
    echo "error: expected 2 ENGINE_BUILD assignments rewritten to $ENGINE_BUILD, found $COUNT" >&2
    exit 1
fi

echo "==> Wrote _site"
