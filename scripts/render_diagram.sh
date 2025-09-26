#!/usr/bin/env bash
set -euo pipefail

SRC=${1:-docs/architecture.mmd}
OUT=${2:-docs/architecture.png}
EXECUTABLE=${3:-}

# Try to detect a local Chrome/Chromium if not explicitly provided
detect_browser() {
  if [[ -n "${PUPPETEER_EXECUTABLE_PATH:-}" ]]; then
    echo "$PUPPETEER_EXECUTABLE_PATH"; return 0
  fi
  if [[ -n "$EXECUTABLE" ]]; then
    echo "$EXECUTABLE"; return 0
  fi
  # Common macOS locations
  for p in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"; do
    [[ -x "$p" ]] && { echo "$p"; return 0; }
  done
  # Common PATH names
  for bin in chromium chromium-browser google-chrome chrome; do
    if command -v "$bin" >/dev/null 2>&1; then
      command -v "$bin"; return 0
    fi
  done
  echo ""; return 1
}

BROWSER="$(detect_browser || true)"
if [[ -n "$BROWSER" ]]; then
  echo "Using system browser: $BROWSER"
  export PUPPETEER_EXECUTABLE_PATH="$BROWSER"
fi

if command -v mmdc >/dev/null 2>&1; then
  echo "Using mmdc from PATH"
  mmdc -i "$SRC" -o "$OUT" -t neutral -b transparent -s 1.25
  exit 0
fi

if command -v npx >/dev/null 2>&1; then
  echo "Using npx to run mermaid-cli"
  npx --yes @mermaid-js/mermaid-cli@10.9.1 -i "$SRC" -o "$OUT" -t neutral -b transparent -s 1.25
  exit 0
fi

echo "Neither mmdc nor npx found. Install one of:\n  - npm i -g @mermaid-js/mermaid-cli\n  - or install Node and use: npx @mermaid-js/mermaid-cli -i $SRC -o $OUT\nOptionally pass a browser path as 3rd arg or set PUPPETEER_EXECUTABLE_PATH." >&2
exit 1
