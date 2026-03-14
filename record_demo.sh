#!/usr/bin/env bash
# record_demo.sh - record a 60-90 s terminal demo and export to GIF
#
# Dependencies:
#   brew install asciinema          # records terminal session
#   cargo install --git https://github.com/asciinema/agg   # converts cast -> GIF
#   (or: pip install agg)
#
# Output: demo_assets/demo.gif  (commit this file for README embedding)

set -euo pipefail

CAST=demo.cast
GIF=demo_assets/demo.gif

echo "==> Recording demo session (type 'exit' or Ctrl-D to stop)..."
asciinema rec "$CAST" \
  --command "bash run_demo.sh" \
  --title "Ticker Disambiguation - 60s Demo" \
  --idle-time-limit 2

echo "==> Converting cast -> GIF..."
agg "$CAST" "$GIF" --speed 1.5 --font-size 14

echo "==> Done: $GIF"
echo "    Add to README with: ![demo](demo_assets/demo.gif)"

# Clean up raw cast file (optional)
rm -f "$CAST"
