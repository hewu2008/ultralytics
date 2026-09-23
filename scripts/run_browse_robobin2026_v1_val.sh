#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Loads the robobin2026-v1 validation split into FiftyOne and serves the App.
# Extra arguments are forwarded to the tool, e.g. `--limit 200`, `--overwrite`, `--no-app`.
python "$REPO_DIR/tools/browse_robobin2026_v1_val.py" "$@"