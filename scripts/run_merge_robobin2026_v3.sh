#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Merges robobin2026-v1 and robobin2026-v2 into robobin2026-v3 as a symlinked view.
# Extra arguments are forwarded to the tool, e.g. `--dry-run`, `--force`.
python "$REPO_DIR/tools/merge_robobin2026_v3.py" "$@"