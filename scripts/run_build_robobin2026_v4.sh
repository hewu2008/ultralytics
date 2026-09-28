#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR=/data/4T-2/dataset/robobin2026/robobin2026-v3
OUT_DIR=/data/4T-2/dataset/robobin2026/robobin2026-v4

python "$REPO_DIR/tools/build_robobin2026_v4.py" --src "$SRC_DIR" --out "$OUT_DIR" "$@"