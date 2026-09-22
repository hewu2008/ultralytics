#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR=/data/4T-2/dataset/robobin2026/robobin2026-v2
OUT_DIR=/data/4T-2/dataset/robobin2026/robobin2026-v2-coco

python "$REPO_DIR/tools/convert_robobin2026_to_coco.py" --data-root "$DATA_DIR" --out "$OUT_DIR" "$@"
