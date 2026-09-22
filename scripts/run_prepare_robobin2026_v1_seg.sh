#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR=/data/4T-2/dataset/robobin2026/robobin2026-v1

python "$REPO_DIR/tools/prepare_robobin2026_seg.py" "$DATA_DIR"
