#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

python "$REPO_DIR/tools/predict_robobin2026_video.py" \
    --source "$REPO_DIR/assets/video" \
    --out "$REPO_DIR/runs/segment/robobin2026-video-pred" \
    --device 0