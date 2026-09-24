#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

python "$REPO_DIR/tools/predict_robobin2026_video.py" \
    --model "$REPO_DIR/runs/segment/ultralytics/yolo26/robobin2026-v1-seg-finetune-2/weights/best.pt" \
    --source "$REPO_DIR/assets/video" \
    --out "$REPO_DIR/runs/segment/robobin2026-v1-seg-video-pred" \
    --device 0

python "$REPO_DIR/tools/predict_robobin2026_video.py" \
    --model "$REPO_DIR/runs/segment/ultralytics/yolo26/robobin2026-v3-seg/weights/best.pt" \
    --source "$REPO_DIR/assets/video" \
    --out "$REPO_DIR/runs/segment/robobin2026-v3-seg-video-pred" \
    --device 0
