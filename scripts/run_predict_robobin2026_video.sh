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

# v4 weights on the 20260924 capture (16 clips, 9313 frames) that v4 contributed to train and test.
python "$REPO_DIR/tools/predict_robobin2026_video.py" \
    --model "$REPO_DIR/runs/segment/ultralytics/yolo26/robobin2026-v4-seg/weights/best.pt" \
    --source /data/4T-2/dataset/robobin2026/20260924/videos \
    --out "$REPO_DIR/runs/segment/robobin2026-v4-seg-video-pred-20260924" \
    --device 0
