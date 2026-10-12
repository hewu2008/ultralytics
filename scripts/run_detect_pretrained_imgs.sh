#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Detect /data/4T-1/hewu/dataset/yolo/imgs with the pretrained robobin2026 weights (best.pt).
# Pass --model .../weights/best.onnx to run the exported end2end graph instead.
python "$REPO_DIR/tools/detect_yolo_imgs.py" \
    --model "$REPO_DIR/runs/pretrained/weights/best.pt" \
    --source /data/4T-1/hewu/dataset/yolo/imgs \
    --out "$REPO_DIR/runs/pretrained/detect-imgs" \
    --conf 0.8 \
    --device 0