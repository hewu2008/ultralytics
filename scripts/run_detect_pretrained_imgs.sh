#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Detect /data/4T-1/hewu/dataset/yolo/imgs with the pretrained robobin2026 weights (best.pt).
# Pass --model .../weights/best.onnx to run the exported end2end graph instead.
python "$REPO_DIR/tools/detect_yolo_imgs.py" \
    --model "$REPO_DIR/runs/segment/ultralytics/yolo26/yolo-seg-finetune-5/weights/best.pt" \
    --source /data/4T-1/hewu/dataset/hewu2008/parts-seg-v2-20261017/test \
    --out "$REPO_DIR/runs/segment/ultralytics/yolo26/yolo-seg-finetune-5/detect-imgs" \
    --conf 0.70 \
    --device 0