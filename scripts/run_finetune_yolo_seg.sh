#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Fine-tune the runs/pretrained robobin2026 segmentation weights on /data/4T-1/hewu/dataset/yolo/imgs.
# tools/finetune_yolo_seg.py builds the train/val symlink split + data.yaml, then trains with the
# assets cfg. Add --prepare-only (or run `yolo train ...` yourself) to build the split without training.
cd "$REPO_DIR"
python "$REPO_DIR/tools/finetune_yolo_seg.py" \
    --model "$REPO_DIR/runs/pretrained/weights/best.pt" \
    --source /data/4T-1/hewu/dataset/yolo/imgs \
    --data /data/4T-1/hewu/dataset/yolo/finetune \
    --cfg "$REPO_DIR/assets/yolo-seg-finetune.yaml" \
    --device 0