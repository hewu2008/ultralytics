#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Ultralytics lives in the 'yolo' env, so activate it rather than relying on an interactive shell.
source /home/jszn/miniconda3/etc/profile.d/conda.sh
conda activate yolo

# Pre-annotates the parts-seg-v4 raw images with the yolo-seg-finetune-6 checkpoint and writes them
# as the flat NNNN dataset that X-AnyLabeling opens directly: a renamed copy of each image plus its
# JSON and Ultralytics label text. The raw images stay untouched in the source, so this batch runs
# 0001-0032, the numbering a fresh parts-seg-v4 set starts from.
python "$REPO_DIR/tools/detect_yolo_imgs.py" \
    --model /data/4T-1/hewu/alg-product/ultralytics/runs/segment/ultralytics/yolo26/yolo-seg-finetune-6/weights/best.pt \
    --source /data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020/images \
    --out /data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020/auto-annotated \
    --start-index 1 \
    --conf 0.25 \
    --device 0
