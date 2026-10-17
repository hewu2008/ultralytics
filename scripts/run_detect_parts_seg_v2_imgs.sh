#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Ultralytics lives in the 'yolo' env, so activate it rather than relying on an interactive shell.
source /home/jszn/miniconda3/etc/profile.d/conda.sh
conda activate yolo

# Pre-annotates the parts-seg-v2 raw images with the yolo-seg-finetune-3 checkpoint and writes them
# as the flat NNNN dataset that X-AnyLabeling opens directly: a renamed copy of each image plus its
# JSON and Ultralytics label text. Numbering continues parts-seg-v1, whose imgs/ stops at 0244, so
# the batch runs 0245-0294 (numbered files first, then the Colour Viewer screenshot, which the
# numbered-first ordering places last).
python "$REPO_DIR/tools/detect_yolo_imgs.py" \
    --model /data/4T-1/hewu/alg-product/ultralytics/runs/segment/ultralytics/yolo26/yolo-seg-finetune-3/weights/best.pt \
    --source /data/4T-1/hewu/dataset/hewu2008/parts-seg-v3-20260117/images \
    --out /data/4T-1/hewu/dataset/hewu2008/parts-seg-v3-20260117/image-auto-annotated \
    --conf 0.25 \
    --device 0
