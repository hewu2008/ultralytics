#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Fine-tune the runs/pretrained/weights/best.pt checkpoint (output_parts_20260723_mixed_14) on the
# merged parts-seg-v5 set. v5 is already a YOLO-format dataset, so no split is built here: its
# finetune/{images,labels}/{train,val} hold real files (7018 train / 754 val - v5's own 6324/676 plus
# 694/78 merged in from parts-seg-v4 by tools/merge_parts_seg_finetune.py), and yolo train reads v5's
# data.yaml directly, leaving both memberships untouched. These images carry no json annotations, so
# there is no flat numbered triple folder to point at and tools/finetune_yolo_seg.py does not apply.
# The hyperparameters come from assets/yolo-seg-finetune.yaml through cfg=, whose project/name/
# exist_ok place the run under runs/segment/ultralytics/yolo26/yolo-seg-finetune-<n>.
cd "$REPO_DIR"
yolo train \
    model="$REPO_DIR/runs/pretrained/weights/best.pt" \
    data=/data/4T-1/hewu/dataset/hewu2008/parts-seg-v5-20261020/finetune/data.yaml \
    cfg="$REPO_DIR/assets/yolo-seg-finetune.yaml" \
    device=0