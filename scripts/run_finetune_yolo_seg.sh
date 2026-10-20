#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Fine-tune the runs/pretrained/weights/best.pt checkpoint (output_parts_20260723_mixed_14) on the
# merged parts-seg-v4 set (772 triples: v3's 740 plus the 32 images captured 2026-10-20).
# tools/finetune_yolo_seg.py rebuilds the train/val symlink split + data.yaml on every run, then
# trains with the assets cfg. Val is every 1/--val-ratio-th file of the numbered order, so a re-run
# over a grown set keeps the images that were already val in val and stays comparable with earlier
# runs. Add --prepare-only (or run `yolo train ...` yourself) to build the split without training.
cd "$REPO_DIR"
python "$REPO_DIR/tools/finetune_yolo_seg.py" \
    --model "$REPO_DIR/runs/pretrained/weights/best.pt" \
    --source /data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020/images \
    --data /data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020/finetune \
    --cfg "$REPO_DIR/assets/yolo-seg-finetune.yaml" \
    --val-ratio 0.1 \
    --device 0