#!/bin/bash
set -e

# Rewrite robobin2026-v1 in place: drop the duplicated test split, merge valid into train,
# remove unlabeled and byte-identical duplicate images, then draw a stratified 10000-image val.
# Pass --dry-run through to preview the plan without touching any file.
conda run -n yolo python tools/rewrite_robobin2026_v1_split.py \
  /data/4T-2/dataset/robobin2026/robobin2026-v1 \
  --val-size 10000 \
  "$@"