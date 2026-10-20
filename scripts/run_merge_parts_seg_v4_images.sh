#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Data prep runs in the repo's env, so activate it rather than relying on an interactive shell.
source /home/jszn/miniconda3/etc/profile.d/conda.sh
conda activate yolo

# Merges the parts-seg-v3 images (0001-0742) into parts-seg-v4. The v4 set's own 32 triples are
# renumbered to 0743-0774 first, so the older v3 numbering stays untouched and the two sets cannot
# collide. The v3 source directory is left as-is.
python "$REPO_DIR/tools/merge_parts_seg_images.py" \
    --src /data/4T-1/hewu/dataset/hewu2008/parts-seg-v3-20260117/images \
    --dst /data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020/images \
    --start-index 743
