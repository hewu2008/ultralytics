#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Data prep runs in the repo's env, so activate it rather than relying on an interactive shell.
source /home/jszn/miniconda3/etc/profile.d/conda.sh
conda activate yolo

# Merges the parts-seg-v2 images (0001-0294) into parts-seg-v3. The v3 set's own 448 triples are
# renumbered to 0295-0742 first, because the two sets overlap on the numbers 16-294 even though
# neither side zero-pads the same way. The v2 source directory is left untouched.
python "$REPO_DIR/tools/merge_parts_seg_images.py" \
    --src /data/4T-1/hewu/dataset/hewu2008/parts-seg-v2-20261017/images \
    --dst /data/4T-1/hewu/dataset/hewu2008/parts-seg-v3-20260117/images \
    --start-index 295
