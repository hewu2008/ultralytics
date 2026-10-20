#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Merge the parts-seg-v4 numbered triples (772 png/json/txt) into the parts-seg-v5 finetune dataset.
# Every v4 image follows the split it already has in v4/finetune: 694 into v5 train, 78 into v5 val,
# with v5's own 6324/676 members left untouched. images/ takes images only, so the json annotations
# stay in v4; tools/merge_parts_seg_finetune.py writes v5/finetune/data.yaml from v4's class names and
# verifies afterwards that no existing member moved and every image still has its label.
python "$REPO_DIR/tools/merge_parts_seg_finetune.py" \
    --src /data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020 \
    --dst /data/4T-1/hewu/dataset/hewu2008/parts-seg-v5-20261020
