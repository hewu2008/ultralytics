#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR=/data/4T-2/dataset/coco2017

python "$REPO_DIR/tools/prepare_coco2017_seg.py" "$DATA_DIR"