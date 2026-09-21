#!/bin/bash
set -e

WEIGHTS_DIR=/data/4T-1/hewu/alg-product/ultralytics/assets
MODEL="$WEIGHTS_DIR/yolo26l-seg.pt"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ -f "$MODEL" ]; then
  echo ">>> $MODEL already exists, nothing to do"
  exit 0
fi

echo ">>> downloading yolo26l-seg.pt from $HF_ENDPOINT ..."
mkdir -p "$WEIGHTS_DIR"
wget -c -O "$MODEL" "$HF_ENDPOINT/Ultralytics/YOLO26/resolve/main/yolo26l-seg.pt"