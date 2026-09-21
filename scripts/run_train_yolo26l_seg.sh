#!/bin/bash
set -e

yolo train \
  cfg=/data/4T-1/hewu/alg-product/ultralytics/assets/yolo26l-seg-train.yaml \
  model=yolo26l-seg.yaml \
  data=/data/4T-1/hewu/alg-product/ultralytics/assets/coco2017-seg.yaml \
  device=0,1 \
  "$@"