#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

yolo train \
  cfg=assets/yolo26l-seg-train.yaml \
  model=yolo26l-seg.yaml \
  data=assets/coco2017-seg.yaml