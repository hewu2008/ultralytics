#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

yolo train \
  cfg=assets/robobin2026-v1-seg-finetune.yaml \
  model=assets/yolo26l-seg.pt \
  data=assets/robobin2026-v1-seg.yaml \
  cls_pw=0.5 \
  device=0
