#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

yolo train \
  cfg=assets/robobin2026-v1-seg-finetune.yaml \
  model=assets/yolo26l-seg.pt \
  data=assets/robobin2026-v1-seg.yaml \
  device=0,1
