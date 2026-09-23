#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

yolo train \
  name=robobin2026-v3-seg \
  cfg=assets/robobin2026-v3-seg-finetune.yaml \
  model=assets/yolo26l-seg.pt \
  data=assets/robobin2026-v3-seg.yaml \
  device=0,1
