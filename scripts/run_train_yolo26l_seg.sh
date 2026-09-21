#!/bin/bash
# YOLO26l-seg 训练启动脚本（COCO2017 instance segmentation，本地数据集）
# 前置步骤：
#   scripts/run_download_yolo26l_seg_weights.sh  下载 $MODEL
#   scripts/run_prepare_coco2017_seg.sh          准备 images/ 与 labels/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_YAML="$REPO_DIR/assets/coco2017-seg.yaml"
WEIGHTS_DIR=/data/4T-1/hewu/alg-product/ultralytics/weights
MODEL="$WEIGHTS_DIR/yolo26l-seg.pt"
DEVICE=0,1

source /home/jszn/miniconda3/etc/profile.d/conda.sh
conda activate yolo

if [ ! -f "$MODEL" ]; then
  echo ">>> $MODEL not found, run scripts/run_download_yolo26l_seg_weights.sh first" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 参考命令中 semseg_loss / o2m / muon_w / sgd_w / cls_w / stride_ratio /
# hungarian / topk 在当前版本(8.4.157)不是合法参数，half / int8 已被 quantize
# 取代，均已剔除；device 由 0,1,2,4 改为本机实际的 0,1。
# ---------------------------------------------------------------------------
yolo train \
  model="$MODEL" \
  data="$DATA_YAML" \
  task=segment \
  project=ultralytics/yolo26 \
  name=yolo26l-seg \
  classes=None \
  epochs=60 \
  batch=128 \
  save=true \
  device="$DEVICE" \
  exist_ok=false \
  optimizer=MuSGD \
  verbose=true \
  multi_scale=false \
  mask_ratio=1 \
  split=val \
  save_json=true \
  conf=None \
  plots=true \
  vid_stride=1 \
  stream_buffer=false \
  visualize=false \
  augment=false \
  agnostic_nms=false \
  retina_masks=false \
  embed=None \
  show=false \
  save_frames=false \
  save_txt=false \
  save_conf=false \
  save_crop=false \
  show_labels=true \
  show_conf=true \
  show_boxes=true \
  line_width=None \
  format=torchscript \
  keras=false \
  optimize=false \
  dynamic=false \
  simplify=true \
  opset=None \
  workspace=None \
  nms=false \
  lr0=0.00038 \
  lrf=0.88219 \
  momentum=0.94751 \
  weight_decay=0.00027 \
  warmup_epochs=1 \
  warmup_momentum=0.54064 \
  warmup_bias_lr=0.05684 \
  box=9.83241 \
  cls=0.64896 \
  dfl=0.95824 \
  hsv_h=0.01315 \
  hsv_s=0.35348 \
  hsv_v=0.19383 \
  translate=0.27484 \
  scale=0.95 \
  fliplr=0.30393 \
  mixup=0.42713 \
  copy_paste=0.40413 \
  cfg=None \
  tracker=botsort.yaml