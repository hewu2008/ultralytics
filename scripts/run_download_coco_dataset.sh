#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATA_DIR=/data/4T-2/dataset/coco2017
MODE=${1:-hf}

mkdir -p $DATA_DIR

case $MODE in
  wget)
    wget -P $DATA_DIR http://images.cocodataset.org/zips/train2017.zip
    wget -P $DATA_DIR http://images.cocodataset.org/zips/val2017.zip
    wget -P $DATA_DIR http://images.cocodataset.org/zips/test2017.zip
    ;;
  hf)
    hf download --repo-type dataset pcuenq/coco-2017-mirror --local-dir $DATA_DIR
    ;;
  *)
    echo "Usage: $0 [wget|hf]"
    exit 1
    ;;
esac
