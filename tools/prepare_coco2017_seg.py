# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare a raw COCO2017 dataset for YOLO segmentation training.

Turns the original COCO layout (annotations/*.json + train2017/ + val2017/) into the
layout Ultralytics expects (images/{train,val}2017 + labels/{train,val}2017).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco


def main(data_dir: Path) -> None:
    """Build images/ and labels/ inside the dataset root, skipping work that is already done."""
    # Images must live under images/ so img2label_paths can map them onto labels/
    for split in ("train2017", "val2017"):
        src, dst = data_dir / split, data_dir / "images" / split
        if not dst.exists() and src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

    labels_dir = data_dir / "labels"
    if labels_dir.exists():
        print(f">>> labels already present in {labels_dir}, skipping conversion")
        return

    # Only instances_*.json is passed to convert_coco; captions/keypoints annotations carry no usable segment labels
    stage = data_dir / ".seg_prep"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "anno").mkdir(parents=True)
    for json_file in sorted((data_dir / "annotations").glob("instances_*.json")):
        (stage / "anno" / json_file.name).symlink_to(json_file)

    convert_coco(labels_dir=str(stage / "anno"), save_dir=str(stage / "out"), use_segments=True)
    shutil.move(str(stage / "out" / "labels"), str(labels_dir))
    shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare COCO2017 YOLO segmentation labels")
    parser.add_argument("data_dir", type=Path, help="dataset root dir, i.e. /data/4T-2/dataset/coco2017")
    main(parser.parse_args().data_dir)