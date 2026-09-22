# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare the robobin2026-v1 COCO export for YOLO segmentation training.

Turns ``annotations/instances_<split>.json`` into the ``labels/<split>`` tree Ultralytics expects
beside ``images/<split>``. ``cls91to80`` is kept off because robobin2026 defines its own five
classes, so the 91-to-80 COCO remap that prepare_coco2017_seg.py relies on would mislabel every
instance.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco

SPLITS = ("train", "valid", "test")


def main(data_dir: Path) -> None:
    """Build labels/ inside the dataset root, skipping work that is already done."""
    labels_dir = data_dir / "labels"
    if labels_dir.exists():
        print(f">>> labels already present in {labels_dir}, skipping conversion")
        return

    # convert_coco writes labels/<json stem minus 'instances_'>/, which is exactly images/<split>
    stage = data_dir / ".seg_prep"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "anno").mkdir(parents=True)
    for split in SPLITS:
        json_file = data_dir / "annotations" / f"instances_{split}.json"
        if not json_file.is_file():
            raise FileNotFoundError(json_file)
        (stage / "anno" / json_file.name).symlink_to(json_file)

    convert_coco(labels_dir=str(stage / "anno"), save_dir=str(stage / "out"), use_segments=True, cls91to80=False)
    shutil.move(str(stage / "out" / "labels"), str(labels_dir))
    shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare robobin2026-v1 YOLO segmentation labels")
    parser.add_argument("data_dir", type=Path, help="dataset root, i.e. /data/4T-2/dataset/robobin2026/robobin2026-v1")
    main(parser.parse_args().data_dir)
