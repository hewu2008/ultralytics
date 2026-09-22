# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert robobin2026-v1 into the standard COCO layout.

The source is a Roboflow export: one ``_annotations.coco.json`` sitting next to the images inside
each split directory, plus a non-standard ``score`` field on every annotation. This tool
restructures it into

    <out>/annotations/instances_<split>.json
    <out>/images/<split>/<file_name>

and normalizes the schema to the CVAT/pycocotools shape -- ``licenses``/``info`` added, ``score``
dropped, per-image ``license``/``flickr_url``/``coco_url``/``date_captured`` added -- so the result
matches robobin2026-v2 and can be re-imported by CVAT or read by pycocotools/mmdetection.

The annotations are rewritten as a stream rather than loaded, because the train split carries
~1.3 M annotations in a ~7 GB json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from coco_stream import iter_array

DEFAULT_DATA_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v1")
ANNOTATION_NAME = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")
INFO = {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}
# Field order mirrors a CVAT COCO export so a diff against robobin2026-v2 stays readable.
IMAGE_FIELDS = ("id", "width", "height", "file_name")
ANNOTATION_FIELDS = ("id", "image_id", "category_id", "area", "bbox", "iscrowd", "segmentation")
PROGRESS_EVERY = 5000


def dump(record: dict) -> str:
    """Serialize one COCO record compactly so the multi-GB output stays near its source size."""
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


def copy_images(src_dir: Path, dst_dir: Path, file_names: list[str]) -> tuple[int, int, int]:
    """Copy referenced images into the standard layout, reusing any that are already there."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = reused = missing = 0
    for name in file_names:
        src, dst = src_dir / name, dst_dir / name
        if not src.is_file():
            missing += 1
            continue
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            reused += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
        if copied % PROGRESS_EVERY == 0:
            print(f"    copied {copied} images...", flush=True)
    return copied, reused, missing


def write_split(src_dir: Path, out_dir: Path, split: str) -> tuple[int, int, Path]:
    """Write ``annotations/instances_<split>.json`` and copy the images it references."""
    anno = src_dir / ANNOTATION_NAME
    if not anno.is_file():
        raise FileNotFoundError(anno)

    images = list(iter_array(anno, "images"))
    copied, reused, missing = copy_images(src_dir, out_dir / "images" / split, [im["file_name"] for im in images])
    print(f"  {split}: {len(images)} images referenced, copied={copied} reused={reused} missing={missing}", flush=True)

    out_json = out_dir / "annotations" / f"instances_{split}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'"licenses":[],\n"info":{dump(INFO)},\n')

        categories = [("" if i == 0 else ",") + "\n" + dump(c) for i, c in enumerate(iter_array(anno, "categories"))]
        f.write('"categories":[')
        f.writelines(categories)
        f.write("\n],\n")

        f.write('"images":[')
        for i, image in enumerate(images):
            record = {k: image[k] for k in IMAGE_FIELDS}
            record.update(license=0, flickr_url="", coco_url="", date_captured=0)
            f.write(("" if i == 0 else ",") + "\n" + dump(record))
        f.write("\n],\n")

        annotations = 0
        f.write('"annotations":[')
        for annotation in iter_array(anno, "annotations"):
            f.write(("" if annotations == 0 else ",") + "\n" + dump({k: annotation[k] for k in ANNOTATION_FIELDS}))
            annotations += 1
        f.write("\n]\n}\n")
    return len(images), annotations, out_json


def main() -> None:
    """Parse arguments and convert the requested splits."""
    parser = argparse.ArgumentParser(description="Convert robobin2026-v1 to the standard COCO layout")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="source dataset root")
    parser.add_argument("--out", type=Path, default=None, help="output root, defaults to <data-root>-coco")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS, help="splits to convert")
    args = parser.parse_args()

    out_dir = args.out or args.data_root.with_name(f"{args.data_root.name}-coco")
    print(f">>> {args.data_root} -> {out_dir}", flush=True)
    for split in args.splits:
        n_images, n_annotations, out_json = write_split(args.data_root / split, out_dir, split)
        print(f"  {split}: wrote {n_annotations} annotations / {n_images} images to {out_json}", flush=True)


if __name__ == "__main__":
    main()
