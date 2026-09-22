# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert robobin2026 into the standard COCO layout.

Two source layouts are recognized from the files present under ``--data-root``:

- **Roboflow export (v1)**: a ``_annotations.coco.json`` sitting next to the images inside each
  ``train``/``valid``/``test`` directory, plus a non-standard ``score`` field on every annotation.
  The export already defines the splits, so they are used as-is.
- **CVAT export (v2)**: a single ``annotations/instances_default.json`` holding every image, whose
  files live flat in ``frames_rename``. The source carries no splits, so the images are shuffled
  and partitioned into train/valid/test following ``SPLIT_RATIOS``.

Both are restructured into

    <out>/annotations/instances_<split>.json
    <out>/images/<split>/<file_name>

with the schema normalized to the CVAT/pycocotools shape -- ``licenses``/``info`` added, extra
fields dropped, per-image ``license``/``flickr_url``/``coco_url``/``date_captured`` added -- so the
result can be re-imported by CVAT or read by pycocotools/mmdetection.

The annotations are rewritten as a stream rather than loaded, because the v1 train split carries
~1.3 M annotations in a ~7 GB json.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import NamedTuple

from coco_stream import iter_array

DEFAULT_DATA_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v1")
ANNOTATION_NAME = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")
# A CVAT export keeps every image in one json beside a flat image directory instead of per-split dirs.
CVAT_ANNOTATION = Path("annotations/instances_default.json")
CVAT_IMAGE_DIR = "frames_rename"
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
SPLIT_SEED = 0
INFO = {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}
# Field order mirrors a CVAT COCO export so a diff against robobin2026-v2 stays readable.
IMAGE_FIELDS = ("id", "width", "height", "file_name")
ANNOTATION_FIELDS = ("id", "image_id", "category_id", "area", "bbox", "iscrowd", "segmentation")
PROGRESS_EVERY = 5000


class SplitPlan(NamedTuple):
    """Where one target split takes its images and annotations from."""

    anno: Path  # source json holding the categories, images and annotations
    image_dir: Path  # directory the referenced file_name values resolve against
    images: list[dict]  # image records assigned to this target split


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


def split_counts(total: int, splits: list[str]) -> list[int]:
    """Cut ``total`` items into per-split sizes following ``SPLIT_RATIOS``, the last split taking the rest."""
    weights = [SPLIT_RATIOS[split] for split in splits]
    counts = [round(total * weight / sum(weights)) for weight in weights[:-1]]
    return [*counts, total - sum(counts)]


def plan_splits(root: Path, splits: list[str], seed: int) -> dict[str, SplitPlan]:
    """Resolve every target split, detecting the Roboflow or CVAT source layout under ``root``."""
    cvat_anno = root / CVAT_ANNOTATION
    if not cvat_anno.is_file():
        plans = {}
        for split in splits:
            anno = root / split / ANNOTATION_NAME
            if not anno.is_file():
                raise FileNotFoundError(anno)
            plans[split] = SplitPlan(anno, root / split, list(iter_array(anno, "images")))
        return plans

    images = sorted(iter_array(cvat_anno, "images"), key=lambda image: image["id"])
    random.Random(seed).shuffle(images)
    plans, start = {}, 0
    for split, count in zip(splits, split_counts(len(images), splits)):
        plans[split] = SplitPlan(cvat_anno, root / CVAT_IMAGE_DIR, images[start : start + count])
        start += count
    return plans


def write_split(plan: SplitPlan, out_dir: Path, split: str) -> tuple[int, int, Path]:
    """Write ``annotations/instances_<split>.json`` and copy the images it references."""
    copied, reused, missing = copy_images(
        plan.image_dir, out_dir / "images" / split, [image["file_name"] for image in plan.images]
    )
    print(
        f"  {split}: {len(plan.images)} images referenced, copied={copied} reused={reused} missing={missing}",
        flush=True,
    )

    image_ids = {image["id"] for image in plan.images}
    out_json = out_dir / "annotations" / f"instances_{split}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'"licenses":[],\n"info":{dump(INFO)},\n')

        categories = [
            ("" if i == 0 else ",") + "\n" + dump(c) for i, c in enumerate(iter_array(plan.anno, "categories"))
        ]
        f.write('"categories":[')
        f.writelines(categories)
        f.write("\n],\n")

        f.write('"images":[')
        for i, image in enumerate(plan.images):
            record = {k: image[k] for k in IMAGE_FIELDS}
            record.update(license=0, flickr_url="", coco_url="", date_captured=0)
            f.write(("" if i == 0 else ",") + "\n" + dump(record))
        f.write("\n],\n")

        annotations = 0
        f.write('"annotations":[')
        for annotation in iter_array(plan.anno, "annotations"):
            if annotation["image_id"] not in image_ids:
                continue
            f.write(("" if annotations == 0 else ",") + "\n" + dump({k: annotation[k] for k in ANNOTATION_FIELDS}))
            annotations += 1
        f.write("\n]\n}\n")
    return len(plan.images), annotations, out_json


def main() -> None:
    """Parse arguments and convert the requested splits."""
    parser = argparse.ArgumentParser(description="Convert robobin2026 to the standard COCO layout")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="source dataset root")
    parser.add_argument("--out", type=Path, default=None, help="output root, defaults to <data-root>-coco")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS, help="splits to write")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="shuffle seed, only used for the CVAT layout")
    args = parser.parse_args()

    out_dir = args.out or args.data_root.with_name(f"{args.data_root.name}-coco")
    print(f">>> {args.data_root} -> {out_dir}", flush=True)
    for split, plan in plan_splits(args.data_root, args.splits, args.seed).items():
        n_images, n_annotations, out_json = write_split(plan, out_dir, split)
        print(f"  {split}: wrote {n_annotations} annotations / {n_images} images to {out_json}", flush=True)


if __name__ == "__main__":
    main()
