# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Merge the 20260924 capture into robobin2026-v4 as a 0.9/0.1 train/test split.

``20260924`` is one X-AnyLabeling COCO export: 297 frames sampled at 1 fps from 15 videos, so the
neighbouring frames of one video are near-duplicates. Its vocabulary is v4's plus an unused id 0
``_background_``, and the only category it uses is ``cardboard_box``, v4's id 1, so nothing needs
remapping. The frames are shuffled with a fixed seed and cut 0.9/0.1 into v4's train and test.

Each target split takes three additions: the images, the annotations, and the YOLO labels the export
never carried. The labels come from ``convert_coco`` over a temporary json holding only the new
records, so polygon handling stays in the single place that already implements it.

v4's json is rewritten rather than patched in place. A new image record has to land inside the
``images`` array and that array sits in front of the multi-GB ``annotations`` array, which truncate
and append cannot reach. The existing records are streamed out through the same ``dump`` that wrote
them, so they come back byte-identical, and the new records continue past the highest existing id.
"""

from __future__ import annotations

import argparse
import json
import operator
import random
import shutil
from pathlib import Path

from coco_stream import iter_array
from convert_robobin2026_to_coco import ANNOTATION_FIELDS, IMAGE_FIELDS, INFO, copy_images, dump

from ultralytics.data.converter import convert_coco

DEFAULT_SRC = Path("/data/4T-2/dataset/robobin2026/20260924")
DEFAULT_OUT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v4")
SOURCE_ANNO = Path("annotations/coco_instance_segmentation.json")
SOURCE_IMAGES = "images"
TRAIN_SPLIT, TEST_SPLIT = "train", "test"
TEST_RATIO = 0.1
SEED = 0
# convert_coco picks the output directory with increment_path and appends to label files, so it is
# given a staging directory that is rebuilt from scratch on every run.
STAGE_NAME = ".merge_prep"


def load_source(src: Path) -> tuple[list[dict], dict[int, list[dict]], dict[int, str]]:
    """Read the export's images, its annotations grouped by image id, and its category names by id."""
    anno = src / SOURCE_ANNO
    images = list(iter_array(anno, "images"))
    if len({image["id"] for image in images}) != len(images):
        raise ValueError(f"{anno}: duplicate image ids")

    known = {image["id"] for image in images}
    grouped: dict[int, list[dict]] = {}
    for annotation in iter_array(anno, "annotations"):
        if annotation["image_id"] not in known:
            raise ValueError(f"{anno}: annotation references undeclared image id {annotation['image_id']}")
        grouped.setdefault(annotation["image_id"], []).append(annotation)
    return images, grouped, {c["id"]: c["name"] for c in iter_array(anno, "categories")}


def check_categories(used: set[int], source_names: dict[int, str], target_json: Path) -> None:
    """Fail when the export uses a category that v4 does not declare under the same name."""
    target_names = {c["id"]: c["name"] for c in iter_array(target_json, "categories")}
    for category_id in sorted(used):
        if category_id not in source_names:
            raise ValueError(f"annotation uses category {category_id}, which the export never declares")
        if source_names[category_id] != target_names.get(category_id):
            raise ValueError(
                f"category {category_id}: export {source_names[category_id]!r} vs v4 {target_names.get(category_id)!r}"
            )


def split_images(images: list[dict], seed: int) -> dict[str, list[dict]]:
    """Shuffle the frames and cut them 0.9/0.1, ordering each side by file name for a stable output."""
    by_name = operator.itemgetter("file_name")
    shuffled = sorted(images, key=by_name)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * TEST_RATIO)
    return {TRAIN_SPLIT: sorted(shuffled[n_test:], key=by_name), TEST_SPLIT: sorted(shuffled[:n_test], key=by_name)}


def copy_split_images(src: Path, out_dir: Path, split: str, images: list[dict]) -> None:
    """Copy one split's frames into v4, refusing names that are already taken."""
    names = [image["file_name"] for image in images]
    dst_dir = out_dir / "images" / split
    clashing = [name for name in names if (dst_dir / name).exists()]
    if clashing:
        raise FileExistsError(f"{split}: {len(clashing)} names already exist in {dst_dir}, e.g. {clashing[:3]}")
    copied, _, missing = copy_images(src / SOURCE_IMAGES, dst_dir, names)
    if missing:
        raise FileNotFoundError(f"{split}: {missing} of {len(names)} frames are missing under {src / SOURCE_IMAGES}")
    print(f"  {split}: copied {copied} images", flush=True)


def write_labels(out_dir: Path, stage: Path, parts: dict[str, list[dict]], grouped: dict[int, list[dict]]) -> None:
    """Convert the new records to YOLO polygons with convert_coco, then move them beside the images."""
    stage.mkdir(parents=True)
    for split, images in parts.items():
        records = [annotation for image in images for annotation in grouped[image["id"]]]
        with open(stage / f"instances_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"images": images, "annotations": records}, f)

    convert_coco(labels_dir=str(stage), save_dir=str(stage / "out"), use_segments=True, cls91to80=False)
    for split, images in parts.items():
        for image in images:
            txt = stage / "out" / "labels" / split / f"{Path(image['file_name']).stem}.txt"
            if not txt.is_file():  # convert_coco skips images whose annotations all get dropped
                raise FileNotFoundError(f"convert_coco produced no label for {image['file_name']}")
            shutil.move(str(txt), str(out_dir / "labels" / split / txt.name))
        print(f"  {split}: wrote {len(images)} labels", flush=True)


def rewrite_annotations(
    src_json: Path, tmp_json: Path, new_images: list[dict], new_annotations: list[dict]
) -> tuple[int, int]:
    """Stream ``src_json`` into ``tmp_json`` with the new records appended to both arrays."""
    n_images = n_annotations = 0
    next_image_id = next_annotation_id = 1
    mapping: dict[int, int] = {}

    with open(tmp_json, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'"licenses":[],\n"info":{dump(INFO)},\n')

        f.write('"categories":[')
        f.writelines(
            ("" if i == 0 else ",") + "\n" + dump(category)
            for i, category in enumerate(iter_array(src_json, "categories"))
        )
        f.write("\n],\n")

        f.write('"images":[')
        for image in iter_array(src_json, "images"):
            f.write(("" if n_images == 0 else ",") + "\n" + dump(image))
            next_image_id = max(next_image_id, image["id"] + 1)
            n_images += 1
        for image in new_images:
            record = {k: image[k] for k in IMAGE_FIELDS}
            record.update(id=next_image_id, license=0, flickr_url="", coco_url="", date_captured=0)
            f.write(("" if n_images == 0 else ",") + "\n" + dump(record))
            mapping[image["id"]] = next_image_id
            next_image_id += 1
            n_images += 1
        f.write("\n],\n")

        f.write('"annotations":[')
        for annotation in iter_array(src_json, "annotations"):
            f.write(("" if n_annotations == 0 else ",") + "\n" + dump(annotation))
            next_annotation_id = max(next_annotation_id, annotation["id"] + 1)
            n_annotations += 1
        for annotation in new_annotations:
            record = {k: annotation[k] for k in ANNOTATION_FIELDS}
            record["id"] = next_annotation_id
            record["image_id"] = mapping[annotation["image_id"]]
            f.write(("" if n_annotations == 0 else ",") + "\n" + dump(record))
            next_annotation_id += 1
            n_annotations += 1
        f.write("\n]\n}\n")
    return n_images, n_annotations


def main() -> None:
    """Parse arguments and merge the capture into the requested target."""
    parser = argparse.ArgumentParser(description="Merge the 20260924 capture into robobin2026-v4")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="20260924 export root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="v4 root to append to")
    parser.add_argument("--seed", type=int, default=SEED, help="shuffle seed for the 0.9/0.1 cut")
    args = parser.parse_args()

    images, grouped, source_names = load_source(args.src)
    parts = split_images(images, args.seed)
    print(
        f">>> {args.src} -> {args.out}\n"
        f">>> {len(parts[TRAIN_SPLIT])} train / {len(parts[TEST_SPLIT])} test frames (seed {args.seed})",
        flush=True,
    )

    used = {annotation["category_id"] for annotations in grouped.values() for annotation in annotations}
    check_categories(used, source_names, args.out / "annotations" / f"instances_{TRAIN_SPLIT}.json")

    stage = args.out / STAGE_NAME
    shutil.rmtree(stage, ignore_errors=True)
    try:
        for split, split_images_ in parts.items():
            copy_split_images(args.src, args.out, split, split_images_)
        write_labels(args.out, stage, parts, grouped)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    for split, split_images_ in parts.items():
        json_path = args.out / "annotations" / f"instances_{split}.json"
        tmp_json = json_path.with_name(f"{json_path.name}.tmp")
        new_annotations = [annotation for image in split_images_ for annotation in grouped[image["id"]]]
        n_images, n_annotations = rewrite_annotations(json_path, tmp_json, split_images_, new_annotations)
        tmp_json.replace(json_path)
        print(f"  {split}: {n_images} images, {n_annotations} annotations -> {json_path}", flush=True)


if __name__ == "__main__":
    main()
