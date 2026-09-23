# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Remove every zero-annotation image from a robobin2026 export, in place.

An image without ``labels/<split>/<stem>.txt`` is not skipped by Ultralytics: it is scanned as a
*background* image, so it enters training as a pure negative and, at validation time, every
prediction on it is counted as a false positive. In robobin2026-v2 that is 667 of 1778 images
(train 522/1422, valid 76/178, test 69/178) and they are not real backgrounds - the model detects
cardboard_box on 100% of them at a mean top confidence of 0.978. Measured on v2 valid, dropping the
76 background images moves the same model, on the same 204 ground-truth instances, from
box P 0.591 / R 0.773 / mAP50 0.513 to P 0.951 / R 0.775 / mAP50 0.843.

The plan is computed read-only first, so ``--dry-run`` reports the real outcome. Each removed image
is unlinked together with its ``.npy`` cache, and ``annotations/instances_<split>.json`` is rewritten
without the dropped images so the COCO files keep describing the tree. The images are checked against
those json files before anything is deleted: a removed image that carries annotations aborts the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")
DRY = False


def labeled_stems(root: Path, split: str) -> set[str]:
    """Stems of the split's images that carry at least one annotation row."""
    label_dir = root / "labels" / split
    if not label_dir.is_dir():
        return set()
    return {p.stem for p in label_dir.glob("*.txt") if p.stat().st_size and p.read_text(encoding="utf-8").strip()}


def build_plan(root: Path) -> dict[str, list[Path]]:
    """Collect, read-only, the images to delete per split."""
    plan = {}
    for split in SPLITS:
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            continue
        keep = labeled_stems(root, split)
        plan[split] = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT and p.stem not in keep)
    return plan


def check_annotations(root: Path, plan: dict[str, list[Path]]) -> None:
    """Abort if an image marked for deletion still carries annotations in its COCO json."""
    for split, paths in plan.items():
        json_file = root / "annotations" / f"instances_{split}.json"
        if not json_file.is_file():
            print(f"    {split}: no {json_file.name}, skipping the annotation check")
            continue
        data = json.loads(json_file.read_text())
        annotated = {a["image_id"] for a in data["annotations"]}
        gone = {p.stem for p in paths}
        conflicts = sorted(
            im["file_name"] for im in data["images"] if Path(im["file_name"]).stem in gone and im["id"] in annotated
        )
        if conflicts:
            raise RuntimeError(
                f"{split}: {len(conflicts)} images marked for deletion carry annotations, e.g. {conflicts[:5]}"
            )
        missing = sorted(gone - {Path(im["file_name"]).stem for im in data["images"]})
        print(f"    {split}: {len(gone)} to delete, {len(conflicts)} annotated, {len(missing)} absent from the json")


def remove_images(paths: list[Path]) -> None:
    """Unlink the images and their ``.npy`` caches."""
    for path in paths:
        for target in (path, path.with_suffix(".npy")):
            if target.exists() and not DRY:
                target.unlink()


def rebuild_coco(root: Path, plan: dict[str, list[Path]]) -> None:
    """Rewrite instances_<split>.json without the deleted images and their annotations."""
    for split, paths in plan.items():
        json_file = root / "annotations" / f"instances_{split}.json"
        if not paths or not json_file.is_file():
            continue
        data = json.loads(json_file.read_text())
        gone = {p.stem for p in paths}
        images = [im for im in data["images"] if Path(im["file_name"]).stem not in gone]
        kept = {im["id"] for im in images}
        annotations = [a for a in data["annotations"] if a["image_id"] in kept]
        if not DRY:
            json_file.write_text(json.dumps({**data, "images": images, "annotations": annotations}))
        print(
            f"    instances_{split}.json: images {len(data['images'])} -> {len(images)}, "
            f"annotations {len(data['annotations'])} -> {len(annotations)}"
        )


def report(root: Path) -> None:
    """Print the resulting per-split composition."""
    for split in SPLITS:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir():
            continue
        images = sum(1 for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        labels = len(list(label_dir.glob("*.txt"))) if label_dir.is_dir() else 0
        print(f"  [{split}] images={images} labels={labels}")


def main() -> None:
    """Parse arguments and clean the dataset."""
    global DRY
    parser = argparse.ArgumentParser(description="Remove zero-annotation images from a robobin2026 export")
    parser.add_argument("data_dir", type=Path, help="dataset root, e.g. /data/4T-2/dataset/robobin2026/robobin2026-v2")
    parser.add_argument("--dry-run", action="store_true", help="report the plan without touching any file")
    args = parser.parse_args()
    root: Path = args.data_dir
    DRY = args.dry_run
    print(f">>> {'DRY RUN — ' if DRY else ''}去除 {root} 的零标注图片\n")

    print("[1/3] 扫描零标注图片")
    plan = build_plan(root)
    for split, paths in plan.items():
        print(f"    {split}: {len(paths)} / {len(paths) + len(labeled_stems(root, split))} 张零标注")
    total = sum(len(p) for p in plan.values())
    if not total:
        print("\n>>> 没有零标注图片，无需处理")
        return

    print("\n[2/3] 校验 COCO json 一致性")
    check_annotations(root, plan)

    print("\n[3/3] 删除并重建 json")
    for paths in plan.values():
        remove_images(paths)
    rebuild_coco(root, plan)
    report(root)
    print("\n>>> DRY RUN 完成，未改动任何文件" if DRY else f"\n>>> 完成：删除 {total} 张图片")


if __name__ == "__main__":
    main()