# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build robobin2026-v4 from robobin2026-v3 as a self-contained standard COCO dataset.

v3 is a symlink farm: every file under ``images/<split>`` and ``labels/<split>`` points back into
robobin2026-v1 or robobin2026-v2, so the dataset is only readable while all three stay in place and
moving or deleting either source silently breaks it. v4 replaces those links with real copies and
adds the COCO annotations v3 never carried.

Both sources already use the standard COCO layout, so the annotations are produced by filtering and
merging their json instead of re-deriving polygons from the YOLO labels:

- **v1** covers 107804 of the images with a five-class vocabulary.
- **v2** covers the remaining 1111 (the ``20260917_lab_*`` capture) with ``cardboard_box`` alone,
  which is v1's category 1, so the two vocabularies compose without remapping.

The ``.npy`` decode caches and ``.cache`` label caches sitting in v3 beside the real payload are not
carried over; Ultralytics rebuilds them on the next run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

from coco_stream import iter_array
from convert_robobin2026_to_coco import ANNOTATION_FIELDS, IMAGE_FIELDS, INFO, copy_images, dump

DEFAULT_SRC = Path("/data/4T-2/dataset/robobin2026/robobin2026-v3")
# Sibling roots searched under --src's parent, in the order their images are numbered in the output.
SOURCES = ("robobin2026-v1", "robobin2026-v2")
SPLITS = ("train", "valid", "test")


class Source(NamedTuple):
    """One COCO source contributing images and annotations to a target split."""

    anno: Path  # annotations/instances_<split>.json of the source
    images: list[dict]  # image records this split takes from the source, ordered by file_name


def plan_split(root: Path, split: str, source_roots: list[Path]) -> tuple[list[Source], list[str]]:
    """Assign every real file of one split to a source, failing loudly when one is unclaimed or doubly claimed."""
    names = {path.name for path in (root / "images" / split).iterdir() if path.is_symlink()}
    claimed, sources = set(), []
    for source_root in source_roots:
        anno = source_root / "annotations" / f"instances_{split}.json"
        if not anno.is_file():
            continue
        images = sorted(
            (image for image in iter_array(anno, "images") if image["file_name"] in names),
            key=lambda image: image["file_name"],
        )
        overlap = claimed & {image["file_name"] for image in images}
        if overlap:
            raise ValueError(f"{split}: {len(overlap)} images claimed by two sources, e.g. {sorted(overlap)[:3]}")
        claimed |= {image["file_name"] for image in images}
        sources.append(Source(anno, images))
    missing = names - claimed
    if missing:
        raise ValueError(f"{split}: {len(missing)} images have no source annotation, e.g. {sorted(missing)[:3]}")
    return sources, sorted(names)


def read_categories(source_roots: list[Path]) -> list[dict]:
    """Merge the category records of every source json into the single dataset vocabulary.

    COCO declares ``categories`` inside each json, but a dataset split from one annotation pool has
    to repeat the same list in every split. Deriving it per split would make the test split, whose
    images all come from the single-class v2, advertise a vocabulary of its own.
    """
    records: dict[int, dict] = {}
    for source_root in source_roots:
        for anno in sorted((source_root / "annotations").glob("instances_*.json")):
            for category in iter_array(anno, "categories"):
                known = records.setdefault(category["id"], category)
                if known["name"] != category["name"]:
                    raise ValueError(
                        f"category {category['id']}: {known['name']!r} vs {category['name']!r} in {anno}"
                    )
    return [records[i] for i in sorted(records)]


def write_annotations(sources: list[Source], categories: list[dict], out_json: Path, split: str) -> tuple[int, int]:
    """Write ``instances_<split>.json``, renumbering image and annotation ids across every source."""
    image_ids = []
    records, next_id = [], 1
    for source in sources:
        mapping = {}
        for image in source.images:
            mapping[image["id"]] = next_id
            record = {k: image[k] for k in IMAGE_FIELDS}
            record.update(id=next_id, license=0, flickr_url="", coco_url="", date_captured=0)
            records.append(record)
            next_id += 1
        image_ids.append(mapping)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    annotations = 0
    with open(out_json, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'"licenses":[],\n"info":{dump(INFO)},\n')
        f.write('"categories":[')
        f.writelines(("" if i == 0 else ",") + "\n" + dump(c) for i, c in enumerate(categories))
        f.write("\n],\n")

        f.write('"images":[')
        for i, record in enumerate(records):
            f.write(("" if i == 0 else ",") + "\n" + dump(record))
        f.write("\n],\n")

        f.write('"annotations":[')
        for source, mapping in zip(sources, image_ids):
            for annotation in iter_array(source.anno, "annotations"):
                image_id = mapping.get(annotation["image_id"])
                if image_id is None:
                    continue
                record = {k: annotation[k] for k in ANNOTATION_FIELDS}
                record["id"] = annotations + 1
                record["image_id"] = image_id
                f.write(("" if annotations == 0 else ",") + "\n" + dump(record))
                annotations += 1
        f.write("\n]\n}\n")
    print(f"  {split}: wrote {len(records)} images, {annotations} annotations to {out_json}", flush=True)
    return len(records), annotations


def write_split(
    root: Path, out_dir: Path, split: str, source_roots: list[Path], categories: list[dict]
) -> tuple[int, int]:
    """Rebuild one split of v4: dereferenced images and labels plus the merged COCO annotations."""
    sources, names = plan_split(root, split, source_roots)
    _, _, missing_images = copy_images(root / "images" / split, out_dir / "images" / split, names)
    _, _, missing_labels = copy_images(
        root / "labels" / split, out_dir / "labels" / split, [f"{Path(name).stem}.txt" for name in names]
    )
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"{split}: {missing_images} images and {missing_labels} labels are unreadable in {root}"
        )
    return write_annotations(sources, categories, out_dir / "annotations" / f"instances_{split}.json", split)


def main() -> None:
    """Parse arguments and build the requested splits."""
    parser = argparse.ArgumentParser(description="Build robobin2026-v4 from robobin2026-v3")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="v3 root holding the symlink farm")
    parser.add_argument("--out", type=Path, default=None, help="output root, defaults to --src with v3 -> v4")
    parser.add_argument(
        "--sources", nargs="+", type=Path, default=None, help="COCO roots, default v1 and v2 beside --src"
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS, help="splits to write")
    args = parser.parse_args()

    source_roots = args.sources or [args.src.with_name(name) for name in SOURCES]
    out_dir = args.out or args.src.with_name(args.src.name.replace("v3", "v4"))
    print(f">>> {args.src} -> {out_dir}\n>>> annotation sources: {[str(root) for root in source_roots]}", flush=True)

    categories = read_categories(source_roots)
    print(f">>> categories: {[category['name'] for category in categories]}", flush=True)

    images = annotations = 0
    for split in args.splits:
        split_images, split_annotations = write_split(args.src, out_dir, split, source_roots, categories)
        images += split_images
        annotations += split_annotations
    print(f">>> done: {images} images, {annotations} annotations", flush=True)


if __name__ == "__main__":
    main()
