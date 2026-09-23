# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Load a robobin2026-v1 split into FiftyOne for interactive label review.

The COCO export is imported as two parallel label fields: ``ground_truth_detections`` keeps the
boxes and ``ground_truth_segmentations`` keeps the original polygons as polylines, so mask quality
(the pallet / conveyor_belt boundary problem) can be reviewed next to the boxes in the App.

The import is cached as a named FiftyOne dataset, so a second run reuses it instead of re-parsing
the 240 MB annotation file; pass ``--overwrite`` to rebuild it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fiftyone as fo

DEFAULT_DATA_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v1")
DEFAULT_NAME = "robobin2026-v1-val"
# Requesting two label types makes the COCO importer emit flat `<field>_<type>` names rather than
# nesting them under a single document field. Each of those fields holds a Detections/Polylines
# label object, so the label itself sits one level deeper.
LABEL_FIELD = "ground_truth"
BOX_FIELD = f"{LABEL_FIELD}_detections.detections"
MASK_FIELD = f"{LABEL_FIELD}_segmentations.polylines"


def import_split(args: argparse.Namespace) -> fo.Dataset:
    """Return the FiftyOne dataset for the requested split, building it on first use."""
    if args.name in fo.list_datasets():
        if not args.overwrite:
            print(f">>> reusing dataset '{args.name}', pass --overwrite to rebuild it")
            return fo.load_dataset(args.name)
        fo.delete_dataset(args.name)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(args.data_root / "images" / args.split),
        labels_path=str(args.data_root / "annotations" / f"instances_{args.split}.json"),
        label_field=LABEL_FIELD,
        label_types=["detections", "segmentations"],  # boxes plus the original polygons
        use_polylines=True,  # keep polygons as polylines rather than rasterising them into masks
        max_samples=args.limit,
        name=args.name,
    )
    dataset.persistent = True
    return dataset


def main() -> None:
    """Import the split, report its contents and optionally serve the App."""
    parser = argparse.ArgumentParser(description="Load a robobin2026-v1 split into FiftyOne")
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="dataset root holding images/ and annotations/"
    )
    parser.add_argument("--split", default="valid", help="split to load")
    parser.add_argument("--name", default=DEFAULT_NAME, help="FiftyOne dataset name")
    parser.add_argument("--limit", type=int, default=None, help="import only the first N samples")
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild an existing dataset")
    parser.add_argument("--no-app", action="store_true", help="import only, do not serve the App")
    parser.add_argument("--no-browser", action="store_true", help="serve the App without opening a browser")
    parser.add_argument("--port", type=int, default=None, help="App port, defaults to 5151")
    args = parser.parse_args()

    dataset = import_split(args)
    boxes = dataset.count_values(f"{BOX_FIELD}.label")
    polys = dataset.count_values(f"{MASK_FIELD}.label")
    print(f">>> '{args.name}': {dataset.count()} samples, {sum(boxes.values())} boxes, {sum(polys.values())} polygons")
    for label, n in sorted(boxes.items(), key=lambda kv: -kv[1]):
        print(f"    {label:<16} {n}")

    if not args.no_app:
        session = fo.launch_app(dataset, port=args.port, auto=not args.no_browser)
        print(f">>> App serving {args.name} at {session.url}, Ctrl-C to stop")
        session.wait()


if __name__ == "__main__":
    main()