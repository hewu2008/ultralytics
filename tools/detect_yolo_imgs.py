# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detect every image in a directory and save the results, either for review or as a dataset.

Two output layouts share one prediction loop:

* Review (default): Ultralytics' own saver writes the rendered ``Results.plot()`` copy (``save=True``)
  plus the machine-readable YOLO label files (``save_txt``), every result is converted to
  X-AnyLabeling's native JSON under ``xanylabeling/``, and a per-image class-count ``summary.json``
  lets a folder be reviewed without re-running inference.
* Rename (``--start-index``): a flat dataset of ``NNNN`` prefixed files -- a byte copy of the source
  image, its X-AnyLabeling JSON and its Ultralytics label text -- so a fresh batch can continue the
  numbering of an earlier batch and be opened in X-AnyLabeling as-is.

Any Ultralytics backend works, so ``--model`` accepts either the ``.pt`` checkpoint or the exported
end2end ``.onnx``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "runs/pretrained/weights/best.pt"
DEFAULT_SOURCE = Path("/data/4T-1/hewu/dataset/yolo/imgs")
DEFAULT_OUT = REPO / "runs/pretrained/detect-imgs"
XANY_VERSION = "2.4.0"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def xanylabeling(result: Results) -> dict:
    """Convert one prediction result into an X-AnyLabeling annotation dict.

    Masks become absolute-pixel ``polygon`` shapes; a detection model without masks falls back to
    ``rectangle`` shapes. ``imagePath``/``imageHeight``/``imageWidth`` reference the original source
    image so the file lines up when it is opened next to it in X-AnyLabeling.
    """
    polygons = result.masks.xy if result.masks is not None else None
    shapes = []
    for i, box in enumerate(result.boxes):
        if polygons is not None:
            points, shape_type = [[float(x), float(y)] for x, y in polygons[i]], "polygon"
        else:
            top, left, bottom, right = (float(v) for v in box.xyxy[0])
            points, shape_type = [[top, left], [bottom, right]], "rectangle"
        shapes.append(
            {
                "label": result.names[int(box.cls)],
                "score": float(box.conf),
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": shape_type,
                "flags": {},
                "attributes": {},
            }
        )
    height, width = result.orig_shape
    return {
        "version": XANY_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(result.path).name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def ordered_images(source: Path) -> list[Path]:
    """Return the images to predict, numbered files in numeric order first and the rest by name."""
    if source.is_file():
        return [source]
    files = [p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    numbered = sorted((p for p in files if p.stem.isdigit()), key=lambda p: int(p.stem))
    return numbered + sorted((p for p in files if not p.stem.isdigit()), key=lambda p: p.name)


def yolo_label(result: Results) -> str:
    """Return the Ultralytics label rows for one result: normalised polygons, or boxes if no masks.

    Polygons with fewer than 3 points are dropped, matching ``Results.save_txt``, because a shorter
    row is not a polygon and no loader accepts it.
    """
    rows = []
    if result.masks is not None:
        for cls, polygon in zip(result.boxes.cls.tolist(), result.masks.xyn):
            points = polygon.reshape(-1)
            if len(points) >= 6:
                rows.append(f"{int(cls)} " + " ".join(str(float(v)) for v in points))
    else:
        for box in result.boxes:
            rows.append(f"{int(box.cls)} " + " ".join(str(float(v)) for v in box.xywhn[0]))
    return "".join(f"{row}\n" for row in rows)


def review(model: YOLO, args: argparse.Namespace) -> None:
    """Write the rendered plots, the Ultralytics labels and the X-AnyLabeling JSON for review."""
    xany_dir = args.out / "xanylabeling"
    xany_dir.mkdir(parents=True, exist_ok=True)
    summary, total, images = {}, Counter(), 0
    for result in model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=False,
        project=str(args.out.parent),
        name=args.out.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    ):
        name = Path(result.path).name
        counts = Counter(model.names[int(c)] for c in result.boxes.cls.tolist())
        summary[name] = dict(counts)
        total.update(counts)
        images += 1
        (xany_dir / f"{Path(name).stem}.json").write_text(json.dumps(xanylabeling(result), indent=2))

    (args.out / "summary.json").write_text(
        json.dumps({"total": dict(total.most_common()), "images": summary}, indent=2)
    )
    detected = ", ".join(f"{name} {n}" for name, n in total.most_common()) or "nothing detected"
    print(f">>> {images} images from {args.source} -> {args.out} ({detected})")


def rename(model: YOLO, args: argparse.Namespace) -> None:
    """Write a flat NNNN dataset: the image copy, its X-AnyLabeling JSON and its label text."""
    images = ordered_images(args.source)
    if not images:
        raise FileNotFoundError(f"no images under {args.source}")

    found = Counter()
    # A list source is streamed back in the order it was given, so `offset` maps to `images[offset]`.
    for offset, result in enumerate(
        model.predict(
            source=[str(p) for p in images],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            stream=True,
            verbose=False,
        )
    ):
        stem = f"{args.start_index + offset:04d}"
        source = Path(result.path)
        target = args.out / f"{stem}{source.suffix.lower()}"
        shutil.copy2(source, target)
        annotation = xanylabeling(result)
        annotation["imagePath"] = target.name  # X-AnyLabeling pairs the JSON with this renamed image
        (args.out / f"{stem}.json").write_text(json.dumps(annotation, indent=2))
        (args.out / f"{stem}.txt").write_text(yolo_label(result))
        counts = Counter(model.names[int(c)] for c in result.boxes.cls.tolist())
        found.update(counts)
        shapes = ", ".join(f"{name} {n}" for name, n in counts.items()) or "nothing"
        print(f">>> {source.name} -> {target.name} ({shapes})")

    detected = ", ".join(f"{name} {n}" for name, n in found.most_common()) or "nothing detected"
    print(
        f">>> {len(images)} images -> {args.start_index:04d}-{args.start_index + len(images) - 1:04d} "
        f"in {args.out} ({detected})"
    )


def main() -> None:
    """Parse arguments, predict the images and write either the review layout or a renamed dataset."""
    parser = argparse.ArgumentParser(description="Detect images with the pretrained robobin2026 weights")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="weights to predict with (.pt or .onnx)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="image file or directory of images")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for the saved results")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1, or cpu")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="write a flat NNNN image copy + JSON + label dataset from this index instead of the review layout",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"no such source {args.source}")

    # Absolute, so that in review mode Ultralytics' saver cannot nest the rendered images under its
    # default runs/<task> dir, away from summary.json/xanylabeling/.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    if args.start_index is None:
        review(model, args)
    else:
        rename(model, args)


if __name__ == "__main__":
    main()
