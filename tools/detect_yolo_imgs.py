# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detect every image in a directory with the pretrained robobin2026 weights and save the results.

Streams each image through ``model.predict(..., stream=True)`` and lets Ultralytics' own saver
write the rendered ``Results.plot()`` copy (``save=True``) plus the machine-readable YOLO label
files (``save_txt``/``save_conf``). A per-image class-count summary is dumped to ``summary.json``
so a folder can be reviewed without re-running inference. Any Ultralytics backend works, so
``--model`` accepts either the ``.pt`` checkpoint or the exported end2end ``.onnx``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "runs/pretrained/weights/best.pt"
DEFAULT_SOURCE = Path("/data/4T-1/hewu/dataset/yolo/imgs")
DEFAULT_OUT = REPO / "runs/pretrained/detect-imgs"


def main() -> None:
    """Parse arguments, predict the image directory and save the rendered and label results."""
    parser = argparse.ArgumentParser(description="Detect images with the pretrained robobin2026 weights")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="weights to predict with (.pt or .onnx)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="image file or directory of images")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for the saved results")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1, or cpu")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"no such source {args.source}")

    model = YOLO(str(args.model))
    args.out.mkdir(parents=True, exist_ok=True)

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
        counts = Counter(model.names[int(c)] for c in result.boxes.cls.tolist())
        summary[Path(result.path).name] = dict(counts)
        total.update(counts)
        images += 1

    (args.out / "summary.json").write_text(json.dumps({"total": dict(total.most_common()), "images": summary}, indent=2))
    detected = ", ".join(f"{name} {n}" for name, n in total.most_common()) or "nothing detected"
    print(f">>> {images} images from {args.source} -> {args.out} ({detected})")


if __name__ == "__main__":
    main()