# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fine-tune the runs/pretrained robobin2026 segmentation weights on a flat image/label folder.

The source folder (``/data/4T-1/hewu/dataset/yolo/imgs``) holds one PNG and one YOLO label ``.txt``
per image with no train/val layout, so this first symlinks the pairs into a small dataset
(``images/{train,val}`` + ``labels/{train,val}``) and writes a ``data.yaml`` whose class names are
read from the pretrained checkpoint, then fine-tunes from that checkpoint with ``--cfg``. Pass
``--prepare-only`` to build the split and stop before training.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/data/4T-1/hewu/dataset/yolo/imgs")
DEFAULT_MODEL = REPO / "runs/pretrained/weights/best.pt"
DEFAULT_DATA = Path("/data/4T-1/hewu/dataset/yolo/finetune")
DEFAULT_CFG = REPO / "assets/yolo-seg-finetune.yaml"
IMG_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def build_split(source: Path, data_root: Path, names: dict, val_ratio: float, seed: int) -> Path:
    """Symlink every image/label pair into a train/val dataset and write its ``data.yaml``."""
    pairs = sorted(
        (p for p in source.iterdir() if p.suffix.lower() in IMG_FORMATS and p.with_suffix(".txt").is_file()),
        key=lambda p: p.stem,
    )
    if not pairs:
        raise FileNotFoundError(f"no image/label pairs found in {source}")

    stems = [p.stem for p in pairs]
    random.Random(seed).shuffle(stems)
    n_val = min(len(stems) - 1, max(1, round(len(stems) * val_ratio)))
    for split, split_stems in {"val": stems[:n_val], "train": stems[n_val:]}.items():
        for sub in ("images", "labels"):
            (data_root / sub / split).mkdir(parents=True, exist_ok=True)
        for stem in split_stems:
            image = next(source / f"{stem}{ext}" for ext in IMG_FORMATS if (source / f"{stem}{ext}").is_file())
            for dst, src in (
                (data_root / "images" / split / image.name, image),
                (data_root / "labels" / split / f"{stem}.txt", source / f"{stem}.txt"),
            ):
                dst.unlink(missing_ok=True)
                dst.symlink_to(src.resolve())

    data_yaml = data_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(data_root), "train": "images/train", "val": "images/val", "names": names}, sort_keys=False
        )
    )
    print(f">>> {len(stems) - n_val} train / {n_val} val pairs from {source} -> {data_yaml}")
    return data_yaml


def main() -> None:
    """Parse arguments, build the train/val split and fine-tune from the pretrained checkpoint."""
    parser = argparse.ArgumentParser(description="Fine-tune the pretrained robobin2026 segmentation weights")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="pretrained weights to fine-tune from")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="flat folder of images and .txt labels")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="output root of the built train/val dataset")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="fine-tuning argument yaml")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of pairs held out for validation")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed for the train/val split")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1, or cpu")
    parser.add_argument("--prepare-only", action="store_true", help="build the split and data.yaml without training")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    data_yaml = build_split(args.source, args.data, model.names, args.val_ratio, args.seed)
    if args.prepare_only:
        print(f">>> prepare-only: run `yolo train model={args.model} data={data_yaml} cfg={args.cfg}` to fine-tune")
        return

    model.train(cfg=str(args.cfg), data=str(data_yaml), device=args.device)


if __name__ == "__main__":
    main()
