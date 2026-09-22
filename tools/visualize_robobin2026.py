# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize the robobin2026-v1 COCO dataset with instance masks and bounding boxes.

Samples a few images per split, overlays every annotation (translucent polygon mask + box +
class label) and writes both the individual frames and a contact-sheet grid.

The COCO json is decoded incrementally rather than with ``json.load`` because the train
annotation file is ~7 GB; only the sampled annotations are retained, so peak memory stays flat
regardless of split size.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from coco_stream import iter_array

DEFAULT_DATA_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v1")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "robobin2026"
ANNOTATION_NAME = "_annotations.coco.json"

# One stable color per class id so a class keeps its color across splits and samples.
PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
]
MASK_ALPHA = 0.35
GAP = 8  # pixel gap between grid tiles


def imread(path: Path) -> np.ndarray:
    """Read an image, tolerating non-ASCII paths that ``cv2.imread`` rejects."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, image: np.ndarray) -> None:
    """Write an image, tolerating non-ASCII paths that ``cv2.imwrite`` rejects."""
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"failed to encode {path}")
    buf.tofile(str(path))


def draw_labels(image: np.ndarray, anns: list[dict], names: dict[int, str]) -> np.ndarray:
    """Overlay masks, boxes and class labels of ``anns`` onto a copy of ``image``."""
    overlay = image.copy()
    for a in anns:
        seg = a.get("segmentation")
        if not isinstance(seg, list):  # RLE (crowd) annotations carry no polygon to fill
            continue
        color = PALETTE[(a["category_id"] - 1) % len(PALETTE)]
        for poly in seg:
            pts = np.asarray(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(overlay, [pts], color)
    canvas = cv2.addWeighted(overlay, MASK_ALPHA, image, 1 - MASK_ALPHA, 0)

    for a in anns:
        color = PALETTE[(a["category_id"] - 1) % len(PALETTE)]
        seg = a.get("segmentation")
        if isinstance(seg, list):
            for poly in seg:
                pts = np.asarray(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
                if pts.shape[0] >= 3:
                    cv2.polylines(canvas, [pts], True, color, 2)
        x, y, w, h = (round(v) for v in a["bbox"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        draw_text(canvas, names.get(a["category_id"], str(a["category_id"])), (x, y - 6), color)
    return canvas


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draw ``text`` with a filled background so it stays readable over busy regions."""
    scale, thickness = 0.6, 2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(0, min(origin[0], image.shape[1] - tw - 4))
    y = max(th + 4, min(origin[1], image.shape[0] - 4))
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(image, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


def make_grid(tiles: list[np.ndarray], cols: int, tile_width: int) -> np.ndarray:
    """Stack resized tiles into a contact sheet padded with a dark background."""
    h, w = tiles[0].shape[:2]
    tile_size = (tile_width, max(1, round(tile_width * h / w)))
    rows = ceil(len(tiles) / cols)
    th, tw = tile_size[1], tile_size[0]
    grid = np.full((rows * th + (rows + 1) * GAP, cols * tw + (cols + 1) * GAP, 3), 32, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = GAP + r * (th + GAP), GAP + c * (tw + GAP)
        grid[y : y + th, x : x + tw] = cv2.resize(tile, tile_size, interpolation=cv2.INTER_AREA)
    return grid


def visualize_split(
    data_root: Path, split: str, num: int, seed: int, cols: int, tile_width: int, out_dir: Path
) -> None:
    """Sample ``num`` images of ``split``, render their annotations and write the contact sheet."""
    split_dir = data_root / split
    anno = split_dir / ANNOTATION_NAME
    if not anno.is_file():
        raise FileNotFoundError(anno)

    names = {c["id"]: c["name"] for c in iter_array(anno, "categories")}
    images = {im["id"]: im for im in iter_array(anno, "images")}
    picked = random.Random(seed).sample(sorted(images), min(num, len(images)))
    target = set(picked)

    anns, counts = defaultdict(list), Counter()
    for a in iter_array(anno, "annotations"):
        counts[a["category_id"]] += 1
        if a["image_id"] in target:
            anns[a["image_id"]].append(a)

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for image_id in picked:
        meta = images[image_id]
        src = split_dir / meta["file_name"]
        image = imread(src)
        if image is None:
            print(f">>> missing image, skipped: {src}")
            continue
        canvas = draw_labels(image, anns[image_id], names)
        draw_text(canvas, meta["file_name"], (8, canvas.shape[0] - 10), (32, 32, 32))
        imwrite(out_dir / f"{Path(meta['file_name']).stem}.jpg", canvas)
        tiles.append(canvas)

    grid_path = out_dir / f"{split}_grid.jpg"
    if tiles:
        imwrite(grid_path, make_grid(tiles, cols, tile_width))

    print(f"{split}: {len(images)} images, {sum(counts.values())} annotations, {len(counts)} classes")
    for cid, name in sorted(names.items()):
        print(f"  {cid} {name:<16} {counts[cid]}")
    print(f">>> wrote {len(tiles)} frames and {grid_path}")


def main() -> None:
    """Parse arguments and visualize the requested split."""
    parser = argparse.ArgumentParser(description="Visualize the robobin2026-v1 COCO dataset")
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="dataset root holding train/valid/test"
    )
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"), help="split to visualize")
    parser.add_argument("--num", type=int, default=16, help="number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--cols", type=int, default=4, help="contact-sheet columns")
    parser.add_argument("--tile-width", type=int, default=640, help="contact-sheet tile width in pixels")
    parser.add_argument("--out", type=Path, default=None, help="output dir, defaults to runs/robobin2026/<split>")
    args = parser.parse_args()
    visualize_split(
        args.data_root,
        args.split,
        args.num,
        args.seed,
        args.cols,
        args.tile_width,
        args.out or DEFAULT_OUT_DIR / args.split,
    )


if __name__ == "__main__":
    main()
