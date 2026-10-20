# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Merge a flat numbered parts-seg set into another dataset's finetune splits.

The source is a flat ``images`` directory of ``NNNN.png`` images each paired with ``NNNN.json``
(X-AnyLabeling) and ``NNNN.txt`` (Ultralytics) labels, the layout parts-seg v2-v4 use. The
destination is a dataset whose ``finetune/{images,labels}/{train,val}`` holds real (non-symlink)
files, the layout parts-seg v5 uses with hash-based frame names, so the two naming schemes cannot
collide and a source triple keeps its filename.

Every source image goes to the split it already belongs to in ``src/finetune``, so the merge only
extends the destination's train/val split and never reshuffles either side. Only images and labels
are copied: ``images`` holds images alone, and the JSON annotations stay in the source, where the
X-AnyLabeling workflow that produced them lives. The destination's data.yaml is written from the
source's class names, or checked against them when it already exists.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path("/data/4T-1/hewu/dataset/hewu2008/parts-seg-v4-20261020")
DEFAULT_DST = Path("/data/4T-1/hewu/dataset/hewu2008/parts-seg-v5-20261020")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SPLITS = ("train", "val")
TRIPLE_SUFFIXES = (".png", ".json", ".txt")


def image_stems(directory: Path) -> set[str]:
    """Return every image stem directly under ``directory``."""
    return {p.stem for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def split_members(root: Path) -> dict[str, set[str]]:
    """Return the image stems of ``root/finetune`` per split."""
    return {split: image_stems(root / "finetune" / "images" / split) for split in SPLITS}


def numbered_triples(src: Path) -> list[str]:
    """Return the source stems in numeric order, failing on a non-numeric name or an incomplete triple."""
    stems = [p.stem for p in (src / "images").glob("*.png")]
    if not stems:
        raise FileNotFoundError(f"no .png images under {src / 'images'}")
    for stem in stems:
        if not stem.isdigit():
            raise ValueError(f"image {stem}.png under {src / 'images'} is not a number")
        for suffix in TRIPLE_SUFFIXES:
            if not (src / "images" / f"{stem}{suffix}").is_file():
                raise FileNotFoundError(src / "images" / f"{stem}{suffix}")
    return sorted(stems, key=int)


def class_names(src: Path) -> dict[int, str]:
    """Read the class names of the source ``finetune/data.yaml`` as an index-to-name mapping."""
    return dict(yaml.safe_load((src / "finetune" / "data.yaml").read_text())["names"])


def check_data_yaml(dst: Path, names: dict[int, str]) -> Path:
    """Write the destination data.yaml, or fail if an existing one disagrees with the source names."""
    root = dst / "finetune"
    lines = [f"path: {root}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {index}: {names[index]}" for index in sorted(names)]
    text = "\n".join(lines) + "\n"
    target = root / "data.yaml"
    if target.exists():
        if target.read_text() != text:
            raise ValueError(f"{target} disagrees with the source class names")
    else:
        target.write_text(text)
    return target


def main() -> None:
    """Copy the source triples into the matching destination split, then write and verify its data.yaml."""
    parser = argparse.ArgumentParser(description="Merge a numbered parts-seg set into the finetune splits")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="flat numbered source dataset")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="finetune-split destination dataset")
    args = parser.parse_args()

    stems = numbered_triples(args.src)
    src_splits = split_members(args.src)
    unsplit = sorted(set(stems) - set().union(*src_splits.values()), key=int)
    if unsplit:
        raise ValueError(f"{len(unsplit)} source triples are in no src/finetune split, e.g. {unsplit[:5]}")
    names = class_names(args.src)
    before = split_members(args.dst)
    for split in SPLITS:
        clashes = sorted(before[split] & src_splits[split], key=int)
        if clashes:
            raise FileExistsError(f"{len(clashes)} source stems already exist in {split}, e.g. {clashes[:5]}")

    for split in SPLITS:
        images = args.dst / "finetune" / "images" / split
        labels = args.dst / "finetune" / "labels" / split
        for stem in sorted(src_splits[split], key=int):
            shutil.copy2(args.src / "images" / f"{stem}.png", images / f"{stem}.png")
            shutil.copy2(args.src / "images" / f"{stem}.txt", labels / f"{stem}.txt")
    data_yaml = check_data_yaml(args.dst, names)

    after = split_members(args.dst)
    for split in SPLITS:
        if not before[split] <= after[split]:
            raise AssertionError(f"{split} lost existing members: {sorted(before[split] - after[split])[:5]}")
        added = after[split] - before[split]
        if added != src_splits[split]:
            raise AssertionError(f"{split} took {len(added)} images, expected the source's {len(src_splits[split])}")
        unpaired = sorted(
            s for s in after[split] if not (args.dst / "finetune" / "labels" / split / f"{s}.txt").is_file()
        )
        if unpaired:
            raise AssertionError(f"{split} images without a label: {unpaired[:5]}")
    strays = {
        p.name
        for split in SPLITS
        for p in (args.dst / "finetune" / "images" / split).iterdir()
        if p.suffix.lower() not in IMAGE_SUFFIXES
    }
    if strays:
        raise AssertionError(f"non-image files under images/: {sorted(strays)[:5]}")
    print(
        f">>> {args.dst / 'finetune'}: train {len(before['train'])} -> {len(after['train'])} "
        f"(+{len(src_splits['train'])}), val {len(before['val'])} -> {len(after['val'])} "
        f"(+{len(src_splits['val'])}), {len(stems)} triples from {args.src}, json left in the source, "
        f"data.yaml -> {data_yaml}"
    )


if __name__ == "__main__":
    main()
