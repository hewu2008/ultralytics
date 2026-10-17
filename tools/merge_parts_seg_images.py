# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Merge one flat numbered parts-seg image set into another without reusing a number.

A parts-seg set is a flat directory of ``NNNN.png`` images each paired with a ``NNNN.json``
(X-AnyLabeling) and a ``NNNN.txt`` (Ultralytics) label. Two sets cannot share a number, so the merge
runs in two steps:

* Renumber: the destination's own triples, taken in numeric order, move to consecutive numbers
  starting at ``--start-index``, and each JSON's ``imagePath`` is rewritten to its new image name so
  X-AnyLabeling keeps pairing JSON and image.
* Merge: the source triples are copied in unchanged, into the numbers the renumber freed up.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path("/data/4T-1/hewu/dataset/hewu2008/parts-seg-v2-20261017/images")
DEFAULT_DST = Path("/data/4T-1/hewu/dataset/hewu2008/parts-seg-v3-20260117/images")
DEFAULT_START_INDEX = 295
SUFFIXES = (".png", ".json", ".txt")


def numbered_stems(directory: Path) -> list[str]:
    """Return the image stems of ``directory`` in numeric order, failing on a non-numeric name."""
    stems = [p.stem for p in directory.glob("*.png")]
    if not stems:
        raise FileNotFoundError(f"no .png images under {directory}")
    for stem in stems:
        if not stem.isdigit():
            raise ValueError(f"image {stem}.png under {directory} is not a number")
    return sorted(stems, key=int)


def check_triples(directory: Path, stems: list[str]) -> None:
    """Fail if any stem is missing one of its png/json/txt companions."""
    for stem in stems:
        for suffix in SUFFIXES:
            if not (directory / f"{stem}{suffix}").is_file():
                raise FileNotFoundError(f"missing {directory / f'{stem}{suffix}'}")


def renumber(directory: Path, start_index: int) -> dict[str, str]:
    """Move every triple to consecutive numbers from ``start_index`` and point its JSON at the new image."""
    stems = numbered_stems(directory)
    check_triples(directory, stems)
    mapping = {stem: f"{start_index + offset:04d}" for offset, stem in enumerate(stems)}
    temporary = {stem: f".renumber-{offset}" for offset, stem in enumerate(stems)}
    # Two phases through throwaway names, so an old name that matches a new number is never clobbered.
    for stem, temp in temporary.items():
        for suffix in SUFFIXES:
            (directory / f"{stem}{suffix}").rename(directory / f"{temp}{suffix}")
    for stem, new in mapping.items():
        for suffix in SUFFIXES:
            shutil.move(directory / f"{temporary[stem]}{suffix}", directory / f"{new}{suffix}")
        annotation = directory / f"{new}.json"
        label = json.loads(annotation.read_text())
        label["imagePath"] = f"{new}.png"
        annotation.write_text(json.dumps(label, indent=2))
    return mapping


def merge(src: Path, dst: Path) -> None:
    """Copy every triple of ``src`` into ``dst``, refusing to overwrite an existing number."""
    stems = numbered_stems(src)
    check_triples(src, stems)
    for stem in stems:
        for suffix in SUFFIXES:
            target = dst / f"{stem}{suffix}"
            if target.exists():
                raise FileExistsError(f"{target} already exists")
    for stem in stems:
        for suffix in SUFFIXES:
            shutil.copy2(src / f"{stem}{suffix}", dst / f"{stem}{suffix}")


def main() -> None:
    """Renumber the destination's own triples, then copy the source triples in."""
    parser = argparse.ArgumentParser(description="Merge one numbered parts-seg image set into another")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="source directory to copy from")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="destination directory to merge into")
    parser.add_argument(
        "--start-index", type=int, default=DEFAULT_START_INDEX, help="first number the destination triples get"
    )
    args = parser.parse_args()

    own = numbered_stems(args.dst)
    mapping = renumber(args.dst, args.start_index)
    merge(args.src, args.dst)

    stems = numbered_stems(args.dst)
    check_triples(args.dst, stems)
    print(
        f">>> {args.dst}: own {len(own)} triples {own[0]}-{own[-1]} -> {min(mapping.values())}-"
        f"{max(mapping.values())}, {len(stems) - len(own)} copied from {args.src}, "
        f"{len(stems)} triples total ({stems[0]}-{stems[-1]})"
    )


if __name__ == "__main__":
    main()
