# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build robobin2026-v3 as a merged, zero-copy view over robobin2026-v1 and robobin2026-v2.

Split rules:
    train = v1/train + v2/train
    val   = v1/valid + v2/valid
    test  = v2/test                 (v1 has no test split)

Images and labels are symlinked rather than copied: building v3 spends no disk, takes no time, and
keeps any in-flight run against v1 or v2 working. The trade-off is that v1 and v2 must stay in place
for v3 to remain valid. Derived caches (.npy next to images, .cache next to labels) are never linked;
they are rebuilt per root, so writing v3's cache can never touch the source trees.

The class maps already agree - index 0 is cardboard_box in v1's 5-class map and the only class in v2 -
so label files concatenate without remapping. v3 therefore keeps v1's 5-class map.

Run with ``--dry-run`` first to print the plan without touching the filesystem.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
from collections import Counter
from pathlib import Path

DEFAULT_V1_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v1")
DEFAULT_V2_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v2")
DEFAULT_OUT_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v3")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NC = 5  # v1's class map is the merged one

# Target split -> the (source root, source split) pairs it is built from.
SPLIT_RULES = {
    "train": (("v1", "train"), ("v2", "train")),
    "valid": (("v1", "valid"), ("v2", "valid")),
    "test": (("v2", "test"),),  # v1 has no test split
}


def find_image(images_dir: Path, stem: str) -> Path | None:
    """Return the image that pairs with ``<stem>.txt``, or None when the split has no such image."""
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_plan(v1_root: Path, v2_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Map each target split to its ``(image, label)`` source pairs, driven by the label files."""
    roots = {"v1": v1_root, "v2": v2_root}
    plan = {}
    for split, sources in SPLIT_RULES.items():
        pairs = []
        for name, src_split in sources:
            images_dir = roots[name] / "images" / src_split
            labels_dir = roots[name] / "labels" / src_split
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"{name}/{src_split}: labels not found at {labels_dir}")
            for label in labels_dir.glob("*.txt"):
                image = find_image(images_dir, label.stem)
                if image is None:
                    raise FileNotFoundError(f"{name}/{src_split}: no image paired with {label.name}")
                pairs.append((image, label))
        pairs.sort()
        plan[split] = pairs
    return plan


def count_instances(labels: list[Path]) -> Counter:
    """Count instances per class id across the given label files."""

    def read(path: Path) -> list[int]:
        return [int(line.split(" ", 1)[0]) for line in path.read_text().splitlines() if line.strip()]

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        return Counter(i for ids in pool.map(read, labels) for i in ids)


def report(plan: dict[str, list[tuple[Path, Path]]]) -> None:
    """Print each target split's size and class distribution."""
    for split, pairs in plan.items():
        ids = count_instances([label for _, label in pairs])
        print(f">>> {split}: {len(pairs)} images, {sum(ids.values())} instances {dict(sorted(ids.items()))}")


def apply_plan(plan: dict[str, list[tuple[Path, Path]]], out_root: Path) -> None:
    """Create the merged tree, symlinking every source image and label into place."""
    seen: dict[str, Path] = {}
    for split, pairs in plan.items():
        images_out = out_root / "images" / split
        labels_out = out_root / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image, label in pairs:
            for src, dst in ((image, images_out / image.name), (label, labels_out / label.name)):
                if seen.setdefault(dst.name, src) != src:
                    raise RuntimeError(f"name collision on '{dst.name}': {seen[dst.name]} vs {src}")
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src, dst)


def verify(plan: dict[str, list[tuple[Path, Path]]], out_root: Path) -> None:
    """Assert the merged tree has the planned size, resolvable links and only valid class ids."""
    for split, pairs in plan.items():
        images = list((out_root / "images" / split).iterdir())
        labels = list((out_root / "labels" / split).glob("*.txt"))
        assert len(images) == len(labels) == len(pairs), f"{split}: expected {len(pairs)} images/labels"
        broken = [p for p in images + labels if not p.resolve().is_file()]
        assert not broken, f"{split}: {len(broken)} broken links, first {broken[0]}"
        bad = sorted(i for i in count_instances(labels) if not 0 <= i < NC)
        assert not bad, f"{split}: class ids outside 0..{NC - 1}: {bad}"
    print(">>> verified: counts, link targets and class ids all check out")


def main() -> None:
    """Build robobin2026-v3, or print the plan with ``--dry-run``."""
    parser = argparse.ArgumentParser(description="Merge robobin2026-v1 and robobin2026-v2 into robobin2026-v3")
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT, help="robobin2026-v1 root")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT, help="robobin2026-v2 root")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="robobin2026-v3 root to create")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing anything")
    parser.add_argument("--force", action="store_true", help="rebuild an existing output root")
    args = parser.parse_args()

    if args.out_root.exists():
        if not args.force:
            raise SystemExit(f"{args.out_root} already exists, pass --force to rebuild it")
        print(f">>> removing existing {args.out_root}")
        if not args.dry_run:
            shutil.rmtree(args.out_root)

    plan = build_plan(args.v1_root, args.v2_root)
    report(plan)
    if args.dry_run:
        print(">>> dry run, nothing written")
        return

    apply_plan(plan, args.out_root)
    verify(plan, args.out_root)
    print(f">>> robobin2026-v3 built at {args.out_root} (symlinked; v1 and v2 must stay in place)")


if __name__ == "__main__":
    main()