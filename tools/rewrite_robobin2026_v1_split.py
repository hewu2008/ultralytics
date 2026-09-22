# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Rewrite robobin2026-v1 in place into a clean, leak-free train/val layout.

The dataset as exported carries three defects, all verified on the live tree before anything is
touched:

1. ``test`` is a byte-identical copy of ``valid`` (1210/1210 images, identical ``.json``), so the
   split is deleted outright rather than kept as a duplicate twin.
2. ``valid`` and ``train`` share 501 basenames that resolve to *different pictures*: two sources
   reused the ``v1_<N>`` numbering (train 512x368, valid 1280x720), with 0/501 content overlap. A
   plain merge would silently overwrite 501 valid images, so the valid side is renamed to
   ``<stem>__v`` first.
3. ``train`` holds 622 groups of byte-identical duplicates under different names, carrying identical
   labels, so the redundant copies are removed.

With the pool de-duplicated and unlabeled images dropped, a new ``valid`` is drawn by stratifying on
the observed class-presence signature, reproducing the pooled marginal and joint class distribution
in the new validation set. ``annotations/instances_{train,valid}.json`` are rebuilt by filtering the
original COCO files, which preserves segmentation polygons and areas exactly.

The whole plan is computed read-only first and only then applied, so ``--dry-run`` reports the real
outcome. Every step is a rename or unlink on one filesystem: nothing is copied, no extra disk is
used, and ``.npy`` caches written by ``cache: disk`` travel with their image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("cardboard_box", "material_bin", "pallet", "conveyor_belt", "human")
COLLISION_SUFFIX = "__v"
CHANGES = Counter()
VACATED: set[Path] = set()
DRY = False


# --------------------------------------------------------------------------- helpers


def scan_images(root: Path, split: str) -> dict[str, Path]:
    """Map stem -> image path for a split, ignoring ``.npy`` caches."""
    return {p.stem: p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT}


def scan_labels(root: Path, split: str) -> dict[str, Path]:
    """Map stem -> label path for a split."""
    return {p.stem: p for p in (root / "labels" / split).iterdir() if p.suffix == ".txt"}


def image_path(root: Path, split: str, stem: str, ext: str) -> Path:
    """Path of a source image."""
    return root / "images" / split / f"{stem}{ext}"


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def signature(root: Path, split: str, stem: str) -> tuple[int, ...]:
    """Sorted tuple of class ids present in a label file."""
    lines = (root / "labels" / split / f"{stem}.txt").read_text().splitlines()
    return tuple(sorted({int(float(x.split()[0])) for x in lines if x.strip()}))


def allocate(counts: dict[tuple[int, ...], int], val_size: int) -> dict[tuple[int, ...], int]:
    """Split ``val_size`` across signatures proportionally, keeping every signature reachable.

    Each signature starts at its proportional share (rounded, floor of 1) and the residual is spread
    onto the largest signatures that still have capacity, so the total lands exactly on ``val_size``
    while every class combination present in the pool also appears in the new validation set.
    """
    total = sum(counts.values())
    if val_size > total:
        raise ValueError(f"val_size {val_size} exceeds the labeled pool {total}")
    alloc = {sig: min(max(1, round(n * val_size / total)), n) for sig, n in counts.items()}
    residual = val_size - sum(alloc.values())
    order = sorted(counts, key=lambda s: (-counts[s], s))
    i = 0
    while residual != 0:
        sig = order[i % len(order)]
        if residual > 0 and alloc[sig] < counts[sig]:
            alloc[sig] += 1
            residual -= 1
        elif residual < 0 and alloc[sig] > 1:
            alloc[sig] -= 1
            residual += 1
        i += 1
        if i > 200 * len(order):
            raise RuntimeError("could not balance the stratified allocation")
    return alloc


def _stem_paths(root: Path, split: str, stem: str, ext: str, has_label: bool = True) -> set[Path]:
    """Every path one dataset entry occupies: image, ``.npy`` cache, and optionally its label."""
    paths = {image_path(root, split, stem, ext), root / "images" / split / f"{stem}.npy"}
    if has_label:
        paths.add(root / "labels" / split / f"{stem}.txt")
    return paths


def _relocate(root: Path, src: str, dst: str, src_stem: str, ext: str, dst_stem: str) -> None:
    """Rename one image, its ``.npy`` cache, and its label from one split into another.

    A destination that a previous phase already vacated is expected to still be on disk under
    ``--dry-run``, so those are accepted rather than reported as a conflict.
    """
    for old, new in (
        (image_path(root, src, src_stem, ext), image_path(root, dst, dst_stem, ext)),
        (root / "images" / src / f"{src_stem}.npy", root / "images" / dst / f"{dst_stem}.npy"),
        (root / "labels" / src / f"{src_stem}.txt", root / "labels" / dst / f"{dst_stem}.txt"),
    ):
        if old == new or not old.exists():
            continue
        if new.exists() and new not in VACATED:
            raise FileExistsError(f"refusing to overwrite {new}")
        if not DRY:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
    CHANGES["relocate"] += 1


def _remove(root: Path, split: str, stem: str, ext: str, *, label: bool) -> None:
    """Delete one image, its ``.npy`` cache, and optionally its label."""
    targets = [image_path(root, split, stem, ext), root / "images" / split / f"{stem}.npy"]
    if label:
        targets.append(root / "labels" / split / f"{stem}.txt")
    for p in targets:
        if p.exists() and not DRY:
            p.unlink()
    CHANGES["remove"] += 1


# --------------------------------------------------------------------------- planning


def build_plan(root: Path, val_size: int, seed: int) -> dict:
    """Decide the entire rewrite read-only: what moves where, what is deleted, how val is sampled."""
    images = {s: scan_images(root, s) for s in SPLITS}
    labels = {s: scan_labels(root, s) for s in SPLITS}
    for s in SPLITS:
        print(f"    {s:5s} images={len(images[s]):7d} labels={len(labels[s]):7d}")

    # (1) test duplicates valid exactly.
    if set(images["test"]) != set(images["valid"]) or set(labels["test"]) != set(labels["valid"]):
        raise RuntimeError("test is not a copy of valid; refusing to delete the split")
    test_delete = [(s, images["test"][s].suffix.lower(), s in labels["test"]) for s in sorted(images["test"])]

    # (2) Build the merged pool keyed by the *final* stem, renaming the valid side of a collision.
    collisions = sorted(set(images["train"]) & set(images["valid"]))
    clash = {f"{c}{COLLISION_SUFFIX}" for c in collisions} & set(images["train"])
    if clash:
        raise RuntimeError(f"rename suffix collides with existing train names: {sorted(clash)[:5]}")
    pool: dict[str, dict] = {}
    for stem, p in images["train"].items():
        pool[stem] = {"src": "train", "src_stem": stem, "ext": p.suffix.lower(), "origin": ("train", stem)}
    for stem, p in images["valid"].items():
        final = f"{stem}{COLLISION_SUFFIX}" if stem in collisions else stem
        pool[final] = {"src": "valid", "src_stem": stem, "ext": p.suffix.lower(), "origin": ("valid", stem)}
    for final, r in pool.items():
        r["file_name"] = f"{final}{r['ext']}"
        r["final"] = final
        r["renamed"] = r["src_stem"] != final  # collided valid entry, renamed in place first

    # (3) Drop images with no label.
    unlabeled = sorted(
        s for s, r in pool.items() if not (root / "labels" / r["src"] / f"{r['src_stem']}.txt").exists()
    )

    # (4) Drop byte-identical duplicates inside the pool, keeping the lexicographically first name.
    labeled = sorted(set(pool) - set(unlabeled))
    by_size: dict[int, list[str]] = defaultdict(list)
    for stem in labeled:
        r = pool[stem]
        by_size[image_path(root, r["src"], r["src_stem"], r["ext"]).stat().st_size].append(stem)
    dup_extras: list[str] = []
    for stems in by_size.values():
        if len(stems) < 2:
            continue
        by_hash: dict[str, list[str]] = defaultdict(list)
        for stem in stems:
            r = pool[stem]
            by_hash[sha256(image_path(root, r["src"], r["src_stem"], r["ext"]))].append(stem)
        for group in by_hash.values():
            if len(group) > 1:
                dup_extras.extend(sorted(group)[1:])
    dup_extras.sort()

    # (5) Stratified sample for the new validation split.
    final_pool = sorted(set(labeled) - set(dup_extras))
    sig_of = {s: signature(root, pool[s]["src"], pool[s]["src_stem"]) for s in final_pool}
    counts = Counter(sig_of.values())
    alloc = allocate(counts, val_size)
    rng = random.Random(seed)
    val_stems: list[str] = []
    for sig in sorted(counts):
        candidates = sorted(s for s, v in sig_of.items() if v == sig)
        val_stems.extend(rng.sample(candidates, alloc[sig]))
    val_set = set(val_stems)
    for stem in final_pool:
        pool[stem]["dst"] = "valid" if stem in val_set else "train"
    for stem in unlabeled + dup_extras:
        pool[stem]["dst"] = "deleted"

    deletes = [(pool[s]["src"], pool[s]["src_stem"], pool[s]["ext"], False) for s in unlabeled]
    deletes += [(pool[s]["src"], pool[s]["src_stem"], pool[s]["ext"], True) for s in dup_extras]
    return {
        "pool": pool,
        "sig_of": sig_of,
        "test_delete": test_delete,
        "deletes": deletes,
        "train_stems": [s for s in final_pool if s not in val_set],
        "val_stems": sorted(val_stems),
        "collisions": collisions,
        "unlabeled": unlabeled,
        "dup_extras": dup_extras,
        "counts": counts,
    }


# --------------------------------------------------------------------------- applying


def apply_plan(root: Path, plan: dict) -> None:
    """Execute a plan produced by :func:`build_plan`.

    Runs in three phases — deletes, then in-split renames, then split moves — because a name a later
    phase needs may currently be held by a file an earlier phase vacates. ``valid/v1_379.png`` is the
    concrete case: it must be renamed to ``v1_379__v.png`` before train's ``v1_379.png`` can move
    into ``images/valid/`` under that freed name.
    """
    kept = {f: r for f, r in plan["pool"].items() if r["dst"] != "deleted"}
    renames = [(f, r) for f, r in kept.items() if r["renamed"]]  # collide -> <stem>__v, in place
    moves = [(f, r) for f, r in kept.items() if r["dst"] != r["src"]]  # now safe to reuse freed names

    # A phase must never fill a name another operation in the same phase still has to vacate: the
    # vacated set below would otherwise mask a silent overwrite.
    for phase, in_place in ((renames, True), (moves, False)):
        src_stem = (lambda r: r["src_stem"]) if in_place else (lambda r: r["final"])
        srcs = {p for _, r in phase for p in _stem_paths(root, r["src"], src_stem(r), r["ext"])}
        dsts = {p for _, r in phase for p in _stem_paths(root, r["src"] if in_place else r["dst"], r["final"], r["ext"])}
        if srcs & dsts:
            raise RuntimeError(f"phase has cyclic renames: {sorted(srcs & dsts)[:5]}")

    for stem, ext, has_label in plan["test_delete"]:
        VACATED.update(_stem_paths(root, "test", stem, ext, has_label))
    for split, stem, ext, has_label in plan["deletes"]:
        VACATED.update(_stem_paths(root, split, stem, ext, has_label))
    for stem, ext, has_label in plan["test_delete"]:
        _remove(root, "test", stem, ext, label=has_label)
    for split, stem, ext, has_label in plan["deletes"]:
        _remove(root, split, stem, ext, label=has_label)

    for _, r in renames:
        VACATED.update(_stem_paths(root, r["src"], r["src_stem"], r["ext"]))
    for _, r in renames:
        _relocate(root, r["src"], r["src"], r["src_stem"], r["ext"], r["final"])

    for _, r in moves:
        VACATED.update(_stem_paths(root, r["src"], r["final"], r["ext"]))
    for _, r in moves:
        _relocate(root, r["src"], r["dst"], r["final"], r["ext"], r["final"])


def rebuild_coco(root: Path, plan: dict) -> None:
    """Write instances_<split>.json for the new splits by filtering the original COCO files."""
    sources, meta = {}, None
    for name in ("train", "valid"):
        d = json.loads((root / "annotations" / f"instances_{name}.json").read_text())
        by_stem: dict[str, dict] = {}
        for im in d["images"]:
            by_stem.setdefault(Path(im["file_name"]).stem, im)  # keep first when a name repeats
        anns: dict[int, list[dict]] = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        sources[name] = (by_stem, anns)
        if name == "train":
            meta = {k: d[k] for k in ("licenses", "info", "categories") if k in d}
        print(f"    载入 instances_{name}.json: images={len(d['images'])} annotations={len(d['annotations'])}")
        del d

    for split, key in (("train", "train_stems"), ("valid", "val_stems")):
        images, annotations, missing = [], [], []
        for stem in plan[key]:
            r = plan["pool"][stem]
            by_stem, anns = sources[r["origin"][0]]
            im = by_stem.get(r["origin"][1])
            if im is None:
                missing.append(stem)
                continue
            new_id = len(images) + 1
            record = {k: v for k, v in im.items() if k not in {"id", "file_name"}}
            record["id"] = new_id
            record["file_name"] = r["file_name"]
            images.append(record)
            for a in anns.get(im["id"], []):
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": new_id,
                        "category_id": a["category_id"],
                        "area": a["area"],
                        "bbox": a["bbox"],
                        "iscrowd": a.get("iscrowd", 0),
                        "segmentation": a["segmentation"],
                    }
                )
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} stems absent from the source COCO json, e.g. {missing[:5]}")
        payload = json.dumps({**meta, "images": images, "annotations": annotations})
        target = root / "annotations" / f"instances_{split}.json"
        if not DRY:
            target.write_text(payload)
        print(f"    instances_{split}.json: images={len(images)} annotations={len(annotations)} -> {len(payload) / 1e6:.1f} MB")
        del payload, images, annotations


def report(plan: dict) -> None:
    """Print the resulting split composition."""
    print(f"\n  类别签名 {len(plan['counts'])} 种")
    for split, key in (("train", "train_stems"), ("valid", "val_stems")):
        stems = plan[key]
        cls = Counter(c for s in stems for c in plan["sig_of"][s])
        inst = sum(cls.values())
        print(f"  [{split}] 图片={len(stems)}  实例={inst}  每图实例={inst / max(len(stems), 1):.2f}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"      cls{c} {name:14s} 图片 {cls[c]:7d} ({100 * cls[c] / max(len(stems), 1):5.2f}%)")


def main() -> None:
    """Rewrite the dataset in place."""
    global DRY
    parser = argparse.ArgumentParser(description="Rewrite robobin2026-v1 into a clean train/val layout")
    parser.add_argument("data_dir", type=Path, help="dataset root, e.g. /data/4T-2/dataset/robobin2026/robobin2026-v1")
    parser.add_argument("--val-size", type=int, default=10000, help="size of the new validation split")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed for the new validation split")
    parser.add_argument("--dry-run", action="store_true", help="report the plan without touching any file")
    args = parser.parse_args()
    root: Path = args.data_dir
    DRY = args.dry_run
    print(f">>> {'DRY RUN — ' if DRY else ''}重写 {root}   val_size={args.val_size} seed={args.seed}\n")

    print("[1/7] 扫描原始划分")
    plan = build_plan(root, args.val_size, args.seed)
    print(f"    同名冲突（内容不同，非重复）={len(plan['collisions'])} -> 重命名为 <stem>{COLLISION_SUFFIX}")
    print(f"    合并池={len(plan['pool'])}  无标注={len(plan['unlabeled'])}  内部逐字节重复={len(plan['dup_extras'])}")
    print(f"    test 待删除={len(plan['test_delete'])}（与 valid 完全重复）")

    print("\n[2/7] 删除 test 与无标注、重复图像")
    print("\n[3/7] 合并 valid 到 train 并重命名冲突项")
    print("\n[4/7] 分层采样新 val")
    print("\n[5/7] 落盘")
    apply_plan(root, plan)
    for split in SPLITS:
        for d in (root / "images" / split, root / "labels" / split):
            if d.exists() and not any(d.iterdir()) and not DRY:
                shutil.rmtree(d)
    print(f"    relocate={CHANGES['relocate']} remove={CHANGES['remove']}")

    print("\n[6/7] 按新划分重建 COCO json")
    rebuild_coco(root, plan)
    stale = root / "annotations" / "instances_test.json"
    if stale.exists() and not DRY:
        stale.unlink()

    print("\n[7/7] 结果")
    report(plan)
    print("\n>>> DRY RUN 完成，未改动任何文件" if DRY else f"\n>>> 完成：relocate={CHANGES['relocate']} remove={CHANGES['remove']}")


if __name__ == "__main__":
    main()