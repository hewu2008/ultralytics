# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Summarize a COCO dataset as a markdown report.

Streams every split (the robobin2026-v1 train json is ~7 GB) and reports per-split totals, the
class distribution, the image resolution distribution and data-quality findings. The report is
written to ``runs/robobin2026/dataset_stats.md`` by default.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from coco_stream import iter_array

DEFAULT_DATA_ROOT = Path("/data/4T-2/dataset/robobin2026/robobin2026-v1-coco")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "runs" / "robobin2026" / "dataset_stats.md"
SPLITS = ("train", "valid", "test")
TOP_RESOLUTIONS = 10  # keep the resolution table readable, the rest is aggregated


def annotation_path(root: Path, split: str) -> Path:
    """Locate a split's COCO json in either the standard layout or the Roboflow layout."""
    standard = root / "annotations" / f"instances_{split}.json"
    return standard if standard.is_file() else root / split / "_annotations.coco.json"


def summarize(path: Path) -> dict:
    """Collect one split's statistics with a single streaming pass per array."""
    names = {c["id"]: c["name"] for c in iter_array(path, "categories")}
    images = list(iter_array(path, "images"))

    instances, class_images = Counter(), defaultdict(set)
    annotated, point_total, point_min, point_max = set(), 0, None, 0
    for a in iter_array(path, "annotations"):
        instances[a["category_id"]] += 1
        class_images[a["category_id"]].add(a["image_id"])
        annotated.add(a["image_id"])
        seg = a.get("segmentation")
        if isinstance(seg, list):
            n = sum(len(p) // 2 for p in seg)
            point_total += n
            point_min = n if point_min is None else min(point_min, n)
            point_max = max(point_max, n)

    resolutions = Counter((im["width"], im["height"]) for im in images)
    file_names = Counter(im["file_name"] for im in images)
    return {
        "path": path,
        "names": names,
        "n_images": len(images),
        "n_annotations": sum(instances.values()),
        "n_annotated_images": len(annotated),
        "instances": instances,
        "class_images": class_images,
        "resolutions": resolutions,
        "duplicate_names": {n: c for n, c in file_names.items() if c > 1},
        "points_total": point_total,
        "points_min": point_min or 0,
        "points_max": point_max,
    }


def render(root: Path, stats: dict[str, dict]) -> str:
    """Render the collected statistics as a markdown document."""
    names = next(iter(stats.values()))["names"]
    splits = list(stats)
    lines = [
        f"# {root.name} 数据集统计",
        "",
        f"- 数据根目录: `{root}`",
        f"- 生成时间: {datetime.now(timezone.utc).astimezone():%Y-%m-%d %H:%M:%S}",
        f"- 类别数: {len(names)}",
        "",
        "## 总览",
        "",
        "| split | 图片数 | 标注数 | 平均每图标注 | 有标注图片 | 无标注图片 | 多边形点数均值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in splits:
        s = stats[split]
        mean_points = s["points_total"] / s["n_annotations"] if s["n_annotations"] else 0
        lines.append(
            f"| {split} | {s['n_images']} | {s['n_annotations']} | {s['n_annotations'] / s['n_images']:.2f} | "
            f"{s['n_annotated_images']} | {s['n_images'] - s['n_annotated_images']} | {mean_points:.1f} |"
        )

    lines += ["", "## 类别分布", "", "| id | 类别 | " + " | ".join(splits) + " | 合计 |"]
    lines.append("| ---: | --- | " + " | ".join("---:" for _ in splits) + " | ---: |")
    for cid in sorted(names):
        cells = [
            f"{stats[s]['instances'][cid]} ({stats[s]['instances'][cid] / stats[s]['n_annotations']:.2%})"
            for s in splits
        ]
        total = sum(stats[s]["instances"][cid] for s in splits)
        lines.append(f"| {cid} | {names[cid]} | " + " | ".join(cells) + f" | {total} |")

    lines += ["", "## 出现该类别的图片数", "", "| id | 类别 | " + " | ".join(splits) + " |"]
    lines.append("| ---: | --- | " + " | ".join("---:" for _ in splits) + " |")
    for cid in sorted(names):
        cells = [str(len(stats[s]["class_images"][cid])) for s in splits]
        lines.append(f"| {cid} | {names[cid]} | " + " | ".join(cells) + " |")

    lines += ["", "## 图片分辨率", "", "| split | 分辨率 | 图片数 | 占比 |", "| --- | --- | ---: | ---: |"]
    for split in splits:
        counts = stats[split]["resolutions"].most_common()
        for (w, h), count in counts[:TOP_RESOLUTIONS]:
            lines.append(f"| {split} | {w}x{h} | {count} | {count / stats[split]['n_images']:.2%} |")
        if len(counts) > TOP_RESOLUTIONS:
            rest = sum(c for _, c in counts[TOP_RESOLUTIONS:])
            lines.append(
                f"| {split} | 其他 {len(counts) - TOP_RESOLUTIONS} 种 | {rest} | "
                f"{rest / stats[split]['n_images']:.2%} |"
            )

    lines += ["", "## 多边形点数", "", "| split | 最小 | 最大 |", "| --- | ---: | ---: |"]
    for split in splits:
        lines.append(f"| {split} | {stats[split]['points_min']} | {stats[split]['points_max']} |")

    lines += ["", "## 数据质量", ""]
    findings = 0
    for split in splits:
        dupes = stats[split]["duplicate_names"]
        if dupes:
            findings += 1
            lines.append(f"- **{split}: {len(dupes)} 个文件名被多条 image 记录重复引用**")
            for name, count in list(dupes.items())[:10]:
                lines.append(f"  - `{name}` 出现 {count} 次")
    unannotated = {s: stats[s]["n_images"] - stats[s]["n_annotated_images"] for s in splits}
    for split, count in unannotated.items():
        if count:
            findings += 1
            lines.append(f"- **{split}: {count} 张图片没有任何标注**")
    if len(splits) > 1:
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                if stats[a]["duplicate_names"] or stats[b]["duplicate_names"]:
                    continue
                same = (
                    stats[a]["n_images"] == stats[b]["n_images"]
                    and stats[a]["n_annotations"] == stats[b]["n_annotations"]
                )
                if same:
                    findings += 1
                    lines.append(f"- **{a} 与 {b} 的图片数和标注数完全一致, 可能是同一批数据的副本**")
    if not findings:
        lines.append("- 未发现异常")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments, collect statistics and write the markdown report."""
    parser = argparse.ArgumentParser(description="Summarize a COCO dataset as a markdown report")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="dataset root to summarize")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown output path")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS, help="splits to summarize")
    args = parser.parse_args()

    stats = {}
    for split in args.splits:
        path = annotation_path(args.data_root, split)
        if not path.is_file():
            print(f">>> skipped {split}: {path} not found", flush=True)
            continue
        stats[split] = summarize(path)
        s = stats[split]
        print(f"  {split}: {s['n_images']} images, {s['n_annotations']} annotations", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.data_root, stats), encoding="utf-8")
    print(f">>> wrote {args.out}")


if __name__ == "__main__":
    main()
