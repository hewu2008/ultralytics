# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Predict every clip in assets/video with the current robobin2026 segment model and save annotated copies.

Streams each source clip through ``model.predict(..., stream=True)``, writes the rendered
``Results.plot()`` frame with ``cv2.VideoWriter`` and reports the per-class instance counts, so a
clip can be reviewed without re-running inference. Ultralytics' own ``save=True`` writes a
Motion-JPEG ``.avi`` on Linux, which is several times larger; this writes ``.mp4`` (``mp4v``)
at the source frame rate instead.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "runs/segment/ultralytics/yolo26/robobin2026-v1-seg-finetune-2/weights/best.pt"
DEFAULT_SOURCE = REPO / "assets/video"
DEFAULT_OUT = REPO / "runs/segment/robobin2026-v1-seg-video-pred"


def source_videos(source: Path) -> list[Path]:
    """Return the clips to run, accepting either one video file or a directory of them."""
    paths = [source] if source.is_file() else sorted(source.iterdir())
    return [p for p in paths if p.suffix.lstrip(".").lower() in VID_FORMATS]


def predict_video(
    model: YOLO, video: Path, out_path: Path, imgsz: int, conf: float, iou: float, device: str, vid_stride: int
) -> tuple[int, Counter]:
    """Render ``video`` frame by frame into ``out_path`` and return the frame count and class counts."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / vid_stride, size)
    if not writer.isOpened():
        raise OSError(f"cannot write {out_path}")

    counts, frames = Counter(), 0
    for result in model.predict(
        source=str(video),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        vid_stride=vid_stride,
        stream=True,
        verbose=False,
    ):
        writer.write(result.plot())
        frames += 1
        counts.update(model.names[int(c)] for c in result.boxes.cls.tolist())
    writer.release()
    return frames, counts


def main() -> None:
    """Parse arguments and render every requested clip."""
    parser = argparse.ArgumentParser(description="Predict robobin2026 videos and save annotated copies")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="weights to predict with")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="video file or directory of videos")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for the rendered clips")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--vid-stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1, or cpu")
    args = parser.parse_args()

    videos = source_videos(args.source)
    if not videos:
        raise FileNotFoundError(f"no videos found in {args.source}")

    model = YOLO(str(args.model))
    args.out.mkdir(parents=True, exist_ok=True)
    for video in videos:
        out_path = args.out / f"{video.stem}.mp4"
        frames, counts = predict_video(
            model, video, out_path, args.imgsz, args.conf, args.iou, args.device, max(1, args.vid_stride)
        )
        summary = ", ".join(f"{name} {n}" for name, n in counts.most_common()) or "nothing detected"
        print(f">>> {video.name}: {frames} frames -> {out_path} ({summary})")


if __name__ == "__main__":
    main()