# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render a synthetic multi-view RGB-D scene for validating multi-view mask fusion.

Builds a small world (a bounded ground plane plus axis-aligned cardboard boxes) and
renders it from several cameras by analytic ray/AABB intersection, so the ground-truth
depth and instance maps are exact. Three camera layouts cover the failure modes this
pipeline targets: a top view where adjacent boxes merge into one blob, and side views
that can see the air gap between them.

Alongside the ground truth, a deliberately degraded copy of the 2D masks is written to
``masks_input.png``: nearby instances are unioned (emulating top-view adhesion) and one
instance is dropped from one view (emulating a miss), which is the error the 3D stage
is expected to undo.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Cardboard is a low-texture, near-white material; shading is kept subtle on purpose so
# the color images are a faithful stand-in for the "weak texture" top-view condition.
BOX_COLOR = np.array([206, 182, 152], np.float32)
GROUND_COLOR = np.array([70, 70, 70], np.float32)
LIGHT_DIR = np.array([0.30, 0.45, 0.84], np.float32)

GROUND_HALF_EXTENT_MM = 1500.0
# How far the stacked boxes are pulled back from the edge of the box below them, which sets how
# much of that lower box stays visible from above.
STACK_INSET_MM = 50.0


def make_camera(name: str, position, target, up_hint, size, fx, fy, cx, cy) -> dict:
    """Build a pinhole camera dict with OpenCV convention (x right, y down, z forward)."""
    h, w = size
    position = np.asarray(position, np.float32)
    forward = np.asarray(target, np.float32) - position
    forward /= np.linalg.norm(forward)

    up_hint = np.asarray(up_hint, np.float32)
    if abs(float(np.dot(forward, up_hint / np.linalg.norm(up_hint)))) > 0.99:
        raise ValueError(f"camera '{name}' up_hint is parallel to its optical axis")

    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = -np.cross(right, forward)

    r_c2w = np.stack([right, down, forward], axis=1)
    t_c2w = np.eye(4, dtype=np.float32)
    t_c2w[:3, :3] = r_c2w
    t_c2w[:3, 3] = position

    return {
        "name": name,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "T_c2w": t_c2w.tolist(),
        "size": [h, w],
    }


def default_cameras(size, fx, fy, cx, cy) -> list[dict]:
    """Return the default three-camera rig: one top view plus two side views.

    The side cameras look down at roughly 30 degrees rather than edge-on, because a flat face
    viewed near grazing incidence produces a depth gradient per pixel as large as a real gap,
    and the depth peel would then cut the face apart instead of cutting at the gap.
    """
    return [
        make_camera("cam_top", (0, 0, 1700), (0, 0, 0), (0, -1, 0), size, fx, fy, cx, cy),
        make_camera("cam_front", (0, -1250, 900), (0, 0, 120), (0, 0, 1), size, fx, fy, cx, cy),
        make_camera("cam_side", (1300, -900, 900), (0, 0, 120), (0, 0, 1), size, fx, fy, cx, cy),
    ]


def default_boxes(gap_mm: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return a tightly packed 2x2 base layer with a smaller box stacked on two of its members.

    The base layer is the case this pipeline targets: four boxes at one height whose gaps are
    only a pixel or two wide seen from above, so the top view merges them into a single blob.
    The stacked boxes are inset by `STACK_INSET_MM`, which keeps a wide ring of each box below
    them visible from every camera and puts a real depth step across the stack boundary.
    """
    half = gap_mm / 2.0
    inset = STACK_INSET_MM
    specs = [
        # name, (x0, x1), (y0, y1), (z0, z1)
        ("box_A", (-400 - half, -half), (-300 - half, -half), (0, 200)),
        ("box_B", (half, 400 + half), (-300 - half, -half), (0, 200)),
        ("box_C", (-400 - half, -half), (half, 300 + half), (0, 200)),
        ("box_D", (half, 400 + half), (half, 300 + half), (0, 200)),
        ("box_E", (-400 + inset - half, -inset - half), (-300 + inset - half, -inset - half), (200, 380)),
        ("box_F", (inset + half, 400 - inset + half), (inset + half, 300 - inset + half), (200, 380)),
    ]
    boxes = []
    for name, (x0, x1), (y0, y1), (z0, z1) in specs:
        boxes.append((name, np.array([x0, y0, z0], np.float32), np.array([x1, y1, z1], np.float32)))
    return boxes


def render(camera: dict, boxes, add_ground: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast the scene from one camera.

    Returns:
        (depth, instance, color): depth in millimeters (uint16, 0 = no return),
        instance id per pixel (uint16, 0 = background), and an RGB image (uint8).
    """
    k = np.asarray(camera["K"], np.float32)
    t_c2w = np.asarray(camera["T_c2w"], np.float32)
    r_c2w, origin = t_c2w[:3, :3], t_c2w[:3, 3]
    h, w = camera["size"]

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size, np.float32)], axis=1)
    dir_cam = pixels @ np.linalg.inv(k).T
    # Not normalized: the ray parameter s then equals the camera-frame z, i.e. the depth.
    dir_world = dir_cam @ r_c2w.T
    # A zero direction component would make the slab test divide by zero; a tiny epsilon keeps
    # the sign-independent min/max structure intact and yields a harmless huge tn or tf.
    dir_world = np.where(np.abs(dir_world) < 1e-9, np.float32(1e-9), dir_world)

    n_pix = pixels.shape[0]
    best_depth = np.full(n_pix, np.inf, np.float32)
    best_id = np.zeros(n_pix, np.uint16)
    best_normal = np.zeros((n_pix, 3), np.float32)

    for box_id, (_, bmin, bmax) in enumerate(boxes, start=1):
        t1 = (bmin - origin) / dir_world
        t2 = (bmax - origin) / dir_world
        tn = np.minimum(t1, t2).max(axis=1)
        tf = np.maximum(t1, t2).min(axis=1)
        hit = (tn <= tf) & (tf > 0) & (tn > 0)
        closer = hit & (tn < best_depth)

        axis = np.minimum(t1, t2).argmax(axis=1)
        normal = np.zeros((n_pix, 3), np.float32)
        rows = np.arange(n_pix)
        normal[rows, axis] = -np.sign(dir_world[rows, axis])

        best_depth = np.where(closer, tn, best_depth)
        best_id = np.where(closer, np.uint16(box_id), best_id)
        best_normal = np.where(closer[:, None], normal, best_normal)

    if add_ground:
        t_ground = -origin[2] / dir_world[:, 2]
        hit_xy = origin[:2] + t_ground[:, None] * dir_world[:, :2]
        on_ground = (t_ground > 0) & (t_ground < best_depth) & (np.abs(hit_xy) <= GROUND_HALF_EXTENT_MM).all(axis=1)
        best_depth = np.where(on_ground, t_ground, best_depth)
        best_id = np.where(on_ground, np.uint16(0), best_id)
        best_normal = np.where(on_ground[:, None], np.array([0, 0, 1], np.float32), best_normal)

    valid = np.isfinite(best_depth)
    ambient = 0.78 + 0.22 * np.abs((best_normal * LIGHT_DIR).sum(axis=1))
    color = np.where(best_id[:, None] > 0, BOX_COLOR, GROUND_COLOR) * ambient[:, None]

    # Depth is quantized to millimeters to match the RealSense z16 convention.
    depth = np.where(valid, np.rint(best_depth), 0.0).reshape(h, w).astype(np.uint16)
    instance = best_id.reshape(h, w)
    image = np.clip(color, 0, 255).reshape(h, w, 3).astype(np.uint8)
    return depth, instance, image


def add_depth_noise(depth: np.ndarray, sigma_mm: float, dropout: float, rng) -> np.ndarray:
    """Perturb depth with gaussian noise and random dropouts, emulating consumer RGB-D sensors."""
    noisy = depth.astype(np.float32)
    valid = noisy > 0
    noisy[valid] += rng.normal(0.0, sigma_mm, int(valid.sum())).astype(np.float32)
    noisy[valid] = np.maximum(noisy[valid], 1.0)
    if dropout > 0:
        drop = valid & (rng.random(depth.shape) < dropout)
        noisy[drop] = 0.0
    return np.where(noisy > 0, np.rint(noisy), 0.0).astype(np.uint16)


def degrade_masks(
    instance: np.ndarray,
    merge_groups: list[set[int]],
    drop_ids: set[int],
    jitter_px: int,
    close_px: int,
    rng,
) -> np.ndarray:
    """Emulate YOLO's typical errors on a ground-truth instance map.

    Each group in `merge_groups` is unioned into the lowest id of that group and then closed
    by `close_px`, which seals the thin background stripe the sensor reports inside the air
    gap. Without that close the merged mask would still carry the gap as a hole and the
    adhesion would be trivially separable. `drop_ids` are removed entirely (a miss), and mask
    boundaries are rounded and roughened by `jitter_px` so the masks are not pixel-perfect.
    """
    remap = {}
    for group in merge_groups:
        for label in group:
            remap[label] = min(group)

    out = np.zeros_like(instance)
    for label in (int(v) for v in np.unique(instance) if v > 0):
        if label in drop_ids:
            continue
        out[instance == label] = remap.get(label, label)

    if close_px > 0:
        kernel = np.ones((2 * close_px + 1, 2 * close_px + 1), np.uint8)
        for group in merge_groups:
            label = min(group)
            # MORPH_CLOSE only bridges thin gaps, so large holes such as the footprint of a
            # box stacked on top are preserved.
            sealed = cv2.morphologyEx((out == label).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            out[(sealed > 0) & (out == 0)] = label

    if jitter_px > 0:
        kernel = np.ones((2 * jitter_px + 1, 2 * jitter_px + 1), np.uint8)
        rounded = np.zeros_like(out)
        for label in (int(v) for v in np.unique(out) if v > 0):
            mask = (out == label).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            rounded[mask > 0] = label
        # Roughen the boundary by eroding a random half of the mask edge.
        edge = cv2.morphologyEx((rounded > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        rounded[(rng.random(rounded.shape) < 0.5) & edge] = 0
        out = rounded

    return out.astype(np.uint16)


def main(args) -> None:
    """Render the default scene and write it to `args.scene_dir` in the documented layout."""
    rng = np.random.default_rng(args.seed)
    scene_dir = Path(args.scene_dir)
    if scene_dir.exists():
        raise FileExistsError(f"{scene_dir} already exists, remove it or choose another scene_dir")

    size = (args.height, args.width)
    fx = fy = args.fx
    cx, cy = args.width / 2.0, args.height / 2.0
    cameras = default_cameras(size, fx, fy, cx, cy)
    boxes = default_boxes(args.gap_mm)

    merge_view = args.merge_view
    merge_group = {int(v) for v in args.merge_ids.split(",") if v}
    degradation = {
        camera["name"]: {
            "merge_groups": [sorted(merge_group)] if camera["name"] == merge_view else [],
            "drop_ids": sorted([args.drop_id]) if camera["name"] == args.drop_view else [],
        }
        for camera in cameras
    }

    box_records = [
        {"id": i, "name": name, "min": bmin.tolist(), "max": bmax.tolist()}
        for i, (name, bmin, bmax) in enumerate(boxes, start=1)
    ]
    scene = {
        "class_id": 0,
        "class_names": ["box"],
        "gap_mm": args.gap_mm,
        "cameras": cameras,
        "boxes": box_records,
        "degradation": degradation,
    }

    for camera in cameras:
        depth, instance, color = render(camera, boxes)
        depth = add_depth_noise(depth, args.noise_mm, args.dropout, rng)

        spec = degradation[camera["name"]]
        input_masks = degrade_masks(
            instance,
            [set(g) for g in spec["merge_groups"]],
            set(spec["drop_ids"]),
            args.jitter_px,
            args.merge_close_px,
            rng,
        )

        out_dir = scene_dir / "cameras" / camera["name"]
        out_dir.mkdir(parents=True)
        # cv2 writes BGR, while render() produces RGB.
        if not cv2.imwrite(str(out_dir / "color.png"), color[..., ::-1]):
            raise RuntimeError(f"failed to write color.png for {camera['name']}")
        cv2.imwrite(str(out_dir / "depth.png"), depth)
        cv2.imwrite(str(out_dir / "instances.png"), instance)
        cv2.imwrite(str(out_dir / "masks_input.png"), input_masks)

        n_gt = len([v for v in np.unique(instance) if v > 0])
        n_in = len([v for v in np.unique(input_masks) if v > 0])
        print(
            f"{camera['name']:10s} gt_instances={n_gt}  input_instances={n_in}  "
            f"merged={spec['merge_groups']}  dropped={spec['drop_ids']}"
        )

    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"scene written to {scene_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a synthetic multi-view RGB-D scene")
    parser.add_argument("scene_dir", type=Path, help="output scene dir, e.g. runs/multiview/scene_boxes")
    parser.add_argument("--width", type=int, default=640, help="image width")
    parser.add_argument("--height", type=int, default=480, help="image height")
    parser.add_argument("--fx", type=float, default=465.0, help="focal length in pixels (fx = fy)")
    parser.add_argument("--gap-mm", type=float, default=6.0, help="air gap between adjacent boxes")
    parser.add_argument("--noise-mm", type=float, default=3.0, help="depth noise sigma in millimeters")
    parser.add_argument("--dropout", type=float, default=0.005, help="fraction of invalid depth pixels")
    parser.add_argument("--jitter-px", type=int, default=1, help="mask boundary roughness in pixels")
    parser.add_argument("--merge-view", default="cam_top", help="camera whose boxes are merged (adhesion)")
    parser.add_argument("--merge-ids", default="1,2,3,4", help="ground-truth ids merged in --merge-view")
    parser.add_argument("--merge-close-px", type=int, default=3, help="close radius that seals the gap stripe")
    parser.add_argument("--drop-view", default="cam_front", help="camera to drop instances from")
    parser.add_argument("--drop-id", type=int, default=4, help="ground-truth id dropped in --drop-view")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    main(parser.parse_args())
