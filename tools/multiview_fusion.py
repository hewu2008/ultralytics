# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fuse per-view 2D segmentation masks through a shared 3D point cloud.

Implements the four-stage closed loop: lift each view's masks into world space with its
depth image, split them by 3D geometry, then reproject every fused instance back into every
camera to obtain refined masks and YOLO segmentation labels.

Splitting and merging are deliberately handled by different stages, because they need
different evidence:

- Each camera's own depth image splits its masks. Two touching boxes still show a sharp
  depth step between them (the gap, or the lower box seen through it), and no other signal
  resolves that boundary at pixel resolution.
- 3D geometry then merges those per-view fragments into global instances by co-visibility: a
  fragment links to another view's fragment when its points reproject into that view onto that
  fragment's mask at the depth that view measured there. This is what associates the surfaces
  of one box seen from different angles, and what lets a view that missed a box recover it.

Neither half can do the other's job. Two boxes stacked flush are one connected solid, so no
voxel-adjacency rule splits them without also splitting single objects; and two different
boxes meet along a contact plane, so their fragments are voxel-adjacent across views and a
proximity rule welds them together.

Reads a scene directory produced by `tools/synth_multiview_scene.py` (or an adapter that
emits the same layout) and writes labels, instance maps, overlays and a report.
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics.utils.ops import masks2segments

# A dense voxel grid keeps the neighbourhood statistics simple and fast; the bound only guards
# against a misconfigured voxel size, since the grid is sized to the point-cloud extent.
MAX_GRID_CELLS = 50_000_000
NEIGHBORS_26 = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
PALETTE = np.array(
    [
        [0, 0, 0],
        [235, 75, 75],
        [75, 210, 105],
        [80, 130, 245],
        [245, 200, 60],
        [205, 90, 220],
        [70, 220, 220],
        [250, 150, 60],
        [160, 160, 160],
    ],
    np.uint8,
)


def load_scene(scene_dir: Path, cfg: dict) -> dict:
    """Read scene.json and confirm every camera the pipeline needs has its inputs on disk."""
    if not (scene_dir / "scene.json").is_file():
        raise FileNotFoundError(f"no scene.json in {scene_dir}, run tools/synth_multiview_scene.py first")
    scene = json.loads((scene_dir / "scene.json").read_text())
    fx, fy, cx, cy = (float(v) for v in cfg["default_K"])
    default_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)
    for camera in scene["cameras"]:
        cam_dir = scene_dir / "cameras" / camera["name"]
        for name in ("depth.png", "instances.png"):
            if not (cam_dir / name).is_file():
                raise FileNotFoundError(f"{cam_dir / name} is missing")
        if "K" not in camera:
            camera["K"] = default_k.tolist()  # placeholder calibration, override in the config
        camera["K"] = np.asarray(camera["K"], np.float64)
        camera["T_c2w"] = np.asarray(camera["T_c2w"], np.float64)
        camera["dir"] = cam_dir
    scene["scene_dir"] = scene_dir
    return scene


def masks_from_synthetic(camera: dict, scene: dict) -> tuple[np.ndarray, dict]:
    """Load the pre-degraded 2D instance map that stands in for a YOLO prediction."""
    masks = cv2.imread(str(camera["dir"] / "masks_input.png"), cv2.IMREAD_UNCHANGED)
    if masks is None:
        raise FileNotFoundError(f"failed to read masks_input.png in {camera['dir']}")
    classes = {int(v): scene["class_id"] for v in np.unique(masks) if v > 0}
    return masks.astype(np.int32), classes


def masks_from_yolo(camera: dict, cfg: dict) -> tuple[np.ndarray, dict]:
    """Rasterize masks predicted by a YOLO segmentation model into an instance map."""
    from ultralytics import YOLO

    result = YOLO(cfg["weights"]).predict(str(camera["dir"] / "color.png"), conf=cfg["conf"], verbose=False)[0]
    height, width = result.orig_shape
    masks = np.zeros((height, width), np.int32)
    classes = {}
    if result.masks is not None:
        for instance_id, (polygon, cls) in enumerate(zip(result.masks.xy, result.boxes.cls.tolist()), start=1):
            cv2.fillPoly(masks, [np.round(polygon).astype(np.int32)], instance_id)
            classes[instance_id] = int(cls)
    return masks, classes


def peel_by_depth(masks: np.ndarray, depth_mm: np.ndarray, cfg: dict) -> np.ndarray:
    """Split each 2D instance wherever the depth image steps, then drop fragments that are too small.

    A box face seen by one camera is nearly fronto-parallel, so the depth changes smoothly across
    it, while the air gap between two boxes shows a much deeper surface within a pixel or two.
    Cutting on that step is what tears a fused 2D blob apart before the 3D stage runs at all.
    """
    depth = depth_mm.astype(np.float32)
    valid = depth > 0
    gradient = np.zeros_like(depth)
    gradient[:, :-1] = np.maximum(gradient[:, :-1], np.abs(depth[:, 1:] - depth[:, :-1]))
    gradient[:-1, :] = np.maximum(gradient[:-1, :], np.abs(depth[1:, :] - depth[:-1, :]))
    edge = cv2.dilate(((gradient > cfg["depth_step_mm"]) & valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    out = np.zeros_like(masks)
    next_id = 0
    for label in (int(v) for v in np.unique(masks) if v > 0):
        kept = ((masks == label) & valid & ~edge & (depth < cfg["max_depth_mm"])).astype(np.uint8)
        count, components = cv2.connectedComponents(kept, connectivity=8)
        for component in range(1, count):
            region = components == component
            if int(region.sum()) >= cfg["min_instance_px"]:
                next_id += 1
                out[region] = next_id
    return out


def lift_to_world(masks: np.ndarray, depth_mm: np.ndarray, camera: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Back-project masked pixels to world coordinates, keeping the source instance id per point."""
    depth = depth_mm.astype(np.float32)
    valid = (masks > 0) & (depth > 0) & (depth < cfg["max_depth_mm"])
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)

    z = depth[rows, cols]
    # Pixel centres must match the convention used when the depth image was rendered.
    pixels = np.stack([cols + 0.5, rows + 0.5, np.ones_like(rows)], axis=1).astype(np.float64)
    points_cam = (pixels @ np.linalg.inv(camera["K"]).T) * z[:, None]
    points_world = points_cam @ camera["T_c2w"][:3, :3].T + camera["T_c2w"][:3, 3]
    return points_world.astype(np.float32), masks[rows, cols].astype(np.int32)


def voxelize(points_mm: np.ndarray, voxel_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize points to a dense occupancy grid sized to their extent."""
    index = np.floor(points_mm / voxel_mm).astype(np.int64)
    origin = index.min(axis=0)
    index -= origin
    dims = index.max(axis=0) + 1
    if int(np.prod(dims)) > MAX_GRID_CELLS:
        raise ValueError(
            f"voxel grid would need {int(np.prod(dims))} cells, exceeding {MAX_GRID_CELLS}; "
            f"raise voxel_mm (currently {voxel_mm})"
        )
    occupancy = np.zeros(tuple(dims), bool)
    occupancy[tuple(index.T)] = True
    return index, origin, dims, occupancy


def neighborhood_count(occupancy: np.ndarray) -> np.ndarray:
    """Count how many of the 26 neighbours of every cell are occupied."""
    padded = np.pad(occupancy.astype(np.uint8), 1)
    counts = np.zeros(occupancy.shape, np.uint8)
    for dx, dy, dz in NEIGHBORS_26:
        counts += padded[
            1 + dx : 1 + dx + occupancy.shape[0],
            1 + dy : 1 + dy + occupancy.shape[1],
            1 + dz : 1 + dz + occupancy.shape[2],
        ]
    return counts


def reprojection_score(points_mm: np.ndarray, camera: dict, mask: np.ndarray, depth_mm: np.ndarray, cfg: dict) -> float:
    """Fraction of `points_mm` that land inside `mask` at the depth the camera measured there.

    This is the co-visibility test that binds two views of one surface: a point another camera
    observed must project into this camera onto the same object, and land at the depth this
    camera really recorded. Points from a different object miss the mask, or hit it at a depth
    that does not match, and background pixels return 0 so they never match either.
    """
    if points_mm.shape[0] == 0:
        return 0.0

    rotation, translation = camera["T_c2w"][:3, :3], camera["T_c2w"][:3, 3]
    points_cam = (points_mm - translation) @ rotation
    z = points_cam[:, 2]
    height, width = camera["size"]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (points_cam @ camera["K"].T)[:, :2] / z[:, None]
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not inside.any():
        return 0.0

    u, v, z = u[inside], v[inside], z[inside]
    hit = mask[v, u] & (np.abs(depth_mm[v, u].astype(np.float32) - z) < cfg["reproj_tol_mm"])
    return float(hit.sum()) / float(points_mm.shape[0])


def cluster_fragments(
    points_mm: np.ndarray,
    unit_index: np.ndarray,
    units: np.ndarray,
    cameras: list[dict],
    view_masks: dict,
    view_depths: dict,
    cfg: dict,
) -> tuple[np.ndarray, int]:
    """Merge per-view fragments into global instances, then drop speckle and tiny clusters.

    Args:
        points_mm (np.ndarray): (N, 3) points in world millimeters.
        unit_index (np.ndarray): (N,) fragment index per point.
        units (np.ndarray): (n_units, 2) `(camera index, instance id in that camera)` per fragment.
        cameras (list[dict]): cameras in the same index order as `units[:, 0]`.
        view_masks (dict): camera name -> peeled instance map for that camera.
        view_depths (dict): camera name -> depth image in millimeters for that camera.
        cfg (dict): pipeline config, uses `voxel_mm`, `min_neighbors`, `reproj_tol_mm`,
            `min_consistency` and `min_cluster_points`.

    Returns:
        (point_cluster, n_clusters): global instance id per input point (0 = discarded) and the count.
    """
    if points_mm.shape[0] == 0:
        return np.zeros(0, np.int32), 0

    index, _, _, occupancy = voxelize(points_mm, cfg["voxel_mm"])
    # A voxel with too few occupied neighbours is sensor speckle rather than a real surface.
    alive = occupancy & (neighborhood_count(occupancy) >= cfg["min_neighbors"])
    point_alive = alive[tuple(index.T)]

    n_units = int(unit_index.max()) + 1
    unit_points = [points_mm[point_alive & (unit_index == u)] for u in range(n_units)]
    unit_masks = [view_masks[cameras[int(view)]["name"]] == frag for view, frag in units]
    unit_depths = [view_depths[cameras[int(view)]["name"]] for view, _ in units]

    parent = list(range(n_units))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n_units):
        if not unit_points[a].size:
            continue
        for b in range(a + 1, n_units):
            view_a, view_b = int(units[a, 0]), int(units[b, 0])
            # Fragments of one camera are kept apart: only their own depth image decides where
            # that camera's objects end, and co-visibility cannot be tested inside one view.
            if view_a == view_b or not unit_points[b].size:
                continue
            scores = (
                reprojection_score(unit_points[a], cameras[view_b], unit_masks[b], unit_depths[b], cfg),
                reprojection_score(unit_points[b], cameras[view_a], unit_masks[a], unit_depths[a], cfg),
            )
            if max(scores) >= cfg["min_consistency"]:
                parent[find(b)] = find(a)

    members = {}
    for unit in range(n_units):
        if unit_points[unit].size:
            members.setdefault(find(unit), []).append(unit)
    sizes = {root: sum(int(unit_points[u].size) for u in member_units) for root, member_units in members.items()}

    # Ascending root == ascending (view, fragment) of the unit that seeded the cluster, which
    # keeps the final ids deterministic and in the same order as the input instances.
    kept = [root for root in sorted(members) if sizes[root] >= cfg["min_cluster_points"]]
    root_of_unit = np.array([find(u) for u in range(n_units)], np.int64)
    cluster_of_root = np.zeros(n_units, np.int32)
    for new_id, root in enumerate(kept, start=1):
        cluster_of_root[root] = new_id
    point_cluster = np.where(point_alive, cluster_of_root[root_of_unit[unit_index]], 0).astype(np.int32)
    return point_cluster, len(kept)


def fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Fill interior holes up to `max_hole_px`, keeping larger ones such as a box footprint."""
    count, components, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    out = mask.copy()
    height, width = mask.shape
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and area <= max_hole_px:
            out[components == i] = 1
    return out


def reproject(cluster_points_mm: np.ndarray, point_cluster: np.ndarray, camera: dict, cfg: dict) -> np.ndarray:
    """Project clusters into one camera with a z-buffer, then solidify each instance mask."""
    height, width = camera["size"]
    result = np.zeros((height, width), np.int32)
    if point_cluster.size == 0:
        return result

    rotation, translation = camera["T_c2w"][:3, :3], camera["T_c2w"][:3, 3]
    points_cam = (cluster_points_mm - translation) @ rotation
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (points_cam @ camera["K"].T)[:, :2] / z[:, None]
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    visible = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not visible.any():
        return result

    flat = v[visible] * width + u[visible]
    depth = z[visible]
    labels = point_cluster[visible]

    # Sort by pixel then depth so the first entry of a run is the nearest point of that pixel.
    order = np.lexsort((depth, flat))
    flat, labels = flat[order], labels[order]
    first = np.concatenate([[True], flat[1:] != flat[:-1]])
    flat, labels = flat[first], labels[first]

    kernel = np.ones((3, 3), np.uint8)
    for cluster in (int(c) for c in np.unique(labels) if c > 0):
        mask = np.zeros(height * width, np.uint8)
        mask[flat[labels == cluster]] = 1
        mask = mask.reshape(height, width)
        if int(mask.sum()) < cfg["min_final_px"]:
            continue
        # The projected point set is dense but not gapless, and its outline is what matters.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if cfg["hole_fill"]:
            mask = fill_small_holes(mask, cfg["fill_hole_max_px"])
        result[mask > 0] = cluster
    return result


def match_instances(gt_map: np.ndarray, pred_map: np.ndarray) -> dict:
    """Greedily match predicted instances to ground-truth instances by IoU, one to one."""
    gt_ids = [int(v) for v in np.unique(gt_map) if v > 0]
    pred_ids = [int(v) for v in np.unique(pred_map) if v > 0]
    pairs = []
    for gt_id in gt_ids:
        gt_region = gt_map == gt_id
        for pred_id in pred_ids:
            pred_region = pred_map == pred_id
            intersection = int((gt_region & pred_region).sum())
            if intersection:
                pairs.append((intersection / int((gt_region | pred_region).sum()), gt_id, pred_id))
    pairs.sort(reverse=True)

    matches, used = {}, set()
    for iou, gt_id, pred_id in pairs:
        if gt_id not in matches and pred_id not in used:
            matches[gt_id] = {"pred_id": pred_id, "iou": round(iou, 4)}
            used.add(pred_id)
    for gt_id in gt_ids:
        matches.setdefault(gt_id, {"pred_id": 0, "iou": 0.0})
    return matches


def write_labels(path: Path, instance_map: np.ndarray, class_of: dict) -> int:
    """Write a YOLO segmentation label file: `cls x1 y1 x2 y2 ... xn yn`, normalized.

    Returns:
        Number of instances written.
    """
    ids = [int(v) for v in np.unique(instance_map) if v > 0]
    if not ids:
        path.write_text("")
        return 0

    height, width = instance_map.shape
    segments = masks2segments(np.stack([instance_map == i for i in ids]), strategy="largest")
    lines = []
    for instance_id, polygon in zip(ids, segments):
        if len(polygon) < 3:
            continue
        polygon = np.clip(polygon, [0, 0], [width - 1, height - 1])
        polygon = polygon / [width, height]
        lines.append(" ".join([str(class_of.get(instance_id, 0)), *(f"{c:.6f}" for c in polygon.ravel())]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def save_overlay(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    """Write a side-by-side comparison of instance maps, labelled above each panel."""
    images = []
    for title, instance_map in panels:
        height, width = instance_map.shape
        image = PALETTE[np.where(instance_map > 0, instance_map % (len(PALETTE) - 1) + 1, 0)]
        canvas = np.zeros((height + 24, width, 3), np.uint8)
        canvas[24:] = image
        cv2.putText(canvas, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(canvas)
    cv2.imwrite(str(path), np.concatenate(images, axis=1)[..., ::-1])


def main(args) -> None:
    """Run the fusion pipeline described in the module docstring."""
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    scene = load_scene(Path(cfg["scene"]), cfg)
    out_dir = Path(cfg["out"])
    for name in ("labels", "instances", "overlay"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(cfg, indent=2, default=str))

    all_points, all_fragments, all_point_classes, raw_masks_by_view = [], [], [], {}
    peeled_masks_by_view, depths_by_view = {}, {}

    for view_index, camera in enumerate(scene["cameras"]):
        depth_path = camera["dir"] / "depth.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"failed to read {depth_path}")
        depth = depth.astype(np.uint16)

        if cfg["masks"] == "synthetic":
            raw_masks, classes = masks_from_synthetic(camera, scene)
        else:
            raw_masks, classes = masks_from_yolo(camera, cfg)
        peeled = peel_by_depth(raw_masks, depth, cfg)

        # Peeling keeps fragment ids stable per view, so the class of the original id carries over.
        peeled_classes = {i: classes.get(i, scene["class_id"]) for i in np.unique(peeled) if i > 0}

        points, point_ids = lift_to_world(peeled, depth, camera, cfg)
        raw_masks_by_view[camera["name"]] = raw_masks
        peeled_masks_by_view[camera["name"]] = peeled
        depths_by_view[camera["name"]] = depth

        all_points.append(points)
        all_fragments.append(np.stack([np.full(point_ids.size, view_index), point_ids], axis=1))
        # Carry the source class per point so a class can be recovered after fusion without
        # relying on view-local ids, which collide across cameras.
        all_point_classes.append(np.array([peeled_classes[int(i)] for i in point_ids], np.int32))

        n_raw = len([v for v in np.unique(raw_masks) if v > 0])
        print(f"{camera['name']:10s} input={n_raw}  after_depth_peel={len(peeled_classes)}  points={points.shape[0]}")

    points = np.concatenate(all_points) if all_points else np.zeros((0, 3), np.float32)
    if points.shape[0] == 0:
        raise RuntimeError("no valid depth pixels inside any mask; check the depth images and max_depth_mm")

    # Every (view, fragment) pair is one unit the 3D stage may merge but never split.
    fragments = np.concatenate(all_fragments)
    units, unit_index = np.unique(fragments, axis=0, return_inverse=True)
    point_cluster, n_clusters = cluster_fragments(
        points, unit_index, units, scene["cameras"], peeled_masks_by_view, depths_by_view, cfg
    )
    print(f"merged {units.shape[0]} view fragments into {n_clusters} global instances from {points.shape[0]} points")

    # Every cluster inherits the majority class of the points that formed it.
    point_classes = np.concatenate(all_point_classes)
    cluster_class = {}
    for cluster in (int(c) for c in np.unique(point_cluster) if c > 0):
        values, counts = np.unique(point_classes[point_cluster == cluster], return_counts=True)
        cluster_class[cluster] = int(values[counts.argmax()])

    report = {"scene": str(scene["scene_dir"]), "voxel_mm": cfg["voxel_mm"], "n_clusters": n_clusters, "views": {}}

    for camera in scene["cameras"]:
        name = camera["name"]
        fused = reproject(points, point_cluster, camera, cfg)
        gt = cv2.imread(str(camera["dir"] / "instances.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        raw = raw_masks_by_view[name]

        # Colour correspondence between panels: paint every fused mask with the ground-truth id it matches.
        matches = match_instances(gt, fused)
        aligned = np.zeros_like(fused)
        for gt_id, match in matches.items():
            if match["pred_id"]:
                aligned[fused == match["pred_id"]] = gt_id
        save_overlay(
            out_dir / "overlay" / f"{name}.jpg",
            [("input 2D", raw), ("fused 3D", aligned), ("ground truth", gt)],
        )
        cv2.imwrite(str(out_dir / "instances" / f"{name}.png"), fused.astype(np.uint16))

        label_class = {i: cluster_class.get(i, scene["class_id"]) for i in np.unique(fused) if i > 0}
        written = write_labels(out_dir / "labels" / f"{name}.txt", fused, label_class)

        dropped = set(scene.get("degradation", {}).get(name, {}).get("drop_ids", []))
        ious = [m["iou"] for m in matches.values()]
        report["views"][name] = {
            "gt_instances": len([v for v in np.unique(gt) if v > 0]),
            "input_instances": len([v for v in np.unique(raw) if v > 0]),
            "fused_instances": len([v for v in np.unique(fused) if v > 0]),
            "labels_written": written,
            "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
            "per_gt_iou": {str(k): v["iou"] for k, v in sorted(matches.items())},
            # A ground-truth instance removed from this view's input that the fusion recovers.
            "recovered_from_other_views": sorted(
                int(k) for k, v in matches.items() if k in dropped and v["iou"] >= 0.5
            ),
        }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["views"], indent=2))
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse multi-view 2D masks through a shared 3D point cloud")
    parser.add_argument("cfg", type=Path, help="config yaml, e.g. assets/multiview-fusion.yaml")
    main(parser.parse_args())