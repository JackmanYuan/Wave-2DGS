#!/usr/bin/env python3
"""Visibility-aware mesh metrics against the Replica ground-truth mesh.

Accuracy is predicted-surface to complete GT-surface distance; completeness is
training-visible GT-surface to predicted-surface distance. Distances are in
Replica world units (metres). GT samples are visible only if they agree with at
least one training RGB-D depth map.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

from scene.dataset_readers import (
    convertReplicaPose,
    loadReplicaCameraParams,
    loadReplicaPoses,
)


def sample_surface(mesh_path, count, rng):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError("Expected a triangle mesh: {}".format(mesh_path))
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
    return np.asarray(points, dtype=np.float32)


def visible_gt_samples(samples, source, pose_convention, frame_stride,
                       depth_abs_tol, depth_rel_tol, chunk_size):
    source = Path(source)
    camera_path = source / "cam_params.json"
    if not camera_path.exists():
        camera_path = source.parent / "cam_params.json"
    camera = loadReplicaCameraParams(camera_path)
    poses = loadReplicaPoses(source / "traj.txt")
    depth_paths = sorted((source / "results").glob("depth*.png"))
    available = min(len(poses), len(depth_paths))
    train_indices = [idx for idx in range(available) if idx % 8 != 0]
    selected = train_indices[::max(1, frame_stride)]
    visible = np.zeros(len(samples), dtype=bool)

    for index in selected:
        depth = np.asarray(Image.open(depth_paths[index]), dtype=np.float32) / camera["scale"]
        c2w = convertReplicaPose(poses[index], pose_convention)
        w2c = np.linalg.inv(c2w)
        rotation, translation = w2c[:3, :3], w2c[:3, 3]
        for start in range(0, len(samples), chunk_size):
            end = min(start + chunk_size, len(samples))
            ids = np.arange(start, end)
            ids = ids[~visible[start:end]]
            if not len(ids):
                continue
            camera_points = samples[ids] @ rotation.T + translation
            z = camera_points[:, 2]
            u = np.rint(camera_points[:, 0] * camera["fx"] / z + camera["cx"]).astype(np.int64)
            v = np.rint(camera_points[:, 1] * camera["fy"] / z + camera["cy"]).astype(np.int64)
            valid = (z > 0.0) & (u >= 0) & (u < camera["width"]) & (v >= 0) & (v < camera["height"])
            if not np.any(valid):
                continue
            observed = np.zeros(len(ids), dtype=np.float32)
            observed[valid] = depth[v[valid], u[valid]]
            tolerance = depth_abs_tol + depth_rel_tol * observed
            matches = valid & (observed > 0.0) & (np.abs(z - observed) <= tolerance)
            visible[ids[matches]] = True
    return samples[visible], len(selected)


def distances(query, reference):
    tree = cKDTree(reference)
    return tree.query(query, workers=-1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Replica scene directory")
    parser.add_argument("--pred-mesh", required=True)
    parser.add_argument("--gt-mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gt-visible-cache", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pred-samples", type=int, default=300000)
    parser.add_argument("--gt-samples", type=int, default=500000)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--depth-abs-tol", type=float, default=0.02)
    parser.add_argument("--depth-rel-tol", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--pose-convention", choices=("raw", "nice"), default="raw")
    parser.add_argument("--thresholds", default="0.01,0.02,0.05")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.gt_visible_cache) if args.gt_visible_cache else None
    rng = np.random.default_rng(args.seed)
    gt_samples = sample_surface(args.gt_mesh, args.gt_samples, rng)
    if cache and cache.exists():
        gt_visible = np.load(cache)["points"].astype(np.float32)
        views_used = None
    else:
        gt_visible, views_used = visible_gt_samples(
            gt_samples, args.source, args.pose_convention, args.frame_stride,
            args.depth_abs_tol, args.depth_rel_tol, args.chunk_size)
        if not len(gt_visible):
            raise RuntimeError("No GT mesh samples passed RGB-D visibility culling")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, points=gt_visible)

    pred_samples = sample_surface(args.pred_mesh, args.pred_samples, np.random.default_rng(args.seed + 1))
    d_pred_to_gt = distances(pred_samples, gt_samples)
    d_gt_to_pred = distances(gt_visible, pred_samples)
    result = {
        "accuracy": float(np.mean(d_pred_to_gt)),
        "completeness": float(np.mean(d_gt_to_pred)),
        "overall": float((np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) * 0.5),
        "accuracy_median": float(np.median(d_pred_to_gt)),
        "completeness_median": float(np.median(d_gt_to_pred)),
        "pred_samples": int(len(pred_samples)),
        "gt_visible_samples": int(len(gt_visible)),
        "visibility_views_used": views_used,
        "units": "metres",
        "pred_mesh": str(Path(args.pred_mesh).resolve()),
        "gt_surface_samples": int(len(gt_samples)),
        "gt_mesh": str(Path(args.gt_mesh).resolve()),
    }
    for threshold in (float(value) for value in args.thresholds.split(",") if value.strip()):
        precision = float(np.mean(d_pred_to_gt <= threshold))
        recall = float(np.mean(d_gt_to_pred <= threshold))
        fscore = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        suffix = "{:g}m".format(threshold).replace(".", "p")
        result.update({"precision_" + suffix: precision, "recall_" + suffix: recall, "fscore_" + suffix: fscore})
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
