#!/usr/bin/env python3
"""Replica mesh metrics using one shared TSDF/visibility manifest."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene.dataset_readers import convertReplicaPose, loadReplicaCameraParams, loadReplicaPoses


def sample_surface(path, count, seed):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError("Expected non-empty triangle mesh: {}".format(path))
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=np.random.default_rng(seed))
    return np.asarray(points, dtype=np.float32)


def camera_data(source):
    source = Path(source)
    params = source / "cam_params.json"
    if not params.exists():
        params = source.parent / "cam_params.json"
    return (loadReplicaCameraParams(params), loadReplicaPoses(source / "traj.txt"),
            sorted((source / "results").glob("depth*.png")))


def fusion_indices(source, manifest_path, stride):
    _, poses, depth_paths = camera_data(source)
    available = min(len(poses), len(depth_paths))
    if manifest_path:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        indices = [int(value) for value in data.get("frame_indices", data["camera_uids"])]
    else:
        indices = [index for index in range(available) if index % 8 != 0][::max(1, stride)]
    invalid = [index for index in indices if index < 0 or index >= available]
    if invalid or not indices:
        raise ValueError("Invalid or empty fusion frame IDs: {}".format(invalid[:8]))
    return indices


def visible_gt(points, source, convention, indices, abs_tol, rel_tol, chunk_size):
    camera, poses, depth_paths = camera_data(source)
    visible = np.zeros(len(points), dtype=bool)
    for index in indices:
        depth = np.asarray(Image.open(depth_paths[index]), dtype=np.float32) / camera["scale"]
        w2c = np.linalg.inv(convertReplicaPose(poses[index], convention))
        rotation, translation = w2c[:3, :3], w2c[:3, 3]
        for start in range(0, len(points), chunk_size):
            end = min(start + chunk_size, len(points))
            ids = np.arange(start, end)[~visible[start:end]]
            if not len(ids):
                continue
            local = points[ids] @ rotation.T + translation
            z = local[:, 2]
            u = np.rint(local[:, 0] * camera["fx"] / z + camera["cx"]).astype(np.int64)
            v = np.rint(local[:, 1] * camera["fy"] / z + camera["cy"]).astype(np.int64)
            valid = ((z > 0.0) & (u >= 0) & (u < camera["width"]) &
                     (v >= 0) & (v < camera["height"]))
            observed = np.zeros(len(ids), dtype=np.float32)
            observed[valid] = depth[v[valid], u[valid]]
            matches = valid & (observed > 0.0) & (np.abs(z - observed) <= abs_tol + rel_tol * observed)
            visible[ids[matches]] = True
    return points[visible]


def frustum_cull(mesh_path, source, convention, indices, output_path):
    camera, poses, _ = camera_data(source)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError("Expected non-empty triangle mesh: {}".format(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    visible = np.zeros(len(vertices), dtype=bool)
    for index in indices:
        w2c = np.linalg.inv(convertReplicaPose(poses[index], convention))
        local = vertices @ w2c[:3, :3].T + w2c[:3, 3]
        z = local[:, 2]
        u = local[:, 0] * camera["fx"] / np.maximum(z, 1e-8) + camera["cx"]
        v = local[:, 1] * camera["fy"] / np.maximum(z, 1e-8) + camera["cy"]
        visible |= ((z > 0.0) & (u >= 0.0) & (u < camera["width"]) &
                    (v >= 0.0) & (v < camera["height"]))
    faces = np.asarray(mesh.faces)
    culled = trimesh.Trimesh(vertices=vertices, faces=faces[visible[faces].all(axis=1)], process=False)
    culled.remove_unreferenced_vertices()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    culled.export(output_path)
    return output_path, {"vertices_before": int(len(vertices)), "vertices_after": int(len(culled.vertices)),
                         "faces_before": int(len(faces)), "faces_after": int(len(culled.faces))}


def nn_distance(query, reference):
    return cKDTree(reference).query(query, workers=-1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--pred-mesh", required=True)
    parser.add_argument("--gt-mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fusion-manifest", required=True)
    parser.add_argument("--gt-visible-cache", default="")
    parser.add_argument("--pose-convention", choices=("raw", "nice"), default="raw")
    parser.add_argument("--pred-cull-mode", choices=("none", "frustum"), default="frustum")
    parser.add_argument("--pred-samples", type=int, default=300000)
    parser.add_argument("--gt-samples", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-abs-tol", type=float, default=0.02)
    parser.add_argument("--depth-rel-tol", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--thresholds", default="0.01,0.02,0.05")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    indices = fusion_indices(args.source, args.fusion_manifest, 1)
    cache = Path(args.gt_visible_cache) if args.gt_visible_cache else None
    if cache and cache.exists():
        cached = np.load(cache)
        if not np.array_equal(cached["camera_uids"], np.asarray(indices, dtype=np.int32)):
            raise ValueError("GT visibility cache was built with a different fusion manifest")
        gt_visible = cached["points"].astype(np.float32)
    else:
        gt_visible = visible_gt(sample_surface(args.gt_mesh, args.gt_samples, args.seed), args.source,
                                args.pose_convention, indices, args.depth_abs_tol,
                                args.depth_rel_tol, args.chunk_size)
        if not len(gt_visible):
            raise RuntimeError("No GT mesh samples passed shared-view visibility culling")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, points=gt_visible, camera_uids=np.asarray(indices, dtype=np.int32))
    pred_mesh, cull_stats = Path(args.pred_mesh), None
    if args.pred_cull_mode == "frustum":
        pred_mesh, cull_stats = frustum_cull(args.pred_mesh, args.source, args.pose_convention,
                                              indices, output / "pred_mesh_frustum_culled.ply")
    pred = sample_surface(pred_mesh, args.pred_samples, args.seed + 1)
    d_pred_to_gt = nn_distance(pred, gt_visible)
    d_gt_to_pred = nn_distance(gt_visible, pred)
    result = {
        "accuracy": float(np.mean(d_pred_to_gt)), "completeness": float(np.mean(d_gt_to_pred)),
        "overall": float((np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) / 2.0),
        "accuracy_median": float(np.median(d_pred_to_gt)),
        "completeness_median": float(np.median(d_gt_to_pred)), "units": "metres",
        "fusion_manifest": str(Path(args.fusion_manifest).resolve()), "fusion_views_used": len(indices),
        "prediction_cull_mode": args.pred_cull_mode, "prediction_cull_stats": cull_stats,
        "pred_mesh": str(pred_mesh.resolve()), "gt_mesh": str(Path(args.gt_mesh).resolve()),
        "pred_samples": int(len(pred)), "gt_visible_samples": int(len(gt_visible)),
    }
    for threshold in (float(value) for value in args.thresholds.split(",") if value.strip()):
        precision, recall = float(np.mean(d_pred_to_gt <= threshold)), float(np.mean(d_gt_to_pred <= threshold))
        fscore = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        suffix = "{:g}m".format(threshold).replace(".", "p")
        result.update({"precision_" + suffix: precision, "recall_" + suffix: recall, "fscore_" + suffix: fscore})
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
