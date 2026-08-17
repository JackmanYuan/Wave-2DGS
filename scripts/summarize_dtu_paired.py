#!/usr/bin/env python3
"""Build paired Original-vs-All-Wave DTU geometry tables for paper reporting."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


GEOMETRY_KEYS = ("accuracy", "completeness", "mean")
RESOURCE_KEYS = ("primitives", "ply_bytes", "train_wall_seconds", "train_peak_gpu_memory_mb")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_row(path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


def model_stats(model_dir):
    config_path = model_dir / "run_config.json"
    if not config_path.exists():
        return {}
    config = read_json(config_path)
    iteration = int(config["iterations"])
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    result = {"iteration": iteration, "model_path": str(model_dir), "ply_path": str(ply_path)}
    if ply_path.exists():
        data = PlyData.read(ply_path).elements[0].data
        result["primitives"] = int(len(data))
        result["ply_bytes"] = int(ply_path.stat().st_size)
        names = set(data.dtype.names)
        required = {"wave_amp_r", "wave_amp_g", "wave_amp_b", "wave_gate"}
        if required.issubset(names):
            amplitude = np.linalg.norm(np.stack([
                data["wave_amp_r"], data["wave_amp_g"], data["wave_amp_b"]], axis=1), axis=1)
            result["wave_gate_active_fraction"] = float(np.mean(np.asarray(data["wave_gate"]) >= 0.5))
            result["wave_amp_gt_001_fraction"] = float(np.mean(amplitude > 0.01))
            result["wave_amp_norm_median"] = float(np.median(amplitude))
            result["wave_amp_norm_p90"] = float(np.quantile(amplitude, 0.90))
    training = read_csv_row(model_dir / "training_summary.csv")
    if training.get("train_wall_seconds", ""):
        result["train_wall_seconds"] = float(training["train_wall_seconds"])
    if training.get("peak_gpu_memory_mb", ""):
        result["train_peak_gpu_memory_mb"] = float(training["peak_gpu_memory_mb"])
    geometry_path = model_dir / "geometry_eval" / "results.json"
    if geometry_path.exists():
        geometry = read_json(geometry_path)
        values = (float(geometry["mean_d2s"]), float(geometry["mean_s2d"]),
                  float(geometry["overall"]))
        # Exclude degenerate/culling-failed meshes that the evaluator reports as NaN.
        if np.isfinite(values).all():
            result.update({"accuracy": values[0], "completeness": values[1],
                           "mean": values[2], "geometry_json": str(geometry_path)})
    return result


def numeric_summary(rows, key):
    values = np.asarray([float(row[key]) for row in rows if row.get(key) not in (None, "")], dtype=float)
    if not values.size:
        return {}
    return {f"{key}_mean": float(values.mean()),
            f"{key}_std": float(values.std(ddof=1)) if values.size > 1 else 0.0}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--per-run-output", type=Path, default=None)
    parser.add_argument("--per-scene-output", type=Path, default=None)
    parser.add_argument("--paper-output", type=Path, default=None)
    args = parser.parse_args()

    pairs = []
    for baseline_dir in sorted(args.root.glob("*/seed_*_baseline")):
        scene = baseline_dir.parent.name
        # The surfel_splatting environment may use Python < 3.9, where
        # str.removesuffix/removeprefix are unavailable.
        dirname = baseline_dir.name
        suffix = "_baseline"
        prefix = dirname[:-len(suffix)] if dirname.endswith(suffix) else dirname
        wave_dir = baseline_dir.parent / f"{prefix}_wave"
        baseline, wave = model_stats(baseline_dir), model_stats(wave_dir)
        if not all(key in baseline for key in GEOMETRY_KEYS) or not all(key in wave for key in GEOMETRY_KEYS):
            continue
        seed_prefix = "seed_"
        seed = int(prefix[len(seed_prefix):] if prefix.startswith(seed_prefix) else prefix)
        row = {"scene": scene, "seed": seed, "iteration": baseline["iteration"],
               "baseline_model_path": baseline["model_path"], "wave_model_path": wave["model_path"],
               "baseline_geometry_json": baseline["geometry_json"], "wave_geometry_json": wave["geometry_json"]}
        for key in GEOMETRY_KEYS + RESOURCE_KEYS:
            if key in baseline:
                row[f"baseline_{key}"] = baseline[key]
            if key in wave:
                row[f"wave_{key}"] = wave[key]
            if key in baseline and key in wave:
                row[f"delta_{key}"] = wave[key] - baseline[key]
        for key in ("wave_gate_active_fraction", "wave_amp_gt_001_fraction", "wave_amp_norm_median", "wave_amp_norm_p90"):
            if key in wave:
                row[key] = wave[key]
        pairs.append(row)

    per_run = args.per_run_output or args.root / "dtu_paired_per_run.csv"
    write_csv(per_run, pairs)
    per_scene = []
    fields = [f"baseline_{name}" for name in GEOMETRY_KEYS + RESOURCE_KEYS]
    fields += [f"wave_{name}" for name in GEOMETRY_KEYS + RESOURCE_KEYS]
    fields += [f"delta_{name}" for name in GEOMETRY_KEYS + RESOURCE_KEYS]
    fields += ["wave_gate_active_fraction", "wave_amp_gt_001_fraction", "wave_amp_norm_median", "wave_amp_norm_p90"]
    for scene in sorted({row["scene"] for row in pairs}):
        members = [row for row in pairs if row["scene"] == scene]
        summary = {"scene": scene, "paired_runs": len(members),
                   "seeds": " ".join(str(row["seed"]) for row in members)}
        for key in fields:
            summary.update(numeric_summary(members, key))
        per_scene.append(summary)
    per_scene_path = args.per_scene_output or args.root / "dtu_paired_by_scene.csv"
    write_csv(per_scene_path, per_scene)

    paper = {"comparison": "Original_2DGS_vs_All_Wave", "scans": len(per_scene), "paired_runs": len(pairs)}
    for key in fields:
        paper.update(numeric_summary(pairs, key))
    paper_path = args.paper_output or args.root / "dtu_paper_summary.csv"
    write_csv(paper_path, [paper])
    print(f"Paired {len(pairs)} completed DTU runs across {len(per_scene)} scenes.")
    print(f"Wrote {per_run}, {per_scene_path}, and {paper_path}")


if __name__ == "__main__":
    main()
