#!/usr/bin/env python3
import argparse
import csv
import os
from plyfile import PlyData
import numpy as np


def read_metrics(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    result = {
        "views": len(rows),
        "psnr": float(np.mean([float(row["psnr"]) for row in rows])),
        "l1": float(np.mean([float(row["l1"]) for row in rows])),
    }
    for key in ("ssim", "lpips_vgg", "render_ms"):
        if rows and key in rows[0] and rows[0][key] != "":
            result[key] = float(np.mean([float(row[key]) for row in rows]))
    return result


def ply_stats(path):
    data = PlyData.read(path).elements[0].data
    result = {"primitives": len(data), "ply_bytes": os.path.getsize(path)}
    names = set(data.dtype.names)
    wave_names = ["wave_amp_r", "wave_amp_g", "wave_amp_b"]
    if all(name in names for name in wave_names):
        amp = np.stack([data[name] for name in wave_names], axis=1)
        amp_norm = np.linalg.norm(amp, axis=1)
        k_norm = np.sqrt(np.asarray(data["wave_ku"]) ** 2 + np.asarray(data["wave_kv"]) ** 2)
        result.update({
            "wave_amp_norm_median": float(np.median(amp_norm)),
            "wave_amp_norm_p90": float(np.quantile(amp_norm, 0.90)),
            "wave_amp_norm_p99": float(np.quantile(amp_norm, 0.99)),
            "wave_k_norm_median": float(np.median(k_norm)),
            "wave_active_fraction_amp_gt_001": float(np.mean(amp_norm > 0.01)),
            "wave_gate_active_fraction": float(np.mean(np.asarray(data["wave_gate"]) >= 0.5)),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    method_dir = f"ours_{args.iteration}/test"
    baseline_metrics = read_metrics(os.path.join(args.baseline, "render_lite", method_dir, "metrics.csv"))
    wave_metrics = read_metrics(os.path.join(args.wave, "render_lite", method_dir, "metrics.csv"))
    baseline_ply = ply_stats(os.path.join(args.baseline, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    wave_ply = ply_stats(os.path.join(args.wave, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    row = {
        "iteration": args.iteration,
        "views": wave_metrics["views"],
        "baseline_psnr": baseline_metrics["psnr"],
        "wave_psnr": wave_metrics["psnr"],
        "delta_psnr": wave_metrics["psnr"] - baseline_metrics["psnr"],
        "baseline_l1": baseline_metrics["l1"],
        "wave_l1": wave_metrics["l1"],
        "delta_l1": wave_metrics["l1"] - baseline_metrics["l1"],
        "baseline_primitives": baseline_ply["primitives"],
        "wave_primitives": wave_ply["primitives"],
        "delta_primitives": wave_ply["primitives"] - baseline_ply["primitives"],
        "baseline_ply_bytes": baseline_ply["ply_bytes"],
        "wave_ply_bytes": wave_ply["ply_bytes"],
    }
    for key in ("ssim", "lpips_vgg", "render_ms"):
        if key in baseline_metrics and key in wave_metrics:
            row[f"baseline_{key}"] = baseline_metrics[key]
            row[f"wave_{key}"] = wave_metrics[key]
            row[f"delta_{key}"] = wave_metrics[key] - baseline_metrics[key]
    row.update({key: value for key, value in wave_ply.items() if key.startswith("wave_")})
    ablation_path = os.path.join(args.wave, "render_lite", method_dir, "wave_ablation_summary.csv")
    if os.path.exists(ablation_path):
        ablation = next(csv.DictReader(open(ablation_path, newline="", encoding="utf-8")))
        row.update({f"ablation_{key}": value for key, value in ablation.items()})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    for key, value in row.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
