#!/usr/bin/env python3
"""Collect official DTU evaluator JSON files into scene/seed/method CSV tables."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--aggregate-output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for result_path in sorted(args.root.glob("*/seed_*_*/geometry_eval/results.json")):
        parts = result_path.relative_to(args.root).parts
        scene, model_dir = parts[0], parts[1]
        if not model_dir.startswith("seed_"):
            continue
        seed_text, method = model_dir[len("seed_"):].split("_", 1)
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        rows.append({
            "scene": scene,
            "seed": int(seed_text),
            "method": method,
            "accuracy": float(result["mean_d2s"]),
            "completeness": float(result["mean_s2d"]),
            "mean": float(result["overall"]),
            "result_json": str(result_path),
        })

    output = args.output or args.root / "dtu_geometry_per_run.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene", "seed", "method", "accuracy", "completeness", "mean", "result_json"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["scene"], row["method"]), []).append(row)
    aggregate_rows = []
    for (scene, method), values in sorted(grouped.items()):
        aggregate_rows.append({
            "scene": scene,
            "method": method,
            "runs": len(values),
            "accuracy": mean(row["accuracy"] for row in values),
            "completeness": mean(row["completeness"] for row in values),
            "mean": mean(row["mean"] for row in values),
        })
    aggregate = args.aggregate_output or args.root / "dtu_geometry_by_scene.csv"
    with aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "method", "runs", "accuracy", "completeness", "mean"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Collected {len(rows)} DTU evaluations: {output}")


if __name__ == "__main__":
    main()
