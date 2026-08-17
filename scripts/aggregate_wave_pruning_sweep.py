#!/usr/bin/env python3
"""Aggregate per-checkpoint post-training wave-pruning sweeps."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = (
    "active_primitives",
    "active_fraction",
    "mean_psnr",
    "mean_ssim",
    "mean_lpips_vgg",
    "mean_l1",
    "delta_psnr_vs_tau0",
    "delta_ssim_vs_tau0",
    "delta_lpips_vs_tau0",
    "delta_l1_vs_tau0",
    "dense_checkpoint_bytes",
    "compact_payload_raw_bytes",
    "compact_payload_compressed_bytes",
    "checkpoint_reduction_raw_fraction",
    "checkpoint_reduction_compressed_fraction",
)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = sorted(root.rglob("sweep_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No sweep summaries found under {root}")

    rows = []
    for path in paths:
        seed_name = path.parent.name
        if seed_name.startswith("seed_"):
            scene = path.parent.parent.name
            seed = seed_name[5:]
        else:
            scene = path.parent.name or "unknown"
            seed = "unknown"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item = {"scene": scene, "seed": seed, **row}
                rows.append(item)

    write_csv(root / "all_runs.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["threshold"])].append(row)

    aggregate = []
    for threshold in sorted(grouped):
        group = grouped[threshold]
        item = {
            "threshold": threshold,
            "runs": len(group),
            "scenes": len({row["scene"] for row in group}),
            "seeds": len({row["seed"] for row in group}),
        }
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_std"] = statistics.pstdev(values)
        aggregate.append(item)

    write_csv(root / "aggregate_by_threshold.csv", aggregate)
    print(f"Collected {len(paths)} runs under {root}")
    print(f"Wrote {root / 'all_runs.csv'}")
    print(f"Wrote {root / 'aggregate_by_threshold.csv'}")


if __name__ == "__main__":
    main()
