#!/usr/bin/env python3
"""Write one paired Original-vs-Wave Replica mesh-evaluation row."""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, wave = load(args.baseline), load(args.wave)
    keys = ("accuracy", "completeness", "overall", "accuracy_median", "completeness_median")
    row = {"baseline_result": str(Path(args.baseline)), "wave_result": str(Path(args.wave))}
    for key in keys:
        row["baseline_" + key] = base[key]
        row["wave_" + key] = wave[key]
        row["delta_" + key] = wave[key] - base[key]
    for key in sorted(key for key in wave if key.startswith(("precision_", "recall_", "fscore_"))):
        row["baseline_" + key] = base[key]
        row["wave_" + key] = wave[key]
        row["delta_" + key] = wave[key] - base[key]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print("Wrote", output)


if __name__ == "__main__":
    main()
