#!/usr/bin/env python3
"""Create the shared Replica held-out rendering-view manifest.

The selection is the same as the local 2DGS Replica loader plus render_lite:
frames whose original index is divisible by eight are the test split, then
``offset::stride`` and ``max_views`` are applied in that ordered test list.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_views < 0 or args.stride < 1 or args.offset < 0:
        parser.error("max_views >= 0, stride >= 1, and offset >= 0 are required")
    if args.holdout < 1 or args.offset >= args.stride:
        parser.error("holdout >= 1 and offset must be smaller than stride")
    source = args.source.resolve()
    frames = sorted((source / "results").glob("frame*.jpg"))
    poses = [line for line in (source / "traj.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    available = min(len(frames), len(poses))
    if not available:
        raise RuntimeError(f"No usable Replica frames under {source}")
    test_indices = [idx for idx in range(available) if idx % args.holdout == 0]
    selected_positions = list(range(len(test_indices)))[args.offset::args.stride]
    if args.max_views > 0:
        selected_positions = selected_positions[:args.max_views]
    selected_indices = [test_indices[pos] for pos in selected_positions]
    payload = {
        "format": "replica-fixed-render-v1",
        "source": str(source),
        "split": "test (original frame index modulo holdout == 0)",
        "holdout": args.holdout,
        "max_views": args.max_views,
        "stride": args.stride,
        "offset": args.offset,
        "available_frames": available,
        "available_test_views": len(test_indices),
        "test_positions": selected_positions,
        "frame_indices": selected_indices,
        "frame_names": [frames[idx].stem for idx in selected_indices],
        # NGS exports its sorted COLMAP test list as 00000.png, 00001.png, ... .
        "ngs_test_indices": selected_positions,
    }
    if args.output.exists() and not args.force:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing == payload:
            print(f"Reusing existing manifest: {args.output}")
            return
        raise FileExistsError(f"Manifest exists with different selection: {args.output}; use --force")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected_indices)} shared Replica test views to {args.output}")


if __name__ == "__main__":
    main()
