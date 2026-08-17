#!/usr/bin/env python3
"""Create a deterministic Replica train-view subset for TSDF fusion/evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Replica scene directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=20,
                        help="Keep every Nth training view (Replica eval split is excluded)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.offset < 0 or args.offset >= args.stride:
        parser.error("--offset must satisfy 0 <= offset < stride")

    source, output = Path(args.source), Path(args.output)
    depth_paths = sorted((source / "results").glob("depth*.png"))
    if not depth_paths:
        raise FileNotFoundError("No Replica depth images under {}".format(source / "results"))
    if output.exists() and not args.force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("format") == "replica-fixed-fusion-v2":
            print("Reusing existing manifest: {}".format(output))
            return
        print("Replacing legacy manifest: {}".format(output))
    train_frame_indices = [index for index in range(len(depth_paths)) if index % 8 != 0]
    # Camera.uid is assigned by cameraList_from_camInfos as this list's
    # contiguous index. Camera.colmap_id retains the original Replica frame.
    selected_camera_uids = list(range(len(train_frame_indices)))[args.offset::args.stride]
    selected_frame_indices = [train_frame_indices[uid] for uid in selected_camera_uids]
    payload = {
        "format": "replica-fixed-fusion-v2",
        "source": str(source.resolve()),
        "split": "train (frame_uid % 8 != 0)",
        "selection": "ordered_train_camera_uids[offset::stride]",
        "stride": args.stride,
        "offset": args.offset,
        "available_train_views": len(train_frame_indices),
        "camera_uids": selected_camera_uids,
        "frame_indices": selected_frame_indices,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Wrote {} selected train views to {}".format(len(selected_camera_uids), output))


if __name__ == "__main__":
    main()
