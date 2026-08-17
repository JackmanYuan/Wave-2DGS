#!/usr/bin/env python3
"""Post-training amplitude-pruning sweep for trained Wave-2DGS checkpoints.

The rendering test keeps the Gaussian set fixed and zeroes all seven wave
fields for primitives whose RGB amplitude norm does not exceed a threshold.
The optional storage export writes a wave-free base PLY once and a sparse
sidecar for every threshold. Active entries retain the six learned wave
scalars; the gate is omitted when it is identically one and recorded in the
sidecar otherwise.
"""

import csv
import json
import os
from argparse import ArgumentParser
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch.modules.lpips import LPIPS
from render_lite import choose_views, image_metrics, read_scene_info
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration


WAVE_FIELDS = (
    "wave_amp_r",
    "wave_amp_g",
    "wave_amp_b",
    "wave_ku",
    "wave_kv",
    "wave_phase",
    "wave_gate",
)


def parse_thresholds(value):
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one pruning threshold is required")
    if values[0] < 0:
        raise ValueError("Pruning thresholds must be non-negative")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def threshold_tag(threshold):
    if threshold == 0:
        return "0"
    return (
        f"{threshold:.12g}"
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_base_ply(source_ply, output_ply):
    """Write the exact Gaussian PLY attributes except the seven wave fields."""
    output_ply = Path(output_ply)
    if output_ply.exists():
        return output_ply.stat().st_size

    ply = PlyData.read(str(source_ply))
    vertex = ply["vertex"].data
    names = [name for name in vertex.dtype.names if name not in WAVE_FIELDS]
    missing = [name for name in WAVE_FIELDS if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"Checkpoint lacks expected wave fields: {missing}")

    dtype = [(name, vertex.dtype.fields[name][0]) for name in names]
    base = np.empty(vertex.shape[0], dtype=dtype)
    for name in names:
        base[name] = vertex[name]

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(base, "vertex")
    PlyData(
        [element],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(str(output_ply))
    return output_ply.stat().st_size


def export_sparse_sidecar(storage_dir, threshold, wave, active_mask):
    """Serialize a reconstructable sparse wave payload for one threshold."""
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    tag = threshold_tag(threshold)
    active_indices = np.flatnonzero(active_mask).astype(np.int32)
    active_wave = wave[active_mask].astype(np.float32, copy=False)
    implicit_unit_gate = bool(
        active_wave.shape[0] == 0
        or np.allclose(active_wave[:, 6], 1.0, rtol=0.0, atol=1e-7)
    )

    fields = [
        ("primitive_index", "<i4"),
        ("wave_amp_r", "<f4"),
        ("wave_amp_g", "<f4"),
        ("wave_amp_b", "<f4"),
        ("wave_ku", "<f4"),
        ("wave_kv", "<f4"),
        ("wave_phase", "<f4"),
    ]
    if not implicit_unit_gate:
        fields.append(("wave_gate", "<f4"))

    sparse = np.empty(active_indices.shape[0], dtype=np.dtype(fields))
    sparse["primitive_index"] = active_indices
    for column, name in enumerate(WAVE_FIELDS[:6]):
        sparse[name] = active_wave[:, column]
    if not implicit_unit_gate:
        sparse["wave_gate"] = active_wave[:, 6]

    raw_path = storage_dir / f"wave_sparse_tau_{tag}.npy"
    compressed_path = storage_dir / f"wave_sparse_tau_{tag}.npz"
    np.save(raw_path, sparse, allow_pickle=False)
    np.savez_compressed(compressed_path, wave=sparse)

    manifest = {
        "format": "wave2dgs_sparse_wave_v1",
        "threshold": threshold,
        "selection": "l2_rgb_amplitude_norm_gt_threshold_and_gate_active",
        "primitive_count": int(wave.shape[0]),
        "active_count": int(active_indices.shape[0]),
        "implicit_wave_gate": 1.0 if implicit_unit_gate else None,
        "base_ply": "base_point_cloud_without_wave.ply",
        "raw_sidecar": raw_path.name,
        "compressed_sidecar": compressed_path.name,
        "wave_fields": list(WAVE_FIELDS),
    }
    manifest_path = storage_dir / f"wave_sparse_tau_{tag}.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "sparse_sidecar_raw_bytes": raw_path.stat().st_size,
        "sparse_sidecar_compressed_bytes": compressed_path.stat().st_size,
        "sparse_manifest_bytes": manifest_path.stat().st_size,
        "sparse_gate_stored": int(not implicit_unit_gate),
    }


@torch.no_grad()
def evaluate_threshold(
    threshold,
    cameras,
    gaussians,
    original_wave,
    pipe,
    background,
    lpips_model,
    output_dir,
    save_images,
):
    amp_norm = torch.linalg.vector_norm(original_wave[:, :3], dim=1)
    gate_active = original_wave[:, 6] >= 0.5
    active = (amp_norm > threshold) & gate_active

    pruned = original_wave.clone()
    pruned[~active] = 0.0
    gaussians._wave_params.data.copy_(pruned)

    rows = []
    render_dir = Path(output_dir) / "renders" / f"tau_{threshold_tag(threshold)}"
    if save_images:
        render_dir.mkdir(parents=True, exist_ok=True)

    for view_index, camera in tqdm(
        list(enumerate(cameras)),
        desc=f"prune tau={threshold:g}",
        leave=False,
    ):
        image = render(camera, gaussians, pipe, background)["render"]
        gt = camera.original_image[0:3, :, :].cuda()
        psnr, l1, ssim_value, lpips_value = image_metrics(image, gt, lpips_model)
        rows.append(
            {
                "threshold": threshold,
                "view_index": view_index,
                "image_name": camera.image_name,
                "psnr": psnr,
                "l1": l1,
                "ssim": ssim_value,
                "lpips_vgg": lpips_value,
            }
        )
        if save_images:
            stem = f"{view_index:04d}_{camera.image_name}.png"
            torchvision.utils.save_image(
                torch.clamp(image, 0.0, 1.0).cpu(),
                str(render_dir / stem),
            )
        del image, gt

    return active.detach().cpu().numpy(), rows


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def main():
    parser = ArgumentParser(description="Post-training Wave-2DGS pruning sweep")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--thresholds",
        default="0,1e-4,3e-4,1e-3,3e-3,1e-2",
        type=str,
    )
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--max_test_views", default=16, type=int)
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--view_offset", default=0, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--skip_sparse_export", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    thresholds = parse_thresholds(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_state(args.quiet, getattr(args, "seed", 0))

    dataset = model.extract(args)
    if not getattr(dataset, "enable_primitive_wave", False):
        raise ValueError("The selected checkpoint/config is not wave-enabled")
    pipe = pipeline.extract(args)

    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    checkpoint = Path(dataset.model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    gaussians = GaussianModel(
        dataset.sh_degree,
        enable_primitive_wave=True,
        wave_k_init=getattr(dataset, "wave_k_init", 6.283185307),
    )
    gaussians.load_ply(str(checkpoint))
    original_wave = gaussians.get_wave_params.detach().clone()
    if original_wave.ndim != 2 or original_wave.shape[1] != 7:
        raise ValueError(f"Expected Nx7 wave parameters, got {tuple(original_wave.shape)}")

    scene_info = read_scene_info(dataset)
    camera_infos = choose_views(
        scene_info.test_cameras,
        max_views=args.max_test_views,
        stride=args.view_stride,
        offset=args.view_offset,
    )
    if not camera_infos:
        raise ValueError("No test cameras selected")
    cameras = cameraList_from_camInfos(camera_infos, 1.0, dataset)

    print(f"Checkpoint: {checkpoint}")
    print(f"Primitives: {original_wave.shape[0]}; test views: {len(cameras)}")
    print(f"Thresholds: {thresholds}")
    print("Loading official 2DGS LPIPS-VGG metric")
    lpips_model = LPIPS(net_type="vgg", version="0.1").cuda().eval()
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    wave_np = original_wave.cpu().numpy()
    primitive_count = int(wave_np.shape[0])
    dense_checkpoint_bytes = checkpoint.stat().st_size
    dense_wave_payload_bytes = primitive_count * len(WAVE_FIELDS) * 4
    storage_dir = output_dir / "storage"
    if args.skip_sparse_export:
        # Logical payload size for a fine sweep without writing every sidecar.
        base_ply_bytes = dense_checkpoint_bytes - dense_wave_payload_bytes
    else:
        base_ply_bytes = export_base_ply(
            checkpoint,
            storage_dir / "base_point_cloud_without_wave.ply",
        )

    all_view_rows = []
    summary_rows = []
    reference = None
    for threshold in thresholds:
        active, view_rows = evaluate_threshold(
            threshold,
            cameras,
            gaussians,
            original_wave,
            pipe,
            background,
            lpips_model,
            output_dir,
            args.save_images,
        )
        all_view_rows.extend(view_rows)

        active_count = int(active.sum())
        active_fraction = active_count / max(primitive_count, 1)
        if args.skip_sparse_export:
            active_wave = wave_np[active]
            gate_stored = int(active_count > 0 and not np.allclose(
                active_wave[:, 6], 1.0, rtol=0.0, atol=1e-7))
            # int32 index + six float32 learned scalars; gate is usually implicit.
            storage = {
                "sparse_sidecar_raw_bytes": active_count * (28 + 4 * gate_stored),
                "sparse_sidecar_compressed_bytes": 0,
                "sparse_manifest_bytes": 0,
                "sparse_gate_stored": gate_stored,
            }
            storage_mode = "estimated_raw_payload"
        else:
            storage = export_sparse_sidecar(storage_dir, threshold, wave_np, active)
            storage_mode = "serialized_sidecar"

        row = {
            "threshold": threshold,
            "views": len(view_rows),
            "primitives": primitive_count,
            "active_primitives": active_count,
            "active_fraction": active_fraction,
            "mean_psnr": mean(view_rows, "psnr"),
            "mean_ssim": mean(view_rows, "ssim"),
            "mean_lpips_vgg": mean(view_rows, "lpips_vgg"),
            "mean_l1": mean(view_rows, "l1"),
            "delta_psnr_vs_tau0": 0.0,
            "delta_ssim_vs_tau0": 0.0,
            "delta_lpips_vs_tau0": 0.0,
            "delta_l1_vs_tau0": 0.0,
            "dense_checkpoint_bytes": dense_checkpoint_bytes,
            "base_ply_without_wave_bytes": base_ply_bytes,
            "dense_wave_payload_bytes": dense_wave_payload_bytes,
            "storage_mode": storage_mode,
            **storage,
        }
        row["compact_payload_raw_bytes"] = (
            base_ply_bytes + row["sparse_sidecar_raw_bytes"]
        )
        row["compact_payload_compressed_bytes"] = (
            base_ply_bytes + row["sparse_sidecar_compressed_bytes"]
            if not args.skip_sparse_export
            else 0
        )
        row["checkpoint_reduction_raw_fraction"] = (
            1.0 - row["compact_payload_raw_bytes"] / dense_checkpoint_bytes
            if row["compact_payload_raw_bytes"]
            else 0.0
        )
        row["checkpoint_reduction_compressed_fraction"] = (
            1.0 - row["compact_payload_compressed_bytes"] / dense_checkpoint_bytes
            if row["compact_payload_compressed_bytes"]
            else 0.0
        )
        if reference is None:
            reference = row
        else:
            row["delta_psnr_vs_tau0"] = row["mean_psnr"] - reference["mean_psnr"]
            row["delta_ssim_vs_tau0"] = row["mean_ssim"] - reference["mean_ssim"]
            row["delta_lpips_vs_tau0"] = row["mean_lpips_vgg"] - reference["mean_lpips_vgg"]
            row["delta_l1_vs_tau0"] = row["mean_l1"] - reference["mean_l1"]
        summary_rows.append(row)
        print(
            f"tau={threshold:g}: active={active_fraction:.1%}, "
            f"PSNR={row['mean_psnr']:.4f}, "
            f"dPSNR={row['delta_psnr_vs_tau0']:+.4f}, "
            f"raw checkpoint reduction={row['checkpoint_reduction_raw_fraction']:.1%}"
        )

    gaussians._wave_params.data.copy_(original_wave)
    write_csv(output_dir / "per_view_metrics.csv", all_view_rows)
    write_csv(output_dir / "sweep_summary.csv", summary_rows)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": str(Path(dataset.model_path).resolve()),
                "source_path": str(Path(dataset.source_path).resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "iteration": iteration,
                "thresholds": thresholds,
                "max_test_views": args.max_test_views,
                "view_stride": args.view_stride,
                "view_offset": args.view_offset,
                "geometry_note": (
                    "Post-training pruning changes only wave appearance fields; "
                    "Gaussian geometry, opacity, and primitive count are unchanged."
                ),
                "storage_note": (
                    "Compact payload sizes include the wave-free base PLY and one "
                    "sparse sidecar, excluding the small JSON manifest and cfg_args."
                ),
            },
            handle,
            indent=2,
        )
    print(f"Wrote pruning sweep to {output_dir}")


if __name__ == "__main__":
    main()
