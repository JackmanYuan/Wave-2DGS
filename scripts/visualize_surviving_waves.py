#!/usr/bin/env python3
"""Visualize post-training surviving Wave-2DGS primitives.

The threshold is the same amplitude-norm rule used by
``evaluate_wave_pruning.py``.  This tool does not modify the checkpoint: it
exports all retained primitive locations/parameters, a colored center-point
PLY, and an image-space projected footprint overlay for one camera view.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import sceneLoadTypeCallbacks
from scripts.visualize_wave_allocation import (
    draw_footprints,
    ensure_dir,
    project_planar_footprint_contours,
    project_points,
    read_scene_info,
    resolve_iteration,
    save_footprint_colorbar,
    save_rgb,
    to_uint8,
)
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--wave", required=True, help="Wave-2DGS model directory")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--view_index", default=0, type=int)
    parser.add_argument("--image_name", default="")
    parser.add_argument("--replica_frame", default=-1, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--active_threshold", default=0.01, type=float,
                        help="Keep primitives with RGB amplitude L2 norm above this value")
    parser.add_argument("--topk", default=250, type=int,
                        help="Maximum strongest surviving footprints drawn in the view overlay")
    parser.add_argument("--heatmap_cmap", default="inferno",
                        choices=("hot", "turbo", "viridis", "plasma", "magma", "inferno"),
                        help="Colormap for the all-survivor image-space heatmap")
    parser.add_argument("--heatmap_gain", default=1.0, type=float,
                        help="Display gain after robust heatmap normalization")
    parser.add_argument("--footprint_cmap", default="autumn",
                        choices=("autumn", "hot", "turbo", "viridis", "plasma", "magma", "inferno"))
    parser.add_argument("--footprint_sigma", default=2.0, type=float)
    parser.add_argument("--footprint_segments", default=48, type=int)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return args, model.extract(args), pipeline.extract(args)


def selected_camera(scene_info, args):
    cam_infos = scene_info.train_cameras if args.split == "train" else scene_info.test_cameras
    requested = args.image_name
    if args.replica_frame >= 0:
        implied = f"frame{args.replica_frame:06d}"
        if requested and Path(requested).stem != implied:
            raise ValueError("--image_name and --replica_frame specify different cameras")
        requested = implied
    if requested:
        name = Path(requested).stem
        matches = [(idx, info) for idx, info in enumerate(cam_infos)
                   if Path(info.image_name).stem == name]
        if not matches:
            examples = ", ".join(Path(info.image_name).stem for info in cam_infos[:5])
            raise ValueError(f"{requested!r} is not in {args.split}; examples: {examples}")
        index, info = matches[0]
    else:
        if not 0 <= args.view_index < len(cam_infos):
            raise IndexError(f"view_index must be in [0, {len(cam_infos)-1}]")
        index, info = args.view_index, cam_infos[args.view_index]
    return index, cameraList_from_camInfos([info], 1.0, args._dataset)[0]


def save_center_ply(path, xyz, strength, cmap_name):
    if xyz.shape[0] == 0:
        return
    colors = (colormaps[cmap_name](np.clip(strength, 0.0, 1.0))[:, :3] * 255.0 + 0.5).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
             ("amplitude", "f4")]
    data = np.empty(xyz.shape[0], dtype=dtype)
    data["x"], data["y"], data["z"] = xyz.astype(np.float32).T
    data["red"], data["green"], data["blue"] = colors.T
    data["amplitude"] = strength.astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


@torch.no_grad()
def main():
    args, dataset, pipe = parse_args()
    args._dataset = dataset
    safe_state(args.quiet, getattr(args, "seed", 0))
    if not dataset.source_path:
        raise ValueError("--source_path/-s is required")
    iteration = resolve_iteration(args.wave, args.iteration)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    scene_info = read_scene_info(dataset)
    selected_index, camera = selected_camera(scene_info, args)
    print(f"Visualizing surviving waves on {args.split}[{selected_index}] = {camera.image_name}")

    wave = GaussianModel(dataset.sh_degree,
                         enable_primitive_wave=True,
                         wave_k_init=getattr(dataset, "wave_k_init", 6.283185307))
    wave.load_ply(os.path.join(args.wave, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    params = wave._wave_params.detach()
    amplitude = torch.linalg.vector_norm(params[:, :3], dim=1)
    active = amplitude > args.active_threshold
    ids = torch.nonzero(active, as_tuple=False).squeeze(1)
    if ids.numel() == 0:
        raise RuntimeError("No primitives survive the requested amplitude threshold")

    strength = amplitude[ids]
    q95 = torch.quantile(strength, 0.95).clamp_min(1e-8)
    strength_norm = torch.clamp(strength / q95, 0.0, 1.0)
    xyz = wave.get_xyz[ids].detach().cpu().numpy()
    save_center_ply(out_dir / "surviving_wave_centers.ply", xyz, strength_norm.cpu().numpy(), args.footprint_cmap)

    with (out_dir / "surviving_wave_primitives.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["primitive_id", "x", "y", "z", "amplitude_norm", "k_u", "k_v", "frequency_norm", "phase"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for primitive_id, amp, pos in zip(ids.cpu().tolist(), strength.cpu().tolist(), xyz.tolist()):
            ku, kv, phase = params[primitive_id, 3:6].cpu().tolist()
            writer.writerow({
                "primitive_id": primitive_id, "x": pos[0], "y": pos[1], "z": pos[2],
                "amplitude_norm": amp, "k_u": ku, "k_v": kv,
                "frequency_norm": math.sqrt(ku * ku + kv * kv), "phase": phase,
            })

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    render_pkg = render(camera, wave, pipe, background)
    visible = render_pkg["visibility_filter"]
    projected_xy, ndc_z = project_points(wave.get_xyz, camera)
    visible = visible & (ndc_z >= 0.0) & (ndc_z <= 1.0) & active
    visible_ids = torch.nonzero(visible, as_tuple=False).squeeze(1)
    visible_amp = amplitude[visible_ids]
    order = torch.argsort(visible_amp, descending=True)[:max(0, args.topk)]
    draw_ids = visible_ids[order]
    draw_amp = visible_amp[order]
    contours = project_planar_footprint_contours(
        wave, camera, draw_ids, sigma=args.footprint_sigma, segments=args.footprint_segments)
    gt_rgb = to_uint8(camera.original_image[:3].cuda().clamp(0.0, 1.0))
    save_rgb(gt_rgb, out_dir / "gt.png", crop=args.crop)
    overlay, color_low, color_high = draw_footprints(
        gt_rgb, contours, draw_amp, args.active_threshold, args.footprint_cmap, crop=args.crop)
    overlay.save(out_dir / "surviving_wave_footprints_overlay.png")
    save_footprint_colorbar(out_dir / "surviving_wave_colorbar.png", args.footprint_cmap,
                            color_low, color_high)

    # Aggregate all visible surviving primitives into an image-space heatmap.
    # The renderer supplies visibility, alpha compositing, and the projected
    # Gaussian support; unlike the footprint overlay, no top-K selection is used.
    strength_scale = torch.quantile(strength, 0.95).clamp_min(1e-8)
    strength_all = torch.clamp(amplitude / strength_scale, 0.0, 1.0)
    heat_color = torch.zeros((params.shape[0], 3), dtype=amplitude.dtype, device=amplitude.device)
    heat_color[active] = strength_all[active, None].expand(-1, 3)
    heat_pkg = render(camera, wave, pipe, torch.zeros(3, dtype=torch.float32, device="cuda"),
                      override_color=heat_color)
    heat_raw = heat_pkg["render"].mean(dim=0)
    heat_scale = torch.quantile(heat_raw[torch.isfinite(heat_raw)], 0.99).clamp_min(1e-8)
    heat = torch.clamp(heat_raw / heat_scale * args.heatmap_gain, 0.0, 1.0)
    heat_rgb = (colormaps[args.heatmap_cmap](heat.detach().cpu().numpy())[..., :3] * 255.0 + 0.5).astype(np.uint8)
    heat_overlay = (
        (1.0 - 0.72 * heat.detach().cpu().numpy()[..., None]) * gt_rgb.astype(np.float32)
        + (0.72 * heat.detach().cpu().numpy()[..., None]) * heat_rgb.astype(np.float32)
    ).clip(0.0, 255.0).astype(np.uint8)
    save_rgb(heat_rgb, out_dir / "surviving_wave_heatmap.png", crop=args.crop)
    save_rgb(heat_overlay, out_dir / "surviving_wave_heatmap_overlay.png", crop=args.crop)
    save_footprint_colorbar(out_dir / "surviving_wave_heatmap_colorbar.png", args.heatmap_cmap,
                            0.0, float(heat_scale.item()))

    # A sparse center-only overlay separates allocation locations from footprint
    # geometry and remains readable even when many primitives survive.
    center_image = Image.fromarray(gt_rgb, mode="RGB").convert("RGBA")
    center_layer = Image.new("RGBA", center_image.size, (0, 0, 0, 0))
    center_draw = ImageDraw.Draw(center_layer)
    center_xy = projected_xy[visible_ids].detach().cpu().numpy()
    center_colors = (colormaps[args.footprint_cmap](
        torch.clamp(visible_amp / strength_scale, 0.0, 1.0).cpu().numpy())[:, :3] * 255.0 + 0.5).astype(np.uint8)
    for (x, y), color in zip(center_xy, center_colors):
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < gt_rgb.shape[1] and 0 <= y < gt_rgb.shape[0]:
            center_draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                                fill=tuple(color.tolist()) + (190,))
    centers_overlay = Image.alpha_composite(center_image, center_layer).convert("RGB")
    if args.crop is not None:
        centers_overlay = centers_overlay.crop(tuple(args.crop))
    centers_overlay.save(out_dir / "surviving_wave_centers_overlay.png")

    metadata = {
        "source_path": dataset.source_path, "wave": args.wave, "iteration": iteration,
        "split": args.split, "view_index": selected_index, "image_name": camera.image_name,
        "active_threshold": args.active_threshold, "total_primitives": int(params.shape[0]),
        "surviving_primitives": int(ids.numel()),
        "surviving_fraction": float(ids.numel() / max(params.shape[0], 1)),
        "visible_surviving_primitives": int(visible_ids.numel()),
        "drawn_footprints": int(draw_ids.numel()), "footprint_sigma": args.footprint_sigma,
        "footprint_cmap": args.footprint_cmap,
        "strength_normalization": "amplitude_l2_q95_over_surviving_primitives",
        "selection": "l2_rgb_amplitude_norm_gt_threshold",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
