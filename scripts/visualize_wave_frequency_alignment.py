#!/usr/bin/env python3
"""Compare surviving-wave allocation with independent image high-frequency evidence."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import build_rotation, safe_state
from utils.system_utils import searchForMaxIteration


def read_scene_info(dataset):
    source = dataset.source_path
    if os.path.exists(os.path.join(source, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](source, dataset.images, dataset.eval)
    if os.path.exists(os.path.join(source, "traj.txt")) and os.path.isdir(os.path.join(source, "results")):
        return sceneLoadTypeCallbacks["Replica"](source, dataset.white_background, dataset.eval, dataset)
    raise RuntimeError(f"Unsupported scene at {source}")


def select_camera(scene_info, args, dataset):
    infos = scene_info.train_cameras if args.split == "train" else scene_info.test_cameras
    name = args.image_name
    if args.replica_frame >= 0:
        implied = f"frame{args.replica_frame:06d}"
        if name and Path(name).stem != implied:
            raise ValueError("--image_name and --replica_frame disagree")
        name = implied
    if name:
        stem = Path(name).stem
        matches = [(i, info) for i, info in enumerate(infos) if Path(info.image_name).stem == stem]
        if not matches:
            raise ValueError(f"{name!r} is not in {args.split} camera split")
        index, info = matches[0]
    else:
        if not 0 <= args.view_index < len(infos):
            raise IndexError(f"view_index must be in [0, {len(infos)-1}]")
        index, info = args.view_index, infos[args.view_index]
    return index, cameraList_from_camInfos([info], 1.0, dataset)[0]


def project_points(xyz, camera):
    ones = torch.ones_like(xyz[:, :1])
    clip = torch.cat((xyz, ones), dim=1) @ camera.full_proj_transform
    safe_w = torch.where(clip[:, 3:4].abs() < 1e-8,
                         torch.full_like(clip[:, 3:4], 1e-8), clip[:, 3:4])
    ndc = clip[:, :3] / safe_w
    x = ((ndc[:, 0] + 1.0) * camera.image_width - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * camera.image_height - 1.0) * 0.5
    return torch.stack((x, y), dim=1), ndc[:, 2]


def project_planar_footprint_contours(gaussians, camera, primitive_ids,
                                      sigma=2.0, segments=48):
    """Project learned 2DGS tangent-plane sigma contours into the image."""
    if primitive_ids.numel() == 0:
        return []
    scales = gaussians.get_scaling[primitive_ids]
    rotations = build_rotation(gaussians.get_rotation[primitive_ids])
    axis_u = rotations[:, :, 0] * scales[:, 0:1]
    axis_v = rotations[:, :, 1] * scales[:, 1:2]
    theta = torch.linspace(0.0, 2.0 * math.pi, segments + 1,
                           device=scales.device, dtype=scales.dtype)[:-1]
    offsets = sigma * (torch.cos(theta)[None, :, None] * axis_u[:, None, :] +
                       torch.sin(theta)[None, :, None] * axis_v[:, None, :])
    points = gaussians.get_xyz[primitive_ids, None, :] + offsets
    xy, _ = project_points(points.reshape(-1, 3), camera)
    return xy.reshape(primitive_ids.numel(), segments, 2).detach().cpu().numpy()


def robust_normalize(x, quantile=0.99):
    finite = x[torch.isfinite(x)]
    scale = torch.quantile(finite, quantile).clamp_min(1e-8) if finite.numel() else x.new_tensor(1.0)
    return torch.clamp(x / scale, 0.0, 1.0), float(scale.item())


def rgb_uint8(x):
    return (x.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def colorize(values, cmap_name):
    return (colormaps[cmap_name](values)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def rank_correlation(a, b):
    a = a.flatten()
    b = b.flatten()
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    return float((ar * br).sum().item() / (torch.linalg.vector_norm(ar).item() * torch.linalg.vector_norm(br).item() + 1e-8))


def make_sobel_high_frequency(gt):
    luminance = (0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2])[None, None]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=gt.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    gx = F.conv2d(luminance, sobel_x, padding=1)
    gy = F.conv2d(luminance, sobel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)[0, 0]


def save_panel(path, gt, highfreq, allocation, alignment, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.1), constrained_layout=True)
    items = [
        ("Ground truth", gt),
        ("GT high-frequency proxy (Sobel magnitude)", highfreq),
        ("Surviving-wave allocation", allocation),
        ("Sobel + surviving-wave centers", alignment),
    ]
    for ax, (title, image) in zip(axes.flat, items):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(
        f"Wave allocation alignment | high-frequency top-20% mass={metrics['wave_mass_in_hf_top20']:.3f}, "
        f"enrichment={metrics['enrichment_top20']:.2f}x, Spearman={metrics['spearman']:.3f}",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_footprint_overlay(base_rgb, contours, amplitudes, cmap_name="autumn",
                           cmap_min=0.0, cmap_max=1.0, width=2):
    """Draw projected planar footprints over an already colorized Sobel map."""
    from PIL import Image, ImageDraw
    base = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if len(contours) == 0:
        return np.asarray(base.convert("RGB")), 0.0, 0.0
    amp = np.asarray(amplitudes, dtype=np.float32)
    finite = amp[np.isfinite(amp)]
    color_low = float(np.quantile(finite, 0.05)) if finite.size else 0.0
    color_high = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    color_high = max(color_high, color_low + 1e-8)
    normalized = np.clip((amp - color_low) / (color_high - color_low), 0.0, 1.0)
    normalized = cmap_min + normalized * (cmap_max - cmap_min)
    colors = (colormaps[cmap_name](normalized)[:, :3] * 255.0 + 0.5).astype(np.uint8)
    for idx in np.argsort(amp):
        contour = contours[idx]
        points = [(float(x), float(y)) for x, y in contour
                  if np.isfinite(x) and np.isfinite(y)]
        if len(points) < 3:
            continue
        draw.line(points + [points[0]], fill=tuple(colors[idx].tolist()) + (230,),
                  width=int(width), joint="curve")
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB")), color_low, color_high


def rasterize_footprint_union(contours, width, height):
    """Rasterize the union of projected footprint interiors."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask)
    for contour in contours:
        points = np.asarray(contour, dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.shape[0] < 3:
            continue
        points[:, 0] = np.clip(points[:, 0], -width, 2 * width)
        points[:, 1] = np.clip(points[:, 1], -height, 2 * height)
        draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def save_three_panel(path, gt, sobel, sobel_footprints, metrics, topk, sigma,
                     omit_suptitle=False):
    """Save GT | Sobel | identical Sobel + projected primitive footprints."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    items = [
        ("Ground truth", gt),
        ("Sobel high-frequency proxy", sobel),
        (f"Sobel + surviving {sigma:g}σ footprints (top-{topk})", sobel_footprints),
    ]
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    if not omit_suptitle:
        fig.suptitle(
            f"Wave footprint / image high-frequency alignment | "
            f"enrichment={metrics['enrichment_top20']:.2f}x, Spearman={metrics['spearman']:.3f}",
            fontsize=14,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--view_index", default=0, type=int)
    parser.add_argument("--image_name", default="")
    parser.add_argument("--replica_frame", default=-1, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--active_threshold", default=0.01, type=float)
    parser.add_argument("--hf_quantile", default=0.99, type=float)
    parser.add_argument("--display_quantile", default=0.99, type=float)
    parser.add_argument("--hf_cmap", default="magma")
    parser.add_argument("--wave_cmap", default="inferno")
    parser.add_argument("--circle_topk", default=500, type=int,
                        help="Legacy strongest visible surviving primitives drawn as circles")
    parser.add_argument("--circle_radius", default=3.0, type=float,
                        help="Fixed projected circle radius in pixels")
    parser.add_argument("--footprint_topk", default=250, type=int,
                        help="Strongest visible surviving primitives drawn as projected ellipses")
    parser.add_argument("--footprint_sigma", default=2.0, type=float,
                        help="Gaussian footprint contour radius in standard deviations")
    parser.add_argument("--footprint_segments", default=48, type=int,
                        help="Polyline samples per projected footprint")
    parser.add_argument("--footprint_cmap", default="autumn")
    parser.add_argument("--footprint_cmap_min", default=0.0, type=float,
                        help="Lower normalized colormap value used for footprint lines")
    parser.add_argument("--footprint_cmap_max", default=1.0, type=float,
                        help="Upper normalized colormap value used for footprint lines")
    parser.add_argument("--omit_panel_suptitle", action="store_true",
                        help="Omit the enrichment/correlation line from the three-panel figure")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return args, model.extract(args), pipeline.extract(args)


@torch.no_grad()
def main():
    args, dataset, pipe = parse_args()
    safe_state(args.quiet, getattr(args, "seed", 0))
    if not dataset.source_path:
        raise ValueError("--source_path/-s is required")
    iteration = args.iteration
    if iteration < 0:
        iteration = searchForMaxIteration(os.path.join(args.wave, "point_cloud"))
    scene_info = read_scene_info(dataset)
    view_index, camera = select_camera(scene_info, args, dataset)

    wave = GaussianModel(dataset.sh_degree, enable_primitive_wave=True,
                         wave_k_init=getattr(dataset, "wave_k_init", 6.283185307))
    wave.load_ply(os.path.join(args.wave, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    gt = camera.original_image[:3].cuda().clamp(0.0, 1.0)
    amplitude = torch.linalg.vector_norm(wave._wave_params[:, :3].detach(), dim=1)
    active = amplitude > args.active_threshold
    if not bool(active.any()):
        raise RuntimeError("No wave primitives pass --active_threshold")

    amp_scale = torch.quantile(amplitude[active], 0.95).clamp_min(1e-8)
    amp_strength = torch.clamp(amplitude / amp_scale, 0.0, 1.0)
    colors = torch.zeros((amplitude.shape[0], 3), dtype=amplitude.dtype, device=amplitude.device)
    colors[active] = amp_strength[active, None].expand(-1, 3)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    allocation_pkg = render(camera, wave, pipe, background, override_color=colors)
    allocation_raw = allocation_pkg["render"].mean(dim=0)
    allocation, allocation_scale = robust_normalize(allocation_raw, args.display_quantile)

    highfreq_raw = make_sobel_high_frequency(gt)
    highfreq, highfreq_scale = robust_normalize(highfreq_raw, args.hf_quantile)
    hf_color = colorize(highfreq.detach().cpu().numpy(), args.hf_cmap)
    wave_color = colorize(allocation.detach().cpu().numpy(), args.wave_cmap)
    gt_rgb = rgb_uint8(gt)
    # Independent display maps.
    highfreq_rgb = hf_color
    allocation_rgb = wave_color
    # Two-channel overlay: cyan = image high frequency, orange = wave allocation.
    hf = highfreq.detach().cpu().numpy()[..., None]
    wa = allocation.detach().cpu().numpy()[..., None]
    base = np.clip(0.28 * gt_rgb.astype(np.float32), 0.0, 255.0)
    cyan = np.array([20., 210., 220.], dtype=np.float32)
    orange = np.array([245., 105., 20.], dtype=np.float32)
    alignment = np.clip(base + 0.85 * hf * cyan + 0.85 * wa * orange, 0.0, 255.0).astype(np.uint8)

    # A more literal alignment view: use a grayscale Sobel background and
    # overlay fixed-size circles for the strongest visible surviving waves.
    projected_xy, ndc_z = project_points(wave.get_xyz, camera)
    visible_active = allocation_pkg["visibility_filter"] & active & (ndc_z >= 0.0) & (ndc_z <= 1.0)
    visible_ids = torch.nonzero(visible_active, as_tuple=False).squeeze(1)
    visible_amp = amplitude[visible_ids]
    circle_order = torch.argsort(visible_amp, descending=True)[:max(0, args.circle_topk)]
    circle_ids = visible_ids[circle_order]
    circle_strength = torch.clamp(amplitude[circle_ids] / amp_scale, 0.0, 1.0).cpu().numpy()
    circle_xy = projected_xy[circle_ids].cpu().numpy()
    sobel_gray = np.repeat((highfreq.detach().cpu().numpy()[..., None] * 255.0), 3, axis=2).astype(np.uint8)
    from PIL import Image, ImageDraw
    circle_image = Image.fromarray(sobel_gray, mode="RGB").convert("RGBA")
    circle_layer = Image.new("RGBA", circle_image.size, (0, 0, 0, 0))
    circle_draw = ImageDraw.Draw(circle_layer)
    circle_colors = (colormaps["autumn"](circle_strength)[:, :3] * 255.0 + 0.5).astype(np.uint8)
    radius = float(args.circle_radius)
    for (x, y), color in zip(circle_xy, circle_colors):
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < gt.shape[2] and 0 <= y < gt.shape[1]:
            circle_draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                                outline=tuple(color.tolist()) + (235,), width=2)
    sobel_circle = np.asarray(Image.alpha_composite(circle_image, circle_layer).convert("RGB"))

    # True projected planar footprints. Panel 3 uses the exact same
    # colorized Sobel base as panel 2; only the footprint overlay is added.
    footprint_order = torch.argsort(visible_amp, descending=True)[:max(0, args.footprint_topk)]
    footprint_ids = visible_ids[footprint_order]
    footprint_amp = visible_amp[footprint_order]
    contours = project_planar_footprint_contours(
        wave, camera, footprint_ids, sigma=args.footprint_sigma,
        segments=max(8, args.footprint_segments))
    sobel_footprints, footprint_color_low, footprint_color_high = draw_footprint_overlay(
        highfreq_rgb, contours, footprint_amp.detach().cpu().numpy(),
        cmap_name=args.footprint_cmap,
        cmap_min=args.footprint_cmap_min,
        cmap_max=args.footprint_cmap_max,
        width=2)

    footprint_union_np = rasterize_footprint_union(
        contours, camera.image_width, camera.image_height)
    footprint_union = torch.from_numpy(footprint_union_np).to(
        device=highfreq_raw.device, dtype=torch.bool)
    footprint_coverage = footprint_union.float().mean()
    global_sobel_mean = highfreq_raw.mean()
    if bool(footprint_union.any()):
        footprint_sobel_mean = highfreq_raw[footprint_union].mean()
    else:
        footprint_sobel_mean = highfreq_raw.new_tensor(0.0)
    footprint_sobel_enrichment = footprint_sobel_mean / global_sobel_mean.clamp_min(1e-8)

    top20 = highfreq >= torch.quantile(highfreq, 0.80)
    if bool(footprint_union.any()):
        footprint_top20_fraction = top20[footprint_union].float().mean()
    else:
        footprint_top20_fraction = highfreq_raw.new_tensor(0.0)
    footprint_top20_enrichment = footprint_top20_fraction / 0.20
    wave_mass = allocation.clamp_min(0.0)
    total_mass = wave_mass.sum().clamp_min(1e-8)
    mass_in = wave_mass[top20].sum() / total_mass
    enrichment = mass_in / 0.20
    allocation_sobel_spearman = rank_correlation(highfreq, allocation)
    metrics = {
        "source_path": dataset.source_path, "wave": args.wave, "iteration": iteration,
        "split": args.split, "view_index": view_index, "image_name": camera.image_name,
        "active_threshold": args.active_threshold,
        "total_primitives": int(amplitude.numel()),
        "active_primitives": int(active.sum().item()),
        "active_fraction": float(active.float().mean().item()),
        "high_frequency_definition": "Sobel luminance gradient magnitude",
        "high_frequency_top20_fraction": 0.20,
        "wave_mass_in_hf_top20": float(mass_in.item()),
        "enrichment_top20": float(enrichment.item()),
        "spearman": allocation_sobel_spearman,
        "allocation_sobel_spearman": allocation_sobel_spearman,
        "high_frequency_p99_raw": highfreq_scale,
        "wave_allocation_p99_raw": allocation_scale,
        "circle_topk": int(args.circle_topk),
        "circle_radius_px": float(args.circle_radius),
        "visible_active_primitives": int(visible_ids.numel()),
        "drawn_circles": int(circle_ids.numel()),
        "footprint_topk": int(args.footprint_topk),
        "footprint_sigma": float(args.footprint_sigma),
        "footprint_segments": int(args.footprint_segments),
        "drawn_footprints": int(len(contours)),
        "footprint_coverage_fraction": float(footprint_coverage.item()),
        "footprint_sobel_mean_raw": float(footprint_sobel_mean.item()),
        "global_sobel_mean_raw": float(global_sobel_mean.item()),
        "footprint_sobel_mean_enrichment": float(footprint_sobel_enrichment.item()),
        "footprint_sobel_top20_fraction": float(footprint_top20_fraction.item()),
        "footprint_sobel_top20_enrichment": float(footprint_top20_enrichment.item()),
        "footprint_cmap_min": float(args.footprint_cmap_min),
        "footprint_cmap_max": float(args.footprint_cmap_max),
        "footprint_cmap": args.footprint_cmap,
        "footprint_color_scale_q05": footprint_color_low,
        "footprint_color_scale_q95": footprint_color_high,
        "interpretation": "alignment is spatial association, not a causal claim",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(gt_rgb).save(out / "gt.png")
    Image.fromarray(highfreq_rgb).save(out / "gt_high_frequency_sobel.png")
    Image.fromarray(allocation_rgb).save(out / "surviving_wave_allocation.png")
    Image.fromarray(alignment).save(out / "wave_frequency_alignment_overlay.png")
    Image.fromarray(sobel_circle).save(out / "sobel_with_surviving_wave_circles.png")
    Image.fromarray(sobel_footprints).save(out / "sobel_with_surviving_wave_footprints.png")
    save_panel(out / "wave_frequency_alignment_panel.pdf", gt_rgb, highfreq_rgb, allocation_rgb, sobel_circle, metrics)
    Image.fromarray(footprint_union_np.astype(np.uint8) * 255).save(out / "footprint_union_mask.png")
    save_three_panel(out / "wave_frequency_alignment_three_panel.pdf", gt_rgb, highfreq_rgb,
                     sobel_footprints, metrics, args.footprint_topk,
                     args.footprint_sigma, args.omit_panel_suptitle)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
