#!/usr/bin/env python3
"""Render paper-ready Wave-2DGS allocation and footprint diagnostics.

The allocation map is a visibility/alpha-weighted proxy: the wave model is
rendered with wave modulation disabled and each primitive colored by its RGB
wave-amplitude norm.  It therefore shows where visible, high-amplitude wave
primitives support the image, rather than merely where projected centres lie.
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
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import build_rotation, safe_state
from utils.system_utils import searchForMaxIteration


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_scene_info(dataset):
    source = dataset.source_path
    if os.path.exists(os.path.join(source, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](source, dataset.images, dataset.eval)
    if os.path.exists(os.path.join(source, "traj.txt")) and os.path.isdir(os.path.join(source, "results")):
        print("Found Replica RGB-D sequence, assuming Replica data set!")
        return sceneLoadTypeCallbacks["Replica"](source, dataset.white_background, dataset.eval, dataset)
    if os.path.exists(os.path.join(source, "transforms_train.json")):
        return sceneLoadTypeCallbacks["Blender"](source, dataset.white_background, dataset.eval)
    raise RuntimeError(f"Could not recognize scene type at {source}")


def resolve_iteration(model_path, iteration):
    if iteration >= 0:
        return iteration
    return searchForMaxIteration(os.path.join(model_path, "point_cloud"))


def project_points(xyz, camera):
    ones = torch.ones_like(xyz[:, :1])
    clip = torch.cat((xyz, ones), dim=1) @ camera.full_proj_transform
    safe_w = torch.where(clip[:, 3:4].abs() < 1e-8, torch.full_like(clip[:, 3:4], 1e-8), clip[:, 3:4])
    ndc = clip[:, :3] / safe_w
    x = ((ndc[:, 0] + 1.0) * camera.image_width - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * camera.image_height - 1.0) * 0.5
    return torch.stack((x, y), dim=1), ndc[:, 2]


def to_uint8(image):
    if torch.is_tensor(image):
        image = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def scalar_colormap(values, cmap_name="magma"):
    values = np.clip(values, 0.0, 1.0)
    return (colormaps[cmap_name](values)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def signed_colormap(values, scale, cmap_name="BrBG"):
    """Map signed values to a diverging colour map centred exactly at zero."""
    scale = max(float(scale), 1e-8)
    normalized = np.clip(0.5 + 0.5 * values / scale, 0.0, 1.0)
    return scalar_colormap(normalized, cmap_name)


def crop_array(array, crop):
    if crop is None:
        return array
    x0, y0, x1, y1 = crop
    return array[y0:y1, x0:x1]


def save_rgb(array, path, crop=None):
    Image.fromarray(crop_array(array, crop), mode="RGB").save(path)


def save_scalar_map(values, path, crop=None, cmap="magma"):
    image = scalar_colormap(values, cmap)
    save_rgb(image, path, crop)


def normalize_map(tensor, quantile=0.99):
    values = tensor.detach().float().squeeze()
    positive = values[torch.isfinite(values)]
    if positive.numel() == 0:
        return torch.zeros_like(values), 1.0
    scale = torch.quantile(positive, quantile).clamp_min(1e-8)
    return torch.clamp(values / scale, 0.0, 1.0), float(scale.item())


def error_map(image, gt, gain):
    return torch.clamp(torch.mean(torch.abs(image - gt), dim=0) * gain, 0.0, 1.0)


def make_overlay(base_rgb, heat, opacity=0.70):
    heat_rgb = scalar_colormap(heat, "magma").astype(np.float32) / 255.0
    alpha = np.clip(heat[..., None] * opacity, 0.0, 1.0)
    result = base_rgb.astype(np.float32) / 255.0
    result = result * (1.0 - alpha) + heat_rgb * alpha
    return (result * 255.0 + 0.5).astype(np.uint8)


def project_planar_footprint_contours(gaussians, camera, primitive_ids, sigma=2.0, segments=48):
    """Project the local planar 2DGS sigma contour through the camera.

    A 2DGS primitive has two learned tangent-plane axes.  Sampling the ellipse
    in that plane and projecting every point retains perspective effects and
    is more faithful than using the rasterizer's scalar bounding radius.
    """
    if primitive_ids.numel() == 0:
        return []
    scales = gaussians.get_scaling[primitive_ids]
    rotations = build_rotation(gaussians.get_rotation[primitive_ids])
    axis_u = rotations[:, :, 0] * scales[:, 0:1]
    axis_v = rotations[:, :, 1] * scales[:, 1:2]
    theta = torch.linspace(0.0, 2.0 * math.pi, segments + 1, device=scales.device, dtype=scales.dtype)[:-1]
    offsets = sigma * (torch.cos(theta)[None, :, None] * axis_u[:, None, :] +
                       torch.sin(theta)[None, :, None] * axis_v[:, None, :])
    points = gaussians.get_xyz[primitive_ids, None, :] + offsets
    xy, _ = project_points(points.reshape(-1, 3), camera)
    return xy.reshape(primitive_ids.numel(), segments, 2).detach().cpu().numpy()


def draw_footprints(gt_rgb, contours, amp, active_threshold, cmap_name, crop=None):
    """Overlay projected 2σ contours colored by thresholded wave strength."""
    base = Image.fromarray(gt_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if amp.numel() == 0:
        result = base.convert("RGB")
        return result.crop(tuple(crop)) if crop is not None else result, 0.0, 0.0

    # Threshold decides eligibility only. The displayed top-K amplitudes occupy
    # a narrow high tail, so their robust local range drives colour variation.
    color_low = torch.quantile(amp, 0.05)
    color_high = torch.quantile(amp, 0.95).clamp_min(color_low + 1e-8)
    normalized = torch.clamp((amp - color_low) / (color_high - color_low), 0.0, 1.0)
    colors = (colormaps[cmap_name](normalized.detach().cpu().numpy())[:, :3] * 255.0 + 0.5).astype(np.uint8)
    order = torch.argsort(amp)
    for local_rank in order.detach().cpu().tolist():
        red, green, blue = colors[local_rank].tolist()
        contour = contours[local_rank]
        points = [(float(x), float(y)) for x, y in contour if np.isfinite(x) and np.isfinite(y)]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(red, green, blue, 235), width=2, joint="curve")
    result = Image.alpha_composite(base, overlay).convert("RGB")
    if crop is not None:
        result = result.crop(tuple(crop))
    return result, float(color_low.item()), float(color_high.item())


def save_footprint_colorbar(path, cmap_name, color_low, color_high):
    width, height = 480, 54
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    rgb = (colormaps[cmap_name](ramp)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    bar = Image.fromarray(np.repeat(rgb, 24, axis=0), mode="RGB")
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(bar, (0, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((0, 2), f"top-K amplitude Q05: {color_low:.3g}", fill="black", font=font)
    right = f"Q95 {color_high:.3g}"
    draw.text((max(0, width - 7 * len(right)), 2), right, fill="black", font=font)
    canvas.save(path)


def save_signed_colorbar(path, cmap_name, scale, label):
    width, height = 600, 66
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    rgb = (colormaps[cmap_name](ramp)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    bar = Image.fromarray(np.repeat(rgb, 24, axis=0), mode="RGB")
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(bar, (0, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((0, 3), f"2DGS better  {-scale:.3g}", fill="black", font=font)
    centre = "similar"
    draw.text(((width - 7 * len(centre)) // 2, 3), centre, fill="black", font=font)
    right = f"{scale:.3g}  Wave-2DGS better"
    draw.text((max(0, width - 7 * len(right)), 3), right, fill="black", font=font)
    draw.text((0, 49), label, fill="black", font=font)
    canvas.save(path)


def contact_sheet(items, path, columns=3):
    font = ImageFont.load_default()
    images = [image.convert("RGB") for _, image in items]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 18
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.rectangle((x, y, x + width, y + label_height), fill="black")
        draw.text((x + 4, y + 4), label, fill="white", font=font)
    sheet.save(path)


def load_model(path, iteration, sh_degree, wave_enabled, wave_k_init):
    model = GaussianModel(sh_degree, enable_primitive_wave=wave_enabled, wave_k_init=wave_k_init)
    model.load_ply(os.path.join(path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Paper visuals for learned Wave-2DGS allocation")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--baseline", required=True, help="Original 2DGS model directory")
    parser.add_argument("--wave", required=True, help="Wave-2DGS model directory")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--view_index", default=0, type=int, help="Index within the selected train/test camera list")
    parser.add_argument("--image_name", default="", help="Exact camera image name, e.g. frame000240")
    parser.add_argument("--replica_frame", default=-1, type=int, help="Replica raw frame index; shorthand for --image_name frameXXXXXX")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--active_threshold", default=0.01, type=float)
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--pattern_topk", default=500, type=int,
                        help="Visible active primitives used by the red/blue learned-wave projection")
    parser.add_argument("--pattern_contrast", default=0.48, type=float,
                        help="Synthetic signed RGB contrast applied to the learned cosine")
    parser.add_argument("--pattern_overlay_alpha", default=0.82, type=float,
                        help="Opacity multiplier when compositing the pattern render over GT")
    parser.add_argument("--pattern_strength_gamma", default=0.50, type=float,
                        help="Gamma applied to normalized learned amplitude for stripe visibility")
    parser.add_argument("--pattern_uniform_strength", action="store_true",
                        help="Display selected stripes uniformly instead of weighting by learned amplitude")
    parser.add_argument("--footprint_cmap", default="autumn_r", choices=("autumn_r", "autumn", "hot", "turbo", "viridis", "plasma", "magma", "inferno"),
                        help="Sequential colormap for thresholded wave footprint strength")
    parser.add_argument("--max_base_footprints", default=1500, type=int,
                        help="Deprecated compatibility option; no longer used by true footprint contours")
    parser.add_argument("--footprint_sigma", default=2.0, type=float,
                        help="Local planar Gaussian contour extent in standard deviations")
    parser.add_argument("--footprint_segments", default=48, type=int,
                        help="Polyline segments used to project each 2σ footprint contour")
    parser.add_argument("--error_gain", default=4.0, type=float)
    parser.add_argument("--contribution_gain", default=1.0, type=float,
                        help="Extra display gain after robust P99 normalization (default: no extra gain)")
    parser.add_argument("--allocation_quantile", default=0.99, type=float)
    parser.add_argument("--improvement_quantile", default=0.99, type=float,
                        help="Absolute quantile used for the symmetric rendering-improvement display range")
    parser.add_argument("--improvement_cmap", default="BrBG",
                        help="Diverging map: low colours favour 2DGS, high colours favour Wave-2DGS")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return args, model.extract(args), pipeline.extract(args)


@torch.no_grad()
def main():
    args, dataset, pipe = parse_args()
    safe_state(args.quiet, getattr(args, "seed", 0))
    if not dataset.source_path:
        raise ValueError("--source_path/-s is required")
    wave_iteration = resolve_iteration(args.wave, args.iteration)
    baseline_iteration = resolve_iteration(args.baseline, args.iteration)
    if wave_iteration != baseline_iteration:
        raise ValueError(f"Iteration mismatch: baseline={baseline_iteration}, wave={wave_iteration}")
    iteration = wave_iteration
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    scene_info = read_scene_info(dataset)
    cam_infos = scene_info.train_cameras if args.split == "train" else scene_info.test_cameras
    requested_name = args.image_name
    if args.replica_frame >= 0:
        implied_name = f"frame{args.replica_frame:06d}"
        if requested_name and requested_name != implied_name:
            raise ValueError("--image_name and --replica_frame specify different cameras")
        requested_name = implied_name
    if requested_name:
        normalized = Path(requested_name).stem
        matches = [(idx, info) for idx, info in enumerate(cam_infos) if Path(info.image_name).stem == normalized]
        if not matches:
            examples = ", ".join(Path(info.image_name).stem for info in cam_infos[:5])
            raise ValueError(f"{requested_name!r} is not in the {args.split} camera split. Examples: {examples}")
        selected_index, selected_info = matches[0]
    else:
        if not 0 <= args.view_index < len(cam_infos):
            raise IndexError(f"view_index {args.view_index} unavailable for {args.split}; valid range is [0, {len(cam_infos)-1}]")
        selected_index, selected_info = args.view_index, cam_infos[args.view_index]
    camera = cameraList_from_camInfos([selected_info], 1.0, dataset)[0]
    print(f"Visualizing {args.split}[{selected_index}] = {camera.image_name}, iteration {iteration}")

    baseline = load_model(args.baseline, iteration, dataset.sh_degree, False, dataset.wave_k_init)
    wave = load_model(args.wave, iteration, dataset.sh_degree, True, dataset.wave_k_init)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    black_background = torch.zeros(3, dtype=torch.float32, device="cuda")

    baseline_pkg = render(camera, baseline, pipe, background)
    wave_pkg = render(camera, wave, pipe, background)
    gt = camera.original_image[:3].cuda().clamp(0.0, 1.0)
    baseline_image = baseline_pkg["render"].clamp(0.0, 1.0)
    wave_image = wave_pkg["render"].clamp(0.0, 1.0)

    wave.enable_primitive_wave = False
    try:
        wave_off = render(camera, wave, pipe, background)["render"].clamp(0.0, 1.0)
        # get_wave_params intentionally returns None while the renderer-side wave switch is off.
        amp = torch.linalg.vector_norm(wave._wave_params[:, :3], dim=1)
        amp_scale = torch.quantile(amp, args.allocation_quantile).clamp_min(1e-8)
        allocation_color = torch.clamp(amp / amp_scale, 0.0, 1.0)[:, None].expand(-1, 3)
        allocation_pkg = render(camera, wave, pipe, black_background, override_color=allocation_color)
    finally:
        wave.enable_primitive_wave = True
    allocation, allocation_scale = normalize_map(allocation_pkg["render"].mean(dim=0), args.allocation_quantile)
    contribution, contribution_scale = normalize_map(torch.mean(torch.abs(wave_image - wave_off), dim=0), args.allocation_quantile)

    baseline_l1 = torch.mean(torch.abs(baseline_image - gt), dim=0)
    wave_l1 = torch.mean(torch.abs(wave_image - gt), dim=0)
    rendering_improvement = baseline_l1 - wave_l1
    finite_improvement = rendering_improvement[torch.isfinite(rendering_improvement)].abs()
    improvement_scale = (float(torch.quantile(finite_improvement, args.improvement_quantile).clamp_min(1e-8).item())
                         if finite_improvement.numel() else 1.0)

    xy, ndc_z = project_points(wave.get_xyz, camera)
    visible = wave_pkg["visibility_filter"] & (ndc_z >= 0.0) & (ndc_z <= 1.0)
    xy_visible = xy[visible]
    radii_visible = wave_pkg["radii"][visible]
    amp_visible = amp[visible]
    gt_rgb = to_uint8(gt)
    active_local = torch.nonzero(amp_visible >= args.active_threshold, as_tuple=False).squeeze(1)
    top_count_for_overlay = min(args.topk, active_local.numel())
    selected_local = (active_local[torch.topk(amp_visible[active_local], top_count_for_overlay).indices]
                      if top_count_for_overlay else torch.empty(0, dtype=torch.long, device=amp_visible.device))
    visible_ids = torch.nonzero(visible, as_tuple=False).squeeze(1)
    selected_ids = visible_ids[selected_local]
    selected_amp = amp_visible[selected_local]
    contours = project_planar_footprint_contours(wave, camera, selected_ids,
                                                 sigma=args.footprint_sigma,
                                                 segments=args.footprint_segments)
    footprint, footprint_color_low, footprint_color_high = draw_footprints(
        gt_rgb, contours, selected_amp, args.active_threshold, args.footprint_cmap, crop=args.crop)

    # Renderer-native diagnostic: preserve each selected primitive's learned
    # frequency, phase, geometry, opacity, and visibility, but replace its RGB
    # amplitude with a common red/blue signed palette. Non-selected primitives
    # remain black occluders, so the diagnostic respects front-to-back
    # transmittance instead of drawing projected patterns over hidden surfaces.
    pattern_count = min(args.pattern_topk, int(active_local.numel()))
    pattern_local = (
        active_local[torch.topk(amp_visible[active_local], pattern_count).indices]
        if pattern_count > 0 else
        torch.empty(0, dtype=torch.long, device=amp_visible.device)
    )
    pattern_ids = visible_ids[pattern_local]
    pattern_strength = torch.ones(
        pattern_count, dtype=amp.dtype, device=amp.device
    )
    if pattern_count > 0 and not args.pattern_uniform_strength:
        pattern_strength = torch.clamp(
            amp_visible[pattern_local] / amp_scale, 0.0, 1.0
        ).pow(args.pattern_strength_gamma)

    original_wave = wave._wave_params.detach().clone()
    pattern_base = torch.zeros(
        (wave.get_xyz.shape[0], 3), dtype=amp.dtype, device=amp.device
    )
    pattern_mask_color = torch.zeros_like(pattern_base)
    if pattern_count > 0:
        pattern_base[pattern_ids] = torch.tensor(
            [0.50, 0.08, 0.50], dtype=amp.dtype, device=amp.device
        )
        pattern_mask_color[pattern_ids] = 1.0
    try:
        wave.enable_primitive_wave = True
        wave._wave_params.data[:, :3].zero_()
        wave._wave_params.data[:, 6].zero_()
        if pattern_count > 0:
            signed_rgb = torch.tensor(
                [args.pattern_contrast, 0.0, -args.pattern_contrast],
                dtype=amp.dtype, device=amp.device,
            )
            wave._wave_params.data[pattern_ids, :3] = signed_rgb
            wave._wave_params.data[pattern_ids, 6] = pattern_strength
        pattern_pkg = render(
            camera, wave, pipe, black_background, override_color=pattern_base
        )
        wave.enable_primitive_wave = False
        pattern_mask_pkg = render(
            camera, wave, pipe, black_background,
            override_color=pattern_mask_color,
        )
    finally:
        wave._wave_params.data.copy_(original_wave)
        wave.enable_primitive_wave = True

    pattern_rgb = pattern_pkg["render"].clamp(0.0, 1.0)
    pattern_mask = pattern_mask_pkg["render"].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    pattern_white = (pattern_rgb + (1.0 - pattern_mask)).clamp(0.0, 1.0)
    pattern_overlay = (
        args.pattern_overlay_alpha * pattern_rgb
        + (1.0 - args.pattern_overlay_alpha * pattern_mask) * gt
    ).clamp(0.0, 1.0)

    crop = args.crop
    gt_rgb = crop_array(gt_rgb, crop)
    baseline_rgb = crop_array(to_uint8(baseline_image), crop)
    wave_rgb = crop_array(to_uint8(wave_image), crop)
    wave_off_rgb = crop_array(to_uint8(wave_off), crop)
    base_error = crop_array(scalar_colormap(error_map(baseline_image, gt, args.error_gain).cpu().numpy(), "inferno"), crop)
    wave_error = crop_array(scalar_colormap(error_map(wave_image, gt, args.error_gain).cpu().numpy(), "inferno"), crop)
    allocation_np = allocation.cpu().numpy()
    contribution_np = contribution.cpu().numpy()
    allocation_rgb = crop_array(scalar_colormap(allocation_np, "magma"), crop)
    allocation_overlay = crop_array(make_overlay(to_uint8(gt), allocation_np), crop)
    contribution_rgb = crop_array(scalar_colormap(np.clip(contribution_np * args.contribution_gain, 0.0, 1.0), "inferno"), crop)
    improvement_np = rendering_improvement.detach().cpu().numpy()
    improvement_rgb = crop_array(signed_colormap(improvement_np, improvement_scale,
                                                 args.improvement_cmap), crop)
    pattern_white_rgb = crop_array(to_uint8(pattern_white), crop)
    pattern_overlay_rgb = crop_array(to_uint8(pattern_overlay), crop)
    pattern_mask_rgb = crop_array(to_uint8(pattern_mask.expand(3, -1, -1)), crop)

    images = {
        "gt": gt_rgb, "baseline": baseline_rgb, "wave": wave_rgb, "wave_off": wave_off_rgb,
        "baseline_error_x4": base_error, "wave_error_x4": wave_error,
        "wave_allocation": allocation_rgb, "wave_allocation_overlay": allocation_overlay,
        "wave_contribution_x8": contribution_rgb,
        "wave_contribution": contribution_rgb,
        "rendering_improvement": improvement_rgb,
        "wave_pattern_projection": pattern_white_rgb,
        "wave_pattern_overlay": pattern_overlay_rgb,
        "wave_pattern_mask": pattern_mask_rgb,
    }
    for name, image in images.items():
        save_rgb(image, out_dir / f"{name}.png")
    footprint.save(out_dir / "wave_footprints_overlay.png")
    save_footprint_colorbar(out_dir / "wave_footprint_colorbar.png", args.footprint_cmap,
                            footprint_color_low, footprint_color_high)
    save_signed_colorbar(out_dir / "rendering_improvement_colorbar.png",
                         args.improvement_cmap, improvement_scale,
                         "Per-pixel mean absolute RGB error difference")
    sheet_items = [
        ("GT", Image.fromarray(gt_rgb)), ("Original 2DGS", Image.fromarray(baseline_rgb)), ("Wave-2DGS", Image.fromarray(wave_rgb)),
        ("Original abs error ×4", Image.fromarray(base_error)), ("Wave abs error ×4", Image.fromarray(wave_error)), ("Wave on-off contribution ×8", Image.fromarray(contribution_rgb)),
        ("Rendering improvement", Image.fromarray(improvement_rgb)), ("Wave allocation", Image.fromarray(allocation_rgb)),
        ("Allocation on GT", Image.fromarray(allocation_overlay)), ("Top active wave footprints", footprint),
        ("Learned red/blue wave projection", Image.fromarray(pattern_white_rgb)),
        ("Learned wave projection on GT", Image.fromarray(pattern_overlay_rgb)),
    ]
    contact_sheet(sheet_items, out_dir / "paper_panel.png")

    top_count = top_count_for_overlay
    top_local = selected_local
    with (out_dir / "top_active_wave_primitives.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank", "primitive_id", "x_px", "y_px", "raster_radius_px", "amplitude_norm", "k_u", "k_v", "phase"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        params = wave.get_wave_params
        for rank, local_idx in enumerate(top_local.detach().cpu().tolist(), start=1):
            primitive_id = int(visible_ids[local_idx].item())
            writer.writerow({
                "rank": rank, "primitive_id": primitive_id,
                "x_px": float(xy_visible[local_idx, 0].item()), "y_px": float(xy_visible[local_idx, 1].item()),
                "raster_radius_px": float(radii_visible[local_idx].item()), "amplitude_norm": float(amp_visible[local_idx].item()),
                "k_u": float(params[primitive_id, 3].item()), "k_v": float(params[primitive_id, 4].item()),
                "phase": float(params[primitive_id, 5].item()),
            })
    metadata = {
        "source_path": dataset.source_path, "baseline": args.baseline, "wave": args.wave,
        "iteration": iteration, "split": args.split, "view_index": selected_index, "image_name": camera.image_name,
        "requested_image_name": requested_name or None, "replica_frame": args.replica_frame if args.replica_frame >= 0 else None,
        "resolution": [int(camera.image_width), int(camera.image_height)], "crop": args.crop,
        "active_threshold": args.active_threshold, "visible_primitives": int(visible.sum().item()),
        "active_visible_primitives": int((amp_visible >= args.active_threshold).sum().item()),
        "topk": top_count, "footprint_cmap": args.footprint_cmap,
        "footprint_geometry": "projected_planar_contour", "footprint_sigma": args.footprint_sigma,
        "footprint_segments": args.footprint_segments,
        "footprint_color_scale_topk_q05": footprint_color_low,
        "footprint_color_scale_topk_q95": footprint_color_high,
        "allocation_color_amp_p99": float(amp_scale.item()),
        "allocation_map_p99": allocation_scale, "contribution_map_p99": contribution_scale,
        "rendering_improvement_definition": "mean_abs_rgb_error_2dgs_minus_wave",
        "rendering_improvement_display_abs_quantile": args.improvement_quantile,
        "rendering_improvement_display_scale": improvement_scale,
        "rendering_improvement_cmap": args.improvement_cmap,
        "rendering_improvement_mean": float(rendering_improvement.mean().item()),
        "rendering_improvement_median": float(rendering_improvement.median().item()),
        "rendering_fraction_wave_better": float((rendering_improvement > 0.0).float().mean().item()),
        "rendering_fraction_2dgs_better": float((rendering_improvement < 0.0).float().mean().item()),
        "pattern_projection": {
            "selected_visible_primitives": pattern_count,
            "selection": "top visible amplitude above active threshold",
            "learned_components": ["k_u", "k_v", "phase", "geometry", "opacity"],
            "synthetic_signed_rgb_amplitude": [args.pattern_contrast, 0.0, -args.pattern_contrast],
            "strength_mode": "uniform" if args.pattern_uniform_strength else "normalized_learned_amplitude",
            "strength_gamma": args.pattern_strength_gamma,
            "gt_overlay_alpha": args.pattern_overlay_alpha,
        },
        "note": "Allocation is an alpha-composited amplitude proxy with wave modulation disabled.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote paper visual panel: {out_dir / 'paper_panel.png'}")


if __name__ == "__main__":
    main()
