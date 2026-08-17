#
# Memory-light renderer for quick 2DGS comparisons.
#

import csv
import json
import os
import time
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
from utils.loss_utils import ssim
from utils.system_utils import searchForMaxIteration
from lpipsPyTorch.modules.lpips import LPIPS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image(tensor, path):
    torchvision.utils.save_image(torch.clamp(tensor, 0.0, 1.0), path)


def normalize_depth(depth):
    depth = depth.detach()
    valid = torch.isfinite(depth) & (depth > 0)
    if not torch.any(valid):
        return torch.zeros_like(depth)
    lo = torch.quantile(depth[valid], 0.02)
    hi = torch.quantile(depth[valid], 0.98)
    return torch.clamp((depth - lo) / (hi - lo).clamp_min(1e-6), 0.0, 1.0)


def choose_views(cameras, max_views, stride, offset):
    stride = max(1, stride)
    offset = max(0, offset)
    selected = cameras[offset::stride]
    if max_views > 0:
        selected = selected[:max_views]
    return selected


def choose_manifest_views(cameras, manifest_path):
    """Select cameras by explicit image name from a shared render manifest."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    names = manifest.get("frame_names")
    if not names:
        raise ValueError(f"Render manifest has no frame_names: {manifest_path}")
    by_name = {camera.image_name: camera for camera in cameras}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(
            f"Render manifest contains unavailable test cameras: {missing[:8]}"
        )
    return [by_name[name] for name in names]


def read_scene_info(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    if os.path.exists(os.path.join(args.source_path, "traj.txt")) and os.path.isdir(os.path.join(args.source_path, "results")):
        print("Found Replica RGB-D sequence, assuming Replica data set!")
        return sceneLoadTypeCallbacks["Replica"](args.source_path, args.white_background, args.eval, args)
    if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        return sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    raise RuntimeError(f"Could not recognize scene type at {args.source_path}")


def resolve_iteration(model_path, iteration):
    if iteration == -1:
        return searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    return iteration


def image_metrics(rendered, gt, lpips_model):
    rendered = torch.clamp(rendered, 0.0, 1.0)
    gt = torch.clamp(gt, 0.0, 1.0)
    mse = torch.mean((rendered - gt) ** 2).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    l1 = torch.mean(torch.abs(rendered - gt))
    rendered_batch = rendered.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    ssim_value = ssim(rendered_batch, gt_batch)
    lpips_value = lpips_model(rendered_batch, gt_batch)
    return psnr.item(), l1.item(), ssim_value.item(), lpips_value.item()


@torch.no_grad()
def render_split(split_name, cameras, gaussians, pipe, background, out_dir, args, lpips_model):
    split_dir = os.path.join(out_dir, split_name)
    render_dir = os.path.join(split_dir, "renders")
    gt_dir = os.path.join(split_dir, "gt")
    diff_dir = os.path.join(split_dir, "diff")
    ensure_dir(render_dir)
    if args.save_gt:
        ensure_dir(gt_dir)
    if args.save_diff:
        ensure_dir(diff_dir)

    depth_dir = os.path.join(split_dir, "depth")
    normal_dir = os.path.join(split_dir, "normal")
    alpha_dir = os.path.join(split_dir, "alpha")
    if args.save_maps:
        ensure_dir(depth_dir)
        ensure_dir(normal_dir)
        ensure_dir(alpha_dir)

    rows = []
    ablation_rows = []
    wave_off_dir = os.path.join(split_dir, "renders_wave_off")
    wave_delta_dir = os.path.join(split_dir, "wave_delta")
    if args.save_wave_ablation:
        if not gaussians.enable_primitive_wave:
            raise ValueError("--save_wave_ablation requires a wave-enabled checkpoint/config")
        ensure_dir(wave_off_dir)
        ensure_dir(wave_delta_dir)
    torch.cuda.reset_peak_memory_stats()
    for view_idx, camera in tqdm(list(enumerate(cameras)), desc=f"render_lite {split_name}"):
        torch.cuda.synchronize()
        render_start = time.perf_counter()
        pkg = render(camera, gaussians, pipe, background)
        torch.cuda.synchronize()
        render_ms = (time.perf_counter() - render_start) * 1000.0
        image = pkg["render"]
        gt = camera.original_image[0:3, :, :].cuda()

        psnr, l1, ssim_value, lpips_value = image_metrics(image, gt, lpips_model)
        stem = f"{view_idx:04d}_{camera.image_name}"
        save_image(image.cpu(), os.path.join(render_dir, f"{stem}.png"))

        if args.save_wave_ablation:
            gaussians.enable_primitive_wave = False
            try:
                image_wave_off = render(camera, gaussians, pipe, background)["render"]
            finally:
                gaussians.enable_primitive_wave = True
            psnr_off, l1_off, ssim_off, lpips_off = image_metrics(image_wave_off, gt, lpips_model)
            wave_delta = torch.abs(image - image_wave_off)
            save_image(image_wave_off.cpu(), os.path.join(wave_off_dir, f"{stem}.png"))
            save_image((wave_delta * args.wave_delta_gain).cpu(), os.path.join(wave_delta_dir, f"{stem}_x{args.wave_delta_gain:g}.png"))
            ablation_rows.append({
                "split": split_name, "view_idx": view_idx, "image_name": camera.image_name,
                "psnr_wave_on": psnr, "psnr_wave_off": psnr_off, "delta_psnr": psnr - psnr_off,
                "l1_wave_on": l1, "l1_wave_off": l1_off, "delta_l1": l1 - l1_off,
                "ssim_wave_on": ssim_value, "ssim_wave_off": ssim_off, "delta_ssim": ssim_value - ssim_off,
                "lpips_wave_on": lpips_value, "lpips_wave_off": lpips_off, "delta_lpips": lpips_value - lpips_off,
                "mean_wave_absdiff": wave_delta.mean().item(),
                "p95_wave_absdiff": torch.quantile(wave_delta, 0.95).item(),
            })

        if args.save_gt:
            save_image(gt.cpu(), os.path.join(gt_dir, f"{stem}.png"))
        if args.save_diff:
            diff = torch.abs(image - gt) * args.diff_gain
            save_image(diff.cpu(), os.path.join(diff_dir, f"{stem}_diff_x{args.diff_gain:g}.png"))

        if args.save_maps:
            if "surf_depth" in pkg:
                save_image(normalize_depth(pkg["surf_depth"]).cpu(), os.path.join(depth_dir, f"{stem}.png"))
            if "rend_normal" in pkg:
                normal = torch.nn.functional.normalize(pkg["rend_normal"], dim=0)
                save_image(((normal + 1.0) * 0.5).cpu(), os.path.join(normal_dir, f"{stem}.png"))
            if "rend_alpha" in pkg:
                save_image(pkg["rend_alpha"].cpu(), os.path.join(alpha_dir, f"{stem}.png"))

        rows.append({
            "split": split_name,
            "view_idx": view_idx,
            "image_name": camera.image_name,
            "psnr": psnr,
            "l1": l1,
            "ssim": ssim_value,
            "lpips_vgg": lpips_value,
            "render_ms": render_ms,
        })

        del pkg, image, gt
        torch.cuda.empty_cache()

    metrics_path = os.path.join(split_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["split", "view_idx", "image_name", "psnr", "l1", "ssim", "lpips_vgg", "render_ms"])
        writer.writeheader()
        writer.writerows(rows)

    if ablation_rows:
        with open(os.path.join(split_dir, "wave_ablation_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ablation_rows)
        summary = {
            "views": len(ablation_rows),
            "mean_psnr_wave_on": sum(r["psnr_wave_on"] for r in ablation_rows) / len(ablation_rows),
            "mean_psnr_wave_off": sum(r["psnr_wave_off"] for r in ablation_rows) / len(ablation_rows),
            "mean_delta_psnr": sum(r["delta_psnr"] for r in ablation_rows) / len(ablation_rows),
            "mean_l1_wave_on": sum(r["l1_wave_on"] for r in ablation_rows) / len(ablation_rows),
            "mean_l1_wave_off": sum(r["l1_wave_off"] for r in ablation_rows) / len(ablation_rows),
            "mean_ssim_wave_on": sum(r["ssim_wave_on"] for r in ablation_rows) / len(ablation_rows),
            "mean_ssim_wave_off": sum(r["ssim_wave_off"] for r in ablation_rows) / len(ablation_rows),
            "mean_delta_ssim": sum(r["delta_ssim"] for r in ablation_rows) / len(ablation_rows),
            "mean_lpips_wave_on": sum(r["lpips_wave_on"] for r in ablation_rows) / len(ablation_rows),
            "mean_lpips_wave_off": sum(r["lpips_wave_off"] for r in ablation_rows) / len(ablation_rows),
            "mean_delta_lpips": sum(r["delta_lpips"] for r in ablation_rows) / len(ablation_rows),
            "mean_wave_absdiff": sum(r["mean_wave_absdiff"] for r in ablation_rows) / len(ablation_rows),
        }
        with open(os.path.join(split_dir, "wave_ablation_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

    if rows:
        mean_psnr = sum(row["psnr"] for row in rows) / len(rows)
        mean_l1 = sum(row["l1"] for row in rows) / len(rows)
        summary = {
            "views": len(rows),
            "mean_psnr": mean_psnr,
            "mean_l1": mean_l1,
            "mean_ssim": sum(row["ssim"] for row in rows) / len(rows),
            "mean_lpips_vgg": sum(row["lpips_vgg"] for row in rows) / len(rows),
            "mean_render_ms": sum(row["render_ms"] for row in rows) / len(rows),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        }
        with open(os.path.join(split_dir, "metrics_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(f"{split_name}: {len(rows)} views, mean PSNR {mean_psnr:.3f}, mean SSIM {summary['mean_ssim']:.4f}, "
              f"mean LPIPS {summary['mean_lpips_vgg']:.4f}, mean L1 {mean_l1:.5f}")
    else:
        print(f"{split_name}: no views selected")


if __name__ == "__main__":
    parser = ArgumentParser(description="Memory-light render script for quick comparisons")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--max_train_views", default=16, type=int)
    parser.add_argument("--max_test_views", default=16, type=int)
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--view_offset", default=0, type=int)
    parser.add_argument("--view_manifest", default="", type=str)
    parser.add_argument("--diff_gain", default=4.0, type=float)
    parser.add_argument("--save_gt", action="store_true", default=True)
    parser.add_argument("--save_diff", action="store_true", default=True)
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--save_wave_ablation", action="store_true")
    parser.add_argument("--wave_delta_gain", default=8.0, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Lite rendering " + args.model_path)
    safe_state(args.quiet, getattr(args, "seed", 0))

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree,
                              enable_primitive_wave=getattr(dataset, "enable_primitive_wave", False),
                              wave_k_init=getattr(dataset, "wave_k_init", 6.283185307))
    loaded_iter = resolve_iteration(dataset.model_path, args.iteration)
    print(f"Loading trained model at iteration {loaded_iter}")
    gaussians.load_ply(os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{loaded_iter}",
        "point_cloud.ply",
    ))

    print("Loading official 2DGS LPIPS-VGG metric")
    lpips_model = LPIPS(net_type="vgg", version="0.1").cuda().eval()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene_info = read_scene_info(dataset)
    out_dir = os.path.join(args.model_path, "render_lite", f"ours_{loaded_iter}")
    ensure_dir(out_dir)

    if not args.skip_train:
        train_cam_infos = choose_views(
            scene_info.train_cameras,
            max_views=args.max_train_views,
            stride=args.view_stride,
            offset=args.view_offset,
        )
        print(f"Loading {len(train_cam_infos)} selected train cameras")
        train_cameras = cameraList_from_camInfos(train_cam_infos, 1.0, dataset)
        render_split("train", train_cameras, gaussians, pipe, background, out_dir, args, lpips_model)

    if not args.skip_test:
        if args.view_manifest:
            test_cam_infos = choose_manifest_views(
                scene_info.test_cameras, args.view_manifest
            )
        else:
            test_cam_infos = choose_views(
                scene_info.test_cameras,
                max_views=args.max_test_views,
                stride=args.view_stride,
                offset=args.view_offset,
            )
        print(f"Loading {len(test_cam_infos)} selected test cameras")
        test_cameras = cameraList_from_camInfos(test_cam_infos, 1.0, dataset)
        render_split("test", test_cameras, gaussians, pipe, background, out_dir, args, lpips_model)

    print(f"Lite render complete: {out_dir}")
