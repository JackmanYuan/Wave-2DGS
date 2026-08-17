#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import csv
import json
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

PRIMITIVE_ACTION_FIELDS = [
    "iteration", "mode", "pre_primitives", "post_primitives", "net_primitive_delta",
    "candidate_clone", "candidate_split", "clone_applied", "split_applied",
    "cap_remaining_before", "capacity_blocked_candidates", "estimated_pruned",
    "gradient_mean", "gradient_p90", "gradient_max", "wave_active_fraction",
    "wave_amp_group_lasso", "wave_amp_norm_mean", "wave_amp_norm_p90",
    "wave_amp_gt_001_fraction",
]

WAVE_LEARNING_FIELDS = [
    "iteration", "primitives", "wave_amp_lr", "wave_freq_lr", "wave_phase_lr",
    "wave_amp_grad_norm_mean", "wave_amp_grad_norm_p90",
    "wave_freq_grad_norm_mean", "wave_freq_grad_norm_p90",
    "wave_phase_grad_abs_mean", "wave_phase_grad_abs_p90",
    "wave_freq_grad_nonzero_fraction", "wave_phase_grad_nonzero_fraction",
]


def append_primitive_action_log(path, row):
    """Write opt-in densification diagnostics without changing model actions."""
    write_header = not os.path.exists(path)
    values = {field: "" for field in PRIMITIVE_ACTION_FIELDS}
    values.update({key: value for key, value in row.items() if key in values})
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMITIVE_ACTION_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(values)


def append_wave_learning_log(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WAVE_LEARNING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in WAVE_LEARNING_FIELDS})


def summarize_wave_gradients(gaussians, iteration):
    grad = gaussians.get_wave_params.grad
    if grad is None:
        return None
    amp = torch.linalg.vector_norm(grad[:, :3], dim=1)
    freq = torch.linalg.vector_norm(grad[:, 3:5], dim=1)
    phase = grad[:, 5].abs()
    return {
        "iteration": int(iteration),
        "primitives": int(grad.shape[0]),
        "wave_amp_lr": gaussians.wave_amp_lr,
        "wave_freq_lr": gaussians.wave_freq_lr,
        "wave_phase_lr": gaussians.wave_phase_lr,
        "wave_amp_grad_norm_mean": float(amp.mean().item()),
        "wave_amp_grad_norm_p90": float(torch.quantile(amp, 0.90).item()),
        "wave_freq_grad_norm_mean": float(freq.mean().item()),
        "wave_freq_grad_norm_p90": float(torch.quantile(freq, 0.90).item()),
        "wave_phase_grad_abs_mean": float(phase.mean().item()),
        "wave_phase_grad_abs_p90": float(torch.quantile(phase, 0.90).item()),
        "wave_freq_grad_nonzero_fraction": float((freq > 1e-12).float().mean().item()),
        "wave_phase_grad_nonzero_fraction": float((phase > 1e-12).float().mean().item()),
    }


def write_wave_parameter_summary(model_path, iteration, gaussians):
    """Persist compact wave diagnostics so later analysis does not need a PLY."""
    if not gaussians.enable_primitive_wave or gaussians.get_wave_params.numel() == 0:
        return

    wave = gaussians.get_wave_params.detach()
    amp_norm = torch.linalg.vector_norm(wave[:, :3], dim=1)
    k_u, k_v = wave[:, 3], wave[:, 4]
    k_norm = torch.linalg.vector_norm(wave[:, 3:5], dim=1)
    k_delta = k_norm - gaussians.wave_k_init
    phase = wave[:, 5]
    gate = wave[:, 6]

    def scalar(value):
        return float(value.item())

    def quantile(values, q):
        return scalar(torch.quantile(values, q))

    phase_resultant = torch.sqrt(torch.cos(phase).mean().square() + torch.sin(phase).mean().square())
    summary = {
        "iteration": int(iteration),
        "primitives": int(wave.shape[0]),
        "wave_k_init": float(gaussians.wave_k_init),
        "wave_amp_lr": float(gaussians.wave_amp_lr),
        "wave_freq_lr": float(gaussians.wave_freq_lr),
        "wave_phase_lr": float(gaussians.wave_phase_lr),
        "wave_gate_active_fraction": scalar((gate >= 0.5).float().mean()),
        "wave_amp_norm_mean": scalar(amp_norm.mean()),
        "wave_amp_norm_median": quantile(amp_norm, 0.50),
        "wave_amp_norm_p90": quantile(amp_norm, 0.90),
        "wave_amp_norm_p99": quantile(amp_norm, 0.99),
        "wave_amp_gt_0001_fraction": scalar((amp_norm > 0.001).float().mean()),
        "wave_amp_gt_001_fraction": scalar((amp_norm > 0.01).float().mean()),
        "wave_amp_gt_005_fraction": scalar((amp_norm > 0.05).float().mean()),
        "wave_k_norm_mean": scalar(k_norm.mean()),
        "wave_k_norm_median": quantile(k_norm, 0.50),
        "wave_k_norm_p10": quantile(k_norm, 0.10),
        "wave_k_norm_p90": quantile(k_norm, 0.90),
        "wave_k_delta_from_init_mean": scalar(k_delta.mean()),
        "wave_k_abs_delta_from_init_median": quantile(k_delta.abs(), 0.50),
        "wave_k_abs_delta_from_init_p90": quantile(k_delta.abs(), 0.90),
        "wave_ku_mean": scalar(k_u.mean()),
        "wave_kv_mean": scalar(k_v.mean()),
        "wave_kv_abs_median": quantile(k_v.abs(), 0.50),
        "wave_phase_resultant_length": scalar(phase_resultant),
    }
    with open(os.path.join(model_path, "wave_parameter_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    history_path = os.path.join(model_path, "wave_parameter_history.csv")
    write_header = not os.path.exists(history_path)
    with open(history_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, runtime_config=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,
                              enable_primitive_wave=getattr(dataset, "enable_primitive_wave", False),
                              wave_k_init=getattr(dataset, "wave_k_init", 6.283185307),
                              wave_amp_init=getattr(dataset, "wave_amp_init", 0.0),
                              wave_random_phase_init=getattr(dataset, "wave_random_phase_init", False),
                              wave_amplitude_only=getattr(dataset, "wave_amplitude_only", False))
    scene = Scene(dataset, gaussians, max_primitives=opt.max_primitives,
                  initialization_seed=getattr(runtime_config, "seed", 0))
    if getattr(dataset, "initial_ply", ""):
        print("Loading explicit training initializer: {}".format(dataset.initial_ply))
        gaussians.load_ply(dataset.initial_ply)
    initial_primitives = int(gaussians.get_xyz.shape[0])
    gaussians.training_setup(opt)
    primitive_action_log_path = os.path.join(dataset.model_path, "primitive_action_logging.csv")
    wave_learning_log_path = os.path.join(dataset.model_path, "wave_learning_diagnostics.csv")
    enable_primitive_action_logging = bool(getattr(runtime_config, "enable_primitive_action_logging", False))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # A zero wave-amplitude vector is exactly the original 2DGS colour
        # model. Group-Lasso therefore lets each primitive opt out of Wave.
        wave_amp_group_lasso = loss.new_zeros(())
        if (dataset.enable_primitive_wave and opt.wave_amp_group_lasso > 0.0
                and iteration >= opt.wave_amp_group_lasso_start_iter):
            wave_amplitude = torch.linalg.vector_norm(gaussians.get_wave_params[:, :3], dim=1)
            wave_amp_group_lasso = opt.wave_amp_group_lasso * wave_amplitude.mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + wave_amp_group_lasso
        

        total_loss.backward()

        # Keep frequency, phase, and gate fixed in the amplitude-only control.
        if dataset.enable_primitive_wave and dataset.wave_amplitude_only and gaussians.get_wave_params.grad is not None:
            gaussians.get_wave_params.grad[:, 3:].zero_()
        if dataset.enable_primitive_wave and dataset.wave_freeze_phase and gaussians.get_wave_params.grad is not None:
            gaussians.get_wave_params.grad[:, 5].zero_()

        iter_end.record()

        wave_gradient_summary = None
        if dataset.enable_primitive_wave:
            wave_gradient_summary = summarize_wave_gradients(gaussians, iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

                if dataset.enable_primitive_wave:
                    amp_norm = torch.linalg.vector_norm(gaussians.get_wave_params[:, :3], dim=1)
                    tb_writer.add_scalar('wave/amplitude_group_lasso', wave_amp_group_lasso.item(), iteration)
                    tb_writer.add_scalar('wave/amplitude_norm_mean', amp_norm.mean().item(), iteration)
                    tb_writer.add_scalar('wave/amplitude_active_gt_001_fraction', (amp_norm > 0.01).float().mean().item(), iteration)
            if (dataset.enable_primitive_wave and opt.wave_parameter_log_interval > 0
                  and iteration % opt.wave_parameter_log_interval == 0 and iteration not in saving_iterations):
                write_wave_parameter_summary(dataset.model_path, iteration, gaussians)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if dataset.enable_primitive_wave:
                    write_wave_parameter_summary(dataset.model_path, iteration, gaussians)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    action_preview = None
                    if enable_primitive_action_logging:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        grad_norm = torch.norm(grads, dim=-1)
                        scales = torch.max(gaussians.get_scaling, dim=1).values
                        clone_mask = (grad_norm >= opt.densify_grad_threshold) & (scales <= gaussians.percent_dense * scene.cameras_extent)
                        split_mask = (grad_norm >= opt.densify_grad_threshold) & (scales > gaussians.percent_dense * scene.cameras_extent)
                        pre_primitives = int(gaussians.get_xyz.shape[0])
                        cap_remaining = max(0, opt.max_primitives - pre_primitives) if opt.max_primitives > 0 else -1
                        candidate_clone = int(clone_mask.sum().item())
                        candidate_split = int(split_mask.sum().item())
                        clone_applied = min(candidate_clone, cap_remaining) if cap_remaining >= 0 else candidate_clone
                        split_budget = max(0, cap_remaining - clone_applied) if cap_remaining >= 0 else candidate_split
                        split_applied = min(candidate_split, split_budget)
                        action_preview = {
                            "pre_primitives": pre_primitives,
                            "candidate_clone": candidate_clone,
                            "candidate_split": candidate_split,
                            "clone_applied": clone_applied,
                            "split_applied": split_applied,
                            "cap_remaining_before": cap_remaining,
                            "capacity_blocked_candidates": candidate_clone + candidate_split - clone_applied - split_applied,
                            "gradient_mean": float(grad_norm.mean().item()),
                            "gradient_p90": float(torch.quantile(grad_norm, 0.90).item()),
                            "gradient_max": float(grad_norm.max().item()),
                            "wave_active_fraction": gaussians.wave_active_fraction,
                            "wave_amp_group_lasso": float(wave_amp_group_lasso.item()),
                        }
                        if dataset.enable_primitive_wave:
                            amp_norm = torch.linalg.vector_norm(gaussians.get_wave_params[:, :3], dim=1)
                            action_preview.update({
                                "wave_amp_norm_mean": float(amp_norm.mean().item()),
                                "wave_amp_norm_p90": float(torch.quantile(amp_norm, 0.90).item()),
                                "wave_amp_gt_001_fraction": float((amp_norm > 0.01).float().mean().item()),
                            })
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold,
                                                 max_primitives=opt.max_primitives)
                    if action_preview is not None:
                        post_primitives = int(gaussians.get_xyz.shape[0])
                        action_preview.update({
                            "iteration": iteration,
                            "mode": "standard_2dgs",
                            "post_primitives": post_primitives,
                            "net_primitive_delta": post_primitives - action_preview["pre_primitives"],
                            "estimated_pruned": max(0, action_preview["pre_primitives"] + action_preview["clone_applied"] + action_preview["split_applied"] - post_primitives),
                        })
                        append_primitive_action_log(primitive_action_log_path, action_preview)
                        if tb_writer is not None:
                            tb_writer.add_scalar("primitive_actions/net_primitive_delta", action_preview["net_primitive_delta"], iteration)
                            tb_writer.add_scalar("primitive_actions/estimated_pruned", action_preview["estimated_pruned"], iteration)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                wave_update_state = gaussians.capture_wave_update_state()
                gaussians.optimizer.step()
                gaussians.apply_wave_component_lrs(wave_update_state)
                gaussians.optimizer.zero_grad(set_to_none = True)


            if wave_gradient_summary is not None and (iteration in saving_iterations or (opt.wave_parameter_log_interval > 0 and iteration % opt.wave_parameter_log_interval == 0)):
                append_wave_learning_log(wave_learning_log_path, wave_gradient_summary)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    return {
        "initial_primitives": initial_primitives,
        "final_primitives": int(gaussians.get_xyz.shape[0]),
    }

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--enable_primitive_action_logging", action="store_true", help="Write densification and Wave diagnostics at every densification event.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization, camera ordering, and densification sampling.")
    args = parser.parse_args(sys.argv[1:])
    if args.initial_ply and not os.path.isfile(args.initial_ply):
        parser.error("--initial_ply must point to an existing PLY file")
    if args.wave_amp_group_lasso < 0.0:
        parser.error("--wave_amp_group_lasso must be non-negative")
    if args.wave_amp_group_lasso_start_iter < 0:
        parser.error("--wave_amp_group_lasso_start_iter must be non-negative")
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    training_result = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
                               args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args)
    torch.cuda.synchronize()
    training_summary = {
        "iterations": args.iterations,
        "seed": args.seed,
        "initial_primitives": training_result["initial_primitives"],
        "final_primitives": training_result["final_primitives"],
        "train_wall_seconds": time.perf_counter() - training_start,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    }
    with open(os.path.join(args.model_path, "training_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_summary.keys()))
        writer.writeheader()
        writer.writerow(training_summary)

    # All done
    print("\nTraining complete.")
