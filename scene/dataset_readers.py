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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def loadReplicaPoses(traj_path):
    poses = []

    with open(traj_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            values = np.fromstring(line, sep=" ", dtype=np.float64)
            if values.size != 16:
                raise ValueError(
                    f"Invalid Replica pose at line {line_idx + 1}: expected 16 values, got {values.size}"
                )

            poses.append(values.reshape(4, 4))

    if not poses:
        raise RuntimeError(f"No poses found in {traj_path}")

    return poses

def loadReplicaCameraParams(cam_params_path):
    """Load original Replica and flat collected-camera JSON schemas."""
    with open(cam_params_path, "r", encoding="utf-8") as f:
        contents = json.load(f)

    camera = contents.get("camera", contents)
    # Original Replica uses w/h/scale; the collected server-room dataset uses
    # width/height/depth_png_scale. Keep both schemas on one loader path.
    return {
        "width": int(camera["w"] if "w" in camera else camera["width"]),
        "height": int(camera["h"] if "h" in camera else camera["height"]),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "scale": float(camera.get("scale", camera.get("depth_png_scale", 1.0))),
    }

def convertReplicaPose(raw_pose, convention="nice"):
    pose = raw_pose.copy()

    if convention == "raw":
        return pose

    if convention == "nice":
        pose[:3, 1] *= -1.0
        pose[:3, 2] *= -1.0
        return pose

    raise ValueError(f"Unknown Replica pose convention: {convention}")

def voxelDownsample(points, colors, voxel_size):
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return points, colors

    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)

    counts = np.bincount(inverse)
    sums_xyz = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    sums_rgb = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    np.add.at(sums_xyz, inverse, points)
    np.add.at(sums_rgb, inverse, colors)

    points_ds = sums_xyz / counts[:, None]
    colors_ds = sums_rgb / counts[:, None]
    return points_ds, colors_ds

def backprojectReplicaRGBD(rgb, depth, c2w, fx, fy, cx, cy, pixel_stride=8, min_depth=0.1, max_depth=10.0):
    height, width = depth.shape
    v, u = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    z = depth[v, u]

    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    u_valid = u[valid].astype(np.float64)
    v_valid = v[valid].astype(np.float64)
    z_valid = z[valid].astype(np.float64)

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    camera_points = np.column_stack((x, y, z_valid))

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    world_points = (rotation @ camera_points.T).T + translation

    colors = rgb[v[valid], u[valid]].astype(np.float64) / 255.0
    return world_points, colors

def readReplicaSceneInfo(path, white_background, eval, args, llffhold=8):
    del white_background

    results_dir = os.path.join(path, "results")
    traj_path = os.path.join(path, "traj.txt")
    cam_params_path = os.path.join(path, "cam_params.json")
    if not os.path.exists(cam_params_path):
        cam_params_path = os.path.join(path, "camera.json")
    if not os.path.exists(cam_params_path):
        cam_params_path = os.path.join(os.path.dirname(path), "cam_params.json")

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Missing Replica results directory: {results_dir}")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Missing Replica trajectory file: {traj_path}")
    if not os.path.exists(cam_params_path):
        raise FileNotFoundError(f"Missing Replica camera params file: {cam_params_path}")

    cam_params = loadReplicaCameraParams(cam_params_path)
    poses = loadReplicaPoses(traj_path)
    pose_convention = getattr(args, "replica_pose_convention", "raw")

    rgb_paths = sorted(Path(results_dir).glob("frame*.jpg"))
    depth_paths = sorted(Path(results_dir).glob("depth*.png"))
    num_available = min(len(rgb_paths), len(depth_paths), len(poses))
    if num_available == 0:
        raise RuntimeError(f"No usable Replica frames found in {results_dir}")

    print(f"Replica frames available: {num_available}")

    all_indices = list(range(num_available))
    if eval:
        train_indices = [idx for idx in all_indices if idx % llffhold != 0]
        test_indices = [idx for idx in all_indices if idx % llffhold == 0]
    else:
        train_indices = all_indices
        test_indices = []

    pointcloud_path = str(getattr(args, "replica_pointcloud_path", "")).strip()
    pointcloud_cache_tag = str(getattr(args, "replica_pointcloud_cache_tag", "")).strip()
    if pointcloud_path:
        ply_path = os.path.abspath(os.path.expanduser(pointcloud_path))
        print(f"Using explicit Replica initializer: {ply_path}")
    elif pointcloud_cache_tag:
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in pointcloud_cache_tag)
        if not safe_tag:
            raise ValueError("--replica_pointcloud_cache_tag must contain at least one safe filename character")
        ply_path = os.path.join(path, f"points3d_{pose_convention}_{safe_tag}.ply")
        print(f"Using tagged Replica initializer cache: {os.path.basename(ply_path)}")
    else:
        plane_seed_path = os.path.join(path, "points3d_plane_seed.ply")
        if os.path.exists(plane_seed_path):
            print("Found plane-seed initializer, using points3d_plane_seed.ply")
            ply_path = plane_seed_path
        else:
            ply_path = os.path.join(path, f"points3d_{pose_convention}.ply")
    if not os.path.exists(ply_path):
        print("Building initial Replica point cloud from RGB-D frames...")
        point_frame_stride = max(1, int(getattr(args, "replica_depth_frame_stride", 20)))
        pixel_stride = max(1, int(getattr(args, "replica_depth_pixel_stride", 8)))
        voxel_size = float(getattr(args, "replica_depth_voxel_size", 0.02))
        min_depth = float(getattr(args, "replica_min_depth", 0.1))
        max_depth = float(getattr(args, "replica_max_depth", 10.0))

        sample_indices = train_indices[::point_frame_stride]
        if not sample_indices:
            sample_indices = train_indices[:1]

        all_points = []
        all_colors = []
        for frame_idx in sample_indices:
            rgb = np.array(Image.open(rgb_paths[frame_idx]).convert("RGB"))
            depth_raw = np.array(Image.open(depth_paths[frame_idx]))
            depth = depth_raw.astype(np.float32) / cam_params["scale"]
            c2w = convertReplicaPose(poses[frame_idx], pose_convention)
            points, colors = backprojectReplicaRGBD(
                rgb=rgb,
                depth=depth,
                c2w=c2w,
                fx=cam_params["fx"],
                fy=cam_params["fy"],
                cx=cam_params["cx"],
                cy=cam_params["cy"],
                pixel_stride=pixel_stride,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)

        if not all_points:
            raise RuntimeError("Could not build a Replica point cloud from the depth maps.")

        merged_points = np.concatenate(all_points, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        merged_points, merged_colors = voxelDownsample(merged_points, merged_colors, voxel_size)
        storePly(ply_path, merged_points, (merged_colors * 255.0).astype(np.uint8))

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    train_cam_infos = []
    test_cam_infos = []
    FovX = focal2fov(cam_params["fx"], cam_params["width"])
    FovY = focal2fov(cam_params["fy"], cam_params["height"])

    for idx in train_indices:
        image = Image.open(rgb_paths[idx]).convert("RGB")
        c2w = convertReplicaPose(poses[idx], pose_convention)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        train_cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=str(rgb_paths[idx]),
                image_name=rgb_paths[idx].stem,
                width=cam_params["width"],
                height=cam_params["height"],
            )
        )

    for idx in test_indices:
        image = Image.open(rgb_paths[idx]).convert("RGB")
        c2w = convertReplicaPose(poses[idx], pose_convention)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        test_cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=str(rgb_paths[idx]),
                image_name=rgb_paths[idx].stem,
                width=cam_params["width"],
                height=cam_params["height"],
            )
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica": readReplicaSceneInfo
}
