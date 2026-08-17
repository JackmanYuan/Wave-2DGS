# Dataset preparation

All examples use repository-relative paths under `data/`. Dataset contents are
ignored by Git; only [`data/README.md`](../data/README.md) is tracked. A typical
layout is:

```text
data/
├── Replica/
├── cull_replica_mesh/
├── DTU/
└── DTU_Official/
```

If the datasets live on a larger disk, symlink individual directories into
`data/`, for example:

```bash
ln -s /large/datasets/Replica data/Replica
ln -s /large/datasets/DTU data/DTU
ln -s /large/datasets/DTU_Official data/DTU_Official
```

## COLMAP scenes

Wave-2DGS retains the original 2DGS COLMAP loader. Follow the upstream 3DGS /
2DGS preparation instructions and pass the scene directory with `-s`.

## Replica

Download the RGB-D sequences from the
[NICE-SLAM Replica release](https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip)
and extract the resulting `Replica/` directory as `data/Replica/`. This release
contains the rendered RGB-D trajectories expected by the Wave-2DGS loader; the
[original Replica release](https://github.com/facebookresearch/Replica-Dataset)
contains the source scene assets and license information.

Each scene directory must contain:

```text
scene/
├── traj.txt
└── results/
    ├── frame000000.jpg
    ├── depth000000.png
    └── ...
```

The loader constructs an RGB-D point cloud when no compatible cached PLY is
available. The main runner uses:

- pose convention: `raw`;
- depth frame stride: 120;
- depth pixel stride: 24;
- voxel size: 0.008;
- test split: original frame index modulo 8 equals zero.

Replica geometry evaluation additionally requires the corresponding cropped or
visible ground-truth mesh. Pass it explicitly through `GT_MESH`; this repository
does not redistribute dataset assets. NICE-SLAM distributes the evaluation
meshes in
[cull_replica_mesh.zip](https://cvg-data.inf.ethz.ch/nice-slam/cull_replica_mesh.zip).
Extract them as `data/cull_replica_mesh/`.

## DTU

Download the preprocessed 2DGS scenes from
[Hugging Face](https://huggingface.co/datasets/dylanebert/2DGS) or the original
[Google Drive folder](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9),
then place the scenes directly under `data/DTU/`:

```text
data/DTU/
├── scan24/
├── scan37/
└── ...
```

Official geometry metrics also require the
[DTU evaluation point clouds and observation masks](https://roboimagedata.compute.dtu.dk/?page_id=36).
Arrange the downloaded evaluation files as follows:

```text
data/DTU_Official/
├── ObsMask/
└── Points/
    └── stl/
```

Set `DTU_ROOT=data/DTU`, `DTU_OFFICIAL=data/DTU_Official`, and enable
`RUN_GEOMETRY=1`.

Dataset contents and trained outputs remain outside Git. Both `data/` and
`output/` may point to larger data volumes through symbolic links.
