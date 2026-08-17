# Dataset preparation

## COLMAP scenes

Wave-2DGS retains the original 2DGS COLMAP loader. Follow the upstream 3DGS /
2DGS preparation instructions and pass the scene directory with `-s`.

## Replica

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
visible ground-truth mesh. Pass it explicitly through `GT_MESH`; the repository
does not redistribute dataset assets.

## DTU

`scripts/run_dtu_wave2dgs.sh` expects the preprocessed DTU scenes distributed
for the original 2DGS evaluation:

```text
DTU_ROOT/
├── scan24/
├── scan37/
└── ...
```

Official geometry metrics also require the DTU evaluation point clouds and
observation masks. Set `DTU_OFFICIAL` to that evaluation root and enable
`RUN_GEOMETRY=1`.

Datasets and trained outputs should remain outside Git. The repository ignores
`output/`, so it may also be a symbolic link to a larger data volume.
