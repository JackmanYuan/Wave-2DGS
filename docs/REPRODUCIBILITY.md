# Reproducibility protocols

## Selected Wave-2DGS configuration

| Setting | Value |
|---|---:|
| Initial frequency magnitude | 3.1416 |
| Amplitude learning rate | 0.001 |
| Frequency learning rate | 0.008 |
| Phase learning rate | 0.001 |
| Group-Lasso weight | 0.03 |
| Group-Lasso start | iteration 500 |

All primitives receive the same wave parameterization. The gate stored in the
dense checkpoint is fixed to one; selective utilization is learned through the
RGB amplitude and group-sparse penalty.

## Matched comparisons

For every baseline/Wave comparison, keep the following identical:

- source images and camera split;
- point-cloud initializer and random seed;
- iteration and densification schedule;
- geometry, opacity, and spherical-harmonic learning rates;
- hard primitive cap;
- rendering and geometry evaluation views.

The only method-specific flags should be the Wave-2DGS parameters listed above.

## Replica rendering

`scripts/run_replica_wave2dgs.sh` uses 3k iterations, a 50k hard primitive cap,
and a manifest containing 16 held-out views. The same manifest is passed to
both methods through `render_lite.py --view_manifest`.

## Replica geometry

`scripts/run_replica_geometry_pair.sh` selects every twentieth training view,
uses that shared manifest for both TSDF fusions, and evaluates both meshes with
the same visible/cropped ground-truth support.

## DTU geometry

`scripts/run_dtu_wave2dgs.sh` defaults to:

- 30k optimization iterations;
- densification from iteration 500 to 15k, every 100 iterations;
- 150k hard primitive cap;
- resolution factor 2 and `depth_ratio=1`;
- the original 2DGS mesh extraction and official DTU evaluator.

## Baseline safety

Wave support is disabled by default. A command without
`--enable_primitive_wave` follows the original appearance path and does not add
Wave parameters to its checkpoint.
