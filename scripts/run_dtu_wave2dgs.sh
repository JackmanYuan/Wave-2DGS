#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DTU_ROOT="${DTU_ROOT:?Set DTU_ROOT to the preprocessed 2DGS DTU directory}"
DTU_OFFICIAL="${DTU_OFFICIAL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/dtu_wave2dgs_30k}"
SCENES="${SCENES:-scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122}"
SEEDS="${SEEDS:-0}"

ITERATIONS="${ITERATIONS:-30000}"
MAX_PRIMITIVES="${MAX_PRIMITIVES:-150000}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"
PORT_BASE="${PORT_BASE:-7000}"
RUN_GEOMETRY="${RUN_GEOMETRY:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_RENDER="${FORCE_RENDER:-0}"

WAVE_K_INIT="${WAVE_K_INIT:-3.1416}"
WAVE_AMP_LR="${WAVE_AMP_LR:-0.001}"
WAVE_FREQ_LR="${WAVE_FREQ_LR:-0.008}"
WAVE_PHASE_LR="${WAVE_PHASE_LR:-0.001}"
WAVE_LAMBDA="${WAVE_LAMBDA:-0.03}"

if [[ "${RUN_GEOMETRY}" == "1" && ! -d "${DTU_OFFICIAL}" ]]; then
  echo "Set DTU_OFFICIAL to the official DTU evaluation data." >&2
  exit 2
fi

train_case() {
  local source="$1" model="$2" seed="$3" port="$4"; shift 4
  local checkpoint="${model}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  if [[ "${FORCE_TRAIN}" == "1" || ! -f "${checkpoint}" ]]; then
    "${PYTHON_BIN}" train.py -s "${source}" -m "${model}" -r 2 \
      --iterations "${ITERATIONS}" --seed "${seed}" --port "${port}" \
      --test_iterations -1 --save_iterations "${ITERATIONS}" \
      --depth_ratio 1.0 --lambda_dist 1000 \
      --max_primitives "${MAX_PRIMITIVES}" --densify_from_iter 500 \
      --densification_interval 100 --densify_until_iter "${DENSIFY_UNTIL_ITER}" "$@"
  else
    echo "[train] reuse ${checkpoint}"
  fi
}

evaluate_geometry() {
  local scene="$1" source="$2" model="$3"; shift 3
  local mesh="${model}/train/ours_${ITERATIONS}/fuse_post.ply"
  if [[ "${FORCE_RENDER}" == "1" || ! -f "${mesh}" ]]; then
    "${PYTHON_BIN}" render.py -s "${source}" -m "${model}" -r 2 \
      --iteration "${ITERATIONS}" --skip_train --skip_test --depth_ratio 1.0 \
      --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0 "$@"
  fi
  "${PYTHON_BIN}" scripts/eval_dtu/evaluate_single_scene.py \
    --input_mesh "${mesh}" --scan_id "${scene#scan}" \
    --output_dir "${model}/geometry_eval" --mask_dir "${DTU_ROOT}" \
    --DTU "${DTU_OFFICIAL}"
}

run_index=0
for scene in ${SCENES}; do
  source="${DTU_ROOT}/${scene}"
  [[ -d "${source}" ]] || { echo "Missing DTU scene: ${source}" >&2; exit 2; }
  for seed in ${SEEDS}; do
    run_root="${OUTPUT_ROOT}/${scene}"
    baseline="${run_root}/seed_${seed}_baseline"
    wave="${run_root}/seed_${seed}_wave"
    port=$((PORT_BASE + 2 * run_index))

    echo "========== ${scene}, seed ${seed}: Original 2DGS =========="
    train_case "${source}" "${baseline}" "${seed}" "${port}"
    echo "========== ${scene}, seed ${seed}: Wave-2DGS =========="
    train_case "${source}" "${wave}" "${seed}" "$((port + 1))" \
      --enable_primitive_wave --wave_k_init "${WAVE_K_INIT}" \
      --wave_amp_lr "${WAVE_AMP_LR}" --wave_freq_lr "${WAVE_FREQ_LR}" \
      --wave_phase_lr "${WAVE_PHASE_LR}" \
      --wave_amp_group_lasso "${WAVE_LAMBDA}" \
      --wave_amp_group_lasso_start_iter 500 --wave_parameter_log_interval 500

    if [[ "${RUN_GEOMETRY}" == "1" ]]; then
      evaluate_geometry "${scene}" "${source}" "${baseline}"
      evaluate_geometry "${scene}" "${source}" "${wave}" \
        --enable_primitive_wave --wave_k_init "${WAVE_K_INIT}"
    fi
    run_index=$((run_index + 1))
  done
done

if [[ "${RUN_GEOMETRY}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/collect_dtu_geometry_results.py --root "${OUTPUT_ROOT}"
  "${PYTHON_BIN}" scripts/summarize_dtu_paired.py --root "${OUTPUT_ROOT}"
fi

echo "DTU paired run complete: ${OUTPUT_ROOT}"
