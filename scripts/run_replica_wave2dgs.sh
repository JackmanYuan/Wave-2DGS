#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_PATH="${SOURCE_PATH:?Set SOURCE_PATH to a Replica scene directory}"
SCENE="${SCENE:-$(basename "${SOURCE_PATH}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/replica_wave2dgs_3k}"
RUN_ROOT="${OUTPUT_ROOT}/${SCENE}/seed_${SEED:-0}"
BASELINE_PATH="${RUN_ROOT}/baseline"
WAVE_PATH="${RUN_ROOT}/wave"
MANIFEST="${RUN_ROOT}/shared_test_views.json"

ITERATIONS="${ITERATIONS:-3000}"
SEED="${SEED:-0}"
MAX_PRIMITIVES="${MAX_PRIMITIVES:-50000}"
PORT_BASE="${PORT_BASE:-6009}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_RENDER="${FORCE_RENDER:-0}"

WAVE_K_INIT="${WAVE_K_INIT:-3.1416}"
WAVE_AMP_LR="${WAVE_AMP_LR:-0.001}"
WAVE_FREQ_LR="${WAVE_FREQ_LR:-0.008}"
WAVE_PHASE_LR="${WAVE_PHASE_LR:-0.001}"
WAVE_LAMBDA="${WAVE_LAMBDA:-0.03}"

[[ -d "${SOURCE_PATH}/results" && -f "${SOURCE_PATH}/traj.txt" ]] || {
  echo "SOURCE_PATH is not a Replica RGB-D scene: ${SOURCE_PATH}" >&2
  exit 2
}

REPLICA_ARGS=(
  --eval
  --replica_pose_convention raw
  --replica_depth_frame_stride 120
  --replica_depth_pixel_stride 24
  --replica_depth_voxel_size 0.008
  --replica_pointcloud_cache_tag lowsplat_f120_p24_v008
)

train_case() {
  local model="$1" port="$2"; shift 2
  local checkpoint="${model}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  if [[ "${FORCE_TRAIN}" == "1" || ! -f "${checkpoint}" ]]; then
    "${PYTHON_BIN}" train.py -s "${SOURCE_PATH}" -m "${model}" \
      "${REPLICA_ARGS[@]}" --iterations "${ITERATIONS}" --seed "${SEED}" \
      --port "${port}" --test_iterations "${ITERATIONS}" \
      --save_iterations "${ITERATIONS}" --max_primitives "${MAX_PRIMITIVES}" \
      --densify_from_iter 500 --densification_interval 100 \
      --densify_until_iter "${ITERATIONS}" "$@"
  else
    echo "[train] reuse ${checkpoint}"
  fi
}

render_case() {
  local model="$1"; shift
  local metrics="${model}/render_lite/ours_${ITERATIONS}/test/metrics.csv"
  if [[ "${FORCE_RENDER}" == "1" || ! -f "${metrics}" ]]; then
    "${PYTHON_BIN}" render_lite.py -s "${SOURCE_PATH}" -m "${model}" \
      "${REPLICA_ARGS[@]}" --iteration "${ITERATIONS}" --skip_train \
      --view_manifest "${MANIFEST}" "$@"
  else
    echo "[render] reuse ${metrics}"
  fi
}

mkdir -p "${RUN_ROOT}"
"${PYTHON_BIN}" scripts/create_replica_render_manifest.py \
  --source "${SOURCE_PATH}" --output "${MANIFEST}" \
  --max-views 16 --stride 8 --offset 0 --force

echo "[1/5] Original 2DGS training"
train_case "${BASELINE_PATH}" "${PORT_BASE}"

echo "[2/5] Wave-2DGS training"
train_case "${WAVE_PATH}" "$((PORT_BASE + 1))" \
  --enable_primitive_wave --wave_k_init "${WAVE_K_INIT}" \
  --wave_amp_lr "${WAVE_AMP_LR}" --wave_freq_lr "${WAVE_FREQ_LR}" \
  --wave_phase_lr "${WAVE_PHASE_LR}" \
  --wave_amp_group_lasso "${WAVE_LAMBDA}" \
  --wave_amp_group_lasso_start_iter 500 --wave_parameter_log_interval 500

echo "[3/5] Original 2DGS rendering"
render_case "${BASELINE_PATH}"

echo "[4/5] Wave-2DGS rendering and branch-disabling analysis"
render_case "${WAVE_PATH}" --enable_primitive_wave --wave_k_init "${WAVE_K_INIT}" \
  --save_wave_ablation

echo "[5/5] Paired summary"
"${PYTHON_BIN}" scripts/summarize_wave2dgs_pair.py \
  --baseline "${BASELINE_PATH}" --wave "${WAVE_PATH}" \
  --iteration "${ITERATIONS}" --output "${RUN_ROOT}/paired_summary.csv"

echo "Replica paired run complete: ${RUN_ROOT}"
