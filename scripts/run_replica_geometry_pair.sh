#!/usr/bin/env bash
# Extract and evaluate a matched Original/Wave Replica mesh pair.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_PATH="${SOURCE_PATH:?Set SOURCE_PATH to a Replica scene directory}"
GT_MESH="${GT_MESH:?Set GT_MESH to the corresponding Replica ground-truth mesh}"
ITERATIONS="${ITERATIONS:-3000}"
BASELINE_PATH="${BASELINE_PATH:?Set BASELINE_PATH to the matched Original checkpoint directory}"
WAVE_PATH="${WAVE_PATH:?Set WAVE_PATH to the matched Wave checkpoint directory}"
VOXEL_SIZE="${VOXEL_SIZE:-0.01}"
SDF_TRUNC="${SDF_TRUNC:-0.04}"
DEPTH_TRUNC="${DEPTH_TRUNC:-10.0}"
NUM_CLUSTER="${NUM_CLUSTER:-20}"
FORCE_MESH="${FORCE_MESH:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FUSION_VIEW_STRIDE="${FUSION_VIEW_STRIDE:-20}"
WAVE_K_INIT="${WAVE_K_INIT:-3.1416}"

GEOMETRY_SUMMARY="${GEOMETRY_SUMMARY:-$(dirname "${WAVE_PATH}")/replica_geometry_summary.csv}"
SOURCE="${SOURCE_PATH}"
[[ -d "${SOURCE}" && -f "${GT_MESH}" ]] || { echo "Missing Replica source or GT mesh" >&2; exit 1; }
BASE_CHECKPOINT="${BASELINE_PATH}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
WAVE_CHECKPOINT="${WAVE_PATH}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
[[ -f "${BASE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASE_CHECKPOINT}" >&2; exit 1; }
[[ -f "${WAVE_CHECKPOINT}" ]] || { echo "Missing Wave checkpoint: ${WAVE_CHECKPOINT}" >&2; exit 1; }
FUSION_MANIFEST="${FUSION_MANIFEST:-$(dirname "${BASELINE_PATH}")/replica_fusion_train_stride${FUSION_VIEW_STRIDE}.json}"
"${PYTHON_BIN}" scripts/eval_replica/create_fusion_manifest.py --source "${SOURCE}" \
  --output "${FUSION_MANIFEST}" --stride "${FUSION_VIEW_STRIDE}"
echo "Replica fixed-view protocol: ${FUSION_MANIFEST}"

extract_mesh() {
  local model="$1"; shift
  local mesh="${model}/train/ours_${ITERATIONS}/fuse_post.ply"
  local mesh_manifest="${mesh}.fixedview_manifest.json"
  if [[ "${FORCE_MESH}" == "1" || ! -f "${mesh}" || ! -f "${mesh_manifest}" ]] || \
     ! cmp -s "${FUSION_MANIFEST}" "${mesh_manifest}"; then
    if ! "${PYTHON_BIN}" render.py -s "${SOURCE}" -m "${model}" --iteration "${ITERATIONS}" \
      --skip_train --skip_test --depth_ratio 1.0 --voxel_size "${VOXEL_SIZE}" \
      --sdf_trunc "${SDF_TRUNC}" --depth_trunc "${DEPTH_TRUNC}" --num_cluster "${NUM_CLUSTER}" \
      --mesh_view_manifest "${FUSION_MANIFEST}" "$@" >&2; then
      echo "Replica mesh extraction failed for ${model}" >&2
      return 1
    fi
    [[ -f "${mesh}" ]] || { echo "Mesh extraction produced no file: ${mesh}" >&2; return 1; }
    cp "${FUSION_MANIFEST}" "${mesh_manifest}"
  fi
  printf '%s\n' "${mesh}"
}

BASE_MESH="$(extract_mesh "${BASELINE_PATH}")"
WAVE_MESH="$(extract_mesh "${WAVE_PATH}" --enable_primitive_wave --wave_k_init "${WAVE_K_INIT}")"
CACHE="$(dirname "${BASELINE_PATH}")/replica_geometry_gt_visible_stride${FUSION_VIEW_STRIDE}.npz"

evaluate() {
  local label="$1" mesh="$2" model="$3"
  local result="${model}/geometry_eval_replica_fixedview/results.json"
  if [[ "${FORCE_EVAL}" == "1" || ! -f "${result}" ]]; then
    [[ -f "${mesh}" ]] || { echo "Cannot evaluate missing mesh: ${mesh}" >&2; return 1; }
    if ! "${PYTHON_BIN}" scripts/eval_replica/evaluate_mesh_protocol.py --source "${SOURCE}" --pred-mesh "${mesh}" --gt-mesh "${GT_MESH}" \
      --output-dir "${model}/geometry_eval_replica_fixedview" --gt-visible-cache "${CACHE}" \
      --fusion-manifest "${FUSION_MANIFEST}" --pose-convention raw --pred-cull-mode frustum >&2; then
      echo "Replica geometry evaluation failed for ${model}" >&2
      return 1
    fi
    [[ -f "${result}" ]] || { echo "Evaluation produced no result: ${result}" >&2; return 1; }
  fi
  printf '%s\n' "${result}"
}

BASE_RESULT="$(evaluate baseline "${BASE_MESH}" "${BASELINE_PATH}")"
WAVE_RESULT="$(evaluate wave "${WAVE_MESH}" "${WAVE_PATH}")"
"${PYTHON_BIN}" scripts/summarize_replica_geometry_pair.py --baseline "${BASE_RESULT}" --wave "${WAVE_RESULT}" \
  --output "${GEOMETRY_SUMMARY}"
