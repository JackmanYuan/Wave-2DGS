#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RASTERIZER="${ROOT}/submodules/diff-surfel-rasterization"
WAVE_PATCH="${ROOT}/patches/0001-feat-add-primitive-local-wave-residual-rasterization.patch"

# Prefer the CUDA toolkit installed beside the selected Python interpreter.
# This avoids accidentally compiling against a system nvcc whose CUDA version
# differs from the PyTorch build in the active conda environment.
PYTHON_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
if [[ -z "${CUDA_HOME:-}" && -x "${PYTHON_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${PYTHON_PREFIX}"
fi
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

if [[ -z "${CC:-}" && -x "${PYTHON_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="${PYTHON_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
fi
if [[ -z "${CXX:-}" && -x "${PYTHON_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]]; then
  export CXX="${PYTHON_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
fi

echo "Python: ${PYTHON_BIN}"
echo "CUDA toolkit: ${CUDA_HOME:-auto-detected by PyTorch}"
echo "Host compiler: ${CXX:-system default}"

git -C "${ROOT}" submodule update --init --recursive

if git -C "${RASTERIZER}" apply --check "${WAVE_PATCH}" >/dev/null 2>&1; then
  echo "Applying Wave-2DGS rasterizer patch"
  git -C "${RASTERIZER}" apply "${WAVE_PATCH}"
elif git -C "${RASTERIZER}" apply --reverse --check "${WAVE_PATCH}" >/dev/null 2>&1; then
  echo "Wave-2DGS rasterizer patch is already applied"
else
  echo "Cannot apply Wave-2DGS rasterizer patch cleanly." >&2
  echo "Reset submodules/diff-surfel-rasterization to the pinned commit and retry." >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install "${RASTERIZER}"
"${PYTHON_BIN}" -m pip install "${ROOT}/submodules/simple-knn"

echo "Wave-2DGS CUDA extensions installed successfully."
