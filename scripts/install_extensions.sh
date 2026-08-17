#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RASTERIZER="${ROOT}/submodules/diff-surfel-rasterization"
WAVE_PATCH="${ROOT}/patches/0001-feat-add-primitive-local-wave-residual-rasterization.patch"

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
