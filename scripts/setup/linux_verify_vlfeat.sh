#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/.pixi/vlfeat_env_linux.sh"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Run: pixi run build-vlfeat-linux"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
  echo "LD_LIBRARY_PATH is not set after sourcing ${ENV_FILE}."
  exit 1
fi

echo "Using VLFeat from: ${VLFEAT_SRC}"
if [[ ! -f "${VLFEAT_SRC}/bin/glnxa64/libvl.so" ]]; then
  echo "libvl.so not found under ${VLFEAT_SRC}/bin/glnxa64"
  exit 1
fi

echo "Benchmark: PHOW (SIMD on)"
python "${ROOT_DIR}/scripts/bench/bench_phow_breakdown.py"

echo "Benchmark: PHOW (SIMD forced off via DENSEVLAD_DISABLE_SIMD=1)"
DENSEVLAD_DISABLE_SIMD=1 python "${ROOT_DIR}/scripts/bench/bench_phow_breakdown.py"

cat <<'NOTE'

If SIMD is active, the second run (SIMD off) should be noticeably slower.
If the timings are similar, the build is not using SIMD paths.
NOTE
