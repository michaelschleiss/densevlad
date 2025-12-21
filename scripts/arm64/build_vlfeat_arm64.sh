#!/usr/bin/env bash
set -euo pipefail

# Reference-only arm64 build helper; not part of the default linux/x86 workflow.
# This script assumes arm-specific patches when arm support resumes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM64_DIR="${ROOT_DIR}/scripts/arm64"
VLFEAT_SRC="${ROOT_DIR}/thirdparty/vlfeat"
SSE2NEON_INCLUDE="${ARM64_DIR}/sse2neon"
EXTRA_CFLAGS="${VLFEAT_EXTRA_CFLAGS:-}"

if [[ ! -d "${VLFEAT_SRC}" ]]; then
  echo "VLFeat submodule not found at ${VLFEAT_SRC}."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

if [[ ! -f "${SSE2NEON_INCLUDE}/sse2neon.h" ]]; then
  echo "Missing sse2neon headers at ${SSE2NEON_INCLUDE}."
  echo "Ensure scripts/arm64/sse2neon is present."
  exit 1
fi

required=(
  "vl/host.h"
  "make/dll.mak"
  "Makefile"
  "vl/generic.c"
  "vl/imopv.c"
  "vl/dsift.c"
)
for path in "${required[@]}"; do
  if [[ ! -f "${VLFEAT_SRC}/${path}" ]]; then
    echo "VLFeat source tree is missing ${path}."
    exit 1
  fi
done

pushd "${VLFEAT_SRC}" >/dev/null

make -C "${VLFEAT_SRC}" clean >/dev/null 2>&1 || true

echo "Building VLFeat (arm64 + sse2neon)..."
make -B -C "${VLFEAT_SRC}" ARCH=maci64 MEX= MKOCTFILE= \
  CFLAGS="-I${SSE2NEON_INCLUDE} -I${VLFEAT_SRC} -D__SSE2__ -D__SSE__ ${EXTRA_CFLAGS}" \
  VLFEAT_SSE2_FLAG= \
  DISABLE_AVX=yes DISABLE_OPENMP=yes dll

LIBVL_PATH="${VLFEAT_SRC}/bin/maci64/libvl.dylib"
if [[ -f "${LIBVL_PATH}" ]]; then
  install_name_tool -id "@rpath/libvl.dylib" "${LIBVL_PATH}"
fi

VLFEAT_REV=""
if command -v git >/dev/null 2>&1; then
  VLFEAT_REV="$(git rev-parse --short HEAD 2>/dev/null || true)"
fi

popd >/dev/null

ENV_DIR="${ROOT_DIR}/.pixi"
ENV_FILE="${ENV_DIR}/vlfeat_env.sh"
mkdir -p "${ENV_DIR}"
cat > "${ENV_FILE}" <<ENV_EOF
export VLFEAT_SRC="${VLFEAT_SRC}"
export CFLAGS="-I${SSE2NEON_INCLUDE} -I${VLFEAT_SRC} -D__SSE2__ -D__SSE__ ${EXTRA_CFLAGS}"
export LDFLAGS="-L${VLFEAT_SRC}/bin/maci64 -Wl,-rpath,${VLFEAT_SRC}/bin/maci64"
export DYLD_LIBRARY_PATH="${VLFEAT_SRC}/bin/maci64:\${DYLD_LIBRARY_PATH:-}"
export PIP_NO_BUILD_ISOLATION=1
ENV_EOF

echo ""
echo "VLFeat built at: ${VLFEAT_SRC}"
if [[ -n "${VLFEAT_REV}" ]]; then
  echo "VLFeat rev: ${VLFEAT_REV}"
fi
echo "libvl.dylib: ${VLFEAT_SRC}/bin/maci64/libvl.dylib"
echo "Env file: ${ENV_FILE}"
echo ""
