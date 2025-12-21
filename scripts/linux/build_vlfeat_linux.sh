#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLFEAT_SRC="${ROOT_DIR}/thirdparty/vlfeat"
EXTRA_CFLAGS="${VLFEAT_EXTRA_CFLAGS:-}"

if [[ ! -d "${VLFEAT_SRC}" ]]; then
  echo "VLFeat submodule not found at ${VLFEAT_SRC}."
  echo "Run: git submodule update --init --recursive"
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

echo "Building VLFeat (linux x86_64)..."
make -B -C "${VLFEAT_SRC}" ARCH=glnxa64 MEX= MKOCTFILE= \
  CFLAGS="-I${VLFEAT_SRC} ${EXTRA_CFLAGS}" \
  DISABLE_OPENMP=yes dll

LIBVL_PATH="${VLFEAT_SRC}/bin/glnxa64/libvl.so"

popd >/dev/null

ENV_DIR="${ROOT_DIR}/.pixi"
ENV_FILE="${ENV_DIR}/vlfeat_env_linux.sh"
mkdir -p "${ENV_DIR}"
cat > "${ENV_FILE}" <<ENV_EOF
export VLFEAT_SRC="${VLFEAT_SRC}"
export CFLAGS="-I${VLFEAT_SRC} ${EXTRA_CFLAGS}"
export LDFLAGS="-L${VLFEAT_SRC}/bin/glnxa64 -Wl,-rpath,${VLFEAT_SRC}/bin/glnxa64"
export LD_LIBRARY_PATH="${VLFEAT_SRC}/bin/glnxa64:\${LD_LIBRARY_PATH:-}"
export PIP_NO_BUILD_ISOLATION=1
ENV_EOF

echo ""
echo "VLFeat built at: ${VLFEAT_SRC}"
echo "libvl.so: ${LIBVL_PATH}"
echo "Env file: ${ENV_FILE}"
echo ""
echo "To install cyvlfeat from the submodule:"
echo "  export CFLAGS=\"-I${VLFEAT_SRC} ${EXTRA_CFLAGS}\""
echo "  export LDFLAGS=\"-L${VLFEAT_SRC}/bin/glnxa64 -Wl,-rpath,${VLFEAT_SRC}/bin/glnxa64\""
echo "  export LD_LIBRARY_PATH=\"${VLFEAT_SRC}/bin/glnxa64:\${LD_LIBRARY_PATH:-}\""
echo "  python -m pip install --no-deps --no-build-isolation ${ROOT_DIR}/thirdparty/cyvlfeat"
