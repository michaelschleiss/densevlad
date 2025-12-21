#!/usr/bin/env bash
set -euo pipefail

# Reference-only arm64 helper; not part of the default linux/x86 workflow.
# Set DVLAD_PATCH_CYVLFEAT=1 to apply the optional cyvlfeat patch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM64_DIR="${ROOT_DIR}/scripts/arm64"
REF_DIR="${ROOT_DIR}/scripts/reference/cyvlfeat"
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/Library/Caches}"
CYVLFEAT_CACHE="${CACHE_BASE}/densevlad/cyvlfeat"
CYVLFEAT_VER="0.7.1"
CYVLFEAT_TAR="cyvlfeat-${CYVLFEAT_VER}.tar.gz"
CYVLFEAT_SRC="${CYVLFEAT_CACHE}/cyvlfeat-${CYVLFEAT_VER}"

mkdir -p "${CYVLFEAT_CACHE}"

if [[ ! -f "${CYVLFEAT_CACHE}/${CYVLFEAT_TAR}" ]]; then
  python -m pip download --no-deps --no-binary=cyvlfeat -d "${CYVLFEAT_CACHE}" "cyvlfeat==${CYVLFEAT_VER}"
fi

if [[ -d "${CYVLFEAT_SRC}" ]]; then
  rm -rf "${CYVLFEAT_SRC}"
fi
tar -xzf "${CYVLFEAT_CACHE}/${CYVLFEAT_TAR}" -C "${CYVLFEAT_CACHE}"

if [[ "${DVLAD_PATCH_CYVLFEAT:-0}" != "0" ]]; then
  python "${REF_DIR}/patch_cyvlfeat_sdist.py" "${CYVLFEAT_SRC}"
fi

python -m pip install --no-deps --force-reinstall --no-cache-dir --no-build-isolation "${CYVLFEAT_SRC}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  python "${ARM64_DIR}/fix_cyvlfeat_rpath.py"
fi
