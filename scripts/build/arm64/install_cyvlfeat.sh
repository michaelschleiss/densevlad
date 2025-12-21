#!/usr/bin/env bash
set -euo pipefail

# Reference-only arm64 helper; not part of the default linux/x86 workflow.
# Set DVLAD_PATCH_CYVLFEAT=1 to apply the optional cyvlfeat patch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM64_DIR="${ROOT_DIR}/scripts/arm64"
REF_DIR="${ROOT_DIR}/scripts/reference/cyvlfeat"
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/Library/Caches}"
CYVLFEAT_CACHE="${CACHE_BASE}/densevlad/cyvlfeat"
CYVLFEAT_SRC="${ROOT_DIR}/thirdparty/cyvlfeat"

if [[ ! -d "${CYVLFEAT_SRC}" ]]; then
  echo "cyvlfeat submodule not found at ${CYVLFEAT_SRC}."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

INSTALL_SRC="${CYVLFEAT_SRC}"

if [[ "${DVLAD_PATCH_CYVLFEAT:-0}" != "0" ]]; then
  mkdir -p "${CYVLFEAT_CACHE}"
  INSTALL_SRC="${CYVLFEAT_CACHE}/cyvlfeat-patched"
  rm -rf "${INSTALL_SRC}"
  rsync -a --delete --exclude ".git" "${CYVLFEAT_SRC}/" "${INSTALL_SRC}/"
  python "${REF_DIR}/patch_cyvlfeat_sdist.py" "${INSTALL_SRC}"
fi

python -m pip install --no-deps --force-reinstall --no-cache-dir --no-build-isolation "${INSTALL_SRC}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  python "${ARM64_DIR}/fix_cyvlfeat_rpath.py"
fi
