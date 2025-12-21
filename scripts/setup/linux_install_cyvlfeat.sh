#!/usr/bin/env bash
set -euo pipefail

# Clean cyvlfeat install for linux/x86; no patches applied.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CYVLFEAT_SRC="${ROOT_DIR}/thirdparty/cyvlfeat"
PREFIX="${CONDA_PREFIX:-}"

if [[ -z "${PREFIX}" ]]; then
  echo "CONDA_PREFIX is not set. Run this via 'pixi run' so the env prefix is available."
  exit 1
fi

if [[ ! -d "${CYVLFEAT_SRC}" ]]; then
  echo "cyvlfeat submodule not found at ${CYVLFEAT_SRC}."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

export CFLAGS="-I${PREFIX}/include ${CFLAGS:-}"
export CPPFLAGS="-I${PREFIX}/include ${CPPFLAGS:-}"
export LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib ${LDFLAGS:-}"
export LIBRARY_PATH="${PREFIX}/lib:${LIBRARY_PATH:-}"

python -m pip install --no-deps --no-build-isolation "${CYVLFEAT_SRC}"
