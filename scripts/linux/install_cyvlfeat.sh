#!/usr/bin/env bash
set -euo pipefail

# Clean cyvlfeat install for linux/x86; no patches applied.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CYVLFEAT_SRC="${ROOT_DIR}/thirdparty/cyvlfeat"

if [[ ! -d "${CYVLFEAT_SRC}" ]]; then
  echo "cyvlfeat submodule not found at ${CYVLFEAT_SRC}."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

python -m pip install --no-deps --no-build-isolation "${CYVLFEAT_SRC}"
