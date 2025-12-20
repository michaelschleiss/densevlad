#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/Library/Caches}"
VLFEAT_CACHE="${CACHE_BASE}/dvlad/vlfeat"
VLFEAT_GIT_URL="https://github.com/vlfeat/vlfeat.git"
VLFEAT_SRC="${VLFEAT_CACHE}/vlfeat-0.9.20-git"
SSE2NEON_URL="https://raw.githubusercontent.com/DLTcollab/sse2neon/master/sse2neon.h"
SSE2NEON_CACHE="${VLFEAT_CACHE}/sse2neon"
SSE2NEON_HEADER="${SSE2NEON_CACHE}/sse2neon.h"
SSE2NEON_INCLUDE="${VLFEAT_SRC}/sse2neon"

mkdir -p "${VLFEAT_CACHE}"

if [[ ! -d "${VLFEAT_SRC}" ]]; then
  echo "Cloning VLFeat..."
  git clone "${VLFEAT_GIT_URL}" "${VLFEAT_SRC}"
fi

pushd "${VLFEAT_SRC}" >/dev/null

echo "Fetching VLFeat tags..."
git fetch --tags >/dev/null 2>&1 || true

VLFEAT_TAG="$(git tag --sort=version:refname | tail -n 1)"
if [[ -z "${VLFEAT_TAG}" ]]; then
  echo "Error: could not determine latest VLFeat tag."
  exit 1
fi

if ! git diff --quiet; then
  echo "Local changes detected in VLFeat; keeping current checkout."
else
  echo "Checking out VLFeat tag ${VLFEAT_TAG}..."
  git checkout "${VLFEAT_TAG}" >/dev/null 2>&1
fi

HOST_H="vl/host.h"
DLL_MAK="make/dll.mak"
MAKEFILE="Makefile"
GENERIC_C="vl/generic.c"

if ! grep -Eq "__aarch64__|__arm64__|_M_ARM64" "${HOST_H}"; then
  echo "Patching VLFeat host.h for arm64..."
  python - <<'PY'
from pathlib import Path

path = Path("vl/host.h")
text = path.read_text()

arm_block = (
    "\n#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)\n"
    "#define VL_ARCH_ARM64\n"
    "#endif\n"
)

if "VL_ARCH_ARM64" not in text:
    marker = "/** @} */"
    ia64 = "#define VL_ARCH_IA64\n#endif\n"
    if ia64 in text and marker in text:
        text = text.replace(ia64 + marker, ia64 + arm_block + marker, 1)
    elif marker in text:
        text = text.replace(marker, arm_block + marker, 1)

if "VL_ARCH_ARM64" not in text:
    raise SystemExit("Failed to inject VL_ARCH_ARM64 into vl/host.h")

inserted = False
needle = "    defined(VL_ARCH_X64)       || \\\n"
if needle in text:
    block = text[text.find(needle):text.find(needle) + 200]
    if "VL_ARCH_ARM64" not in block:
        text = text.replace(
            needle,
            needle + "    defined(VL_ARCH_ARM64)     || \\\n",
            1,
        )
        inserted = True

if not inserted:
    le_marker = "    defined(__DOXYGEN__)\n"
    if le_marker in text:
        text = text.replace(
            le_marker,
            "    defined(VL_ARCH_ARM64)     || \\\n" + le_marker,
            1,
        )

path.write_text(text)
PY
fi

if ! grep -q "VLFEAT_SSE2_FLAG" "${DLL_MAK}"; then
  echo "Patching VLFeat dll.mak to allow arm64 SSE2 flags..."
  patch -p0 -N <<'PATCH'
--- make/dll.mak
+++ make/dll.mak
@@ -34,6 +34,8 @@
 LINK_DLL_LDFLAGS =\
 -L$(BINDIR) -lvl
 
+VLFEAT_SSE2_FLAG ?= -msse2
+
 DLL_CFLAGS = \
 $(STD_CFLAGS) \
 -fvisibility=hidden -fPIC -DVL_BUILD_DLL \
@@ -41,7 +43,7 @@
 $(LINK_DLL_CFLAGS) \
-$(call if-like,%_sse2,$*, $(if $(DISABLE_SSE2),,-msse2)) \
+$(call if-like,%_sse2,$*, $(if $(DISABLE_SSE2),,$(VLFEAT_SSE2_FLAG))) \
 $(call if-like,%_avx,$*, $(if $(DISABLE_AVX),,-mavx)) \
 $(if $(DISABLE_THREADS),,-pthread) \
 $(if $(DISABLE_OPENMP),,-fopenmp)
PATCH
fi

if grep -q 'STD_CLFAGS = $(CFLAGS)' "${MAKEFILE}"; then
  echo "Patching VLFeat Makefile to honor CFLAGS..."
  patch -p0 -N <<'PATCH'
--- Makefile
+++ Makefile
@@ -150,7 +150,7 @@
 LIBTOOL ?= libtool
 
-STD_CLFAGS = $(CFLAGS)
+STD_CFLAGS = $(CFLAGS)
 STD_CFLAGS += -std=c99
 STD_CFLAGS += -Wall -Wextra
 STD_CFLAGS += -Wno-unused-function -Wno-long-long -Wno-variadic-macros
PATCH
fi

if ! grep -q "VL_ARCH_ARM64" "${GENERIC_C}"; then
  echo "Patching VLFeat generic.c to enable SSE2 dispatch on arm64..."
  python - <<'PY'
from pathlib import Path

path = Path("vl/generic.c")
text = path.read_text()

old = (
    "#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)\n"
    "  return vl_get_state()->cpuInfo.hasSSE2 ;\n"
    "#else\n"
    "  return VL_FALSE ;\n"
    "#endif\n"
)
new = (
    "#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)\n"
    "  return vl_get_state()->cpuInfo.hasSSE2 ;\n"
    "#elif defined(VL_ARCH_ARM64)\n"
    "  return VL_TRUE ;\n"
    "#else\n"
    "  return VL_FALSE ;\n"
    "#endif\n"
)

if old not in text:
    raise SystemExit("Expected vl_cpu_has_sse2 block not found in vl/generic.c")

text = text.replace(old, new, 1)
path.write_text(text)
PY
fi

mkdir -p "${SSE2NEON_CACHE}"
if [[ ! -f "${SSE2NEON_HEADER}" ]]; then
  echo "Downloading sse2neon..."
  curl -L "${SSE2NEON_URL}" -o "${SSE2NEON_HEADER}"
fi

mkdir -p "${SSE2NEON_INCLUDE}"
cp "${SSE2NEON_HEADER}" "${SSE2NEON_INCLUDE}/sse2neon.h"
cat > "${SSE2NEON_INCLUDE}/emmintrin.h" <<'EOF'
#ifndef SSE2NEON_EMMINTRIN_H
#define SSE2NEON_EMMINTRIN_H
#include "sse2neon.h"
#endif
EOF

make -C "${VLFEAT_SRC}" clean >/dev/null 2>&1 || true

echo "Building VLFeat (arm64 + sse2neon)..."
make -B -C "${VLFEAT_SRC}" ARCH=maci64 MEX= MKOCTFILE= \
  CFLAGS="-I${SSE2NEON_INCLUDE} -I${VLFEAT_SRC} -D__SSE2__ -D__SSE__" \
  VLFEAT_SSE2_FLAG= \
  DISABLE_AVX=yes DISABLE_OPENMP=yes dll

LIBVL_PATH="${VLFEAT_SRC}/bin/maci64/libvl.dylib"
if [[ -f "${LIBVL_PATH}" ]]; then
  install_name_tool -id "@rpath/libvl.dylib" "${LIBVL_PATH}"
fi

popd >/dev/null

ENV_DIR="${ROOT_DIR}/.pixi"
ENV_FILE="${ENV_DIR}/vlfeat_env.sh"
mkdir -p "${ENV_DIR}"
cat > "${ENV_FILE}" <<EOF
export VLFEAT_SRC="${VLFEAT_SRC}"
export CFLAGS="-I${SSE2NEON_INCLUDE} -I${VLFEAT_SRC} -D__SSE2__ -D__SSE__"
export LDFLAGS="-L${VLFEAT_SRC}/bin/maci64 -Wl,-rpath,${VLFEAT_SRC}/bin/maci64"
export DYLD_LIBRARY_PATH="${VLFEAT_SRC}/bin/maci64:\${DYLD_LIBRARY_PATH:-}"
export PIP_NO_BUILD_ISOLATION=1
EOF

echo ""
echo "VLFeat built at: ${VLFEAT_SRC}"
echo "VLFeat tag: ${VLFEAT_TAG}"
echo "libvl.dylib: ${VLFEAT_SRC}/bin/maci64/libvl.dylib"
echo "Env file: ${ENV_FILE}"
echo ""
echo "To install cyvlfeat:"
echo "  export CFLAGS=\"-I${SSE2NEON_INCLUDE} -I${VLFEAT_SRC} -D__SSE2__ -D__SSE__\""
echo "  export LDFLAGS=\"-L${VLFEAT_SRC}/bin/maci64 -Wl,-rpath,${VLFEAT_SRC}/bin/maci64\""
echo "  export DYLD_LIBRARY_PATH=\"${VLFEAT_SRC}/bin/maci64:\${DYLD_LIBRARY_PATH:-}\""
echo "  pip install --no-build-isolation cyvlfeat"
