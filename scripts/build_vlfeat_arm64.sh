#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/Library/Caches}"
VLFEAT_CACHE="${CACHE_BASE}/dvlad/vlfeat"
VLFEAT_VER="0.9.20"
VLFEAT_TAR="vlfeat-${VLFEAT_VER}.tar.gz"
VLFEAT_URL="http://www.vlfeat.org/download/${VLFEAT_TAR}"
VLFEAT_SRC="${VLFEAT_CACHE}/vlfeat-${VLFEAT_VER}"

mkdir -p "${VLFEAT_CACHE}"

if [[ ! -f "${VLFEAT_CACHE}/${VLFEAT_TAR}" ]]; then
  echo "Downloading VLFeat ${VLFEAT_VER}..."
  curl -L "${VLFEAT_URL}" -o "${VLFEAT_CACHE}/${VLFEAT_TAR}"
fi

if [[ ! -d "${VLFEAT_SRC}" ]]; then
  echo "Extracting VLFeat..."
  tar -xzf "${VLFEAT_CACHE}/${VLFEAT_TAR}" -C "${VLFEAT_CACHE}"
fi

pushd "${VLFEAT_SRC}" >/dev/null

HOST_H="vl/host.h"
HOST_C="vl/host.c"

if ! grep -Eq "__aarch64__|__arm64__|_M_ARM64" "${HOST_H}"; then
  echo "Patching VLFeat host.h for arm64..."
  patch -p0 -N <<'PATCH'
--- vl/host.h
+++ vl/host.h
@@ -260,6 +260,10 @@
 #define VL_ARCH_IA64
 #endif
 /** @} */
+
+#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
+#define VL_ARCH_ARM64
+#endif
@@ -295,6 +299,7 @@
 #if defined(__LITTLE_ENDIAN__) || \
     defined(VL_ARCH_IX86)      || \
     defined(VL_ARCH_IA64)      || \
     defined(VL_ARCH_X64)       || \
+    defined(VL_ARCH_ARM64)     || \
     defined(__DOXYGEN__)
 #define VL_ARCH_LITTLE_ENDIAN
 #endif
PATCH
fi

if ! grep -q "non-x86" "${HOST_C}"; then
  echo "Patching VLFeat host.c for non-x86 cpuid..."
  patch -p0 -N <<'PATCH'
--- vl/host.c
+++ vl/host.c
@@ -395,6 +395,7 @@
 #include "host.h"
 #include "generic.h"
 #include <stdio.h>
+#include <string.h>
@@ -444,6 +445,7 @@
 void
 _vl_x86cpu_info_init (VlX86CpuInfo *self)
 {
+#if defined(HAS_CPUID)
   vl_int32 info [4] ;
   int max_func = 0 ;
   _vl_cpuid(info, 0) ;
   max_func = info[0] ;
@@ -465,6 +467,9 @@
     self->hasSSE42 = info[2] & (1 << 20) ;
     self->hasAVX   = info[2] & (1 << 28) ;
   }
+#else
+  memset(self, 0, sizeof(*self));
+  snprintf(self->vendor.string, sizeof(self->vendor.string), "non-x86");
+#endif
 }
PATCH
fi

echo "Building VLFeat (arm64-safe settings)..."
make -C "${VLFEAT_SRC}" ARCH=maci64 MEX= MKOCTFILE= \
  DISABLE_SSE2=yes DISABLE_AVX=yes DISABLE_OPENMP=yes dll

popd >/dev/null

ENV_DIR="${ROOT_DIR}/.pixi"
ENV_FILE="${ENV_DIR}/vlfeat_env.sh"
mkdir -p "${ENV_DIR}"
cat > "${ENV_FILE}" <<EOF
export VLFEAT_SRC="${VLFEAT_SRC}"
export CFLAGS="-I${VLFEAT_SRC}"
export LDFLAGS="-L${VLFEAT_SRC}/bin/maci64"
export DYLD_LIBRARY_PATH="${VLFEAT_SRC}/bin/maci64:\${DYLD_LIBRARY_PATH:-}"
export PIP_NO_BUILD_ISOLATION=1
EOF

echo ""
echo "VLFeat built at: ${VLFEAT_SRC}"
echo "libvl.dylib: ${VLFEAT_SRC}/bin/maci64/libvl.dylib"
echo "Env file: ${ENV_FILE}"
echo ""
echo "To install cyvlfeat:"
echo "  export CFLAGS=\"-I${VLFEAT_SRC}\""
echo "  export LDFLAGS=\"-L${VLFEAT_SRC}/bin/maci64\""
echo "  export DYLD_LIBRARY_PATH=\"${VLFEAT_SRC}/bin/maci64:\${DYLD_LIBRARY_PATH:-}\""
echo "  pip install --no-build-isolation cyvlfeat"
