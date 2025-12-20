from __future__ import annotations

from pathlib import Path
import os
from typing import Tuple
import ctypes
import ctypes.util
import math
import sys

import numpy as np
from PIL import Image

VL_EPSILON_F = np.float32(1.19209290e-07)
_VL_PAD_BY_CONTINUITY = 1 << 0
_VL_TRANSPOSE = 1 << 2


def _load_libvl():
    candidates = []
    prefix = Path(sys.prefix)
    if sys.platform == "darwin":
        candidates.append(prefix / "lib" / "libvl.dylib")
    elif sys.platform.startswith("linux"):
        candidates.append(prefix / "lib" / "libvl.so")
        candidates.append(prefix / "lib64" / "libvl.so")
    elif sys.platform.startswith("win"):
        candidates.append(prefix / "Library" / "bin" / "vl.dll")
    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))
    libname = ctypes.util.find_library("vl")
    if libname:
        try:
            return ctypes.CDLL(libname)
        except OSError:
            return None
    return None


_LIBVL = _load_libvl()
if _LIBVL is not None:
    _PTR_FLOAT = ctypes.POINTER(ctypes.c_float)
    _LIBVL.vl_imconvcol_vf.argtypes = [
        _PTR_FLOAT,
        ctypes.c_size_t,
        _PTR_FLOAT,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _PTR_FLOAT,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    _LIBVL.vl_imconvcol_vf.restype = None
    _LIBVL.vl_set_simd_enabled.argtypes = [ctypes.c_int]
    _LIBVL.vl_set_simd_enabled.restype = None
    _LIBVL.vl_get_simd_enabled.argtypes = []
    _LIBVL.vl_get_simd_enabled.restype = ctypes.c_int
    _VLFEAT_IMCONV_AVAILABLE = True
else:
    _VLFEAT_IMCONV_AVAILABLE = False


def set_simd_enabled(enabled: bool) -> None:
    if _LIBVL is None:
        return
    _LIBVL.vl_set_simd_enabled(ctypes.c_int(1 if enabled else 0))


def get_simd_enabled() -> bool:
    if _LIBVL is None:
        return False
    return bool(_LIBVL.vl_get_simd_enabled())


if _LIBVL is not None and os.environ.get("DVLAD_DISABLE_SIMD") == "1":
    set_simd_enabled(False)


def _rgb_to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    # Match MATLAB rgb2gray for uint8 input: weighted sum + round half up.
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    # MATLAB uses double-precision coefficients internally.
    gray = 0.298936021293776 * r + 0.587043074451121 * g + 0.114020904255103 * b
    gray = np.floor(gray + 0.5).astype(np.uint8)
    return gray


def _vl_imdown_sample(gray: np.ndarray) -> np.ndarray:
    # Mirrors VLFeat vl_imdown(I, 'method', 'sample') with 1-based indexing:
    # I(1:2:floor(end-.5), 1:2:floor(end-.5))
    h, w = gray.shape[:2]
    return gray[: max(h - 1, 0) : 2, : max(w - 1, 0) : 2]


def _resize_max_dim_gray(gray: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = gray.shape[:2]
    max_hw = max(h, w)
    if max_dim <= 0 or max_hw <= max_dim:
        return gray
    scale = float(max_dim) / float(max_hw)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    im = Image.fromarray(gray)
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=gray.dtype)


def read_gray_im2single(
    path: str | Path,
    *,
    max_dim: int | None = None,
    apply_imdown: bool = True,
) -> np.ndarray:
    """
    Loads an image, converts to grayscale (uint8), applies vl_imdown (sample),
    and converts to float32 in [0, 1], matching Torii15's preprocessing:

        img = imread(); rgb2gray(); vl_imdown(); im2single()
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
    gray = _rgb_to_gray_uint8(rgb)
    if max_dim is not None:
        gray = _resize_max_dim_gray(gray, max_dim)
    if apply_imdown:
        gray = _vl_imdown_sample(gray)
    return gray.astype(np.float32) / np.float32(255.0)


def read_gray_uint8(
    path: str | Path,
    *,
    max_dim: int | None = None,
    apply_imdown: bool = True,
) -> np.ndarray:
    """
    Loads an image and returns the grayscale uint8 after vl_imdown.
    Useful for debugging integer-space operations.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
    gray = _rgb_to_gray_uint8(rgb)
    if max_dim is not None:
        gray = _resize_max_dim_gray(gray, max_dim)
    if apply_imdown:
        gray = _vl_imdown_sample(gray)
    return gray


def image_shape_after_imdown(path: str | Path) -> Tuple[int, int]:
    g = read_gray_uint8(path)
    return int(g.shape[0]), int(g.shape[1])


def _vl_imsmooth_gaussian_py(im: np.ndarray, sigma: float) -> np.ndarray:
    im = np.asarray(im, dtype=np.float32)
    if sigma <= 0:
        return im.copy()
    sigma_f = np.float32(sigma)
    w = int(np.ceil(float(4.0 * sigma_f)))
    if w < 1:
        return im.copy()
    z = np.arange(-w, w + 1, dtype=np.float32)
    denom = sigma_f + VL_EPSILON_F
    filt = np.exp(-0.5 * (z / denom) ** 2).astype(np.float32)
    filt /= np.sum(filt, dtype=np.float32)
    radius = int((len(filt) - 1) // 2)

    def imconvcol(src: np.ndarray) -> np.ndarray:
        h, w_ = src.shape
        out = np.empty((h, w_), dtype=np.float32)
        for x in range(w_):
            col0 = src[0, x]
            coln = src[h - 1, x]
            for y in range(h):
                acc = np.float32(0.0)
                for k in range(-radius, radius + 1):
                    p = y - k
                    if p < 0:
                        v = col0
                    elif p >= h:
                        v = coln
                    else:
                        v = src[p, x]
                    acc += v * filt[k + radius]
                out[y, x] = acc
        return out

    if im.ndim == 2:
        tmp = imconvcol(im)
        tmp = imconvcol(tmp.T).T
        return tmp

    if im.ndim == 3:
        channels = []
        for c in range(im.shape[2]):
            tmp = imconvcol(im[:, :, c])
            tmp = imconvcol(tmp.T).T
            channels.append(tmp)
        return np.stack(channels, axis=2)

    raise ValueError("Expected a 2D or 3D image array.")


def _vl_imsmooth_gaussian_libvl(im: np.ndarray, sigma: float) -> np.ndarray:
    im = np.asarray(im, dtype=np.float32)
    if sigma <= 0:
        return im.copy()
    sigma_d = float(sigma)
    radius = int(math.ceil(4.0 * sigma_d))
    if radius < 1:
        return im.copy()
    size = 2 * radius + 1
    filt = np.empty(size, dtype=np.float32)
    acc = np.float32(0.0)
    denom = sigma_d + float(VL_EPSILON_F)
    for j in range(size):
        z = np.float32((float(j) - radius) / denom)
        val = math.exp(-0.5 * float(z * z))
        filt[j] = np.float32(val)
        acc = np.float32(acc + filt[j])
    for j in range(size):
        filt[j] = np.float32(filt[j] / acc)

    if im.ndim == 3:
        channels = []
        for c in range(im.shape[2]):
            channels.append(_vl_imsmooth_gaussian_libvl(im[:, :, c], sigma))
        return np.stack(channels, axis=2)
    if im.ndim != 2:
        raise ValueError("Expected a 2D or 3D image array.")

    # Match MATLAB's column-major path by smoothing on the transposed image.
    im_t = np.ascontiguousarray(im.T, dtype=np.float32)
    h, w_ = im_t.shape
    buffer = np.empty((w_, h), dtype=np.float32)
    out = np.empty((h, w_), dtype=np.float32)

    flags = _VL_PAD_BY_CONTINUITY | _VL_TRANSPOSE
    _LIBVL.vl_imconvcol_vf(
        buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(h),
        im_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(w_),
        ctypes.c_size_t(h),
        ctypes.c_size_t(w_),
        filt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_long(-radius),
        ctypes.c_long(radius),
        ctypes.c_int(1),
        ctypes.c_uint(flags),
    )
    _LIBVL.vl_imconvcol_vf(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(w_),
        buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(h),
        ctypes.c_size_t(w_),
        ctypes.c_size_t(h),
        filt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_long(-radius),
        ctypes.c_long(radius),
        ctypes.c_int(1),
        ctypes.c_uint(flags),
    )
    return out.T


def vl_imsmooth_gaussian(im: np.ndarray, sigma: float) -> np.ndarray:
    """
    VLFeat-compatible Gaussian smoothing (vl_imsmooth) with continuity padding.
    """
    if _VLFEAT_IMCONV_AVAILABLE:
        return _vl_imsmooth_gaussian_libvl(im, sigma)
    return _vl_imsmooth_gaussian_py(im, sigma)
