from __future__ import annotations

from dataclasses import dataclass
import importlib.resources as importlib_resources
from pathlib import Path
from typing import Optional
import ctypes

import numpy as np

from .image import read_gray_im2single, vl_imsmooth_gaussian
from . import image as _image
from .matio import load_mat_v5


@dataclass(frozen=True)
class Torii15Vocab:
    centers: np.ndarray


def _l2normalize_cols(x: np.ndarray) -> np.ndarray:
    denom = np.sqrt(np.sum(x * x, axis=0, keepdims=True))
    denom[denom == 0] = 1.0
    return x / denom


def _load_packaged_centers() -> np.ndarray | None:
    try:
        data_path = importlib_resources.files("dvlad.torii15").joinpath(
            "data/dnscnt_RDSIFT_K128.cx_norm.npy"
        )
    except Exception:
        return None
    if not data_path.is_file():
        return None
    with importlib_resources.as_file(data_path) as path:
        centers = np.load(path)
    if centers.ndim != 2:
        return None
    if centers.shape[0] != 128 and centers.shape[1] == 128:
        centers = centers.T
    if centers.shape[1] != 128:
        return None
    return np.asarray(centers, dtype=np.float32)


def _vocab_norm_cache_path(vocab_mat_path: Path) -> Path:
    return vocab_mat_path.with_suffix(".cx_norm.npy")


def _load_centers_from_matlab_dump(dump_path: Path) -> np.ndarray | None:
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        return None
    with h5py.File(dump_path, "r") as mat:
        if "CX" not in mat:
            return None
        cx = np.array(mat["CX"], dtype=np.float32)
    if cx.ndim != 2:
        return None
    # MATLAB stores centers as D x K; HDF5 loads transposed, so this is K x D.
    if cx.shape[0] != 128 and cx.shape[1] == 128:
        cx = cx.T
    if cx.shape[1] != 128:
        return None
    return cx


def _dsift_transpose_perm(num_bin_x: int, num_bin_y: int, num_bin_t: int) -> np.ndarray:
    perm = np.empty(num_bin_x * num_bin_y * num_bin_t, dtype=np.int32)
    for y in range(num_bin_y):
        for x in range(num_bin_x):
            offset = num_bin_t * (x + y * num_bin_x)
            offset_t = num_bin_t * (y + x * num_bin_y)
            for t in range(num_bin_t):
                t_t = num_bin_t // 4 - t
                tt = (t_t + num_bin_t) % num_bin_t
                perm[offset_t + tt] = offset + t
    return perm


_DSIFT_GEOM = (4, 4, 8)
_DSIFT_TRANSPOSE_PERM = _dsift_transpose_perm(_DSIFT_GEOM[1], _DSIFT_GEOM[0], _DSIFT_GEOM[2])


def load_torii15_vocab(vocab_mat_path: str | Path) -> Torii15Vocab:
    vocab_mat_path = Path(vocab_mat_path)
    packaged = _load_packaged_centers()
    if packaged is not None:
        return Torii15Vocab(centers=packaged)
    norm_cache = _vocab_norm_cache_path(vocab_mat_path)
    if norm_cache.exists():
        centers = np.load(norm_cache)
        return Torii15Vocab(centers=np.asarray(centers, dtype=np.float32))

    # If MATLAB dump exists, prefer its CX to match VLFeat/kd-tree bit-for-bit.
    try:
        from .assets import Torii15Assets

        dump_path = (
            Torii15Assets.default_cache_dir()
            / "matlab_dump"
            / "densevlad_dump.mat"
        )
    except Exception:
        dump_path = None
    if dump_path and dump_path.exists():
        centers = _load_centers_from_matlab_dump(dump_path)
        if centers is not None:
            norm_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(norm_cache, centers)
            return Torii15Vocab(centers=centers)

    mat = load_mat_v5(vocab_mat_path)
    if "CX" not in mat:
        raise KeyError(f"Expected variable 'CX' in {vocab_mat_path}")
    cx = np.asarray(mat["CX"], dtype=np.float32)
    if cx.shape[0] != 128 and cx.shape[1] == 128:
        cx = cx.T
    if cx.shape[0] != 128:
        raise ValueError(f"Expected 128-D centers, got shape {cx.shape}")
    # MATLAB stores centers as columns (D x K). Normalize columns then transpose to K x D.
    cx = _l2normalize_cols(cx).T
    return Torii15Vocab(centers=cx)


def _rootsift(descs: np.ndarray) -> np.ndarray:
    x = np.asarray(descs, dtype=np.float32)
    denom = np.sum(np.abs(x), axis=1, keepdims=True) + np.float32(1e-12)
    x = x / denom
    return np.sqrt(x, dtype=np.float32)


def _hard_assignments(descs: np.ndarray, centers: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    # Exact nearest neighbors by brute-force L2 distance (deterministic).
    # Use direct squared-distance accumulation to better match VLFeat's numeric path.
    n, d = descs.shape
    k = centers.shape[0]
    assigns = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = descs[start:end]
        diff = block[:, None, :] - centers[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        nn = np.argmin(d2, axis=1)
        assigns[np.arange(start, end), nn] = 1.0
    return assigns


_VL_TYPE_FLOAT = 1
_VL_DIST_L2 = 1
_VL_KDTREE_MEDIAN = 0

if _image._LIBVL is not None:
    _image._LIBVL.vl_kdforest_new.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _image._LIBVL.vl_kdforest_new.restype = ctypes.c_void_p
    _image._LIBVL.vl_kdforest_set_thresholding_method.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _image._LIBVL.vl_kdforest_set_thresholding_method.restype = None
    _image._LIBVL.vl_kdforest_build.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    _image._LIBVL.vl_kdforest_build.restype = None
    _image._LIBVL.vl_kdforest_set_max_num_comparisons.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    _image._LIBVL.vl_kdforest_set_max_num_comparisons.restype = None
    _image._LIBVL.vl_kdforest_query_with_array.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _image._LIBVL.vl_kdforest_query_with_array.restype = ctypes.c_size_t
    _image._LIBVL.vl_kdforest_delete.argtypes = [ctypes.c_void_p]
    _image._LIBVL.vl_kdforest_delete.restype = None


def _kdtree_assignments(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if _image._LIBVL is None:
        return _hard_assignments(descs, centers)

    descs_f = np.ascontiguousarray(descs, dtype=np.float32)
    centers_f = np.ascontiguousarray(centers, dtype=np.float32)
    num_queries = descs_f.shape[0]
    num_centers = centers_f.shape[0]

    forest = _image._LIBVL.vl_kdforest_new(
        _VL_TYPE_FLOAT,
        centers_f.shape[1],
        1,
        _VL_DIST_L2,
    )
    _image._LIBVL.vl_kdforest_set_thresholding_method(forest, _VL_KDTREE_MEDIAN)
    _image._LIBVL.vl_kdforest_build(
        forest,
        num_centers,
        centers_f.ctypes.data_as(ctypes.c_void_p),
    )
    _image._LIBVL.vl_kdforest_set_max_num_comparisons(forest, 0)

    indexes = np.empty((num_queries, 1), dtype=np.uint32)
    _image._LIBVL.vl_kdforest_query_with_array(
        forest,
        indexes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        1,
        num_queries,
        None,
        descs_f.ctypes.data_as(ctypes.c_void_p),
    )
    _image._LIBVL.vl_kdforest_delete(forest)

    assigns = np.zeros((num_queries, num_centers), dtype=np.float32)
    idx = indexes.reshape(-1).astype(np.int64)
    assigns[np.arange(num_queries), idx] = 1.0
    return assigns


def _vl_vlad(
    descs: np.ndarray, centers: np.ndarray, assignments: np.ndarray
) -> np.ndarray:
    try:
        from cyvlfeat.vlad import vlad as cy_vlad
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cyvlfeat is required for VLAD encoding; install VLFeat + cyvlfeat."
        ) from e

    return cy_vlad(
        descs.astype(np.float32),
        centers.astype(np.float32),
        assignments.astype(np.float32),
        normalize_components=True,
    )


def _phow_descs(im: np.ndarray) -> np.ndarray:
    try:
        from cyvlfeat.sift import dsift
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cyvlfeat is required for PHOW descriptors; install VLFeat + cyvlfeat."
        ) from e

    sizes = (4, 6, 8, 10)
    step = 2
    magnification = 6
    window_size = 1.5
    contrast_threshold = 0.005
    max_size = max(sizes)
    descs_all = []

    for size in sizes:
        off = int(np.floor(1.0 + 1.5 * (max_size - size)))
        off0 = max(off - 1, 0)
        ims = vl_imsmooth_gaussian(im, sigma=size / magnification)
        ims_t = ims.T
        frames, descs = dsift(
            ims_t,
            step=step,
            size=size,
            bounds=np.array(
                [off0, off0, ims_t.shape[0] - 1, ims_t.shape[1] - 1], dtype=np.int32
            ),
            window_size=window_size,
            norm=True,
            fast=True,
            float_descriptors=False,
        )
        descs = descs[:, _DSIFT_TRANSPOSE_PERM]
        contrast = frames[:, 2]
        descs[contrast < contrast_threshold] = 0
        descs_all.append(descs)

    return np.vstack(descs_all)


def compute_densevlad_pre_pca(
    image_path: str | Path,
    vocab: Torii15Vocab,
    *,
    reuse_assignments: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Replicates Torii15 `at_image2densevlad.m` to produce the pre-PCA VLAD vector:
      - load image, rgb2gray, vl_imdown, im2single
      - vl_phow (sizes 4/6/8/10, step 2)
      - RootSIFT (L1 + sqrt)
      - hard assignment to vocab centers
      - VLAD with intra-normalization (NormalizeComponents)
    """
    im = read_gray_im2single(image_path)
    descs = _phow_descs(im)
    descs = _rootsift(descs)
    assignments = reuse_assignments or _kdtree_assignments(descs, vocab.centers)
    vlad = _vl_vlad(descs, vocab.centers, assignments)
    return np.asarray(vlad, dtype=np.float32).reshape(-1)
