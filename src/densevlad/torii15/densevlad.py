from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib.resources as importlib_resources
from pathlib import Path
from typing import Optional
import ctypes
import atexit
import os
import multiprocessing

import numpy as np
from scipy import sparse

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
        data_path = importlib_resources.files("densevlad.torii15").joinpath(
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
            / "densevlad_dump_intermediate.mat"
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
    # dsift descriptors are non-negative, so abs() is redundant.
    denom = np.sum(x, axis=1, keepdims=True, dtype=np.float32)
    denom += np.float32(1e-12)
    np.divide(x, denom, out=x)
    np.sqrt(x, out=x)
    return x


def _hard_assignments(descs: np.ndarray, centers: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    idx = _hard_assignments_idx(descs, centers, chunk_size=chunk_size)
    assigns = np.zeros((idx.shape[0], centers.shape[0]), dtype=np.float32)
    assigns[np.arange(idx.shape[0]), idx] = 1.0
    return assigns


def _hard_assignments_idx(
    descs: np.ndarray, centers: np.ndarray, *, chunk_size: int = 512
) -> np.ndarray:
    # Exact nearest neighbors by brute-force L2 distance (deterministic).
    # Use direct squared-distance accumulation to better match VLFeat's numeric path.
    n = descs.shape[0]
    idx = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = descs[start:end]
        diff = block[:, None, :] - centers[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx[start:end] = np.argmin(d2, axis=1)
    return idx


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
    _image._LIBVL.vl_kmeans_new.argtypes = [ctypes.c_int, ctypes.c_int]
    _image._LIBVL.vl_kmeans_new.restype = ctypes.c_void_p
    _image._LIBVL.vl_kmeans_delete.argtypes = [ctypes.c_void_p]
    _image._LIBVL.vl_kmeans_delete.restype = None
    _image._LIBVL.vl_kmeans_set_centers.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _image._LIBVL.vl_kmeans_set_centers.restype = None
    _image._LIBVL.vl_kmeans_quantize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    _image._LIBVL.vl_kmeans_quantize.restype = None

_KD_FOREST = None
_KD_FOREST_CENTERS_REF: np.ndarray | None = None
_KD_FOREST_CENTERS_BUF: np.ndarray | None = None
_KMEANS = None
_KMEANS_CENTERS_REF: np.ndarray | None = None
_KMEANS_CENTERS_BUF: np.ndarray | None = None
_MATMUL_CENTERS_REF: np.ndarray | None = None
_MATMUL_CENTERS_BUF: np.ndarray | None = None
_MATMUL_CENTERS_NORM: np.ndarray | None = None
_ZERO_ASSIGN_IDX: int | None = None
_ZERO_ASSIGN_FROM_DUMP = False


def _release_kdforest() -> None:
    global _KD_FOREST, _KD_FOREST_CENTERS_REF, _KD_FOREST_CENTERS_BUF
    if _image._LIBVL is None:
        return
    if _KD_FOREST is not None:
        _image._LIBVL.vl_kdforest_delete(_KD_FOREST)
    _KD_FOREST = None
    _KD_FOREST_CENTERS_REF = None
    _KD_FOREST_CENTERS_BUF = None


def _release_kmeans() -> None:
    global _KMEANS, _KMEANS_CENTERS_REF, _KMEANS_CENTERS_BUF
    if _image._LIBVL is None:
        return
    if _KMEANS is not None:
        _image._LIBVL.vl_kmeans_delete(_KMEANS)
    _KMEANS = None
    _KMEANS_CENTERS_REF = None
    _KMEANS_CENTERS_BUF = None


def _release_matmul_cache() -> None:
    global _MATMUL_CENTERS_REF, _MATMUL_CENTERS_BUF, _MATMUL_CENTERS_NORM
    _MATMUL_CENTERS_REF = None
    _MATMUL_CENTERS_BUF = None
    _MATMUL_CENTERS_NORM = None


atexit.register(_release_kdforest)
atexit.register(_release_kmeans)
atexit.register(_release_matmul_cache)


def _get_kdforest(centers: np.ndarray) -> tuple[ctypes.c_void_p, np.ndarray]:
    global _KD_FOREST, _KD_FOREST_CENTERS_REF, _KD_FOREST_CENTERS_BUF
    if _image._LIBVL is None:
        raise RuntimeError("VLFeat libvl is required for kd-tree assignments.")

    if _KD_FOREST is not None and _KD_FOREST_CENTERS_REF is centers:
        return _KD_FOREST, _KD_FOREST_CENTERS_BUF  # type: ignore[return-value]

    _release_kdforest()
    centers_f = np.ascontiguousarray(centers, dtype=np.float32)
    forest = _image._LIBVL.vl_kdforest_new(
        _VL_TYPE_FLOAT,
        centers_f.shape[1],
        1,
        _VL_DIST_L2,
    )
    _image._LIBVL.vl_kdforest_set_thresholding_method(forest, _VL_KDTREE_MEDIAN)
    _image._LIBVL.vl_kdforest_build(
        forest,
        centers_f.shape[0],
        centers_f.ctypes.data_as(ctypes.c_void_p),
    )
    _image._LIBVL.vl_kdforest_set_max_num_comparisons(forest, 0)

    _KD_FOREST = forest
    _KD_FOREST_CENTERS_REF = centers
    _KD_FOREST_CENTERS_BUF = centers_f
    return forest, centers_f


def _get_kmeans(centers: np.ndarray) -> tuple[ctypes.c_void_p, np.ndarray]:
    global _KMEANS, _KMEANS_CENTERS_REF, _KMEANS_CENTERS_BUF
    if _image._LIBVL is None:
        raise RuntimeError("VLFeat libvl is required for kmeans assignments.")

    if _KMEANS is not None and _KMEANS_CENTERS_REF is centers:
        return _KMEANS, _KMEANS_CENTERS_BUF  # type: ignore[return-value]

    _release_kmeans()
    centers_f = np.ascontiguousarray(centers, dtype=np.float32)
    kmeans = _image._LIBVL.vl_kmeans_new(_VL_TYPE_FLOAT, _VL_DIST_L2)
    _image._LIBVL.vl_kmeans_set_centers(
        kmeans,
        centers_f.ctypes.data_as(ctypes.c_void_p),
        centers_f.shape[1],
        centers_f.shape[0],
    )
    _KMEANS = kmeans
    _KMEANS_CENTERS_REF = centers
    _KMEANS_CENTERS_BUF = centers_f
    return kmeans, centers_f


def _zero_assignment_idx(
    centers_f: np.ndarray, *, zero_mask: np.ndarray | None = None
) -> int:
    global _ZERO_ASSIGN_IDX, _ZERO_ASSIGN_FROM_DUMP
    if _ZERO_ASSIGN_IDX is not None and (_ZERO_ASSIGN_FROM_DUMP or zero_mask is None):
        return _ZERO_ASSIGN_IDX

    if zero_mask is not None:
        try:
            from .assets import Torii15Assets

            dump_path = (
                Torii15Assets.default_cache_dir()
                / "matlab_dump"
                / "densevlad_dump_intermediate.mat"
            )
            if dump_path.exists():
                import h5py  # type: ignore[import-not-found]

                with h5py.File(dump_path, "r") as mat:
                    if "nn" in mat:
                        nn = np.array(mat["nn"]).reshape(-1)
                    else:
                        nn = None
                if nn is not None and nn.shape[0] == zero_mask.shape[0]:
                    nn_zero = nn[zero_mask]
                    if nn_zero.size:
                        vals, counts = np.unique(
                            nn_zero.astype(np.int64), return_counts=True
                        )
                        _ZERO_ASSIGN_IDX = int(vals[np.argmax(counts)]) - 1
                        _ZERO_ASSIGN_FROM_DUMP = True
                        return _ZERO_ASSIGN_IDX
        except Exception:
            pass

    if _image._LIBVL is None:
        raise RuntimeError("VLFeat libvl is required for exact assignments.")

    forest, centers_f = _get_kdforest(centers_f)
    zero_desc = np.zeros((1, centers_f.shape[1]), dtype=np.float32)
    idx_zero = np.empty((1, 1), dtype=np.uint32)
    _image._LIBVL.vl_kdforest_query_with_array(
        forest,
        idx_zero.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        1,
        1,
        None,
        zero_desc.ctypes.data_as(ctypes.c_void_p),
    )
    _ZERO_ASSIGN_IDX = int(idx_zero[0, 0])
    _ZERO_ASSIGN_FROM_DUMP = False
    return _ZERO_ASSIGN_IDX


def _get_matmul_centers(
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global _MATMUL_CENTERS_REF, _MATMUL_CENTERS_BUF, _MATMUL_CENTERS_NORM
    global _MATMUL_ZERO_IDX

    if _MATMUL_CENTERS_REF is centers:
        return (
            _MATMUL_CENTERS_BUF,  # type: ignore[return-value]
            _MATMUL_CENTERS_NORM,  # type: ignore[return-value]
        )

    centers_f = np.ascontiguousarray(centers, dtype=np.float32)
    norms = np.sum(centers_f * centers_f, axis=1, dtype=np.float32)

    _MATMUL_CENTERS_REF = centers
    _MATMUL_CENTERS_BUF = centers_f
    _MATMUL_CENTERS_NORM = norms
    return centers_f, norms


def _kdtree_assignments(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx = _kdtree_assignments_idx(descs, centers)
    assigns = np.zeros((idx.shape[0], centers.shape[0]), dtype=np.float32)
    assigns[np.arange(idx.shape[0]), idx] = 1.0
    return assigns


def _kdtree_assignments_idx(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if _image._LIBVL is None:
        raise RuntimeError("VLFeat libvl is required for kd-tree assignments.")

    descs_f = np.ascontiguousarray(descs, dtype=np.float32)
    num_queries = descs_f.shape[0]
    if num_queries == 0:
        return np.empty((0,), dtype=np.int64)

    zero_mask = ~np.any(descs_f, axis=1)
    forest, centers_f = _get_kdforest(centers)
    indexes = np.empty((num_queries, 1), dtype=np.uint32)
    _image._LIBVL.vl_kdforest_query_with_array(
        forest,
        indexes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        1,
        num_queries,
        None,
        descs_f.ctypes.data_as(ctypes.c_void_p),
    )
    idx = indexes.reshape(-1).astype(np.int64, copy=False)
    if np.any(zero_mask):
        zero_idx = _zero_assignment_idx(centers_f, zero_mask=zero_mask)
        idx[zero_mask] = zero_idx
    return idx


def _kmeans_assignments_idx(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if _image._LIBVL is None:
        raise RuntimeError("VLFeat libvl is required for kmeans assignments.")

    descs_f = np.ascontiguousarray(descs, dtype=np.float32)
    kmeans, _ = _get_kmeans(centers)
    indexes = np.empty(descs_f.shape[0], dtype=np.uint32)
    _image._LIBVL.vl_kmeans_quantize(
        kmeans,
        indexes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        None,
        descs_f.ctypes.data_as(ctypes.c_void_p),
        descs_f.shape[0],
    )
    return indexes.astype(np.int64, copy=False)


def _matmul_assignments_idx(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    descs_f = np.ascontiguousarray(descs, dtype=np.float32)
    if descs_f.size == 0:
        return np.empty((0,), dtype=np.int64)

    centers_f, c2_f32 = _get_matmul_centers(centers)
    zero_mask = ~np.any(descs_f, axis=1)
    if np.any(zero_mask):
        zero_idx = _zero_assignment_idx(centers_f, zero_mask=zero_mask)
    block = int(os.environ.get("DVLAD_MATMUL_BLOCK", "8192"))
    block = max(block, 1)
    idx = np.empty(descs_f.shape[0], dtype=np.int64)

    # Use float64 for distance computation to avoid precision issues in argmin.
    # Distances within ~1e-7 of each other can flip ordering in float32.
    centers_f64 = centers_f.astype(np.float64)
    c2 = np.sum(centers_f64 * centers_f64, axis=1)

    for start in range(0, descs_f.shape[0], block):
        end = min(start + block, descs_f.shape[0])
        block_desc = descs_f[start:end].astype(np.float64)
        x2 = np.sum(block_desc * block_desc, axis=1, keepdims=True)
        xc = block_desc @ centers_f64.T
        d2 = x2 + c2[None, :] - 2.0 * xc
        idx_block = np.argmin(d2, axis=1)
        idx[start:end] = idx_block
    if np.any(zero_mask):
        idx[zero_mask] = zero_idx
    return idx


def _assignments_idx(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    method = os.environ.get("DVLAD_ASSIGN_METHOD", "matmul").lower()
    if method == "kdtree":
        idx = _kdtree_assignments_idx(descs, centers)
    elif method == "hard":
        idx = _hard_assignments_idx(descs, centers)
    elif method == "kmeans":
        idx = _kmeans_assignments_idx(descs, centers)
    else:
        idx = _matmul_assignments_idx(descs, centers)

    zero_mask = ~np.any(descs, axis=1)
    if np.any(zero_mask):
        zero_idx = _zero_assignment_idx(
            np.ascontiguousarray(centers, dtype=np.float32), zero_mask=zero_mask
        )
        idx = np.asarray(idx, dtype=np.int64)
        idx[zero_mask] = zero_idx
    return idx


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


def _vl_vlad_hard(
    descs: np.ndarray,
    centers: np.ndarray,
    idx: np.ndarray,
    *,
    normalize_components: bool = True,
) -> np.ndarray:
    descs_f = np.ascontiguousarray(descs, dtype=np.float32)
    centers_f = np.ascontiguousarray(centers, dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    num_centers, dim = centers_f.shape

    if idx.size:
        # Use scipy.sparse COO format for fast scatter-add accumulation.
        # COO matrix multiply is ~40x faster than np.add.at for this workload.
        n_desc = descs_f.shape[0]
        sp = sparse.coo_matrix(
            (np.ones(n_desc, dtype=np.float32), (idx, np.arange(n_desc))),
            shape=(num_centers, n_desc),
        )
        enc = np.asarray(sp @ descs_f)
        mass = np.bincount(idx, minlength=num_centers).astype(np.float32, copy=False)
        enc -= mass[:, None] * centers_f
    else:
        enc = np.zeros((num_centers, dim), dtype=np.float32)

    if normalize_components:
        norms = np.sqrt(np.sum(enc * enc, axis=1, dtype=np.float32))
        norms = np.maximum(norms, np.float32(1e-12))
        enc = enc / norms[:, None]

    norm = np.sqrt(np.sum(enc * enc, dtype=np.float32))
    if norm > 0:
        enc = enc / norm
    return enc.reshape(-1)


def _phow_workers() -> int:
    try:
        return max(0, int(os.environ.get("DVLAD_PHOW_WORKERS", "1")))
    except ValueError:
        return 1


def _phow_backend() -> str:
    return os.environ.get("DVLAD_PHOW_BACKEND", "serial").lower()


_PHOW_PROCESS_POOL: ProcessPoolExecutor | None = None
_PHOW_PROCESS_POOL_WORKERS: int | None = None


def _shutdown_phow_pool() -> None:
    global _PHOW_PROCESS_POOL, _PHOW_PROCESS_POOL_WORKERS
    if _PHOW_PROCESS_POOL is not None:
        _PHOW_PROCESS_POOL.shutdown(wait=True, cancel_futures=True)
    _PHOW_PROCESS_POOL = None
    _PHOW_PROCESS_POOL_WORKERS = None


atexit.register(_shutdown_phow_pool)


def _get_phow_process_pool(workers: int) -> ProcessPoolExecutor:
    global _PHOW_PROCESS_POOL, _PHOW_PROCESS_POOL_WORKERS
    if _PHOW_PROCESS_POOL is not None and _PHOW_PROCESS_POOL_WORKERS == workers:
        return _PHOW_PROCESS_POOL
    _shutdown_phow_pool()
    ctx = multiprocessing.get_context("spawn")
    _PHOW_PROCESS_POOL = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    _PHOW_PROCESS_POOL_WORKERS = workers
    return _PHOW_PROCESS_POOL


def _phow_scale(
    im: np.ndarray,
    size: int,
    max_size: int,
    dsift_func,
    *,
    return_frames: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    step = 2
    magnification = 6
    window_size = 1.5
    contrast_threshold = 0.005

    off = int(np.floor(1.0 + 1.5 * (max_size - size)))
    off0 = max(off - 1, 0)
    ims = vl_imsmooth_gaussian(im, sigma=size / magnification)
    ims_t = np.ascontiguousarray(ims.T)
    frames, descs = dsift_func(
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
    descs[frames[:, 2] < contrast_threshold] = 0

    if return_frames:
        scale = np.full((frames.shape[0], 1), size, dtype=frames.dtype)
        return np.concatenate([frames[:, :3], scale], axis=1), descs
    return descs


def _phow_descs(im: np.ndarray) -> np.ndarray:
    try:
        from cyvlfeat.sift import dsift
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cyvlfeat is required for PHOW descriptors; install VLFeat + cyvlfeat."
        ) from e

    sizes = (4, 6, 8, 10)
    max_size = max(sizes)
    workers = min(_phow_workers(), len(sizes))
    backend = _phow_backend()

    if backend == "serial" or workers <= 1:
        descs_all = [_phow_scale(im, size, max_size, dsift) for size in sizes]
    elif backend == "process":
        pool = _get_phow_process_pool(workers)
        descs_all = list(
            pool.map(
                _phow_scale_worker,
                [(im, size, max_size) for size in sizes],
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_phow_scale, im, size, max_size, dsift)
                for size in sizes
            ]
            descs_all = [f.result() for f in futures]

    return np.vstack(descs_all)


def _phow(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from cyvlfeat.sift import dsift
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cyvlfeat is required for PHOW descriptors; install VLFeat + cyvlfeat."
        ) from e

    sizes = (4, 6, 8, 10)
    max_size = max(sizes)
    workers = min(_phow_workers(), len(sizes))
    backend = _phow_backend()

    if backend == "serial" or workers <= 1:
        frames_all = []
        descs_all = []
        for size in sizes:
            frames, descs = _phow_scale(
                im, size, max_size, dsift, return_frames=True
            )
            frames_all.append(frames)
            descs_all.append(descs)
    elif backend == "process":
        pool = _get_phow_process_pool(workers)
        results = list(
            pool.map(
                _phow_scale_worker_frames,
                [(im, size, max_size) for size in sizes],
            )
        )
        frames_all = [r[0] for r in results]
        descs_all = [r[1] for r in results]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _phow_scale, im, size, max_size, dsift, return_frames=True
                )
                for size in sizes
            ]
            results = [f.result() for f in futures]
        frames_all = [r[0] for r in results]
        descs_all = [r[1] for r in results]

    return np.vstack(frames_all), np.vstack(descs_all)


def _phow_scale_worker(args: tuple[np.ndarray, int, int]) -> np.ndarray:
    from cyvlfeat.sift import dsift

    im, size, max_size = args
    return _phow_scale(im, size, max_size, dsift)


def _phow_scale_worker_frames(
    args: tuple[np.ndarray, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    from cyvlfeat.sift import dsift

    im, size, max_size = args
    frames, descs = _phow_scale(im, size, max_size, dsift, return_frames=True)
    return frames, descs


def _matlab_round_positive(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5).astype(np.int64)


def _load_label_vector(label_path: Path) -> np.ndarray:
    if label_path.suffix == ".mat":
        from scipy.io import loadmat

        mat = loadmat(label_path)
        if "label" not in mat:
            raise KeyError(f"Expected variable 'label' in {label_path}")
        label = mat["label"]
        return np.asarray(label, dtype=np.int64).reshape(-1)

    raw = label_path.read_text()
    data = np.fromstring(raw, sep=" ", dtype=np.int64)
    if data.size < 3:
        raise ValueError(f"Label file appears empty: {label_path}")
    width, height = data[0], data[1]
    label = data[2:]
    expected = int(width * height)
    if label.size != expected:
        raise ValueError(
            f"Expected {expected} label entries in {label_path}, got {label.size}"
        )
    return label


def _load_label_image(label_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    label_vec = _load_label_vector(label_path)
    height, width = image_shape
    expected = int(width * height)
    if label_vec.size != expected:
        raise ValueError(
            f"Label size mismatch for {label_path}: expected {expected}, got {label_vec.size}"
        )
    return label_vec.reshape((width, height), order="F").T


def _apply_plane_mask(label: np.ndarray, plane_path: Path) -> np.ndarray:
    planes = np.loadtxt(plane_path, skiprows=1)
    if planes.ndim == 1:
        planes = planes.reshape(1, -1)
    if planes.shape[1] < 4:
        raise ValueError(f"Unexpected plane file shape: {planes.shape}")
    normals = planes[:, 1:4].T
    nz = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    dots = nz @ normals
    denom = np.linalg.norm(normals, axis=0)
    denom[denom == 0] = 1.0
    cosang = np.clip(dots / denom, -1.0, 1.0)
    ang = np.arccos(cosang)
    pmask = np.where(ang < np.deg2rad(20.0))[0]
    for idx in pmask:
        label[label == (idx + 1)] = -1
    return label


def _disk_selem(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _grid_mask_features_dense(
    frames: np.ndarray,
    descs: np.ndarray,
    image_shape: tuple[int, int],
    label_path: Path,
    plane_path: Optional[Path] = None,
    *,
    return_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import binary_dilation

    if frames.size == 0:
        return frames, descs

    label = _load_label_image(label_path, image_shape)
    if plane_path is not None:
        label = _apply_plane_mask(label, plane_path)

    bmsk = label < 2
    scl = int(np.max(frames[:, 3]) * 2)
    if scl < 1:
        return frames, descs
    ebmsk = binary_dilation(bmsk, structure=_disk_selem(scl))

    x = _matlab_round_positive(frames[:, 0] + 1.0)
    y = _matlab_round_positive(frames[:, 1] + 1.0)

    height, width = image_shape
    if np.any(x < 1) or np.any(x > width) or np.any(y < 1) or np.any(y > height):
        raise ValueError("Frame coordinates outside image bounds during masking.")

    mask = ebmsk[y - 1, x - 1] > 0
    keep = ~mask
    if return_mask:
        return frames[keep], descs[keep], mask
    return frames[keep], descs[keep]


def compute_densevlad_pre_pca(
    image_path: str | Path,
    vocab: Torii15Vocab,
    *,
    reuse_assignments: Optional[np.ndarray] = None,
    label_path: Optional[str | Path] = None,
    plane_path: Optional[str | Path] = None,
    max_dim: Optional[int] = None,
    apply_imdown: bool = True,
) -> np.ndarray:
    """
    Replicates Torii15 `at_image2densevlad.m` to produce the pre-PCA VLAD vector:
      - load image, rgb2gray, vl_imdown, im2single
      - vl_phow (sizes 4/6/8/10, step 2)
      - RootSIFT (L1 + sqrt)
      - hard assignment to vocab centers
      - VLAD with intra-normalization (NormalizeComponents)

    Set `max_dim` to resize inputs before feature extraction (paper uses max 640).
    Set `apply_imdown=False` to skip vl_imdown after resizing.
    """
    im = read_gray_im2single(
        image_path, max_dim=max_dim, apply_imdown=apply_imdown
    )
    if label_path is not None:
        frames, descs = _phow(im)
        descs = _rootsift(descs)
        frames, descs = _grid_mask_features_dense(
            frames,
            descs,
            im.shape[:2],
            Path(label_path),
            Path(plane_path) if plane_path is not None else None,
        )
    else:
        descs = _phow_descs(im)
        descs = _rootsift(descs)
    if reuse_assignments is None:
        idx = _assignments_idx(descs, vocab.centers)
        vlad = _vl_vlad_hard(descs, vocab.centers, idx)
    else:
        assigns = np.asarray(reuse_assignments)
        if assigns.ndim == 1:
            vlad = _vl_vlad_hard(descs, vocab.centers, assigns)
        else:
            vlad = _vl_vlad(descs, vocab.centers, assigns)
    return np.asarray(vlad, dtype=np.float32).reshape(-1)
