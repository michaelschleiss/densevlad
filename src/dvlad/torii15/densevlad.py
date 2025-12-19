from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .image import read_gray_im2single
from .matio import load_mat_v5


@dataclass(frozen=True)
class Torii15Vocab:
    centers: np.ndarray


def _l2normalize_cols(x: np.ndarray) -> np.ndarray:
    denom = np.sqrt(np.sum(x * x, axis=0, keepdims=True))
    denom[denom == 0] = 1.0
    return x / denom


def load_torii15_vocab(vocab_mat_path: str | Path) -> Torii15Vocab:
    mat = load_mat_v5(vocab_mat_path)
    if "CX" not in mat:
        raise KeyError(f"Expected variable 'CX' in {vocab_mat_path}")
    cx = np.asarray(mat["CX"], dtype=np.float32)
    if cx.shape[0] != 128 and cx.shape[1] == 128:
        cx = cx.T
    cx = _l2normalize_cols(cx)
    return Torii15Vocab(centers=cx)


def _rootsift(descs: np.ndarray) -> np.ndarray:
    x = np.asarray(descs, dtype=np.float32)
    denom = np.sum(np.abs(x), axis=1, keepdims=True) + 1e-12
    x = x / denom
    return np.sqrt(x)


def _hard_assignments(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Exact nearest neighbors by brute-force L2 distance (deterministic).
    # descs: (N, 128), centers: (K, 128)
    # Compute squared distances efficiently: ||x||^2 + ||c||^2 - 2 x c^T
    x2 = np.sum(descs * descs, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (descs @ centers.T)
    nn = np.argmin(d2, axis=1)
    assigns = np.zeros((descs.shape[0], centers.shape[0]), dtype=np.float32)
    assigns[np.arange(descs.shape[0]), nn] = 1.0
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
        from cyvlfeat.sift import phow
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cyvlfeat is required for PHOW descriptors; install VLFeat + cyvlfeat."
        ) from e

    frames, descs = phow(
        im,
        fast=True,
        sizes=(4, 6, 8, 10),
        step=2,
        color="gray",
        float_descriptors=False,
        magnification=6,
        window_size=1.5,
        contrast_threshold=0.005,
    )
    return descs


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
    assignments = reuse_assignments or _hard_assignments(descs, vocab.centers)
    vlad = _vl_vlad(descs, vocab.centers, assignments)
    return np.asarray(vlad, dtype=np.float32).reshape(-1)

