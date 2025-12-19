from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .matio import load_mat73_dataset, load_mat_v5


@dataclass(frozen=True)
class Torii15PCAWhitening:
    proj: np.ndarray
    lambdas: np.ndarray


def load_torii15_pca_whitening(pca_mat_path: str | Path, dim: int = 4096) -> Torii15PCAWhitening:
    """
    Loads Torii15 PCA/whitening parameters and returns the projection matrix in the same
    orientation used by `247code/test_densevlad.m`:

        vlad_proj = vlad_proj(:,1:dim)';   % => shape (dim, 16384)
        vlad_wht  = diag(1./sqrt(vlad_lambda(1:dim)))

    Note: MATLAB v7.3 .mat is HDF5; data is typically stored transposed w.r.t. MATLAB.
    We exploit this to slice without loading the full 16384x16384 matrix into memory.
    """
    # Stored as MATLAB (16384, 16384), but reading via HDF5 yields a transposed view.
    # We want: vlad_proj(:,1:dim)' which corresponds to first `dim` rows of the transposed storage.
    proj = load_mat73_dataset(pca_mat_path, "vlad_proj")[:dim, :]
    lambdas = load_mat73_dataset(pca_mat_path, "vlad_lambda")
    lambdas = np.asarray(lambdas).reshape(-1)
    return Torii15PCAWhitening(proj=np.asarray(proj), lambdas=lambdas[:dim])


def load_reference_pre_pca_vlad(mat_path: str | Path) -> np.ndarray:
    mat = load_mat_v5(mat_path)
    if "vlad" not in mat:
        raise KeyError(f"Expected variable 'vlad' in {mat_path}")
    vlad = np.asarray(mat["vlad"]).reshape(-1)
    return vlad


def apply_pca_whitening(
    vlad: np.ndarray,
    pca: Torii15PCAWhitening,
    *,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Mirrors the Torii15 MATLAB steps:

        z = vlad_proj * vlad
        z = diag(1/sqrt(lambda)) * z
        z = l2_normalize(z)
    """
    v = np.asarray(vlad, dtype=np.float32).reshape(-1)
    proj = np.asarray(pca.proj, dtype=np.float32)
    lambdas = np.asarray(pca.lambdas, dtype=np.float32).reshape(-1)

    z = proj @ v
    denom = np.sqrt(lambdas + eps)
    z = z / denom

    norm = np.linalg.norm(z)
    if norm == 0:
        return z
    return z / norm

