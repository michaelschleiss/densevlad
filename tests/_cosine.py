from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def assert_cosine_similarity(
    a: np.ndarray, b: np.ndarray, min_cos: float = 0.999, label: str | None = None
) -> float:
    cos = cosine_similarity(a, b)
    name = f"{label}: " if label else ""
    assert cos >= min_cos, f"{name}cosine {cos:.6f} < {min_cos}"
    return cos


def assert_cosine_similarity_rows(
    a: np.ndarray, b: np.ndarray, min_cos: float = 0.999, label: str | None = None
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label or 'arrays'} shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise AssertionError(f"{label or 'arrays'} must be 2D for row-wise cosine")
    num = np.einsum("ij,ij->i", a, b)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    # Treat rows that are all-zero in both arrays as perfect matches.
    zero_rows = (np.linalg.norm(a, axis=1) == 0) & (np.linalg.norm(b, axis=1) == 0)
    if np.any(zero_rows):
        cos[zero_rows] = 1.0
    min_cosine = float(np.min(cos)) if cos.size else float("nan")
    name = f"{label}: " if label else ""
    assert min_cosine >= min_cos, f"{name}min cosine {min_cosine:.6f} < {min_cos}"
    return cos


def assert_fraction_equal(
    a: np.ndarray, b: np.ndarray, min_fraction: float = 0.999, label: str | None = None
) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{label or 'arrays'} shape mismatch: {a.shape} vs {b.shape}")
    frac = float(np.mean(a == b)) if a.size else float("nan")
    name = f"{label}: " if label else ""
    assert frac >= min_fraction, f"{name}match fraction {frac:.6f} < {min_fraction}"
    return frac
