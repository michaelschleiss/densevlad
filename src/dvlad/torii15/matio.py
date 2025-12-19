from __future__ import annotations

from pathlib import Path
from typing import Any


def load_mat_v5(path: str | Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "scipy is required to load MATLAB v5 .mat files (pip install dvlad[dev] or scipy)."
        ) from e

    return loadmat(path)


def load_mat73_dataset(path: str | Path, name: str):
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "h5py is required to load MATLAB v7.3 .mat files (pip install h5py)."
        ) from e

    with h5py.File(path, "r") as f:
        return f[name][()]

