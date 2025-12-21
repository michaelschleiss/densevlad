from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from densevlad.torii15 import Torii15Assets, compute_densevlad_pre_pca, load_torii15_vocab
from densevlad.torii15 import image as image_mod
from densevlad.torii15.tokyo247 import Tokyo247Paths
from densevlad.torii15.whitening import apply_pca_whitening, load_torii15_pca_whitening

image_mod.set_simd_enabled(False)


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        pytest.fail(
            "SETUP REQUIRED: h5py is required for Tokyo247 golden parity tests.\n"
            "Install it and rerun.",
            pytrace=False,
        )
    return h5py


def _golden_paths() -> tuple[Path, Path]:
    base = Torii15Assets.default_cache_dir() / "matlab_dump"
    return base / "tokyo247_golden.mat", base / "tokyo247_golden_list.txt"


def _require_golden_assets() -> tuple[object, Path, Path]:
    h5py = _require_h5py()
    mat_path, list_path = _golden_paths()
    if not mat_path.exists() or not list_path.exists():
        pytest.fail(
            "SETUP REQUIRED: Tokyo247 golden references not found.\n"
            f"  Expected: {mat_path}\n"
            f"            {list_path}\n"
            "Generate them with:\n"
            "  matlab -batch \"run('scripts/dump_densevlad_all_intermediate.m'); dump_densevlad_all_intermediate('tokyo247')\"",
            pytrace=False,
        )
    return h5py, mat_path, list_path


def _load_matlab_matrix(mat, name: str, *, dim: int) -> np.ndarray:
    arr = np.array(mat[name])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got shape {arr.shape}")
    if arr.shape[0] == dim:
        arr = arr.T
    elif arr.shape[1] != dim:
        raise ValueError(f"Unexpected {name} shape: {arr.shape} (dim {dim})")
    return np.asarray(arr, dtype=np.float32)


def _load_matlab_scalar(mat, name: str) -> float:
    if name not in mat:
        raise KeyError(name)
    arr = np.array(mat[name])
    if arr.size != 1:
        raise ValueError(f"Expected scalar for {name}, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _load_golden_list(list_path: Path, paths: Tokyo247Paths) -> list[Path]:
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    image_paths: list[Path] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid golden list line: {line}")
        kind, rel = parts
        if kind == "db":
            image_paths.append(paths.db_dir / rel)
        elif kind == "query":
            image_paths.append(paths.query_dir / rel)
        else:
            raise ValueError(f"Unexpected entry kind: {kind}")
    return image_paths


def test_tokyo247_golden_vectors_match_matlab():
    h5py, mat_path, list_path = _require_golden_assets()
    paths = Tokyo247Paths.default()
    image_paths = _load_golden_list(list_path, paths)

    assets = Torii15Assets.default()
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    pca = load_torii15_pca_whitening(assets.pca_mat_path(), dim=4096)

    with h5py.File(mat_path, "r") as mat:
        pre_ref = _load_matlab_matrix(mat, "vlad_pre", dim=16384)
        v_ref = _load_matlab_matrix(mat, "vlad_4096", dim=4096)
        try:
            max_dim = int(_load_matlab_scalar(mat, "max_dim"))
            use_imdown = bool(_load_matlab_scalar(mat, "use_imdown"))
        except KeyError:
            pytest.fail(
                "SETUP REQUIRED: tokyo247_golden.mat missing max_dim/use_imdown.\n"
                "Regenerate with:\n"
                "  matlab -batch \"run('scripts/dump_densevlad_all_intermediate.m'); dump_densevlad_all_intermediate('tokyo247')\"",
                pytrace=False,
            )

    orig_assign = os.environ.get("DVLAD_ASSIGN_METHOD")
    os.environ["DVLAD_ASSIGN_METHOD"] = "kdtree"
    try:
        pre_list = []
        v_list = []
        for img_path in image_paths:
            v_pre = compute_densevlad_pre_pca(
                img_path,
                vocab,
                max_dim=max_dim,
                apply_imdown=use_imdown,
            )
            v_pre = np.asarray(v_pre, dtype=np.float32).reshape(-1)
            pre_list.append(v_pre)
            v_list.append(apply_pca_whitening(v_pre, pca))
        pre = np.vstack(pre_list)
        v = np.vstack(v_list)
    finally:
        if orig_assign is None:
            os.environ.pop("DVLAD_ASSIGN_METHOD", None)
        else:
            os.environ["DVLAD_ASSIGN_METHOD"] = orig_assign

    assert pre.shape == pre_ref.shape
    assert v.shape == v_ref.shape
    # Element-wise tolerance: 1e-6 is appropriate for float32 after ~20 operations
    np.testing.assert_allclose(pre, pre_ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(v, v_ref, rtol=0, atol=1e-6)
    # Vector-level sanity check: cosine similarity for each row should be nearly perfect
    for i in range(pre.shape[0]):
        cos_pre = np.dot(pre[i], pre_ref[i]) / (np.linalg.norm(pre[i]) * np.linalg.norm(pre_ref[i]))
        cos_v = np.dot(v[i], v_ref[i]) / (np.linalg.norm(v[i]) * np.linalg.norm(v_ref[i]))
        assert cos_pre > 0.999999, f"Pre-PCA cosine similarity {cos_pre} too low for image {i}"
        assert cos_v > 0.999999, f"Whitened cosine similarity {cos_v} too low for image {i}"
