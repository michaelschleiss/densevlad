from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from dvlad.torii15 import Torii15Assets, compute_densevlad_pre_pca, load_torii15_vocab
from dvlad.torii15 import image as image_mod


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        pytest.fail(
            "h5py is required for MATLAB dump parity tests. Install it and rerun.",
            pytrace=False,
        )
    return h5py


def _matlab_dump_path(name: str) -> Path:
    return (
        Path.home()
        / "Library"
        / "Caches"
        / "dvlad"
        / "torii15"
        / "matlab_dump"
        / name
    )


def _require_matlab_dump(name: str) -> tuple[object, Path]:
    h5py = _require_h5py()
    dump_path = _matlab_dump_path(name)
    if not dump_path.exists():
        pytest.fail(
            f"MATLAB dump not found: {dump_path}. Run scripts/matlab/dump_densevlad.m "
            "or scripts/matlab/dump_densevlad_grid.m to generate it.",
            pytrace=False,
        )
    return h5py, dump_path


def _load_matlab_vector(mat, name: str) -> np.ndarray:
    arr = np.array(mat[name])
    if arr.ndim == 2:
        arr = arr.T
    return np.asarray(arr).reshape(-1)


def _require_cyvlfeat():
    try:
        import cyvlfeat  # noqa: F401
    except Exception:
        pytest.fail(
            "cyvlfeat is required for DenseVLAD pre-PCA parity tests. Build/install "
            "it (after VLFeat) and rerun.",
            pytrace=False,
        )


def _require_libvl():
    if image_mod._LIBVL is None:
        pytest.fail(
            "VLFeat libvl (kdforest) is required for pre-PCA VLAD parity. Ensure "
            "libvl is built and discoverable.",
            pytrace=False,
        )


def test_torii15_pre_pca_vlad_matches_reference():
    _require_cyvlfeat()
    _require_libvl()
    h5py, dump_path = _require_matlab_dump("densevlad_dump.mat")
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    vlad = compute_densevlad_pre_pca(image_path, vocab)
    with h5py.File(dump_path, "r") as mat:
        expected = _load_matlab_vector(mat, "vlad")

    assert vlad.shape == expected.shape
    np.testing.assert_allclose(vlad, expected, rtol=0, atol=1e-4)


def test_torii15_pre_pca_vlad_matches_reference_grid():
    _require_cyvlfeat()
    _require_libvl()
    h5py, dump_path = _require_matlab_dump("densevlad_grid_dump.mat")
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_grid/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )
    label_path = assets.extract_member(
        "247code/data/example_grid/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.label.mat"
    )
    plane_path = assets.extract_member("247code/data/example_grid/planes.txt")
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    vlad = compute_densevlad_pre_pca(
        image_path,
        vocab,
        label_path=label_path,
        plane_path=plane_path,
    )
    with h5py.File(dump_path, "r") as mat:
        expected = _load_matlab_vector(mat, "vlad")

    assert vlad.shape == expected.shape
    np.testing.assert_allclose(vlad, expected, rtol=0, atol=1e-4)
