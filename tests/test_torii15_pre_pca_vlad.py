from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from densevlad.torii15 import Torii15Assets, compute_densevlad_pre_pca, load_torii15_vocab
from densevlad.torii15 import image as image_mod

image_mod.set_simd_enabled(False)

_VLFEAT_SETUP_HINT = (
    "Setup (linux/pixi example):\n"
    "  pixi install -e dev\n"
    "  pixi run build-vlfeat-linux\n"
    "  pixi run install-cyvlfeat-linux\n"
    "  pixi run install-densevlad-linux\n"
    "  source .pixi/vlfeat_env_linux.sh"
)


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        pytest.fail(
            "SETUP REQUIRED: h5py is required for MATLAB dump parity tests.\n"
            "Install it and rerun.",
            pytrace=False,
        )
    return h5py


def _matlab_dump_path(name: str) -> Path:
    return Torii15Assets.default_cache_dir() / "matlab_dump" / name


def _require_matlab_dump(name: str) -> tuple[object, Path]:
    h5py = _require_h5py()
    dump_path = _matlab_dump_path(name)
    if not dump_path.exists():
        pytest.fail(
            "SETUP REQUIRED: MATLAB dump not found.\n"
            f"  Expected: {dump_path}\n"
            "Generate it with:\n"
            "  matlab -batch \"run('scripts/dump_densevlad_all.m'); dump_densevlad_all('densevlad')\"",
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
            "SETUP REQUIRED: cyvlfeat is required for DenseVLAD pre-PCA parity tests.\n"
            "Build/install it (after VLFeat) and rerun.\n"
            f"{_VLFEAT_SETUP_HINT}",
            pytrace=False,
        )


def _require_libvl():
    if image_mod._LIBVL is None:
        pytest.fail(
            "SETUP REQUIRED: VLFeat libvl (kdforest) is required for pre-PCA VLAD parity.\n"
            "Ensure libvl is built and discoverable.\n"
            f"{_VLFEAT_SETUP_HINT}",
            pytrace=False,
        )


def test_torii15_pre_pca_vlad_matches_reference():
    _require_cyvlfeat()
    _require_libvl()
    h5py, dump_path = _require_matlab_dump("densevlad_dump_intermediate.mat")
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    vlad = compute_densevlad_pre_pca(image_path, vocab)
    with h5py.File(dump_path, "r") as mat:
        expected = _load_matlab_vector(mat, "vlad")

    assert vlad.shape == expected.shape
    # Element-wise tolerance: 1e-6 is appropriate for float32 after ~20 operations
    np.testing.assert_allclose(vlad, expected, rtol=0, atol=1e-6)
    # Vector-level sanity check: cosine similarity should be nearly perfect
    cosine_sim = np.dot(vlad, expected) / (np.linalg.norm(vlad) * np.linalg.norm(expected))
    assert cosine_sim > 0.999999, f"Cosine similarity {cosine_sim} too low"
