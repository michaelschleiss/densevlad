from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from scipy.io import loadmat

from densevlad.torii15 import Torii15Assets, compute_densevlad_pre_pca, load_torii15_vocab
from densevlad.torii15 import image as image_mod
from tests._cosine import assert_cosine_similarity

image_mod.set_simd_enabled(False)

_VLFEAT_SETUP_HINT = (
    "Setup (linux/pixi example):\n"
    "  pixi install -e dev\n"
    "  pixi run build-vlfeat-linux\n"
    "  pixi run install-cyvlfeat-linux\n"
    "  pixi run install-densevlad-linux\n"
    "  source .pixi/vlfeat_env_linux.sh"
)


def _load_shipped_vlad(path: Path) -> np.ndarray:
    mat = loadmat(path)
    if "vlad" not in mat:
        raise KeyError(f"Expected vlad in {path}")
    return np.asarray(mat["vlad"], dtype=np.float32).reshape(-1)


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
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )
    shipped_path = Path(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.dict_grid.dnsvlad.mat"
    )
    expected = _load_shipped_vlad(shipped_path)
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    vlad = compute_densevlad_pre_pca(image_path, vocab)

    assert vlad.shape == expected.shape
    assert_cosine_similarity(vlad, expected, min_cos=0.999, label="pre-PCA VLAD")
