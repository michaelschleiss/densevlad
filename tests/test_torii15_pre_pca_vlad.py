from __future__ import annotations

import pytest
import numpy as np

from dvlad.torii15 import Torii15Assets, compute_densevlad_pre_pca, load_torii15_vocab
from dvlad.torii15.whitening import load_reference_pre_pca_vlad


def test_torii15_pre_pca_vlad_matches_reference():
    pytest.importorskip("cyvlfeat")
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    vlad = compute_densevlad_pre_pca(image_path, vocab)
    expected = load_reference_pre_pca_vlad(assets.example_pre_pca_vlad_path())

    assert vlad.shape == expected.shape
    np.testing.assert_allclose(vlad, expected, rtol=0, atol=1e-4)

