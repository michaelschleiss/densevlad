from __future__ import annotations

import base64

import numpy as np

from densevlad.torii15 import Torii15Assets
from densevlad.torii15.whitening import apply_pca_whitening, load_reference_pre_pca_vlad, load_torii15_pca_whitening


def main() -> None:
    assets = Torii15Assets.default()
    vlad = load_reference_pre_pca_vlad(assets.example_pre_pca_vlad_path())
    pca = load_torii15_pca_whitening(assets.pca_mat_path(), dim=4096)
    v = apply_pca_whitening(vlad, pca).astype(np.float32, copy=False)
    b64 = base64.b64encode(v.tobytes()).decode("ascii")
    print(b64)


if __name__ == "__main__":
    main()

