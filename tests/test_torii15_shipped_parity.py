from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pytest
from scipy.io import loadmat

from densevlad.torii15 import Torii15Assets, compute_densevlad_pre_pca, load_torii15_vocab
from densevlad.torii15 import image as image_mod
from tests._cosine import cosine_similarity

# Compares Python DenseVLAD output directly against shipped 247code assets.


_VLFEAT_SETUP_HINT = (
    "Setup (linux/pixi example):\n"
    "  pixi install -e dev\n"
    "  pixi run build-vlfeat-linux\n"
    "  pixi run install-cyvlfeat-linux\n"
    "  pixi run install-densevlad-linux\n"
    "  source .pixi/vlfeat_env_linux.sh"
)


def _load_shipped_vlad(mat_path: Path) -> np.ndarray:
    mat = loadmat(mat_path)
    if "vlad" not in mat:
        raise KeyError(f"Expected vlad in {mat_path}")
    return np.asarray(mat["vlad"], dtype=np.float32).reshape(-1)


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> str:
    diff = a.astype(np.float64) - b.astype(np.float64)
    max_diff = float(np.max(np.abs(diff)))
    mean_diff = float(np.mean(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom != 0 else float("nan")
    nz = int(np.count_nonzero(np.abs(diff) > 1e-4))
    return (
        f"max|Δ|={max_diff:.6g} mean|Δ|={mean_diff:.6g} "
        f"L2={l2:.6g} cosine={cosine:.6g} >1e-4={nz}"
    )


def _require_cyvlfeat():
    try:
        import cyvlfeat  # noqa: F401
    except Exception:
        pytest.fail(
            "SETUP REQUIRED: cyvlfeat is required for DenseVLAD parity tests.\n"
            "Build/install it (after VLFeat) and rerun.\n"
            f"{_VLFEAT_SETUP_HINT}",
            pytrace=False,
        )


def _require_libvl():
    if image_mod._LIBVL is None:
        pytest.fail(
            "SETUP REQUIRED: VLFeat libvl (kdforest) is required for parity.\n"
            "Ensure libvl is built and discoverable.\n"
            f"{_VLFEAT_SETUP_HINT}",
            pytrace=False,
        )


def _shipped_vlad_paths() -> list[Path]:
    roots = [Path("247code/data/example_gsv"), Path("247code/data/example_grid")]
    mats: list[Path] = []
    for root in roots:
        if root.is_dir():
            mats.extend(sorted(root.rglob("*.dict_grid.dnsvlad.mat")))
    return sorted(mats)


def _resolve_image_and_masks(mat_path: Path) -> tuple[Path, Path | None, Path | None]:
    image_name = mat_path.name.replace(".dict_grid.dnsvlad.mat", ".jpg")
    if "example_grid" in mat_path.parts:
        root = Path("247code/data/example_grid")
        label = root / image_name.replace(".jpg", ".label.mat")
        plane = root / "planes.txt"
        return root / image_name, label, plane
    root = Path("247code/data/example_gsv")
    return root / image_name, None, None


def test_shipped_vlad_matches_python_strict():
    image_mod.set_simd_enabled(False)
    _require_cyvlfeat()
    _require_libvl()
    min_cos = 0.999
    shipped = _shipped_vlad_paths()
    if not shipped:
        pytest.fail("No shipped .dict_grid.dnsvlad.mat files found under 247code/data", pytrace=False)

    assets = Torii15Assets.default()
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    orig_assign = os.environ.get("DVLAD_ASSIGN_METHOD")
    os.environ["DVLAD_ASSIGN_METHOD"] = "kdtree"

    errors = []
    try:
        for mat_path in shipped:
            shipped_vlad = _load_shipped_vlad(mat_path)
            image_path, label_path, plane_path = _resolve_image_and_masks(mat_path)
            if not image_path.is_file():
                errors.append(f"Missing image for shipped vlad: {image_path}")
                continue
            if label_path is not None and not label_path.is_file():
                errors.append(f"Missing label for shipped vlad: {label_path}")
                continue
            if plane_path is not None and not plane_path.is_file():
                errors.append(f"Missing plane file for shipped vlad: {plane_path}")
                continue

            vlad = compute_densevlad_pre_pca(
                image_path,
                vocab,
                label_path=label_path,
                plane_path=plane_path,
                max_dim=None,
                apply_imdown=True,
            )
            if cosine_similarity(shipped_vlad, vlad) < min_cos:
                metrics = _diff_metrics(shipped_vlad, vlad)
                errors.append(f"Shipped vs Python mismatch for {mat_path}: {metrics}")
    finally:
        if orig_assign is None:
            os.environ.pop("DVLAD_ASSIGN_METHOD", None)
        else:
            os.environ["DVLAD_ASSIGN_METHOD"] = orig_assign

    assert not errors, "\n".join(errors)
