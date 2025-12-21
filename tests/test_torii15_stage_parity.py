from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from densevlad.torii15 import Torii15Assets
from densevlad.torii15 import image as image_mod
from densevlad.torii15.densevlad import _DSIFT_TRANSPOSE_PERM, _kdtree_assignments, _rootsift

image_mod.set_simd_enabled(False)

_VLFEAT_SETUP_HINT = (
    "Setup (linux/pixi example):\n"
    "  pixi install -e dev\n"
    "  pixi run build-vlfeat-linux\n"
    "  pixi run install-cyvlfeat-linux\n"
    "  pixi run install-densevlad-linux\n"
    "  source .pixi/vlfeat_env_linux.sh"
)


def _matlab_dump_path() -> Path:
    return Torii15Assets.default_cache_dir() / "matlab_dump" / "densevlad_dump.mat"


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


def _require_vlfeat_imconv():
    if not image_mod._VLFEAT_IMCONV_AVAILABLE:
        pytest.fail(
            "SETUP REQUIRED: VLFeat imconv is required for exact imsmooth/dsift parity.\n"
            "Build and load libvl with imconv enabled.\n"
            f"{_VLFEAT_SETUP_HINT}",
            pytrace=False,
        )


def _require_libvl():
    if image_mod._LIBVL is None:
        pytest.fail(
            "SETUP REQUIRED: VLFeat libvl (kdforest) is required for assignment parity.\n"
            "Ensure libvl is built and discoverable.\n"
            f"{_VLFEAT_SETUP_HINT}",
            pytrace=False,
        )


def _require_matlab_dump():
    h5py = _require_h5py()
    dump_path = _matlab_dump_path()
    if not dump_path.exists():
        pytest.fail(
            "SETUP REQUIRED: MATLAB dump not found.\n"
            f"  Expected: {dump_path}\n"
            "Generate it with:\n"
            "  matlab -batch \"run('scripts/matlab/dump_densevlad_all.m'); dump_densevlad_all('densevlad')\"",
            pytrace=False,
        )
    return h5py, dump_path



def _load_matlab_array(mat, name: str) -> np.ndarray:
    arr = np.array(mat[name])
    if arr.ndim == 2:
        return arr.T
    return arr


def test_preprocess_matches_matlab_dump():
    h5py, dump_path = _require_matlab_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    img_down = image_mod.read_gray_uint8(image_path)
    img_single = image_mod.read_gray_im2single(image_path)

    with h5py.File(dump_path, "r") as mat:
        img_down_ref = _load_matlab_array(mat, "img_down").astype(np.uint8, copy=False)
        img_single_ref = _load_matlab_array(mat, "img_single").astype(
            np.float32, copy=False
        )

    np.testing.assert_array_equal(img_down, img_down_ref)
    np.testing.assert_array_equal(img_single, img_single_ref)


@pytest.mark.parametrize("size", [4, 6, 8, 10])
def test_imsmooth_matches_matlab_dump(size: int):
    _require_vlfeat_imconv()
    h5py, dump_path = _require_matlab_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    img_single = image_mod.read_gray_im2single(image_path)
    ims = image_mod.vl_imsmooth_gaussian(img_single, sigma=size / 6)

    with h5py.File(dump_path, "r") as mat:
        ims_ref = _load_matlab_array(mat, f"ims_{size}").astype(np.float32, copy=False)

    np.testing.assert_array_equal(ims, ims_ref)


@pytest.mark.parametrize("size", [4, 6, 8, 10])
def test_dsift_scale_matches_matlab_dump(size: int):
    _require_cyvlfeat()
    _require_vlfeat_imconv()
    h5py, dump_path = _require_matlab_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    img_single = image_mod.read_gray_im2single(image_path)
    ims = image_mod.vl_imsmooth_gaussian(img_single, sigma=size / 6)
    ims_t = ims.T

    max_size = 10
    off = int(np.floor(1.0 + 1.5 * (max_size - size)))
    off0 = max(off - 1, 0)
    bounds = np.array([off0, off0, ims_t.shape[0] - 1, ims_t.shape[1] - 1], dtype=np.int32)

    from cyvlfeat.sift import dsift

    frames, descs = dsift(
        ims_t,
        step=2,
        size=size,
        bounds=bounds,
        window_size=1.5,
        norm=True,
        fast=True,
        float_descriptors=False,
    )
    descs = descs[:, _DSIFT_TRANSPOSE_PERM]
    descs[frames[:, 2] < 0.005] = 0

    with h5py.File(dump_path, "r") as mat:
        f_ref = _load_matlab_array(mat, f"f_{size}")
        d_ref = _load_matlab_array(mat, f"d_{size}")

    if f_ref.shape[0] == 3:
        f_ref = f_ref.T
    if d_ref.shape[0] == 128:
        d_ref = d_ref.T

    frames_py = np.stack([frames[:, 0] + 1, frames[:, 1] + 1, frames[:, 2]], axis=1)
    order_py = np.lexsort((frames_py[:, 1], frames_py[:, 0]))
    order_ref = np.lexsort((f_ref[:, 1], f_ref[:, 0]))
    frames_py = frames_py[order_py]
    f_ref = f_ref[order_ref]

    np.testing.assert_array_equal(frames_py[:, :2], f_ref[:, :2])
    np.testing.assert_allclose(frames_py[:, 2], f_ref[:, 2], atol=1e-6)
    np.testing.assert_array_equal(descs[order_py], d_ref[order_ref])


def test_phow_descs_match_matlab_dump():
    _require_cyvlfeat()
    _require_vlfeat_imconv()
    h5py, dump_path = _require_matlab_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    from densevlad.torii15.densevlad import _phow_descs

    img_single = image_mod.read_gray_im2single(image_path)
    descs = _phow_descs(img_single)

    with h5py.File(dump_path, "r") as mat:
        desc_ref = _load_matlab_array(mat, "desc")
    if desc_ref.shape[0] == 128:
        desc_ref = desc_ref.T

    np.testing.assert_array_equal(descs, desc_ref)


def test_rootsift_and_assignments_match_matlab_dump():
    _require_cyvlfeat()
    _require_libvl()
    h5py, dump_path = _require_matlab_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    from densevlad.torii15.densevlad import _phow_descs

    img_single = image_mod.read_gray_im2single(image_path)
    descs = _phow_descs(img_single)
    desc_rs = _rootsift(descs)

    with h5py.File(dump_path, "r") as mat:
        desc_rs_ref = _load_matlab_array(mat, "desc_rs")
        nn_ref = np.array(mat["nn"]).reshape(-1)
        assigns_ref = np.array(mat["assigns"])

    if desc_rs_ref.shape[0] == 128:
        desc_rs_ref = desc_rs_ref.T

    np.testing.assert_array_equal(desc_rs, desc_rs_ref)

    from densevlad.torii15 import load_torii15_vocab

    centers = load_torii15_vocab(assets.vocab_mat_path()).centers
    assigns = _kdtree_assignments(desc_rs, centers)

    np.testing.assert_array_equal(assigns, assigns_ref)
    nn = np.argmax(assigns, axis=1) + 1
    np.testing.assert_array_equal(nn, nn_ref)



def test_matmul_kdtree_assignment_equivalence():
    """Verify that matmul and kdtree assignment methods produce identical results.

    This is critical because matmul is the default (faster) method, but kdtree
    is what MATLAB uses. They must be equivalent for parity guarantees to hold.
    """
    _require_cyvlfeat()
    _require_libvl()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    from densevlad.torii15 import load_torii15_vocab
    from densevlad.torii15.densevlad import (
        _phow_descs,
        _kdtree_assignments_idx,
        _matmul_assignments_idx,
    )

    img_single = image_mod.read_gray_im2single(image_path)
    descs = _phow_descs(img_single)
    desc_rs = _rootsift(descs)
    centers = load_torii15_vocab(assets.vocab_mat_path()).centers

    idx_kdtree = _kdtree_assignments_idx(desc_rs, centers)
    idx_matmul = _matmul_assignments_idx(desc_rs, centers)

    np.testing.assert_array_equal(
        idx_matmul,
        idx_kdtree,
        err_msg="matmul and kdtree assignments must be identical",
    )
