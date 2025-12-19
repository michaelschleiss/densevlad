from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dvlad.torii15 import Torii15Assets
from dvlad.torii15 import image as image_mod
from dvlad.torii15.densevlad import _DSIFT_TRANSPOSE_PERM, _kdtree_assignments, _rootsift


def _matlab_dump_path() -> Path:
    return (
        Path.home()
        / "Library"
        / "Caches"
        / "dvlad"
        / "torii15"
        / "matlab_dump"
        / "densevlad_dump.mat"
    )


def _matlab_grid_dump_path() -> Path:
    return (
        Path.home()
        / "Library"
        / "Caches"
        / "dvlad"
        / "torii15"
        / "matlab_dump"
        / "densevlad_grid_dump.mat"
    )


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        pytest.fail(
            "h5py is required for MATLAB dump parity tests. Install it and rerun.",
            pytrace=False,
        )
    return h5py


def _require_cyvlfeat():
    try:
        import cyvlfeat  # noqa: F401
    except Exception:
        pytest.fail(
            "cyvlfeat is required for DenseVLAD parity tests. Build/install it "
            "(after VLFeat) and rerun.",
            pytrace=False,
        )


def _require_vlfeat_imconv():
    if not image_mod._VLFEAT_IMCONV_AVAILABLE:
        pytest.fail(
            "VLFeat imconv is required for exact imsmooth/dsift parity. Build and "
            "load libvl with imconv enabled.",
            pytrace=False,
        )


def _require_libvl():
    if image_mod._LIBVL is None:
        pytest.fail(
            "VLFeat libvl (kdforest) is required for assignment parity. Ensure "
            "libvl is built and discoverable.",
            pytrace=False,
        )


def _require_matlab_dump():
    h5py = _require_h5py()
    dump_path = _matlab_dump_path()
    if not dump_path.exists():
        pytest.fail(
            f"MATLAB dump not found: {dump_path}. Run scripts/matlab/dump_densevlad.m "
            "to generate it.",
            pytrace=False,
        )
    return h5py, dump_path


def _require_matlab_grid_dump():
    h5py = _require_h5py()
    dump_path = _matlab_grid_dump_path()
    if not dump_path.exists():
        pytest.fail(
            f"MATLAB grid dump not found: {dump_path}. Run "
            "scripts/matlab/dump_densevlad_grid.m to generate it.",
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

    np.testing.assert_array_equal(frames_py, f_ref)
    np.testing.assert_array_equal(descs[order_py], d_ref[order_ref])


def test_phow_descs_match_matlab_dump():
    _require_cyvlfeat()
    _require_vlfeat_imconv()
    h5py, dump_path = _require_matlab_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    from dvlad.torii15.densevlad import _phow_descs

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

    from dvlad.torii15.densevlad import _phow_descs

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

    from dvlad.torii15 import load_torii15_vocab

    centers = load_torii15_vocab(assets.vocab_mat_path()).centers
    assigns = _kdtree_assignments(desc_rs, centers)

    np.testing.assert_array_equal(assigns, assigns_ref)
    nn = np.argmax(assigns, axis=1) + 1
    np.testing.assert_array_equal(nn, nn_ref)


def test_grid_mask_matches_matlab_dump():
    _require_cyvlfeat()
    _require_vlfeat_imconv()
    h5py, dump_path = _require_matlab_grid_dump()
    assets = Torii15Assets.default()
    image_path = assets.extract_member(
        "247code/data/example_grid/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )
    label_path = assets.extract_member(
        "247code/data/example_grid/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.label.mat"
    )
    plane_path = assets.extract_member("247code/data/example_grid/planes.txt")

    from dvlad.torii15.densevlad import _grid_mask_features_dense, _phow

    img_single = image_mod.read_gray_im2single(image_path)
    frames, descs = _phow(img_single)
    descs = _rootsift(descs)
    _, _, mask = _grid_mask_features_dense(
        frames,
        descs,
        img_single.shape[:2],
        Path(label_path),
        Path(plane_path),
        return_mask=True,
    )

    with h5py.File(dump_path, "r") as mat:
        msk_ref = _load_matlab_array(mat, "msk")
    msk_ref = np.asarray(msk_ref).reshape(-1).astype(bool)

    assert mask.shape == msk_ref.shape
    np.testing.assert_array_equal(mask, msk_ref)
