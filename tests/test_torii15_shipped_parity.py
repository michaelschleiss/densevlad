from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from densevlad.torii15 import Torii15Assets

# Requires MATLAB dumps in ./assets/torii15/matlab_dump (generated with MATLAB)
# and shipped VLAD vectors under ./247code/data. Uses h5py (v7.3) and
# scipy.io.loadmat (v5) readers.


def _decode_matlab_str(arr: np.ndarray) -> str:
    flat = np.asarray(arr).reshape(-1)
    if flat.dtype.kind in {"u", "i"}:
        return "".join(chr(int(c)) for c in flat if int(c) != 0)
    if flat.dtype.kind == "S":
        return b"".join(flat.tolist()).decode("utf-8", errors="ignore")
    if flat.dtype.kind == "U":
        return "".join(flat.tolist())
    return str(flat)


def _load_matlab_vlad(mat_path: Path, key: str) -> tuple[np.ndarray, str]:
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception as exc:
        pytest.fail(f"h5py required to read MATLAB v7.3 dumps: {exc}", pytrace=False)
    with h5py.File(mat_path, "r") as mat:
        vlad_key = key
        imfn_key = "imfn" if key == "vlad" else "imfn_030"
        if vlad_key not in mat or imfn_key not in mat:
            raise KeyError(f"Expected {vlad_key}/{imfn_key} in {mat_path}")
        vlad = np.array(mat[vlad_key], dtype=np.float32).reshape(-1)
        imfn = _decode_matlab_str(np.array(mat[imfn_key]))
    return vlad, imfn


def _load_shipped_vlad(mat_path: Path) -> np.ndarray:
    mat = loadmat(mat_path)
    if "vlad" not in mat:
        raise KeyError(f"Expected vlad in {mat_path}")
    vlad = np.asarray(mat["vlad"], dtype=np.float32).reshape(-1)
    return vlad


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


def _shipped_vlad_paths() -> list[Path]:
    return sorted(Path("247code/data/example_gsv").rglob("*.dict_grid.dnsvlad.mat"))


def test_shipped_vlad_matches_matlab_dumps_strict():
    assets_dir = Torii15Assets.default_cache_dir()
    dump_intermediate = assets_dir / "matlab_dump" / "densevlad_dump_intermediate.mat"
    dump_blackbox = assets_dir / "matlab_dump" / "densevlad_dump_blackbox.mat"
    missing = [p for p in (dump_intermediate, dump_blackbox) if not p.exists()]
    if missing:
        pytest.fail(
            "Missing MATLAB dumps; generate them with MATLAB via "
            "dump_densevlad_intermediate('densevlad') and dump_densevlad_all_blackbox('densevlad')",
            pytrace=False,
        )

    vlad_intermediate, imfn_intermediate = _load_matlab_vlad(dump_intermediate, "vlad")
    vlad_intermediate_030, imfn_intermediate_030 = _load_matlab_vlad(
        dump_intermediate, "vlad_030"
    )
    vlad_blackbox, imfn_blackbox = _load_matlab_vlad(dump_blackbox, "vlad")
    vlad_blackbox_030, imfn_blackbox_030 = _load_matlab_vlad(
        dump_blackbox, "vlad_030"
    )
    im_gsv = Path(imfn_intermediate).name
    im_gsv_030 = Path(imfn_intermediate_030).name

    shipped = _shipped_vlad_paths()
    if not shipped:
        pytest.fail("No shipped .dict_grid.dnsvlad.mat files found under 247code/data", pytrace=False)

    errors = []
    if imfn_intermediate != imfn_blackbox or imfn_intermediate_030 != imfn_blackbox_030:
        errors.append(
            "MATLAB dumps disagree on input image paths: "
            f"intermediate={imfn_intermediate},{imfn_intermediate_030} "
            f"blackbox={imfn_blackbox},{imfn_blackbox_030}"
        )
    if not np.array_equal(vlad_intermediate, vlad_blackbox):
        metrics = _diff_metrics(vlad_intermediate, vlad_blackbox)
        errors.append(f"Intermediate vs blackbox mismatch (gsv): {metrics}")
    if not np.array_equal(vlad_intermediate_030, vlad_blackbox_030):
        metrics = _diff_metrics(vlad_intermediate_030, vlad_blackbox_030)
        errors.append(f"Intermediate vs blackbox mismatch (gsv_030): {metrics}")

    for mat_path in shipped:
        shipped_vlad = _load_shipped_vlad(mat_path)
        image_name = mat_path.name.replace(".dict_grid.dnsvlad.mat", ".jpg")
        if "example_gsv" in mat_path.parts:
            if image_name == im_gsv:
                ref_intermediate = vlad_intermediate
                ref_blackbox = vlad_blackbox
            elif image_name == im_gsv_030:
                ref_intermediate = vlad_intermediate_030
                ref_blackbox = vlad_blackbox_030
            else:
                errors.append(
                    f"No MATLAB dump for shipped GSV image {image_name}; "
                    f"MATLAB dumps are for {im_gsv} and {im_gsv_030}."
                )
                continue
        else:
            errors.append(f"Unknown shipped vlad location: {mat_path}")
            continue

        if not np.array_equal(shipped_vlad, ref_intermediate):
            metrics = _diff_metrics(shipped_vlad, ref_intermediate)
            errors.append(f"Shipped vs intermediate mismatch for {mat_path}: {metrics}")
        if not np.array_equal(shipped_vlad, ref_blackbox):
            metrics = _diff_metrics(shipped_vlad, ref_blackbox)
            errors.append(f"Shipped vs blackbox mismatch for {mat_path}: {metrics}")

    assert not errors, "\n".join(errors)
