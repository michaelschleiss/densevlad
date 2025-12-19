#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _default_dump_path() -> Path:
    return (
        Path.home()
        / "Library"
        / "Caches"
        / "dvlad"
        / "torii15"
        / "matlab_dump"
        / "densevlad_dump.mat"
    )


def _default_out_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "src" / "dvlad" / "torii15" / "data" / "dnscnt_RDSIFT_K128.cx_norm.npy"


def _load_cx(dump_path: Path) -> np.ndarray:
    with h5py.File(dump_path, "r") as mat:
        if "CX" not in mat:
            raise KeyError(f"Expected variable 'CX' in {dump_path}")
        cx = np.array(mat["CX"], dtype=np.float32)
    if cx.ndim != 2 or cx.shape != (128, 128):
        raise ValueError(f"Unexpected CX shape: {cx.shape} (expected 128x128)")
    # MATLAB v7.3 data is stored column-major; HDF5 reads it transposed.
    # We keep the HDF5 orientation (K x D) to match the Python pipeline.
    return cx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the MATLAB-normalized CX asset from a densevlad_dump.mat file."
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=_default_dump_path(),
        help="Path to densevlad_dump.mat (MATLAB v7.3).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_out_path(),
        help="Output .npy path for normalized centers.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output against an existing .npy file before overwriting.",
    )
    args = parser.parse_args()

    if not args.dump.exists():
        raise SystemExit(f"Dump not found: {args.dump}")

    cx = _load_cx(args.dump)

    if args.verify and args.out.exists():
        existing = np.load(args.out)
        if existing.shape != cx.shape:
            raise SystemExit(
                f"Shape mismatch: existing {existing.shape}, dump {cx.shape}"
            )
        if not np.array_equal(existing, cx):
            diff = existing.astype(np.float64) - cx.astype(np.float64)
            max_diff = float(np.max(np.abs(diff)))
            nz = int(np.count_nonzero(diff))
            raise SystemExit(
                f"Mismatch with existing asset: max_diff={max_diff} nonzero={nz}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, cx)

    dump_hash = _sha256_path(args.dump)
    out_hash = _sha256_path(args.out)
    arr_hash = _sha256_array(cx)
    print(f"dump: {args.dump}")
    print(f"dump_sha256: {dump_hash}")
    print(f"out: {args.out}")
    print(f"out_sha256: {out_hash}")
    print(f"cx_sha256: {arr_hash}")
    print(f"shape: {cx.shape} dtype: {cx.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
