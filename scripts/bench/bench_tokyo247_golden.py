from __future__ import annotations

import os
from pathlib import Path
import time

import h5py
import numpy as np

from densevlad.torii15 import Torii15Assets, load_torii15_vocab
from densevlad.torii15.densevlad import _assignments_idx, _phow_descs, _rootsift, _vl_vlad_hard
from densevlad.torii15.image import read_gray_im2single
from densevlad.torii15.tokyo247 import Tokyo247Paths


def _golden_paths() -> tuple[Path, Path]:
    base = Path.home() / "Library" / "Caches" / "densevlad" / "torii15" / "matlab_dump"
    return base / "tokyo247_golden.mat", base / "tokyo247_golden_list.txt"


def _load_golden_list(list_path: Path, paths: Tokyo247Paths) -> list[Path]:
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    image_paths: list[Path] = []
    for line in lines:
        kind, rel = line.split("\t")
        if kind == "db":
            image_paths.append(paths.db_dir / rel)
        elif kind == "query":
            image_paths.append(paths.query_dir / rel)
        else:
            raise ValueError(f"Unexpected entry kind: {kind}")
    return image_paths


def _load_resize_opts(mat_path: Path) -> tuple[int, bool]:
    with h5py.File(mat_path, "r") as mat:
        max_dim = int(np.array(mat["max_dim"]).reshape(-1)[0])
        use_imdown = bool(np.array(mat["use_imdown"]).reshape(-1)[0])
    return max_dim, use_imdown


def main() -> None:
    os.environ.setdefault("DVLAD_ASSIGN_METHOD", "kdtree")

    mat_path, list_path = _golden_paths()
    if not mat_path.exists() or not list_path.exists():
        raise SystemExit(
            "Golden references not found. Run scripts/matlab/dump_tokyo247_golden.m first."
        )

    paths = Tokyo247Paths.default()
    image_paths = _load_golden_list(list_path, paths)
    max_dim, use_imdown = _load_resize_opts(mat_path)

    assets = Torii15Assets.default()
    vocab = load_torii15_vocab(assets.vocab_mat_path())

    totals = {
        "read": 0.0,
        "phow": 0.0,
        "rootsift": 0.0,
        "assign": 0.0,
        "vlad": 0.0,
        "total": 0.0,
    }

    for img_path in image_paths:
        t0 = time.perf_counter()
        im = read_gray_im2single(img_path, max_dim=max_dim, apply_imdown=use_imdown)
        t1 = time.perf_counter()
        descs = _phow_descs(im)
        t2 = time.perf_counter()
        descs = _rootsift(descs)
        t3 = time.perf_counter()
        idx = _assignments_idx(descs, vocab.centers)
        t4 = time.perf_counter()
        _vl_vlad_hard(descs, vocab.centers, idx)
        t5 = time.perf_counter()

        totals["read"] += t1 - t0
        totals["phow"] += t2 - t1
        totals["rootsift"] += t3 - t2
        totals["assign"] += t4 - t3
        totals["vlad"] += t5 - t4
        totals["total"] += t5 - t0

    count = len(image_paths)
    print(f"Images: {count}")
    for key in ("read", "phow", "rootsift", "assign", "vlad", "total"):
        avg = totals[key] / count
        print(f"{key:>8}: {avg:.4f} s/img (total {totals[key]:.2f}s)")


if __name__ == "__main__":
    main()
