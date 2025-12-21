#!/usr/bin/env python3
"""Detailed timing breakdown of PHOW/dsift pipeline."""
from __future__ import annotations

import os
import time
from pathlib import Path

import h5py
import numpy as np

from densevlad.torii15 import Torii15Assets
from densevlad.torii15.image import read_gray_im2single, vl_imsmooth_gaussian
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
    from cyvlfeat.sift import dsift

    mat_path, list_path = _golden_paths()
    if not mat_path.exists() or not list_path.exists():
        raise SystemExit(
            "Golden references not found. Run scripts/matlab/dump_tokyo247_golden.m first."
        )

    paths = Tokyo247Paths.default()
    image_paths = _load_golden_list(list_path, paths)
    max_dim, use_imdown = _load_resize_opts(mat_path)

    sizes = (4, 6, 8, 10)
    step = 2
    magnification = 6
    window_size = 1.5
    contrast_threshold = 0.005
    max_size = max(sizes)

    totals = {
        "read": 0.0,
        "smooth": {s: 0.0 for s in sizes},
        "dsift": {s: 0.0 for s in sizes},
        "phow_total": 0.0,
    }
    desc_counts = {s: 0 for s in sizes}

    for img_path in image_paths:
        t0 = time.perf_counter()
        im = read_gray_im2single(img_path, max_dim=max_dim, apply_imdown=use_imdown)
        t1 = time.perf_counter()
        totals["read"] += t1 - t0

        phow_t0 = time.perf_counter()
        for size in sizes:
            off = int(np.floor(1.0 + 1.5 * (max_size - size)))
            off0 = max(off - 1, 0)

            ts0 = time.perf_counter()
            ims = vl_imsmooth_gaussian(im, sigma=size / magnification)
            ts1 = time.perf_counter()
            totals["smooth"][size] += ts1 - ts0

            ims_t = np.ascontiguousarray(ims.T)
            td0 = time.perf_counter()
            frames, descs = dsift(
                ims_t,
                step=step,
                size=size,
                bounds=np.array(
                    [off0, off0, ims_t.shape[0] - 1, ims_t.shape[1] - 1], dtype=np.int32
                ),
                window_size=window_size,
                norm=True,
                fast=True,
                float_descriptors=False,
            )
            td1 = time.perf_counter()
            totals["dsift"][size] += td1 - td0
            desc_counts[size] += descs.shape[0]

        phow_t1 = time.perf_counter()
        totals["phow_total"] += phow_t1 - phow_t0

    count = len(image_paths)
    print(f"Images: {count}")
    print(f"\nread: {totals['read']/count:.4f} s/img")
    print(f"phow_total: {totals['phow_total']/count:.4f} s/img")
    print(f"\nPer-scale breakdown:")

    total_smooth = 0.0
    total_dsift = 0.0
    total_descs = 0

    for size in sizes:
        smooth_avg = totals["smooth"][size] / count
        dsift_avg = totals["dsift"][size] / count
        descs_avg = desc_counts[size] / count
        us_per_desc = (dsift_avg * 1e6) / descs_avg if descs_avg > 0 else 0
        total_smooth += smooth_avg
        total_dsift += dsift_avg
        total_descs += descs_avg
        print(f"  size={size:2d}: smooth={smooth_avg:.4f}s  dsift={dsift_avg:.4f}s  "
              f"descs={descs_avg:.0f}  µs/desc={us_per_desc:.2f}")

    total_us_per_desc = (total_dsift * 1e6) / total_descs if total_descs > 0 else 0
    print(f"\nAggregated:")
    print(f"  total_smooth: {total_smooth:.4f} s/img")
    print(f"  total_dsift:  {total_dsift:.4f} s/img")
    print(f"  total_descs:  {total_descs:.0f} desc/img")
    print(f"  µs/desc:      {total_us_per_desc:.2f}")


if __name__ == "__main__":
    main()
