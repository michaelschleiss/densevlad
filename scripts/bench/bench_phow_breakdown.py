#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from densevlad.torii15 import Torii15Assets, load_torii15_vocab
from densevlad.torii15 import image as image_mod
from densevlad.torii15.densevlad import (
    _DSIFT_TRANSPOSE_PERM,
    _assignments_idx,
    _rootsift,
    _vl_vlad_hard,
)


def _mean(times: List[float]) -> float:
    return float(sum(times) / len(times)) if times else 0.0


def _timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _load_image(path: str, max_dim: int | None, apply_imdown: bool) -> np.ndarray:
    return image_mod.read_gray_im2single(path, max_dim=max_dim, apply_imdown=apply_imdown)


def bench_phow_breakdown(
    *,
    image_path: str,
    max_dim: int | None,
    apply_imdown: bool,
    reps: int,
    warmup: int,
    simd: bool,
    assign_method: str,
) -> Dict[str, float]:
    image_mod.set_simd_enabled(simd)
    os.environ["DVLAD_PHOW_WORKERS"] = "1"
    os.environ["DVLAD_PHOW_BACKEND"] = "serial"
    os.environ["DVLAD_ASSIGN_METHOD"] = assign_method

    vocab = load_torii15_vocab(Torii15Assets.default().vocab_mat_path())

    sizes = (4, 6, 8, 10)
    max_size = max(sizes)
    step = 2
    magnification = 6
    window_size = 1.5
    contrast_threshold = 0.005

    try:
        from cyvlfeat.sift import dsift
    except Exception as exc:  # pragma: no cover - bench-only
        raise SystemExit(f"cyvlfeat is required for benchmarks: {exc}")

    # Warmup
    for _ in range(max(0, warmup)):
        im = _load_image(image_path, max_dim, apply_imdown)
        for size in sizes:
            sigma = size / magnification
            ims = image_mod.vl_imsmooth_gaussian(im, sigma=sigma)
            ims_t = np.ascontiguousarray(ims.T)
            off = int(np.floor(1.0 + 1.5 * (max_size - size)))
            off0 = max(off - 1, 0)
            _ = dsift(
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

    acc: Dict[str, float] = {}

    def add_time(key: str, dt: float) -> None:
        acc[key] = acc.get(key, 0.0) + dt

    for _ in range(reps):
        per: Dict[str, float] = {}

        def add_per(key: str, dt: float) -> None:
            per[key] = per.get(key, 0.0) + dt

        t0 = time.perf_counter()
        im = _load_image(image_path, max_dim, apply_imdown)
        add_per("preprocess", time.perf_counter() - t0)

        descs_all = []
        for size in sizes:
            sigma = size / magnification
            t0 = time.perf_counter()
            ims = image_mod.vl_imsmooth_gaussian(im, sigma=sigma)
            add_per(f"imsmooth_{size}", time.perf_counter() - t0)

            ims_t = np.ascontiguousarray(ims.T)
            off = int(np.floor(1.0 + 1.5 * (max_size - size)))
            off0 = max(off - 1, 0)

            t0 = time.perf_counter()
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
            add_per(f"dsift_{size}", time.perf_counter() - t0)

            t0 = time.perf_counter()
            descs = descs[:, _DSIFT_TRANSPOSE_PERM]
            descs[frames[:, 2] < contrast_threshold] = 0
            add_per(f"post_{size}", time.perf_counter() - t0)
            descs_all.append(descs)

        t0 = time.perf_counter()
        descs = np.vstack(descs_all)
        add_per("phow_concat", time.perf_counter() - t0)

        t0 = time.perf_counter()
        descs_rs = _rootsift(descs)
        add_per("rootsift", time.perf_counter() - t0)

        t0 = time.perf_counter()
        idx = _assignments_idx(descs_rs, vocab.centers)
        add_per("assign", time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = _vl_vlad_hard(descs_rs, vocab.centers, idx)
        add_per("vlad", time.perf_counter() - t0)

        per["total"] = sum(
            per.get(k, 0.0) for k in (
                "preprocess",
                *[f"imsmooth_{s}" for s in sizes],
                *[f"dsift_{s}" for s in sizes],
                *[f"post_{s}" for s in sizes],
                "phow_concat",
                "rootsift",
                "assign",
                "vlad",
            )
        )

        for k, v in per.items():
            add_time(k, v)

    # average
    for k in list(acc.keys()):
        acc[k] /= reps
    return acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PHOW + VLAD breakdown (Python).")
    parser.add_argument("--image", default="", help="Override image path.")
    parser.add_argument("--max-dim", type=int, default=640, help="Max side before imdown.")
    parser.add_argument("--no-imdown", action="store_true", help="Disable vl_imdown.")
    parser.add_argument("--reps", type=int, default=3, help="Benchmark repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions.")
    parser.add_argument("--no-simd", action="store_true", help="Disable SIMD paths.")
    parser.add_argument(
        "--assign-method",
        default="kdtree",
        choices=["kdtree", "matmul", "hard", "kmeans"],
        help="Assignment method for VLAD.",
    )
    args = parser.parse_args()

    assets = Torii15Assets.default()
    image_path = args.image or assets.extract_member(
        "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg"
    )

    res = bench_phow_breakdown(
        image_path=str(image_path),
        max_dim=None if args.max_dim <= 0 else args.max_dim,
        apply_imdown=not args.no_imdown,
        reps=max(1, args.reps),
        warmup=max(0, args.warmup),
        simd=not args.no_simd,
        assign_method=args.assign_method,
    )

    print("BENCH python")
    for key in sorted(res.keys()):
        print(f"BENCH {key} {res[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
