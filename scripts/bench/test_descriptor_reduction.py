#!/usr/bin/env python3
"""Test impact of reducing descriptor count on performance."""
from __future__ import annotations

import os
import time
from pathlib import Path

import h5py
import numpy as np

from densevlad.torii15 import Torii15Assets, load_torii15_vocab
from densevlad.torii15.densevlad import _assignments_idx, _rootsift, _vl_vlad_hard
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


def _phow_descs_variable(
    im: np.ndarray,
    sizes: tuple[int, ...] = (4, 6, 8, 10),
    step: int = 2,
) -> np.ndarray:
    """Modified PHOW with configurable parameters."""
    from cyvlfeat.sift import dsift
    from densevlad.torii15.densevlad import _DSIFT_TRANSPOSE_PERM

    magnification = 6
    window_size = 1.5
    contrast_threshold = 0.005
    max_size = max(sizes)
    descs_all: list[np.ndarray] = []

    for size in sizes:
        off = int(np.floor(1.0 + 1.5 * (max_size - size)))
        off0 = max(off - 1, 0)
        ims = vl_imsmooth_gaussian(im, sigma=size / magnification)
        ims_t = np.ascontiguousarray(ims.T)
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
        descs = descs[:, _DSIFT_TRANSPOSE_PERM]
        contrast = frames[:, 2]
        descs[contrast < contrast_threshold] = 0
        descs_all.append(descs)

    return np.vstack(descs_all)


def benchmark_config(
    image_paths: list[Path],
    vocab,
    max_dim: int,
    use_imdown: bool,
    sizes: tuple[int, ...],
    step: int,
    label: str,
) -> dict:
    """Benchmark a specific configuration."""
    totals = {
        "read": 0.0,
        "phow": 0.0,
        "rootsift": 0.0,
        "assign": 0.0,
        "vlad": 0.0,
        "total": 0.0,
    }
    desc_count = 0

    for img_path in image_paths:
        t0 = time.perf_counter()
        im = read_gray_im2single(img_path, max_dim=max_dim, apply_imdown=use_imdown)
        t1 = time.perf_counter()
        descs = _phow_descs_variable(im, sizes=sizes, step=step)
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
        desc_count += descs.shape[0]

    count = len(image_paths)
    avg_descs = desc_count / count

    return {
        "config": label,
        "sizes": sizes,
        "step": step,
        "avg_descs": avg_descs,
        "time_per_img": totals["total"] / count,
        "read": totals["read"] / count,
        "phow": totals["phow"] / count,
        "rootsift": totals["rootsift"] / count,
        "assign": totals["assign"] / count,
        "vlad": totals["vlad"] / count,
    }


def main() -> None:
    os.environ.setdefault("DVLAD_ASSIGN_METHOD", "matmul")

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

    # Test different configurations
    configs = [
        ("Baseline (4 scales, step=2)", (4, 6, 8, 10), 2),
        ("Reduced scales (2 scales, step=2)", (4, 8), 2),
        ("Larger step (4 scales, step=3)", (4, 6, 8, 10), 3),
        ("Larger step (4 scales, step=4)", (4, 6, 8, 10), 4),
        ("Both (2 scales, step=3)", (4, 8), 3),
        ("Both (2 scales, step=4)", (4, 8), 4),
        ("Aggressive (1 scale, step=4)", (6,), 4),
    ]

    results = []
    baseline_time = None

    for label, sizes, step in configs:
        print(f"\nTesting: {label}")
        result = benchmark_config(
            image_paths, vocab, max_dim, use_imdown, sizes, step, label
        )
        results.append(result)

        if baseline_time is None:
            baseline_time = result["time_per_img"]

        speedup = baseline_time / result["time_per_img"]
        desc_ratio = result["avg_descs"] / results[0]["avg_descs"]

        print(f"  Descriptors: {result['avg_descs']:.0f} ({desc_ratio:.2f}x)")
        print(f"  Time/image:  {result['time_per_img']:.4f}s ({speedup:.2f}x speedup)")
        print(f"    phow:      {result['phow']:.4f}s")
        print(f"    assign:    {result['assign']:.4f}s")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'Descs':>10} {'Time':>10} {'Speedup':>10} {'vs Baseline':>12}")
    print("-" * 80)

    for result in results:
        speedup = baseline_time / result["time_per_img"]
        desc_ratio = result["avg_descs"] / results[0]["avg_descs"]
        print(
            f"{result['config']:<30} {result['avg_descs']:>10.0f} "
            f"{result['time_per_img']:>10.4f}s {speedup:>9.2f}x "
            f"{desc_ratio:>11.2f}x"
        )

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    best = max(results[1:], key=lambda r: baseline_time / r["time_per_img"])
    speedup = baseline_time / best["time_per_img"]
    print(f"Best alternative: {best['config']}")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time: {best['time_per_img']:.4f}s (vs baseline {baseline_time:.4f}s)")
    print(f"  Implementation effort: 1 line of code change")
    print(f"\nCompare to rewriting dsift:")
    print(f"  Expected speedup: 1.24x (optimistic)")
    print(f"  Development time: 2-12 weeks")
    print(f"  Risk: High (parity, maintenance)")


if __name__ == "__main__":
    main()
