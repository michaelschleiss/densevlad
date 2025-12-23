#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from densevlad.torii15.assets import Torii15Assets
from densevlad.torii15.densevlad import (
    Torii15Vocab,
    compute_densevlad_pre_pca,
    load_torii15_vocab,
)
from densevlad.torii15.tokyo247 import (
    Tokyo247Paths,
    load_query_time_of_day,
    load_tokyo247_dbstruct,
    resolve_db_image_paths,
    resolve_query_image_paths,
)
from densevlad.torii15.whitening import apply_pca_whitening, load_torii15_pca_whitening


def _ensure_memmap(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(f"Unexpected shape in {path}: {arr.shape} != {shape}")
        return arr
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _ensure_mask(path: Path, length: int, *, assume_complete: bool) -> np.memmap:
    if path.exists():
        mask = np.load(path, mmap_mode="r+")
        if mask.shape != (length,):
            raise ValueError(f"Unexpected mask shape in {path}: {mask.shape}")
        return mask
    mask = np.lib.format.open_memmap(path, mode="w+", dtype="uint8", shape=(length,))
    mask[:] = 1 if assume_complete else 0
    mask.flush()
    return mask


_WORKER_VOCAB: Torii15Vocab | None = None
_WORKER_MAX_DIM: int | None = None
_WORKER_APPLY_IMDOWN: bool = True


def _init_worker(centers: np.ndarray, max_dim: int | None, apply_imdown: bool) -> None:
    global _WORKER_VOCAB, _WORKER_MAX_DIM, _WORKER_APPLY_IMDOWN
    _WORKER_VOCAB = Torii15Vocab(centers=np.asarray(centers, dtype=np.float32))
    _WORKER_MAX_DIM = max_dim
    _WORKER_APPLY_IMDOWN = apply_imdown


def _worker_compute_pre_pca(task: tuple[int, Path]) -> tuple[int, np.ndarray]:
    if _WORKER_VOCAB is None:
        raise RuntimeError("Worker not initialized with vocab centers.")
    idx, img_path = task
    v_pre = compute_densevlad_pre_pca(
        img_path,
        _WORKER_VOCAB,
        max_dim=_WORKER_MAX_DIM,
        apply_imdown=_WORKER_APPLY_IMDOWN,
    )
    return idx, v_pre


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu))


def _compute_densevlad_features(
    image_paths: list[Path],
    vocab,
    pca,
    *,
    out_path: Path,
    mask_path: Path,
    label: str,
    max_dim: int | None,
    apply_imdown: bool,
    workers: int,
    worker_chunksize: int,
) -> np.ndarray:
    dim = int(pca.proj.shape[0])
    assume_complete = out_path.exists() and not mask_path.exists()
    feats = _ensure_memmap(out_path, (len(image_paths), dim), "float32")
    mask = _ensure_mask(mask_path, len(image_paths), assume_complete=assume_complete)

    done = int(mask.sum())
    pending = [(idx, path) for idx, path in enumerate(image_paths) if not mask[idx]]
    if not pending:
        feats.flush()
        mask.flush()
        return np.load(out_path, mmap_mode="r")

    if workers <= 1:
        for idx, img_path in pending:
            v_pre = compute_densevlad_pre_pca(
                img_path,
                vocab,
                max_dim=max_dim,
                apply_imdown=apply_imdown,
            )
            v = apply_pca_whitening(v_pre, pca)
            feats[idx] = v.astype(np.float32, copy=False)
            mask[idx] = 1
            done += 1
            if done % 50 == 0 or done == len(image_paths):
                print(f"{label}: {done}/{len(image_paths)}")
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(vocab.centers, max_dim, apply_imdown),
        ) as executor:
            for idx, v_pre in executor.map(
                _worker_compute_pre_pca, pending, chunksize=worker_chunksize
            ):
                v = apply_pca_whitening(v_pre, pca)
                feats[idx] = v.astype(np.float32, copy=False)
                mask[idx] = 1
                done += 1
                if done % 50 == 0 or done == len(image_paths):
                    print(f"{label}: {done}/{len(image_paths)}")

    feats.flush()
    mask.flush()
    return np.load(out_path, mmap_mode="r")


def _topk_indices(
    db_desc: np.ndarray,
    q_desc: np.ndarray,
    k: int,
    *,
    chunk_size: int,
) -> np.ndarray:
    num_db = db_desc.shape[0]
    num_q = q_desc.shape[0]
    top_scores = np.full((num_q, k), -np.inf, dtype=np.float32)
    top_idx = np.full((num_q, k), -1, dtype=np.int64)
    row = np.arange(num_q)[:, None]

    for start in range(0, num_db, chunk_size):
        end = min(start + chunk_size, num_db)
        scores = db_desc[start:end] @ q_desc.T
        scores_t = scores.T
        idx_chunk = np.arange(start, end, dtype=np.int64)
        idx_chunk = np.broadcast_to(idx_chunk, (num_q, idx_chunk.size))

        combined_scores = np.concatenate([top_scores, scores_t], axis=1)
        combined_idx = np.concatenate([top_idx, idx_chunk], axis=1)

        part = np.argpartition(combined_scores, -k, axis=1)[:, -k:]
        part_scores = combined_scores[row, part]
        order = np.argsort(part_scores, axis=1)[:, ::-1]
        top_scores = part_scores[row, order]
        top_idx = combined_idx[row, part][row, order]

    return top_idx


def _apply_pano_nms(
    topk_idx: np.ndarray,
    k: int,
    *,
    group_size: int,
) -> np.ndarray:
    num_q = topk_idx.shape[0]
    out = np.full((num_q, k), -1, dtype=np.int64)
    for qi in range(num_q):
        seen: set[int] = set()
        write = 0
        for idx in topk_idx[qi]:
            if idx < 0:
                continue
            pano = int(idx) // group_size
            if pano in seen:
                continue
            seen.add(pano)
            out[qi, write] = idx
            write += 1
            if write >= k:
                break
    return out


def _compute_positives(utm_db: np.ndarray, utm_q: np.ndarray, radius: float) -> list[np.ndarray]:
    tree = cKDTree(utm_db)
    return [np.asarray(idx, dtype=np.int64) for idx in tree.query_ball_point(utm_q, r=radius)]


def _recall_at_n(
    topk_idx: np.ndarray,
    positives: list[np.ndarray],
    ns: list[int],
    *,
    mask: np.ndarray | None = None,
) -> dict[int, float]:
    if mask is None:
        idxs = np.arange(len(positives))
    else:
        idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        return {n: float("nan") for n in ns}
    recalls: dict[int, float] = {}
    for n in ns:
        hits = 0
        for qi in idxs:
            pos = positives[qi]
            if pos.size == 0:
                continue
            if np.isin(topk_idx[qi, :n], pos).any():
                hits += 1
        recalls[n] = hits / float(len(idxs))
    return recalls


def _parse_ns(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute DenseVLAD baseline recall@N on Tokyo 24/7."
    )
    parser.add_argument("--dbstruct", type=Path, help="Path to tokyo247.mat")
    parser.add_argument("--db-dir", type=Path, help="Database image root (03814/...)")
    parser.add_argument("--query-dir", type=Path, help="Query image root (247query_subset_v2)")
    parser.add_argument("--out-dir", type=Path, help="Output directory for cached descriptors")
    parser.add_argument("--recall-n", default="1,5,10,20", help="Comma-separated N list")
    parser.add_argument("--max-dim", type=int, default=640, help="Resize max dimension (0 disables)")
    parser.add_argument(
        "--use-imdown",
        action="store_true",
        help="Apply VLFeat vl_imdown after resizing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for descriptor extraction (0=auto)",
    )
    parser.add_argument(
        "--worker-chunksize",
        type=int,
        default=4,
        help="Number of images per worker task chunk",
    )
    parser.add_argument("--chunk-size", type=int, default=4096, help="DB chunk size for scoring")
    parser.add_argument("--limit-db", type=int, default=0, help="Limit DB images (debug)")
    parser.add_argument("--limit-q", type=int, default=0, help="Limit query images (debug)")
    parser.add_argument("--force", action="store_true", help="Recompute descriptors")
    parser.add_argument(
        "--nms",
        action="store_true",
        help=(
            "Apply Tokyo24/7 panorama non-max suppression (unique panoramas). "
            "Assumes 12 cutouts per panorama ordered contiguously."
        ),
    )
    parser.add_argument(
        "--nms-group-size",
        type=int,
        default=12,
        help="Cutouts per panorama for NMS grouping (default: 12)",
    )
    args = parser.parse_args()

    paths = Tokyo247Paths.default()
    dbstruct_path = args.dbstruct or paths.dbstruct_path
    db_dir = args.db_dir or paths.db_dir
    query_dir = args.query_dir or paths.query_dir
    out_dir = args.out_dir or (paths.root_dir / "densevlad_cache")

    max_dim = args.max_dim if args.max_dim > 0 else None
    apply_imdown = bool(args.use_imdown)
    workers = args.workers if args.workers > 0 else _default_workers()
    worker_chunksize = max(1, args.worker_chunksize)
    cache_tag = f"md{max_dim or 0}_imdown{int(apply_imdown)}"

    dbstruct = load_tokyo247_dbstruct(dbstruct_path)
    db_images = dbstruct.db_images
    q_images = dbstruct.q_images
    utm_db = dbstruct.utm_db
    utm_q = dbstruct.utm_q
    if abs(dbstruct.pos_dist_thr - 25.0) > 1e-6:
        print(
            f"Warning: pos_dist_thr is {dbstruct.pos_dist_thr},"
            " expected 25.0 per Torii15 evaluation protocol."
        )

    if args.limit_db:
        db_images = db_images[: args.limit_db]
        utm_db = utm_db[: args.limit_db]
    if args.limit_q:
        q_images = q_images[: args.limit_q]
        utm_q = utm_q[: args.limit_q]

    db_paths = resolve_db_image_paths(db_images, db_dir)
    q_paths = resolve_query_image_paths(q_images, query_dir)

    if args.force:
        for suffix in ("db.npy", "db.done.npy", "q.npy", "q.done.npy"):
            candidate = out_dir / f"densevlad_4096_{cache_tag}_{suffix}"
            if candidate.exists():
                candidate.unlink()

    assets = Torii15Assets.default()
    vocab = load_torii15_vocab(assets.vocab_mat_path())
    pca = load_torii15_pca_whitening(assets.pca_mat_path(), dim=4096)
    db_desc = _compute_densevlad_features(
        db_paths,
        vocab,
        pca,
        out_path=out_dir / f"densevlad_4096_{cache_tag}_db.npy",
        mask_path=out_dir / f"densevlad_4096_{cache_tag}_db.done.npy",
        label="DB",
        max_dim=max_dim,
        apply_imdown=apply_imdown,
        workers=workers,
        worker_chunksize=worker_chunksize,
    )
    q_desc = _compute_densevlad_features(
        q_paths,
        vocab,
        pca,
        out_path=out_dir / f"densevlad_4096_{cache_tag}_q.npy",
        mask_path=out_dir / f"densevlad_4096_{cache_tag}_q.done.npy",
        label="Q",
        max_dim=max_dim,
        apply_imdown=apply_imdown,
        workers=workers,
        worker_chunksize=worker_chunksize,
    )

    positives = _compute_positives(utm_db, utm_q, dbstruct.pos_dist_thr)

    ns = _parse_ns(args.recall_n)
    max_n = max(ns)
    if args.nms:
        topk_raw = _topk_indices(
            db_desc,
            q_desc,
            max_n * args.nms_group_size,
            chunk_size=args.chunk_size,
        )
        topk_idx = _apply_pano_nms(topk_raw, max_n, group_size=args.nms_group_size)
    else:
        topk_idx = _topk_indices(db_desc, q_desc, max_n, chunk_size=args.chunk_size)

    time_by_name = load_query_time_of_day(query_dir)
    time_labels = np.array([time_by_name[name] for name in q_images])
    mask_day = time_labels == "D"
    mask_night = np.isin(time_labels, ["S", "N"])

    recall_all = _recall_at_n(topk_idx, positives, ns)
    recall_day = _recall_at_n(topk_idx, positives, ns, mask=mask_day)
    recall_night = _recall_at_n(topk_idx, positives, ns, mask=mask_night)

    print("Recall@N (all):", recall_all)
    print("Recall@N (day):", recall_day)
    print("Recall@N (sunset+night):", recall_night)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
