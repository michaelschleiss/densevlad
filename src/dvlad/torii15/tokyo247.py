from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path

import numpy as np

from .assets import Torii15Assets
from .matio import load_mat_v5


@dataclass(frozen=True)
class Tokyo247Paths:
    root_dir: Path
    db_dir: Path
    query_dir: Path
    dbstruct_path: Path

    @classmethod
    def default(cls) -> "Tokyo247Paths":
        cache_dir = Torii15Assets.default_cache_dir()
        root_env = os.environ.get("DVLAD_TOKYO247_ROOT")
        root_dir = Path(root_env) if root_env else cache_dir / "tokyo247"

        db_env = os.environ.get("DVLAD_TOKYO247_DB_DIR")
        db_dir = Path(db_env) if db_env else root_dir / "database_gsv_vga"

        query_env = os.environ.get("DVLAD_TOKYO247_QUERY_DIR")
        if query_env:
            query_dir = Path(query_env)
        else:
            candidate = root_dir / "queries"
            if candidate.is_dir():
                query_dir = candidate
            else:
                query_dir = cache_dir / "247query_subset_v2" / "247query_subset_v2"

        dbstruct_env = os.environ.get("DVLAD_TOKYO247_DBSTRUCT")
        dbstruct_path = Path(dbstruct_env) if dbstruct_env else root_dir / "tokyo247.mat"

        return cls(
            root_dir=root_dir,
            db_dir=db_dir,
            query_dir=query_dir,
            dbstruct_path=dbstruct_path,
        )


@dataclass(frozen=True)
class Tokyo247DbStruct:
    db_images: list[str]
    utm_db: np.ndarray
    q_images: list[str]
    utm_q: np.ndarray
    num_db: int
    num_q: int
    pos_dist_thr: float
    pos_dist_sq_thr: float
    nontriv_pos_dist_sq_thr: float


def _matlab_str_list(arr: np.ndarray) -> list[str]:
    flat = np.asarray(arr).reshape(-1)
    return [str(item[0]) for item in flat]


def load_tokyo247_dbstruct(mat_path: str | Path) -> Tokyo247DbStruct:
    mat = load_mat_v5(mat_path)
    if "dbStruct" not in mat:
        raise KeyError(f"Expected variable 'dbStruct' in {mat_path}")
    raw = mat["dbStruct"]
    if raw.shape != (1, 1):
        raise ValueError(f"Unexpected dbStruct shape: {raw.shape}")
    fields = raw[0, 0]
    if len(fields) < 10:
        raise ValueError("dbStruct is missing expected fields.")

    db_images = _matlab_str_list(fields[1])
    utm_db = np.asarray(fields[2], dtype=np.float64).T
    q_images = _matlab_str_list(fields[3])
    utm_q = np.asarray(fields[4], dtype=np.float64).T

    num_db = int(np.asarray(fields[5]).squeeze())
    num_q = int(np.asarray(fields[6]).squeeze())
    pos_dist_thr = float(np.asarray(fields[7]).squeeze())
    pos_dist_sq_thr = float(np.asarray(fields[8]).squeeze())
    nontriv_pos_dist_sq_thr = float(np.asarray(fields[9]).squeeze())

    if utm_db.shape[0] != len(db_images):
        raise ValueError(
            f"dbStruct mismatch: {utm_db.shape[0]} UTMs for {len(db_images)} db images."
        )
    if utm_q.shape[0] != len(q_images):
        raise ValueError(
            f"dbStruct mismatch: {utm_q.shape[0]} UTMs for {len(q_images)} queries."
        )
    if num_db != len(db_images) or num_q != len(q_images):
        raise ValueError("dbStruct counts do not match image lists.")

    return Tokyo247DbStruct(
        db_images=db_images,
        utm_db=utm_db,
        q_images=q_images,
        utm_q=utm_q,
        num_db=num_db,
        num_q=num_q,
        pos_dist_thr=pos_dist_thr,
        pos_dist_sq_thr=pos_dist_sq_thr,
        nontriv_pos_dist_sq_thr=nontriv_pos_dist_sq_thr,
    )


def resolve_db_image_paths(
    db_images: list[str],
    db_dir: str | Path,
    *,
    extension: str = ".png",
) -> list[Path]:
    root = Path(db_dir)
    return [root / name.replace(".jpg", extension) for name in db_images]


def resolve_query_image_paths(
    q_images: list[str],
    query_dir: str | Path,
) -> list[Path]:
    root = Path(query_dir)
    return [root / name for name in q_images]


def load_query_time_of_day(query_dir: str | Path) -> dict[str, str]:
    root = Path(query_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Query directory not found: {root}")
    time_by_name: dict[str, str] = {}
    for csv_path in root.glob("*.csv"):
        with csv_path.open(newline="") as f:
            row = next(csv.reader(f))
        if len(row) < 6:
            raise ValueError(f"Unexpected query CSV format: {csv_path}")
        time_by_name[row[0]] = row[5]
    return time_by_name
