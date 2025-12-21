from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import tarfile
import urllib.request
import zipfile

import numpy as np

from .assets import Torii15Assets
from .matio import load_mat_v5

TOKYO247_DBSTRUCT_URL = (
    "https://raw.githubusercontent.com/devanshigarg01/pittsburghdata/main/tokyo247.mat"
)
TOKYO247_QUERY_ZIP_URLS = {
    "subset": "https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/queries/247query_subset_v2.zip",
    "full": "https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/queries/247query_v3.zip",
}
TOKYO247_DB_BASE_URL = (
    "https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/database_gsv_vga/"
)
TOKYO247_DB_TARS = [f"{idx:05d}.tar" for idx in range(3814, 3830)]
TOKYO247_DB_TARS_MIN = ["03829.tar"]


@dataclass(frozen=True)
class Tokyo247Paths:
    root_dir: Path
    db_dir: Path
    query_dir: Path
    dbstruct_path: Path

    @classmethod
    def default(cls) -> "Tokyo247Paths":
        cache_dir = Torii15Assets.default_cache_dir()
        root_dir = cache_dir / "tokyo247"
        db_dir = root_dir / "database_gsv_vga"

        candidate = root_dir / "queries"
        if candidate.is_dir():
            query_dir = candidate
        else:
            query_dir = cache_dir / "247query_subset_v2"

        dbstruct_path = root_dir / "tokyo247.mat"

        return cls(
            root_dir=root_dir,
            db_dir=db_dir,
            query_dir=query_dir,
            dbstruct_path=dbstruct_path,
        )


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir)


def _minimal_db_tars(cache_dir: Path) -> list[str]:
    golden_list = cache_dir / "matlab_dump" / "tokyo247_golden_list.txt"
    if golden_list.is_file():
        prefixes = set()
        for line in golden_list.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2 or parts[0] != "db":
                continue
            rel = parts[1]
            prefix = rel.split("/", 1)[0]
            if prefix.isdigit() and len(prefix) == 5:
                prefixes.add(prefix)
        if prefixes:
            return sorted(f"{p}.tar" for p in prefixes)
    return TOKYO247_DB_TARS_MIN


def ensure_tokyo247_assets(
    *,
    download: bool = False,
    include_db: bool = True,
    db_mode: str = "full",
    query_mode: str = "subset",
    verbose: bool = False,
) -> Tokyo247Paths:
    paths = Tokyo247Paths.default()
    cache_dir = Torii15Assets.default_cache_dir()

    if not paths.dbstruct_path.exists():
        if not download:
            raise FileNotFoundError(f"Missing tokyo247 dbStruct: {paths.dbstruct_path}")
        if verbose:
            print(f"Downloading dbStruct -> {paths.dbstruct_path}")
        _download_file(TOKYO247_DBSTRUCT_URL, paths.dbstruct_path)

    if query_mode not in ("subset", "full", "none"):
        raise ValueError(f"Unknown query_mode: {query_mode}")
    if query_mode != "none":
        if query_mode == "subset":
            query_root = cache_dir / "247query_subset_v2"
            zip_name = "247query_subset_v2.zip"
        else:
            query_root = cache_dir / "queries"
            zip_name = "247query_v3.zip"
        if not query_root.is_dir():
            if not download:
                raise FileNotFoundError(f"Missing Tokyo247 queries: {query_root}")
            zip_path = cache_dir / zip_name
            if not zip_path.exists():
                if verbose:
                    print(f"Downloading queries ({query_mode}) -> {zip_path}")
                _download_file(TOKYO247_QUERY_ZIP_URLS[query_mode], zip_path)
            if verbose:
                print(f"Extracting {zip_path} -> {cache_dir}")
            _extract_zip(zip_path, cache_dir)
            try:
                zip_path.unlink()
            except Exception:
                pass
            paths = Tokyo247Paths.default()
            query_root = paths.query_dir
            if not query_root.is_dir():
                raise FileNotFoundError(
                    "Tokyo247 queries extracted but expected directory is missing: "
                    f"{query_root}"
                )

    if include_db:
        if db_mode not in ("full", "minimal"):
            raise ValueError(f"Unknown db_mode: {db_mode}")
        if db_mode == "full":
            db_tars = TOKYO247_DB_TARS
        else:
            db_tars = _minimal_db_tars(cache_dir)
        if verbose:
            if db_mode == "full":
                print(f"DB tars (full): {db_tars[0]} .. {db_tars[-1]}")
            else:
                print(f"DB tars (minimal): {', '.join(db_tars)}")
        paths.db_dir.mkdir(parents=True, exist_ok=True)
        for name in db_tars:
            tar_path = paths.db_dir / name
            if not tar_path.exists():
                if not download:
                    raise FileNotFoundError(f"Missing Tokyo247 DB archive: {tar_path}")
                if verbose:
                    print(f"Downloading DB tar -> {tar_path}")
                _download_file(f"{TOKYO247_DB_BASE_URL}{name}", tar_path)
            extract_marker = paths.db_dir / name.replace(".tar", "")
            if not extract_marker.exists():
                if verbose:
                    print(f"Extracting {tar_path} -> {paths.db_dir}")
                _extract_tar(tar_path, paths.db_dir)
            try:
                tar_path.unlink()
            except Exception:
                pass

    return paths


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
