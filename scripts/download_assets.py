#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import tarfile
import urllib.request
import zipfile


TOKYO247_DBSTRUCT_URL = (
    "https://raw.githubusercontent.com/devanshigarg01/pittsburghdata/main/tokyo247.mat"
)
TOKYO247_QUERY_ZIP_URL = (
    "https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/queries/247query_subset_v2.zip"
)
TOKYO247_DB_BASE_URL = (
    "https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/database_gsv_vga/"
)
TOKYO247_DB_TARS = [f"{idx:05d}.tar" for idx in range(3814, 3830)]

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = REPO_ROOT / "assets" / "torii15"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {tar_path} -> {dest_dir}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir)
    tar_path.unlink(missing_ok=True)


def ensure_247code() -> None:
    """Download 247code and extract the full archive."""
    root = REPO_ROOT / "247code"
    data_dir = root / "data"
    thirdparty_dir = root / "thirdparty"
    code_dir = root / "code"

    # Check if we need to download
    if data_dir.is_dir() and thirdparty_dir.is_dir() and code_dir.is_dir():
        print(f"247code already present: {root}")
        return

    zip_path = REPO_ROOT / "247code.zip"
    if not zip_path.exists():
        _download("http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip", zip_path)

    # Extract full archive (code/, data/, thirdparty/, etc.)
    print(f"Extracting 247code from {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(REPO_ROOT)
    # Make the extracted tree read-only to avoid accidental edits.
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(path.stat().st_mode & ~0o222)
        except PermissionError:
            continue
    zip_path.unlink(missing_ok=True)
    print(f"247code ready: {root}")


def ensure_tokyo247_dbstruct() -> Path:
    tokyo_root = ASSETS_ROOT / "tokyo247"
    dbstruct = tokyo_root / "tokyo247.mat"
    if not dbstruct.exists():
        _download(TOKYO247_DBSTRUCT_URL, dbstruct)
    else:
        print(f"tokyo247.mat already present: {dbstruct}")
    return dbstruct


def ensure_tokyo247_queries() -> Path:
    queries_root = ASSETS_ROOT / "247query_subset_v2"
    if queries_root.is_dir():
        print(f"Tokyo247 queries already present: {queries_root}")
        return queries_root
    zip_path = ASSETS_ROOT / "247query_subset_v2.zip"
    if not zip_path.exists():
        _download(TOKYO247_QUERY_ZIP_URL, zip_path)
    _extract_zip(zip_path, ASSETS_ROOT)
    if not queries_root.is_dir():
        raise FileNotFoundError(f"Expected queries dir after extract: {queries_root}")
    return queries_root


def ensure_tokyo247_db() -> Path:
    db_dir = ASSETS_ROOT / "tokyo247" / "database_gsv_vga"
    db_dir.mkdir(parents=True, exist_ok=True)
    for name in TOKYO247_DB_TARS:
        tar_path = db_dir / name
        if not tar_path.exists():
            _download(f"{TOKYO247_DB_BASE_URL}{name}", tar_path)
        _extract_tar(tar_path, db_dir)
    return db_dir


def main() -> int:
    ensure_247code()
    ensure_tokyo247_dbstruct()
    ensure_tokyo247_queries()
    ensure_tokyo247_db()
    print("All assets downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
