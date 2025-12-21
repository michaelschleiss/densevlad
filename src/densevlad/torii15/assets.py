from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

TORII15_CODE_ZIP_URL = "http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        "Could not locate repo root (pyproject.toml). "
        "This project now requires repo-local 247code assets."
    )


def _download_zip(url: str, dest: Path) -> None:
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


def _extract_zip(zip_path: Path, repo_root: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(repo_root)
    try:
        zip_path.unlink()
    except Exception:
        pass


def ensure_local_247code(*, download: bool = False) -> Path:
    repo_root = _find_repo_root()
    local_root = repo_root / "247code"
    expected_vocab = local_root / "data" / "dnscnt_RDSIFT_K128.mat"
    if expected_vocab.is_file():
        return local_root
    if local_root.is_dir():
        raise FileNotFoundError(
            "247code is present in the repo but missing expected data files. "
            f"Expected: {expected_vocab}"
        )
    zip_path = repo_root / "247code.zip"
    if not zip_path.is_file():
        if not download:
            raise FileNotFoundError(
                "Missing repo-local 247code assets. Expected directory: "
                f"{local_root}. Download/extract 247code.zip to the repo root."
            )
        _download_zip(TORII15_CODE_ZIP_URL, zip_path)
    if not local_root.exists():
        if not download:
            raise FileNotFoundError(
                "Found 247code.zip in the repo but it is not extracted. "
                f"Extract it so {local_root} exists."
            )
        _extract_zip(zip_path, repo_root)
    if not expected_vocab.is_file():
        raise FileNotFoundError(
            "Downloaded/extracted 247code but expected data files are still missing. "
            f"Expected: {expected_vocab}"
        )
    return local_root


def repo_assets_dir() -> Path:
    return _find_repo_root() / "assets"


@dataclass(frozen=True)
class Torii15Assets:
    cache_dir: Path

    @staticmethod
    def default_cache_dir() -> Path:
        return repo_assets_dir() / "torii15"

    @classmethod
    def default(cls) -> "Torii15Assets":
        return cls(cache_dir=cls.default_cache_dir())

    def zip_path(self) -> Path:
        return _find_repo_root() / "247code.zip"

    def ensure_zip(self) -> Path:
        zp = self.zip_path()
        if not (zp.exists() and zp.stat().st_size > 0):
            raise FileNotFoundError(
                "247code.zip not found in the repo root. "
                "Download it (and extract) before running."
            )
        return zp

    def extract_member(self, member: str) -> Path:
        local_root = ensure_local_247code(download=False)
        out_path = _find_repo_root() / member
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        raise FileNotFoundError(
            "Requested 247code member is missing from repo-local assets: "
            f"{out_path}. Ensure {local_root} contains the original 247code tree."
        )

    def vocab_mat_path(self) -> Path:
        return self.extract_member("247code/data/dnscnt_RDSIFT_K128.mat")

    def pca_mat_path(self) -> Path:
        return self.extract_member("247code/data/dnscnt_RDSIFT_K128_vlad_pcaproj.mat")

    def example_pre_pca_vlad_path(self) -> Path:
        return self.extract_member(
            "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.dict_grid.dnsvlad.mat"
        )

    def describe(self) -> dict[str, object]:
        repo_root = _find_repo_root()
        local_root = repo_root / "247code"
        zp = self.zip_path()
        info: dict[str, object] = {
            "cache_dir": str(self.cache_dir),
            "repo_root": str(repo_root),
            "local_247code_dir": str(local_root),
            "local_247code_exists": local_root.exists(),
            "zip_path": str(zp),
            "zip_exists": zp.exists(),
        }
        if zp.exists():
            info["zip_size_bytes"] = zp.stat().st_size
            info["zip_sha256"] = _sha256(zp)
        return info
