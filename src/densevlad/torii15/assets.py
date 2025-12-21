from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

TORII15_CODE_ZIP_URL = "http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dest)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


@dataclass(frozen=True)
class Torii15Assets:
    cache_dir: Path

    @staticmethod
    def default_cache_dir() -> Path:
        return Path(user_cache_dir("densevlad")) / "torii15"

    @classmethod
    def default(cls) -> "Torii15Assets":
        return cls(cache_dir=cls.default_cache_dir())

    def zip_path(self) -> Path:
        return self.cache_dir / "247code.zip"

    def ensure_zip(self) -> Path:
        zp = self.zip_path()
        if zp.exists() and zp.stat().st_size > 0:
            return zp
        _download(TORII15_CODE_ZIP_URL, zp)
        return zp

    def extract_member(self, member: str) -> Path:
        zp = self.ensure_zip()
        out_path = self.cache_dir / member
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            with zf.open(member) as src, out_path.open("wb") as dst:
                dst.write(src.read())
        return out_path

    def vocab_mat_path(self) -> Path:
        return self.extract_member("247code/data/dnscnt_RDSIFT_K128.mat")

    def pca_mat_path(self) -> Path:
        return self.extract_member("247code/data/dnscnt_RDSIFT_K128_vlad_pcaproj.mat")

    def example_pre_pca_vlad_path(self) -> Path:
        return self.extract_member(
            "247code/data/example_gsv/L-NLvGeZ6JHX6JO8Xnf_BA_012_000.dict_grid.dnsvlad.mat"
        )

    def describe(self) -> dict[str, object]:
        zp = self.zip_path()
        info: dict[str, object] = {
            "cache_dir": str(self.cache_dir),
            "zip_path": str(zp),
            "zip_exists": zp.exists(),
        }
        if zp.exists():
            info["zip_size_bytes"] = zp.stat().st_size
            info["zip_sha256"] = _sha256(zp)
        return info

