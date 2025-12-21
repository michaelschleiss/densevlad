from __future__ import annotations

from pathlib import Path

import pytest

from densevlad.torii15.assets import ensure_local_247code
from densevlad.torii15.tokyo247 import Tokyo247Paths


def pytest_sessionstart(session) -> None:
    issues, missing = _collect_setup_issues()
    session.config._densevlad_setup_issues = issues
    session.config._densevlad_missing = missing


def _collect_setup_issues() -> tuple[list[str], dict[str, bool]]:
    issues: list[str] = []
    missing = {
        "247code": False,
        "matlab": False,
        "tokyo247": False,
        "tokyo247_db": False,
    }

    try:
        ensure_local_247code(download=False)
    except Exception as exc:
        missing["247code"] = True
        issues.append(
            "Missing 247code assets:\n"
            f"  {exc}\n"
            "Download them with:\n"
            "  python scripts/download_assets.py"
        )

    paths = Tokyo247Paths.default()
    tokyo_missing: list[str] = []
    if not paths.dbstruct_path.is_file():
        tokyo_missing.append(f"  Missing dbStruct: {paths.dbstruct_path}")
    if not paths.query_dir.is_dir():
        tokyo_missing.append(f"  Missing query dir: {paths.query_dir}")
    if tokyo_missing:
        missing["tokyo247"] = True
        tokyo_lines = "\n".join(tokyo_missing)
        issues.append(
            "Missing Tokyo247 query assets:\n"
            f"{tokyo_lines}\n"
            "Download them with:\n"
            "  python scripts/download_assets.py"
        )

    tokyo_list = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "torii15"
        / "matlab_dump"
        / "tokyo247_golden_list.txt"
    )
    if tokyo_list.is_file():
        missing_imgs = []
        for line in tokyo_list.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            kind, rel = parts
            if kind == "db":
                img_path = paths.db_dir / rel
            elif kind == "query":
                img_path = paths.query_dir / rel
            else:
                continue
            if not img_path.is_file():
                missing_imgs.append(img_path)
        if missing_imgs:
            missing["tokyo247_db"] = True
            missing_lines = "\n".join(f"  {p}" for p in missing_imgs)
            issues.append(
                "Missing Tokyo247 images referenced by golden list:\n"
                f"{missing_lines}\n"
                "Download them with:\n"
                "  python scripts/download_assets.py"
            )
    base = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "torii15"
        / "matlab_dump"
    )
    densevlad_dump = base / "densevlad_dump.mat"
    tokyo_mat = base / "tokyo247_golden.mat"
    tokyo_list = base / "tokyo247_golden_list.txt"
    missing_dumps = [p for p in (densevlad_dump, tokyo_mat, tokyo_list) if not p.exists()]
    if missing_dumps:
        missing["matlab"] = True
        missing_lines = "\n".join(f"  {p}" for p in missing_dumps)
        issues.append(
            "Missing MATLAB assets:\n"
            f"{missing_lines}\n"
            "Generate with:\n"
            "  matlab -batch \"run('scripts/matlab/dump_densevlad_all.m'); dump_densevlad_all('all')\""
        )

    return issues, missing


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    issues = getattr(config, "_densevlad_setup_issues", [])
    if not issues:
        return
    terminalreporter.write_sep("-", "DenseVLAD setup issues")
    for issue in issues:
        terminalreporter.write_line(issue)


def pytest_collection_modifyitems(config, items) -> None:
    return
