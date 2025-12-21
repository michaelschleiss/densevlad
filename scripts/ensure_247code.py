#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def main() -> int:
    _add_repo_src()
    from densevlad.torii15.assets import ensure_local_247code

    try:
        local_root = ensure_local_247code(download=True)
    except Exception as exc:
        print(f"Failed to ensure 247code assets: {exc}")
        return 1
    print(f"247code assets ready at: {local_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
