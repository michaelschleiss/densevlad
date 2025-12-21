#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download required DenseVLAD assets.")
    parser.add_argument(
        "--tokyo-db",
        choices=("none", "minimal", "full"),
        default="minimal",
        help="Tokyo247 DB download size (none, minimal=single tar, full=all tars).",
    )
    parser.add_argument(
        "--tokyo-queries",
        choices=("none", "subset", "full"),
        default="subset",
        help="Tokyo247 query set to download (none, subset, full).",
    )
    args = parser.parse_args()

    _add_repo_src()
    from densevlad.torii15.assets import ensure_local_247code
    from densevlad.torii15.tokyo247 import (
        TOKYO247_DB_TARS,
        TOKYO247_DB_TARS_MIN,
        TOKYO247_QUERY_ZIP_URLS,
        TOKYO247_DB_BASE_URL,
        ensure_tokyo247_assets,
    )

    print("Ensuring 247code assets...")
    try:
        local_root = ensure_local_247code(download=True)
    except Exception as exc:
        print(f"Failed to ensure 247code assets: {exc}")
        return 1
    print(f"247code assets ready at: {local_root}")

    print(
        "Ensuring Tokyo247 assets "
        f"(db={args.tokyo_db}, queries={args.tokyo_queries})..."
    )
    if args.tokyo_queries == "none":
        print("Skipping Tokyo247 queries.")
    else:
        print(f"Query URL: {TOKYO247_QUERY_ZIP_URLS[args.tokyo_queries]}")
    if args.tokyo_db == "none":
        print("Skipping Tokyo247 DB images.")
    elif args.tokyo_db == "minimal":
        print("DB mode: minimal (uses golden list if present, otherwise smallest tar).")
    elif args.tokyo_db == "full":
        print(f"DB tars (full): {TOKYO247_DB_TARS[0]} .. {TOKYO247_DB_TARS[-1]}")
        print(f"DB base URL: {TOKYO247_DB_BASE_URL}")
    try:
        include_db = args.tokyo_db != "none"
        paths = ensure_tokyo247_assets(
            download=True,
            include_db=include_db,
            db_mode=args.tokyo_db if include_db else "full",
            query_mode=args.tokyo_queries,
            verbose=True,
        )
    except Exception as exc:
        print(f"Failed to ensure Tokyo247 assets: {exc}")
        return 1
    print(f"Tokyo247 dbStruct: {paths.dbstruct_path}")
    print(f"Tokyo247 query dir: {paths.query_dir}")
    if args.tokyo_db != "none":
        print(f"Tokyo247 db dir: {paths.db_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
