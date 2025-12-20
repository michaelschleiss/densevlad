from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _rpaths_for(path: Path) -> set[str]:
    output = subprocess.check_output(["otool", "-l", str(path)], text=True)
    rpaths: set[str] = set()
    lines = output.splitlines()
    for idx, line in enumerate(lines):
        if "LC_RPATH" not in line:
            continue
        for offset in range(idx + 1, min(idx + 4, len(lines))):
            candidate = lines[offset].strip()
            if candidate.startswith("path "):
                rpaths.add(candidate.split("path ", 1)[1].split(" (", 1)[0])
                break
    return rpaths


def _needs_change(path: Path) -> bool:
    output = subprocess.check_output(["otool", "-L", str(path)], text=True)
    return "@loader_path/libvl.dylib" in output


def main() -> int:
    if sys.platform != "darwin":
        return 0

    vlfeat_src = os.environ.get("VLFEAT_SRC")
    if not vlfeat_src:
        raise SystemExit("VLFEAT_SRC is not set; source .pixi/vlfeat_env.sh first.")

    libvl = Path(vlfeat_src) / "bin" / "maci64" / "libvl.dylib"
    if not libvl.exists():
        raise SystemExit(f"libvl.dylib not found at {libvl}")

    try:
        import cyvlfeat  # type: ignore
    except Exception as exc:
        raise SystemExit(f"cyvlfeat import failed: {exc}") from exc

    base = Path(cyvlfeat.__file__).resolve().parent
    so_paths = list(base.rglob("*.so"))
    if not so_paths:
        raise SystemExit(f"No cyvlfeat .so files found under {base}")

    patched = 0
    for so_path in so_paths:
        if not _needs_change(so_path):
            continue
        subprocess.check_call(
            [
                "install_name_tool",
                "-change",
                "@loader_path/libvl.dylib",
                "@rpath/libvl.dylib",
                str(so_path),
            ]
        )
        rpaths = _rpaths_for(so_path)
        if str(libvl.parent) not in rpaths:
            subprocess.check_call(
                ["install_name_tool", "-add_rpath", str(libvl.parent), str(so_path)]
            )
        patched += 1

    print(f"Patched {patched} cyvlfeat extension(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
