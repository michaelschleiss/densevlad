#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolbox_root="$repo_root/247code/thirdparty/vlfeat-0.9.20/toolbox"
toolbox_backup="$repo_root/247code/thirdparty/vlfeat-0.9.20/toolbox.toolbox"
external_toolbox="$repo_root/thirdparty/vlfeat/toolbox"
toolbox_dir="$toolbox_root/mex/mexa64"
toolbox_lib="$toolbox_dir/libvl.so"
external_lib="$repo_root/thirdparty/vlfeat/bin/glnxa64/libvl.so"

usage() {
  cat <<'EOF'
Usage:
  scripts/switch_vlfeat_matlab.sh use-external
  scripts/switch_vlfeat_matlab.sh use-toolbox

Notes:
  - use-external swaps the MATLAB toolbox libvl.so to the optimized external build.
  - use-toolbox restores the original toolbox libvl.so (safe default).
EOF
}

ensure_paths() {
  if [[ ! -d "$toolbox_root" && ! -L "$toolbox_root" ]]; then
    echo "ERROR: toolbox directory not found: $toolbox_root" >&2
    exit 1
  fi
  if [[ ! -d "$external_toolbox" ]]; then
    echo "ERROR: external toolbox not found: $external_toolbox" >&2
    exit 1
  fi
  if [[ ! -f "$external_lib" ]]; then
    echo "ERROR: external libvl.so not found: $external_lib" >&2
    exit 1
  fi
}

use_external() {
  ensure_paths
  if [[ -L "$toolbox_root" ]]; then
    rm -f "$toolbox_root"
  elif [[ -d "$toolbox_root" ]]; then
    if [[ ! -d "$toolbox_backup" ]]; then
      mv "$toolbox_root" "$toolbox_backup"
    else
      rm -rf "$toolbox_root"
    fi
  fi
  ln -s "$external_toolbox" "$toolbox_root"
  echo "Switched MATLAB toolbox to external VLFeat toolbox:"
  echo "  $toolbox_root -> $external_toolbox"
}

use_toolbox() {
  if [[ -L "$toolbox_root" ]]; then
    rm -f "$toolbox_root"
  fi
  if [[ -d "$toolbox_backup" ]]; then
    mv "$toolbox_backup" "$toolbox_root"
    echo "Restored toolbox directory from backup:"
    echo "  $toolbox_root"
  elif [[ -d "$toolbox_root" ]]; then
    echo "Toolbox directory already in place."
  else
    echo "ERROR: No toolbox directory or backup found to restore." >&2
    exit 1
  fi
}

case "${1:-}" in
  use-external)
    use_external
    ;;
  use-toolbox)
    use_toolbox
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "ERROR: Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
