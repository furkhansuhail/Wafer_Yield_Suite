#!/usr/bin/env python3
"""
seed_data.py — put the datasets on disk, in the exact place the models read from.

Why this exists
---------------
The dashboard's Download buttons call the same code this script does, but they run
in a background thread and write into the platform *workspace*. In Docker that
workspace is the /data mount — now a HOST bind mount (./workspace by default), so
whatever this writes shows up on your disk and every model picks it up from there
with no extra wiring.

What it does
------------
  • SECOM   — downloads from UCI (public, no credentials) and writes
              <workspace>/data/secom/secom.data (+ secom_labels.data).
  • WM-811K — writes <workspace>/data/wm811k/LSWMD.pkl, either by:
                --wm811k-file PATH   register a pickle you already have, or
                --kaggle             auto-download via the Kaggle API
                                     (needs KAGGLE_USERNAME/KAGGLE_KEY or
                                      ~/.kaggle/kaggle.json).

Both are idempotent: if the file is already on disk it is reused, never re-fetched.

Usage
-----
  # inside the running container (recommended):
  docker compose exec dashboard python seed_data.py --secom
  docker compose exec dashboard python seed_data.py --wm811k-file /data/incoming/LSWMD.pkl
  docker compose exec dashboard python seed_data.py --secom --kaggle      # both

  # locally (writes to wafer_mcp_platform/_workspace unless WAFER_PLATFORM_WORKSPACE is set):
  python seed_data.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `platform_config` and the `shared` package importable no matter the cwd.
PLATFORM_ROOT = Path(__file__).resolve().parent
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import platform_config as cfg          # noqa: E402
from shared import data_registry as reg  # noqa: E402


def _fmt(path: Path) -> str:
    if path.exists():
        if path.is_file():
            return f"  ✅ {path}  ({path.stat().st_size / 1e6:.1f} MB)"
        files = list(path.glob("*"))
        size = sum(f.stat().st_size for f in files if f.is_file())
        return f"  ✅ {path}/  ({len(files)} files, {size / 1e6:.1f} MB)"
    return f"  ❌ {path}  (not present)"


def seed_secom() -> bool:
    print("→ SECOM: downloading from UCI (no credentials needed)…")
    try:
        reg.ensure_secom()
        print("  done.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("  (Most common cause: the container/host has no outbound network "
              "to archive.ics.uci.edu.)")
        return False


def seed_wm811k(from_file: str | None, allow_kaggle: bool) -> bool:
    print("→ WM-811K: ensuring LSWMD.pkl on disk…")
    if not from_file and not allow_kaggle:
        print("  skipped — pass --wm811k-file PATH to register a copy, or --kaggle "
              "to auto-download.")
        print(f"  Or just drop the pickle at: {cfg.WM811K_PICKLE}")
        return False
    try:
        reg.ensure_wm811k(from_file=from_file, allow_download=allow_kaggle)
        print("  done.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed datasets onto disk for the platform.")
    ap.add_argument("--secom", action="store_true", help="Download SECOM (UCI).")
    ap.add_argument("--wm811k-file", metavar="PATH",
                    help="Register an existing LSWMD.pkl into the workspace.")
    ap.add_argument("--kaggle", action="store_true",
                    help="Allow WM-811K auto-download via the Kaggle API.")
    ap.add_argument("--all", action="store_true",
                    help="Seed everything possible (SECOM + WM-811K via --kaggle).")
    args = ap.parse_args(argv)

    do_secom = args.secom or args.all
    do_wm = bool(args.wm811k_file) or args.kaggle or args.all
    if not (do_secom or do_wm):
        ap.error("nothing to do — pass --secom, --wm811k-file PATH, --kaggle, or --all")

    cfg.ensure_dirs()
    print(f"Workspace on disk: {cfg.WORKSPACE}\n")

    ok = True
    if do_secom:
        ok &= seed_secom()
    if do_wm:
        ok &= seed_wm811k(args.wm811k_file, args.kaggle or args.all)

    print("\nOn-disk dataset locations:")
    print(_fmt(cfg.SECOM_DIR))
    print(_fmt(cfg.WM811K_PICKLE))
    print(f"\nManifest (download-once ledger): {cfg.DATA_MANIFEST}")
    print("The models read straight from these paths — nothing else to wire up.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
