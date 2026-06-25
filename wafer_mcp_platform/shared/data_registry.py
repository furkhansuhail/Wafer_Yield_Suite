"""
shared/data_registry.py
=======================

The download-ONCE data layer.

Problem this solves
-------------------
Before the platform, two of the three projects each downloaded the same ~1.5 GB
WM-811K pickle on their own, and SECOM was fetched independently again. The
registry centralises that: there is exactly one canonical `LSWMD.pkl` and one
SECOM cache, both recorded in a small JSON manifest so repeated calls are no-ops.

Public API
----------
    ensure_wm811k(from_file=None)  -> Path     # the single shared LSWMD.pkl
    ensure_secom()                 -> object    # cached SECOM (X, y, timestamps)
    dataset_status()               -> dict      # for the dashboard / list_models
    wafer_examples(n)              -> list[2-D map]   # example data to send to models
    secom_examples(n)              -> DataFrame       # example rows to send to models
    yield_area_examples()          -> list[float]     # example die areas

Both WM-811K consumers (the CNN classifier and the yield-curve models) are meant
to call `ensure_wm811k()` and read the returned path, instead of running their
own downloaders. `register_path()` lets you point the registry at a copy you
already have on disk (the manual `--from-file` flow), so nothing re-downloads.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make the platform package importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import platform_config as cfg  # noqa: E402

logger = logging.getLogger("data_registry")

_PARTIAL_FLOOR_BYTES = 50 * 1024 * 1024  # guard against truncated WM-811K copies


# --------------------------------------------------------------------------- #
# Manifest ledger
# --------------------------------------------------------------------------- #
def _read_manifest() -> dict:
    if cfg.DATA_MANIFEST.exists():
        try:
            return json.loads(cfg.DATA_MANIFEST.read_text())
        except json.JSONDecodeError:
            logger.warning("Manifest corrupt; starting a fresh one.")
    return {"datasets": {}}


def _write_manifest(man: dict) -> None:
    cfg.DATA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    cfg.DATA_MANIFEST.write_text(json.dumps(man, indent=2, default=str))


def _record(name: str, path: Path, **extra) -> None:
    man = _read_manifest()
    man["datasets"][name] = {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else None,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    _write_manifest(man)


def _quick_fingerprint(path: Path) -> str:
    """Cheap content fingerprint: size + first/last 1 MB. Avoids hashing 1.5 GB."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)
            h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# WM-811K  (shared by the CNN classifier AND the yield-curve models)
# --------------------------------------------------------------------------- #
def register_path(src: str | Path) -> Path:
    """Copy/move a WM-811K pickle you already have into the canonical location.

    This is the offline, download-nothing path. After this, both WM-811K
    projects find the data via `ensure_wm811k()` and never hit the network.
    """
    src = Path(src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"No file at {src}")
    if src.stat().st_size < _PARTIAL_FLOOR_BYTES:
        logger.warning(
            "Registered file is only %.1f MB — smaller than a real WM-811K dump; "
            "continuing anyway (fine for synthetic/test pickles).",
            src.stat().st_size / 1e6,
        )
    cfg.WM811K_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    if src != cfg.WM811K_PICKLE:
        shutil.copy2(src, cfg.WM811K_PICKLE)
    _record("wm811k", cfg.WM811K_PICKLE, source=str(src),
            fingerprint=_quick_fingerprint(cfg.WM811K_PICKLE))
    logger.info("WM-811K registered -> %s", cfg.WM811K_PICKLE)
    return cfg.WM811K_PICKLE


def ensure_wm811k(from_file: Optional[str | Path] = None,
                  allow_download: bool = True) -> Path:
    """Return the path to the single shared LSWMD.pkl, fetching it once if needed.

    Resolution order:
      1. already cached at the canonical path  -> return it (the common case)
      2. `from_file` provided                  -> register that copy
      3. `allow_download` and Kaggle available -> delegate to the CNN downloader,
         which knows the Kaggle dataset slug and verification logic, then move the
         result into the canonical path.
    Raises with actionable guidance if none of those produce a file.
    """
    cfg.ensure_dirs()

    if cfg.WM811K_PICKLE.exists() and cfg.WM811K_PICKLE.stat().st_size > 0:
        logger.info("WM-811K already cached (%.0f MB) -> reuse, no download.",
                    cfg.WM811K_PICKLE.stat().st_size / 1e6)
        return cfg.WM811K_PICKLE

    if from_file is not None:
        return register_path(from_file)

    if allow_download:
        fetched = _download_wm811k_via_cnn_downloader()
        if fetched is not None:
            return register_path(fetched)

    raise FileNotFoundError(
        "WM-811K (LSWMD.pkl) is not available yet.\n"
        f"  Expected at: {cfg.WM811K_PICKLE}\n"
        "  Fix it once, then every model that needs wafer maps reuses it:\n"
        "    • have the file already?  data_registry.register_path('/path/LSWMD.pkl')\n"
        "    • want to download it?     set up ~/.kaggle/kaggle.json and retry\n"
    )


def _download_wm811k_via_cnn_downloader() -> Optional[Path]:
    """Reuse the CNN project's Kaggle downloader, into a temp dir.

    Raises a RuntimeError carrying the *real* reason on failure (kaggle CLI not
    installed, missing credentials, bad slug, verify failed) so the caller — and
    the dashboard's job log — can show something actionable instead of a generic
    'not available'.
    """
    try:
        sys.path.insert(0, str(cfg.CNN_PROJECT / "src"))
        import importlib
        dl = importlib.import_module("track1_downloader")
    except Exception as exc:
        raise RuntimeError(
            f"WM-811K auto-download unavailable: could not import the CNN downloader ({exc})."
        ) from exc

    raw_dir = cfg.WM811K_DIR / "_raw_download"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        # The downloader exposes a CLI-style main; it raises with a clear message
        # if the kaggle CLI/credentials are missing or the download fails.
        dl.main(["--raw-dir", str(raw_dir)])  # type: ignore[attr-defined]
    except SystemExit:
        pass  # argparse/`main` may sys.exit; the file check below is the source of truth

    candidate = raw_dir / "LSWMD.pkl"
    if candidate.exists():
        return candidate
    raise RuntimeError(
        "Kaggle download did not produce LSWMD.pkl. The most common causes inside "
        "a container are that the `kaggle` package isn't installed or API "
        "credentials (~/.kaggle/kaggle.json) aren't available. Either enable the "
        "Kaggle path (install kaggle + mount creds), or register a pickle you "
        "downloaded manually via register_path()/the register_wm811k tool."
    )


# --------------------------------------------------------------------------- #
# SECOM  (tabular sensor data; independent domain)
# --------------------------------------------------------------------------- #
def ensure_secom():
    """Load SECOM through its own cached loader; cache lives under DATA_ROOT/secom.

    Returns the project's SecomData object (X, y, timestamps). The loader already
    caches raw files, so this is download-once by construction.
    """
    cfg.ensure_dirs()
    # PROJECTS_ROOT (suite root) first: the SECOM package uses absolute
    # `yield_excursion_fail_prediction.src.*` imports, which need the project's
    # parent on sys.path. Then the project dir + its src for the bare-import path.
    sys.path.insert(0, str(cfg.PROJECTS_ROOT))
    sys.path.insert(0, str(cfg.SECOM_PROJECT))
    sys.path.insert(0, str(cfg.SECOM_PROJECT / "src"))
    try:
        from data_downloader import load_secom  # type: ignore
    except Exception:
        from yield_excursion_fail_prediction.src.data_downloader import load_secom  # type: ignore

    data = load_secom(data_dir=str(cfg.SECOM_DIR))
    _record("secom", cfg.SECOM_DIR, n_samples=int(getattr(data, "X", []).shape[0])
            if hasattr(data, "X") else None)
    return data


# --------------------------------------------------------------------------- #
# Status + example-data helpers (the dashboard sends these to the models)
# --------------------------------------------------------------------------- #
def dataset_status() -> dict:
    man = _read_manifest()
    return {
        "wm811k": {
            "cached": cfg.WM811K_PICKLE.exists(),
            "path": str(cfg.WM811K_PICKLE),
            "bytes": cfg.WM811K_PICKLE.stat().st_size if cfg.WM811K_PICKLE.exists() else 0,
            "shared_by": [cfg.DOMAIN_WAFER_CNN, cfg.DOMAIN_YIELD],
        },
        "secom": {
            "cached": any(cfg.SECOM_DIR.glob("*.data")),
            "path": str(cfg.SECOM_DIR),
            "shared_by": [cfg.DOMAIN_SECOM],
        },
        "manifest": man.get("datasets", {}),
    }


def wafer_examples(n: int = 4):
    """A few real wafer maps to feed the CNN / yield models. Falls back to
    synthetic maps when the dataset is not present yet, so the dashboard is
    always demonstrable."""
    try:
        path = ensure_wm811k(allow_download=False)
        import pandas as pd
        df = pd.read_pickle(path)
        maps = [df["waferMap"].iloc[i] for i in range(min(n, len(df)))]
        return [list(map(lambda r: list(map(int, r)), m)) for m in maps]
    except Exception:
        return _synthetic_wafer_maps(n)


def _synthetic_wafer_maps(n: int = 4):
    """Categorical {0:no-die,1:good,2:bad} maps with simple defect patterns."""
    import numpy as np
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        size = 26
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
        m = np.where(r <= size / 2, 1, 0).astype(int)
        if k % 3 == 0:          # center blob
            m[r < 4] = 2
        elif k % 3 == 1:        # edge ring
            m[(r > size / 2 - 3) & (r <= size / 2)] = 2
        else:                   # random
            m[(m == 1) & (rng.random((size, size)) < 0.08)] = 2
        out.append(m.tolist())
    return out


def secom_examples(n: int = 5):
    """A few SECOM rows (real if cached, else synthetic 590-wide rows)."""
    try:
        data = ensure_secom()
        X = data.X if hasattr(data, "X") else data[0]
        return X.head(n)
    except Exception:
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(11)
        cols = [f"feature_{i:03d}" for i in range(590)]
        return pd.DataFrame(rng.normal(size=(n, 590)), columns=cols)


def yield_area_examples():
    """Example die areas (unit-die areas) to ask the yield models about."""
    return [1.0, 4.0, 9.0, 16.0, 25.0]
