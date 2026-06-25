"""
Track 1 — Module 1: Data Downloader for the WM-811K wafer-map dataset.

Responsibility
--------------
Get the raw `LSWMD.pkl` file onto disk, verify it is intact and has the
expected structure, and never re-download if it is already present.

The WM-811K dataset ships as a single pickled pandas DataFrame (~1.5 GB)
with ~811,457 rows. Expected columns (note the dataset's own misspellings):
    waferMap, dieSize, lotName, waferIndex, trianTestLabel, failureType

Usage
-----
    # Download via the Kaggle API (requires ~/.kaggle/kaggle.json credentials):
    python track1_downloader.py --raw-dir data/raw

    # Register a file you downloaded manually:
    python track1_downloader.py --raw-dir data/raw --from-file ~/Downloads/LSWMD.pkl

    # Deep-verify an existing file (loads the pickle, checks columns/row count):
    python track1_downloader.py --raw-dir data/raw --verify-deep

    # Run the self-test (no network, no big download needed):
    python track1_downloader.py --self-test

Testable individually
---------------------
`--self-test` fabricates a tiny fake LSWMD-shaped pickle in a temp dir and
runs the full verify logic against it, so you can confirm this module works
before committing to the real 1.5 GB download.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration (overridable via CLI; kept here so the module is standalone)
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = "qingyi/wm811k-wafer-map"   # public mirror of WM-811K
PICKLE_NAME = "LSWMD.pkl"
EXPECTED_COLUMNS = {
    "waferMap",
    "dieSize",
    "lotName",
    "waferIndex",
    "trianTestLabel",   # sic — the dataset misspells "train"
    "failureType",
}
EXPECTED_MIN_ROWS = 800_000          # real set is ~811,457
MIN_PLAUSIBLE_BYTES = 100 * 1024 * 1024   # 100 MB floor; guards truncated files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("downloader")


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #

@dataclass
class VerifyResult:
    ok: bool
    path: Path
    size_bytes: int
    n_rows: int | None = None       # None when only a shallow check ran
    missing_columns: set[str] | None = None
    message: str = ""

    def __bool__(self) -> bool:      # lets callers write `if verify(...):`
        return self.ok


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def pickle_path(raw_dir: Path) -> Path:
    """Canonical location of the dataset pickle."""
    return Path(raw_dir) / PICKLE_NAME


def is_present(raw_dir: Path) -> bool:
    """Idempotency guard: is a plausibly-complete pickle already on disk?"""
    p = pickle_path(raw_dir)
    return p.exists() and p.stat().st_size >= MIN_PLAUSIBLE_BYTES


def _read_legacy_pickle(path):
    """Read LSWMD.pkl, falling back to compatibility shims for the legacy
    (pandas ~0.x, Python 2) pickle: maps the removed 'pandas.indexes' module
    and Int64Index/Float64Index classes, and decodes Python-2 byte strings
    via encoding='latin1'."""
    import pickle
    import sys
    import types
    try:
        return pd.read_pickle(path)
    except (ModuleNotFoundError, AttributeError, UnicodeDecodeError):
        import pandas.core.indexes as _new_idx
        import pandas.core.indexes.base as _idx_base
        sys.modules.setdefault("pandas.indexes", _new_idx)
        sys.modules.setdefault("pandas.indexes.base", _idx_base)
        sys.modules.setdefault("pandas.core.index", _new_idx)
        _numeric = types.ModuleType("pandas_compat_numeric")
        for _n in ("Int64Index", "Float64Index", "UInt64Index"):
            setattr(_numeric, _n, getattr(pd, _n, pd.Index))
        sys.modules.setdefault("pandas.indexes.numeric", _numeric)
        sys.modules.setdefault("pandas.core.indexes.numeric", _numeric)
        try:
            from pandas.compat.pickle_compat import Unpickler as _PdUnpickler
        except Exception:
            _PdUnpickler = None
        with open(path, "rb") as fh:
            if _PdUnpickler is not None:
                try:
                    return _PdUnpickler(fh, encoding="latin1").load()
                except TypeError:
                    fh.seek(0)
            return pickle.load(fh, encoding="latin1")


def verify(raw_dir: Path, deep: bool = False,
           min_bytes: int = MIN_PLAUSIBLE_BYTES) -> VerifyResult:
    """
    Validate the pickle.

    Shallow (default): file exists and is above the size floor (`min_bytes`).
    Deep: additionally loads the DataFrame and checks columns + row count.
    Deep is slow/memory-heavy (loads ~1.5 GB) so it is opt-in.

    `min_bytes` is injectable so tests can validate the deep path on small
    fixtures without fabricating a 100 MB file.
    """
    p = pickle_path(raw_dir)
    if not p.exists():
        return VerifyResult(False, p, 0, message=f"Not found: {p}")

    size = p.stat().st_size
    if size < min_bytes:
        return VerifyResult(
            False, p, size,
            message=f"File too small ({size:,} bytes) — likely truncated.",
        )

    if not deep:
        return VerifyResult(True, p, size, message="Shallow check passed.")

    log.info("Deep verify: loading pickle (this can take a minute)...")
    try:
        df = _read_legacy_pickle(p)
    except Exception as exc:  # noqa: BLE001 - surface any unpickling failure
        return VerifyResult(False, p, size, message=f"Unpickling failed: {exc}")

    missing = EXPECTED_COLUMNS - set(df.columns)
    n_rows = len(df)
    ok = not missing and n_rows >= EXPECTED_MIN_ROWS
    msg = "Deep check passed." if ok else "Deep check failed."
    if missing:
        msg += f" Missing columns: {sorted(missing)}."
    if n_rows < EXPECTED_MIN_ROWS:
        msg += f" Only {n_rows:,} rows (expected >= {EXPECTED_MIN_ROWS:,})."
    return VerifyResult(ok, p, size, n_rows=n_rows,
                        missing_columns=missing or None, message=msg)


def register_manual_file(src: Path, raw_dir: Path) -> Path:
    """Copy a manually-downloaded pickle into the raw dir."""
    src = Path(src).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = pickle_path(raw_dir)
    log.info("Copying %s -> %s", src, dst)
    shutil.copy2(src, dst)
    return dst


def _kaggle_available() -> bool:
    """Is the kaggle CLI installed and importable?"""
    return shutil.which("kaggle") is not None


def download(raw_dir: Path, force: bool = False) -> Path:
    """
    Download the dataset via the Kaggle API and unpack the pickle.

    Requires the `kaggle` package and valid credentials at ~/.kaggle/kaggle.json.
    Idempotent: skips the download if a complete file is already present
    (unless force=True).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if is_present(raw_dir) and not force:
        log.info("Already present, skipping download: %s", pickle_path(raw_dir))
        return pickle_path(raw_dir)

    if not _kaggle_available():
        raise RuntimeError(
            "kaggle CLI not found. Install with `pip install kaggle` and place "
            "your API token at ~/.kaggle/kaggle.json. Alternatively, download "
            f"'{KAGGLE_DATASET}' manually and use --from-file."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        log.info("Downloading %s via Kaggle API...", KAGGLE_DATASET)
        # --unzip handles the archive; we still guard for nested zips below.
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(tmp_dir), "--unzip"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Kaggle download failed. Check credentials / dataset slug.\n"
                f"stderr: {result.stderr.strip()}"
            )
        _collect_pickle(tmp_dir, raw_dir)

    final = verify(raw_dir, deep=False)
    if not final:
        raise RuntimeError(f"Post-download verify failed: {final.message}")
    log.info("Download complete: %s (%.2f GB)",
             final.path, final.size_bytes / 1e9)
    return final.path


def _collect_pickle(search_dir: Path, raw_dir: Path) -> None:
    """Find LSWMD.pkl anywhere under search_dir (unzipping nested zips) and move it."""
    for zf in search_dir.rglob("*.zip"):
        log.info("Unzipping nested archive: %s", zf.name)
        with zipfile.ZipFile(zf) as z:
            z.extractall(search_dir)

    matches = list(search_dir.rglob(PICKLE_NAME))
    if not matches:
        raise FileNotFoundError(
            f"{PICKLE_NAME} not found in downloaded archive under {search_dir}"
        )
    shutil.move(str(matches[0]), str(pickle_path(raw_dir)))


# --------------------------------------------------------------------------- #
# Self-test (no network, no big download)
# --------------------------------------------------------------------------- #

def _make_fake_pickle(path: Path, n_rows: int = 5, pad_to_bytes: int = 0) -> None:
    """Create a tiny LSWMD-shaped pickle for testing the verify logic."""
    import numpy as np

    df = pd.DataFrame({
        "waferMap": [np.zeros((26, 26), dtype=np.uint8) for _ in range(n_rows)],
        "dieSize": [676] * n_rows,
        "lotName": [f"lot{i}" for i in range(n_rows)],
        "waferIndex": list(range(n_rows)),
        "trianTestLabel": [["Training"]] * n_rows,
        "failureType": [["Center"]] * n_rows,
    })
    df.to_pickle(path)
    if pad_to_bytes and path.stat().st_size < pad_to_bytes:
        with open(path, "ab") as fh:        # pad so it clears the size floor
            fh.write(b"\0" * (pad_to_bytes - path.stat().st_size))


def self_test() -> bool:
    """Exercise every code path that does not require Kaggle. Returns True if all pass."""
    print("=" * 60)
    print("SELF-TEST: track1_downloader")
    print("=" * 60)
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        raw.mkdir()

        # 1. is_present on empty dir -> False
        t1 = not is_present(raw)
        print(f"[{'PASS' if t1 else 'FAIL'}] is_present() False on empty dir")
        passed &= t1

        # 2. verify on missing file -> not ok
        t2 = not verify(raw)
        print(f"[{'PASS' if t2 else 'FAIL'}] verify() fails when file absent")
        passed &= t2

        # 3. truncated file (below size floor) -> shallow verify fails
        small = pickle_path(raw)
        _make_fake_pickle(small, n_rows=3, pad_to_bytes=0)
        t3 = not verify(raw)  # tiny pickle is under the 100 MB floor
        print(f"[{'PASS' if t3 else 'FAIL'}] verify() flags too-small file")
        passed &= t3

        # 4. register_manual_file copies correctly
        ext = Path(tmp) / "external_LSWMD.pkl"
        _make_fake_pickle(ext, n_rows=4, pad_to_bytes=0)
        register_manual_file(ext, raw)
        t4 = pickle_path(raw).exists()
        print(f"[{'PASS' if t4 else 'FAIL'}] register_manual_file() copies file")
        passed &= t4

        # 5. deep verify on a structurally-valid (but tiny) pickle:
        #    columns match -> column check passes; row count is low -> overall fails,
        #    and crucially missing_columns must be empty. Lower the size floor so the
        #    deep path actually runs on our small fixture.
        res = verify(raw, deep=True, min_bytes=1)
        t5 = res.missing_columns is None and res.n_rows == 4
        print(f"[{'PASS' if t5 else 'FAIL'}] deep verify reads columns + row count "
              f"(rows={res.n_rows}, missing={res.missing_columns})")
        passed &= t5

    print("-" * 60)
    print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
    print("=" * 60)
    return passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WM-811K data downloader (Track 1, Module 1)")
    ap.add_argument("--raw-dir", default="data/raw", help="Where LSWMD.pkl lives")
    ap.add_argument("--from-file", help="Register a manually-downloaded pickle")
    ap.add_argument("--force", action="store_true", help="Re-download even if present")
    ap.add_argument("--verify-deep", action="store_true",
                    help="Load the pickle and check columns + row count")
    ap.add_argument("--self-test", action="store_true",
                    help="Run offline self-test and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    raw_dir = Path(args.raw_dir)

    if args.from_file:
        register_manual_file(args.from_file, raw_dir)
    elif not is_present(raw_dir) or args.force:
        download(raw_dir, force=args.force)
    else:
        log.info("Dataset already present: %s", pickle_path(raw_dir))

    result = verify(raw_dir, deep=args.verify_deep)
    log.info("Verify: %s | %s", "OK" if result.ok else "FAILED", result.message)
    if result.n_rows is not None:
        log.info("Rows: %s | Size: %.2f GB", f"{result.n_rows:,}",
                 result.size_bytes / 1e9)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())