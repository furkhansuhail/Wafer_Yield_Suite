"""
data_downloader.py
==================

Module 1 of the SECOM yield-prediction pipeline.

Responsibilities
----------------
* Fetch the raw SECOM dataset (sensor measurements + pass/fail labels).
* Cache it locally so we only hit the network once.
* Parse the raw whitespace-delimited files into tidy pandas objects.
* Hand the next stage a clean, well-typed object: features ``X``, target
  ``y`` (0 = pass, 1 = fail), and the per-sample ``timestamps``.

Design notes
------------
* Two acquisition paths, tried in order:
    1. ``ucimlrepo`` (the maintained UCI API, dataset id=179).
    2. Direct download of the legacy UCI flat files as a fallback.
  Whichever succeeds writes a raw cache, so re-runs are offline.
* The label convention is remapped on load. UCI uses ``-1 = pass`` and
  ``1 = fail``. For yield/defect prediction the *rare* event (a failing
  unit) is the thing we care about, so we make it the positive class:
  ``pass -> 0``, ``fail -> 1``.
* Parsing is deliberately split out from downloading so it can be unit
  tested against small synthetic files without any network access.

CLI
---
    python -m secom_pipeline.data_downloader --data-dir data
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Legacy UCI flat-file locations (fallback path).
_SECOM_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
)
_SECOM_LABELS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
)
_UCI_DATASET_ID = 179

# Filenames used inside the cache directory.
_RAW_DATA_FILE = "secom.data"
_RAW_LABELS_FILE = "secom_labels.data"


@dataclass
class SecomData:
    """Container returned by :func:`load_secom`.

    Attributes
    ----------
    X : pd.DataFrame
        Sensor measurements, shape (n_samples, n_features). Columns named
        ``feature_000`` ... ``feature_589``. May contain NaNs (handled later
        in the cleaning/balancing stage, not here).
    y : pd.Series
        Target. 0 = pass (in-spec), 1 = fail (yield excursion).
    timestamps : pd.Series
        datetime64 stamp for each sample (when the test was taken).
    """

    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        n_fail = int(self.y.sum())
        n_pass = int(self.n_samples - n_fail)
        ratio = (n_pass / n_fail) if n_fail else float("nan")
        missing = float(self.X.isna().mean().mean()) * 100
        return (
            f"SECOM: {self.n_samples} samples x {self.n_features} features | "
            f"pass={n_pass}, fail={n_fail} (ratio {ratio:.1f}:1) | "
            f"mean missing per cell={missing:.2f}%"
        )


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def _download_url(url: str, dest: Path) -> None:
    """Stream a single URL to ``dest``."""
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def _download_via_urls(data_dir: Path) -> None:
    """Fallback acquisition: pull the two legacy UCI flat files."""
    _download_url(_SECOM_DATA_URL, data_dir / _RAW_DATA_FILE)
    _download_url(_SECOM_LABELS_URL, data_dir / _RAW_LABELS_FILE)


def _download_via_ucimlrepo(data_dir: Path) -> None:
    """Primary acquisition: the maintained UCI API.

    We persist the result into the same raw flat-file format the URL path
    produces, so :func:`parse_raw_secom` is the single source of truth for
    parsing regardless of how the bytes arrived.
    """
    from ucimlrepo import fetch_ucirepo  # imported lazily; optional dependency

    logger.info("Fetching SECOM via ucimlrepo (id=%d)", _UCI_DATASET_ID)
    ds = fetch_ucirepo(id=_UCI_DATASET_ID)

    features: pd.DataFrame = ds.data.features
    targets: pd.DataFrame = ds.data.targets

    # The API exposes the label and a timestamp column among the targets.
    # Normalise to the flat-file layout: features -> secom.data,
    # "<label> <timestamp>" -> secom_labels.data.
    label_col = _pick_column(targets, ["Pass/Fail", "label", "class"])
    time_col = _pick_column(targets, ["Time", "timestamp", "date"], required=False)

    features.to_csv(
        data_dir / _RAW_DATA_FILE, sep=" ", header=False, index=False, na_rep="NaN"
    )

    labels = targets[[label_col]].copy()
    if time_col is not None:
        labels["__ts__"] = (
            pd.to_datetime(targets[time_col], errors="coerce")
            .dt.strftime("%d/%m/%Y %H:%M:%S")
            .fillna("01/01/1970 00:00:00")
        )
    else:
        labels["__ts__"] = "01/01/1970 00:00:00"
    labels.to_csv(
        data_dir / _RAW_LABELS_FILE, sep=" ", header=False, index=False
    )


def _pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True):
    """Return the first column in ``candidates`` that exists (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        # Fall back to the first column rather than failing outright.
        return df.columns[0]
    return None


def download_secom(
    data_dir: str | Path = "data",
    force: bool = False,
    prefer_ucimlrepo: bool = True,
) -> Path:
    """Ensure the raw SECOM files exist locally; return the cache directory.

    Parameters
    ----------
    data_dir : str or Path
        Where to cache the raw files.
    force : bool
        Re-download even if a cache is present.
    prefer_ucimlrepo : bool
        Try the ucimlrepo API first; fall back to direct URLs on any error.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / _RAW_DATA_FILE
    labels_path = data_dir / _RAW_LABELS_FILE

    if data_path.exists() and labels_path.exists() and not force:
        logger.info("Raw cache present in %s (use force=True to refresh)", data_dir)
        return data_dir

    errors: list[str] = []
    order = (
        [_download_via_ucimlrepo, _download_via_urls]
        if prefer_ucimlrepo
        else [_download_via_urls, _download_via_ucimlrepo]
    )
    for fn in order:
        try:
            fn(data_dir)
            logger.info("Acquired SECOM via %s", fn.__name__)
            return data_dir
        except Exception as exc:  # noqa: BLE001 - we want to try the next path
            logger.warning("%s failed: %s", fn.__name__, exc)
            errors.append(f"{fn.__name__}: {exc}")

    raise RuntimeError(
        "Could not download SECOM by any method.\n  " + "\n  ".join(errors)
    )


# --------------------------------------------------------------------------- #
# Parsing (network-free, unit-testable)
# --------------------------------------------------------------------------- #
def parse_raw_secom(data_path: str | Path, labels_path: str | Path) -> SecomData:
    """Parse the two raw whitespace-delimited files into a :class:`SecomData`.

    ``secom.data``   : n_samples rows x n_features cols, NaNs as the token ``NaN``.
    ``secom_labels.data`` : ``<label> <dd/mm/yyyy> <HH:MM:SS>`` per row.
    """
    data_path, labels_path = Path(data_path), Path(labels_path)

    X = pd.read_csv(
        data_path, sep=r"\s+", header=None, na_values=["NaN", "nan", ""], engine="python"
    )
    X.columns = [f"feature_{i:03d}" for i in range(X.shape[1])]
    X = X.astype(np.float64)

    raw_labels = pd.read_csv(
        labels_path, sep=r"\s+", header=None, engine="python"
    )
    raw_label_vals = raw_labels.iloc[:, 0].astype(int)

    # Remap UCI convention (-1 pass / 1 fail) -> (0 pass / 1 fail).
    y = raw_label_vals.map({-1: 0, 1: 1})
    if y.isna().any():
        raise ValueError(
            "Unexpected label values; expected only {-1, 1}, "
            f"saw {sorted(raw_label_vals.unique())}"
        )
    y = y.astype(int)
    y.name = "fail"

    # Reconstruct the timestamp from the trailing date/time tokens, if present.
    if raw_labels.shape[1] >= 3:
        ts_str = (
            raw_labels.iloc[:, 1].astype(str) + " " + raw_labels.iloc[:, 2].astype(str)
        )
        timestamps = pd.to_datetime(ts_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    else:
        timestamps = pd.Series(pd.NaT, index=X.index)
    timestamps.name = "timestamp"

    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError(
            f"Row count mismatch: X={len(X)}, y={len(y)}, ts={len(timestamps)}"
        )

    return SecomData(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        timestamps=timestamps.reset_index(drop=True),
    )


def load_secom(
    data_dir: str | Path = "data",
    force_download: bool = False,
    prefer_ucimlrepo: bool = True,
) -> SecomData:
    """High-level entry point: download if needed, then parse and return.

    This is the function the rest of the pipeline imports.
    """
    data_dir = download_secom(
        data_dir, force=force_download, prefer_ucimlrepo=prefer_ucimlrepo
    )
    return parse_raw_secom(
        data_dir / _RAW_DATA_FILE, data_dir / _RAW_LABELS_FILE
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and inspect the SECOM dataset.")
    p.add_argument("--data-dir", default="data", help="Cache directory (default: data)")
    p.add_argument("--force", action="store_true", help="Force re-download")
    p.add_argument(
        "--no-ucimlrepo",
        action="store_true",
        help="Skip the ucimlrepo API and use direct URLs",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_arg_parser().parse_args(argv)
    data = load_secom(
        data_dir=args.data_dir,
        force_download=args.force,
        prefer_ucimlrepo=not args.no_ucimlrepo,
    )
    print(data.summary())
    print("\nClass balance:")
    print(data.y.value_counts().rename({0: "pass", 1: "fail"}).to_string())


if __name__ == "__main__":
    main()
