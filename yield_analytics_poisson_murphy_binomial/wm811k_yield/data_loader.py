"""data_loader.py — load and normalize the WM-811K dataset.

Responsibilities (and *only* these):
  * read the raw pickle into a tidy DataFrame
  * coerce the messy nested label fields into plain strings
  * compute lightweight per-wafer summaries (die count, defect count)
  * apply coarse filtering (size, failure type, sampling)

It does NOT compute yield curves — that belongs to features.py.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd

from .config import Config, NO_DIE, GOOD_DIE, BAD_DIE


def _flatten_label(value) -> str:
    """WM-811K stores labels as ragged nested arrays like [['Center']] or [].

    Returns a clean lowercase-free string, or 'unlabeled' when empty.
    """
    arr = np.asarray(value, dtype=object).ravel()
    if arr.size == 0:
        return "unlabeled"
    item = arr[0]
    if isinstance(item, (bytes, bytearray)):
        item = item.decode("utf-8", errors="ignore")
    item = str(item).strip()
    return item if item else "unlabeled"


class WM811KLoader:
    """Loads WM-811K and exposes a clean DataFrame of wafer maps."""

    REQUIRED_COLS = ("waferMap",)

    def __init__(self, config: Config):
        self.cfg = config
        self.df: pd.DataFrame | None = None

    # -- public API ----------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Read, clean, summarize and filter. Returns the working DataFrame."""
        df = self._read_raw()
        self._validate(df)
        df = self._clean_labels(df)
        df = self._add_summaries(df)
        df = self._filter(df)
        self.df = df.reset_index(drop=True)
        return self.df

    def wafer_maps(self) -> list[np.ndarray]:
        """Return the list of 2-D integer wafer-map arrays after load()."""
        if self.df is None:
            raise RuntimeError("call load() first")
        return [np.asarray(m, dtype=np.int8) for m in self.df["waferMap"]]

    # -- internals -----------------------------------------------------------
    def _read_raw(self) -> pd.DataFrame:
        df = pd.read_pickle(self.cfg.data_path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected a DataFrame in {self.cfg.data_path}")
        return df.copy()

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"WM-811K is missing required columns: {missing}")

    def _clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if "failureType" in df.columns:
            df["failure_type"] = df["failureType"].apply(_flatten_label)
        else:
            df["failure_type"] = "unlabeled"
        if "trianTestLabel" in df.columns:        # original misspelling kept
            df["split"] = df["trianTestLabel"].apply(_flatten_label)
        else:
            df["split"] = "unlabeled"
        return df

    def _add_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        die, bad = [], []
        for m in df["waferMap"]:
            a = np.asarray(m)
            n_die = int(np.count_nonzero(a != NO_DIE))
            n_bad = int(np.count_nonzero(a == BAD_DIE))
            die.append(n_die)
            bad.append(n_bad)
        df["n_die"] = die
        df["n_bad"] = bad
        # guard against division by zero on empty maps
        df["defect_rate"] = np.where(
            df["n_die"] > 0, np.array(bad) / np.maximum(np.array(die), 1), np.nan
        )
        return df

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        mask = df["n_die"] >= c.min_die_count
        if c.failure_types is not None:
            mask &= df["failure_type"].isin(set(c.failure_types))
        df = df[mask]
        if c.max_wafers is not None and len(df) > c.max_wafers:
            df = df.sample(n=c.max_wafers, random_state=c.random_seed)
        return df
