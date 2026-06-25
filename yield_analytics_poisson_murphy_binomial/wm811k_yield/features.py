"""features.py — turn wafer maps into a yield-vs-area curve (Stapper method).

The three yield models are all functions Y(A). To fit them we need an
*empirical* Y(A): for a window of w x w dies (area A = w^2 unit dies), what
fraction of fully-die windows contain zero defects?

A window is:
  * VALID   if every cell is a real die (no NO_DIE cells inside it)
  * YIELDING if it is valid AND every cell is a GOOD die

    Y(A) = yielding_windows / valid_windows   pooled over all wafers.

We also keep per-area window counts so the trainer can weight points by
their statistical reliability (binomial variance ~ Y(1-Y)/n).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import Config, NO_DIE, GOOD_DIE, BAD_DIE


@dataclass
class YieldCurve:
    """Empirical yield as a function of window area (in unit-die areas)."""
    area: np.ndarray          # A_i  (= w_i**2)
    window: np.ndarray        # w_i  (edge length in dies)
    yield_frac: np.ndarray    # Y_i in [0, 1]
    n_valid: np.ndarray       # number of valid windows at this area
    n_yield: np.ndarray       # number of yielding windows at this area

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window": self.window,
                "area": self.area,
                "yield": self.yield_frac,
                "n_valid": self.n_valid,
                "n_yield": self.n_yield,
            }
        )

    def weights(self) -> np.ndarray:
        """1/sigma for weighted least squares, from binomial std error.

        sigma_i = sqrt(Y(1-Y)/n); floored so Y in {0,1} stays finite.
        """
        p = np.clip(self.yield_frac, 1e-6, 1 - 1e-6)
        sigma = np.sqrt(p * (1 - p) / np.maximum(self.n_valid, 1))
        sigma = np.maximum(sigma, 1e-4)
        return 1.0 / sigma


class WindowExtractor:
    """Builds a pooled YieldCurve from a collection of wafer maps."""

    def __init__(self, config: Config):
        self.cfg = config

    def extract(self, wafer_maps: list[np.ndarray]) -> YieldCurve:
        sizes = self.cfg.window_sizes
        valid_tot = {w: 0 for w in sizes}
        yield_tot = {w: 0 for w in sizes}

        for m in wafer_maps:
            a = np.asarray(m, dtype=np.int8)
            for w in sizes:
                v, y = self._count_windows(a, w)
                valid_tot[w] += v
                yield_tot[w] += y

        return self._assemble(valid_tot, yield_tot)

    # -- internals -----------------------------------------------------------
    def _count_windows(self, a: np.ndarray, w: int) -> tuple[int, int]:
        """Count valid and yielding non-overlapping (or strided) w x w blocks."""
        H, W = a.shape
        if H < w or W < w:
            return 0, 0
        stride = self.cfg.stride or w
        n_valid = n_yield = 0
        for i in range(0, H - w + 1, stride):
            for j in range(0, W - w + 1, stride):
                block = a[i:i + w, j:j + w]
                if np.any(block == NO_DIE):
                    continue                      # window spills off the die grid
                n_valid += 1
                if not np.any(block == BAD_DIE):  # all GOOD_DIE
                    n_yield += 1
        return n_valid, n_yield

    def _assemble(self, valid_tot: dict, yield_tot: dict) -> YieldCurve:
        win, area, yld, nv, ny = [], [], [], [], []
        for w in self.cfg.window_sizes:
            v = valid_tot[w]
            if v < self.cfg.min_windows_per_area:
                continue
            win.append(w)
            area.append(w * w)
            yld.append(yield_tot[w] / v)
            nv.append(v)
            ny.append(yield_tot[w])
        if not win:
            raise ValueError(
                "No area points survived min_windows_per_area; "
                "lower the threshold or use more wafers."
            )
        return YieldCurve(
            area=np.array(area, dtype=float),
            window=np.array(win, dtype=int),
            yield_frac=np.array(yld, dtype=float),
            n_valid=np.array(nv, dtype=int),
            n_yield=np.array(ny, dtype=int),
        )
