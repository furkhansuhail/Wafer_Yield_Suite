"""models/poisson.py — Poisson (Seeds) yield model.

    Y = exp(-D0 * A)

Assumes defects fall uniformly at random with no clustering. This is the
limiting case of the Negative Binomial model as alpha -> infinity, and it is
the most pessimistic of the three for large die.
"""
from __future__ import annotations
import numpy as np
from .base import YieldModel


class PoissonYield(YieldModel):
    name = "poisson"
    param_names = ("D0",)

    @staticmethod
    def yield_fn(area: np.ndarray, D0: float) -> np.ndarray:
        return np.exp(-D0 * area)

    def initial_guess(self, cfg) -> list[float]:
        return [cfg.d0_init]
