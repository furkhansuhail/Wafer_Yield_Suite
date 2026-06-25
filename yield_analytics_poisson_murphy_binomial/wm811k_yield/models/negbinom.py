"""models/negbinom.py — Negative Binomial (Stapper) yield model.

    Y = (1 + D0*A/alpha)^(-alpha)

The general clustering model: defect density is Gamma-distributed with
clustering parameter alpha. Small alpha => strong clustering (defects bunch
together, so large die survive better than Poisson predicts). As
alpha -> infinity the expression collapses to exp(-D0*A), i.e. Poisson.

This is the only one of the three with a free clustering parameter, so it is
expected to win on real WM-811K wafers that show Center/Edge/Scratch patterns.
"""
from __future__ import annotations
import numpy as np
from .base import YieldModel


class NegativeBinomialYield(YieldModel):
    name = "negative_binomial"
    param_names = ("D0", "alpha")

    @staticmethod
    def yield_fn(area: np.ndarray, D0: float, alpha: float) -> np.ndarray:
        area = np.asarray(area, dtype=float)
        return np.power(1.0 + (D0 * area) / alpha, -alpha)

    def initial_guess(self, cfg) -> list[float]:
        return [cfg.d0_init, cfg.alpha_init]

    def bounds(self):
        # D0 > 0 ; alpha in (0, large). Cap alpha so the optimizer doesn't
        # wander toward the Poisson limit numerically.
        return [1e-9, 1e-3], [np.inf, 1e6]
