"""predictor.py — apply a fitted yield model to engineering questions.

Given a fitted model you typically want to answer:
  * "what yield do I get for a die of area A?"         -> predict_yield
  * "in physical units, given die area in mm^2?"        -> predict_yield(..., die_area_mm2, d0_per_mm2)
  * "largest die that still hits a target yield Y*?"    -> max_area_for_yield

Areas are in unit-die areas by default (matching the training curve). To work
in physical units, pass die_area_mm2 so A is scaled accordingly.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

from .models.base import YieldModel, FitResult


class YieldPredictor:
    """Thin wrapper binding a fitted model to prediction utilities."""

    def __init__(self, model: YieldModel, fit: FitResult):
        if not fit.success:
            raise ValueError(f"cannot predict from failed fit: {fit.message}")
        self.model = model
        self.fit = fit

    @property
    def params(self) -> dict[str, float]:
        return dict(self.fit.params)

    # -- forward prediction --------------------------------------------------
    def predict_yield(self, area) -> np.ndarray:
        """Yield at the given area(s), in unit-die areas."""
        return self.model.predict(np.atleast_1d(area).astype(float), self.fit)

    def predict_from_physical(
        self, die_area_mm2: float, ref_die_area_mm2: float = 1.0
    ) -> float:
        """Yield for a die whose physical area is die_area_mm2.

        The training curve measures area in units of the reference die; here we
        convert physical mm^2 into that unit before evaluating.
        """
        a = die_area_mm2 / ref_die_area_mm2
        return float(self.predict_yield(a)[0])

    # -- inverse prediction --------------------------------------------------
    def max_area_for_yield(self, target_yield: float,
                           a_hi: float = 1e6) -> float:
        """Largest area (unit-die areas) whose predicted yield >= target.

        Solves Y(A) = target by bracketed root finding. Y is monotonically
        decreasing in A for all three models, so the root is unique.
        """
        if not (0.0 < target_yield < 1.0):
            raise ValueError("target_yield must be in (0, 1)")
        f = lambda A: float(self.predict_yield(A)[0]) - target_yield
        # Y(0+) ~ 1 > target ; expand a_hi until Y drops below target
        lo, hi = 1e-9, a_hi
        if f(hi) > 0:                      # never reaches target within range
            return float("inf")
        return float(brentq(f, lo, hi, maxiter=200))
