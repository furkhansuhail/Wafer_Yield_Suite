"""models/murphy.py — Murphy's yield model.

    Y = [ (1 - exp(-D0*A)) / (D0*A) ]^2

Murphy assumed defect density itself varies across wafers and approximated
that variation with a triangular distribution. The result sits between the
optimistic and pessimistic extremes and introduces mild clustering relative
to pure Poisson. Has a removable singularity at A -> 0 (limit = 1), handled
explicitly below.
"""
from __future__ import annotations
import numpy as np
from .base import YieldModel


class MurphyYield(YieldModel):
    name = "murphy"
    param_names = ("D0",)

    @staticmethod
    def yield_fn(area: np.ndarray, D0: float) -> np.ndarray:
        area = np.asarray(area, dtype=float)
        lam = D0 * area
        # safe evaluation: as lam -> 0, (1 - e^-lam)/lam -> 1, so Y -> 1
        with np.errstate(divide="ignore", invalid="ignore"):
            core = np.where(lam > 1e-12, (1.0 - np.exp(-lam)) / lam, 1.0)
        return core ** 2

    def initial_guess(self, cfg) -> list[float]:
        return [cfg.d0_init]
