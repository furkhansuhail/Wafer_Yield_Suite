"""trainer.py — fit a single yield model to an empirical YieldCurve.

One trainer works for all three models because they share the YieldModel
interface (yield_fn + initial_guess + bounds). Fitting is weighted nonlinear
least squares (scipy.curve_fit), with weights from each area point's binomial
standard error so reliable points (many windows) dominate.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit

from .config import Config
from .features import YieldCurve
from .models.base import YieldModel, FitResult


class YieldModelTrainer:
    """Fits a YieldModel to a YieldCurve and returns a FitResult."""

    def __init__(self, config: Config):
        self.cfg = config

    def fit(self, model: YieldModel, curve: YieldCurve) -> FitResult:
        p0 = model.initial_guess(self.cfg)
        lo, hi = model.bounds()
        sigma = 1.0 / curve.weights()    # curve.weights() returns 1/sigma

        try:
            popt, pcov = curve_fit(
                model.yield_fn,
                curve.area,
                curve.yield_frac,
                p0=p0,
                bounds=(lo, hi),
                sigma=sigma,
                absolute_sigma=True,
                maxfev=self.cfg.fit_maxfev,
            )
            return FitResult(
                model_name=model.name,
                params=model.pack(popt),
                covariance=pcov,
                success=True,
                message="converged",
            )
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            return FitResult(
                model_name=model.name,
                params={k: float("nan") for k in model.param_names},
                covariance=None,
                success=False,
                message=f"fit failed: {exc}",
            )

    def fit_all(
        self, models: list[YieldModel], curve: YieldCurve
    ) -> dict[str, FitResult]:
        return {m.name: self.fit(m, curve) for m in models}
