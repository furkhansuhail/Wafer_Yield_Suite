"""evaluator.py — score fitted models and rank them.

Metrics on the yield-vs-area curve:
  * RMSE / weighted RMSE   (lower better)
  * R^2                    (higher better)
  * AIC / BIC              (lower better; penalize NB's extra parameter)

AIC/BIC use the least-squares (Gaussian-residual) approximation
    AIC = n*ln(RSS/n) + 2k,   BIC = n*ln(RSS/n) + k*ln(n)
which is the standard form for comparing curve fits with different k.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .features import YieldCurve
from .models.base import YieldModel, FitResult


@dataclass
class Scores:
    model_name: str
    n_params: int
    rmse: float
    weighted_rmse: float
    r2: float
    aic: float
    bic: float
    params: dict[str, float]


class ModelEvaluator:
    """Computes goodness-of-fit for fitted models and assembles a leaderboard."""

    def __init__(self, models: list[YieldModel]):
        self._models = {m.name: m for m in models}

    def score(
        self, model: YieldModel, fit: FitResult, curve: YieldCurve
    ) -> Scores:
        y = curve.yield_frac
        yhat = model.predict(curve.area, fit)
        resid = y - yhat
        n = len(y)
        k = model.n_params

        rss = float(np.sum(resid ** 2))
        rmse = float(np.sqrt(rss / n))

        w = curve.weights() ** 2          # 1/sigma^2
        wrmse = float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))

        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")

        # guard rss=0
        rss_safe = max(rss, 1e-12)
        aic = n * np.log(rss_safe / n) + 2 * k
        bic = n * np.log(rss_safe / n) + k * np.log(n)

        return Scores(
            model_name=model.name,
            n_params=k,
            rmse=rmse,
            weighted_rmse=wrmse,
            r2=r2,
            aic=float(aic),
            bic=float(bic),
            params=fit.params,
        )

    def leaderboard(
        self, fits: dict[str, FitResult], curve: YieldCurve
    ) -> pd.DataFrame:
        rows = []
        for name, fit in fits.items():
            model = self._models[name]
            if not fit.success:
                continue
            rows.append(asdict(self.score(model, fit, curve)))
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # best fit = lowest AIC, tie-break on weighted RMSE
        df = df.sort_values(["aic", "weighted_rmse"]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df
