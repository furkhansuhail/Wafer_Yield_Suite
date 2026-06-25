"""eda.py — exploratory analysis of wafers and the yield curve.

Two layers:
  * wafer-level: distributions of die count, defect rate, failure-type mix,
    and the per-wafer overdispersion check (variance vs mean of defect counts)
    which is the quick Poisson-vs-NegBinom diagnostic.
  * curve-level: plot empirical Y(A) with the fitted models overlaid.

All plots are saved to cfg.output_dir; summary stats are returned as frames.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless / file output
import matplotlib.pyplot as plt

from .config import Config
from .features import YieldCurve
from .models.base import YieldModel, FitResult


class YieldEDA:
    def __init__(self, config: Config):
        self.cfg = config

    # -- wafer-level ---------------------------------------------------------
    def wafer_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descriptive stats on die counts and defect rates."""
        return df[["n_die", "n_bad", "defect_rate"]].describe()

    def failure_mix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Counts and shares of each failure-type label."""
        counts = df["failure_type"].value_counts()
        return pd.DataFrame(
            {"count": counts, "share": counts / counts.sum()}
        )

    def overdispersion(self, df: pd.DataFrame) -> dict[str, float]:
        """Variance-to-mean ratio of per-wafer defect counts.

        VMR ~ 1  => Poisson-like (no clustering)
        VMR >> 1 => overdispersed => Negative Binomial favored.
        """
        x = df["n_bad"].to_numpy(dtype=float)
        mean = float(x.mean())
        var = float(x.var(ddof=1))
        return {
            "mean_defects": mean,
            "var_defects": var,
            "variance_to_mean_ratio": var / mean if mean > 0 else float("nan"),
        }

    def plot_distributions(self, df: pd.DataFrame,
                           fname: str = "eda_distributions.png") -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(df["n_die"], bins=40, color="#4C72B0")
        axes[0].set_title("Die count per wafer")
        axes[1].hist(df["defect_rate"].dropna(), bins=40, color="#C44E52")
        axes[1].set_title("Defect rate per wafer")
        mix = self.failure_mix(df)
        axes[2].bar(mix.index, mix["count"], color="#55A868")
        axes[2].set_title("Failure-type mix")
        axes[2].tick_params(axis="x", rotation=60)
        fig.tight_layout()
        out = self.cfg.output_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    # -- curve-level ---------------------------------------------------------
    def plot_curve_with_fits(
        self,
        curve: YieldCurve,
        models: list[YieldModel],
        fits: dict[str, FitResult],
        fname: str = "yield_curve_fits.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(curve.area, curve.yield_frac, s=40, color="black",
                   zorder=5, label="empirical Y(A)")
        a_grid = np.linspace(curve.area.min(), curve.area.max(), 200)
        for m in models:
            f = fits.get(m.name)
            if f is None or not f.success:
                continue
            ax.plot(a_grid, m.predict(a_grid, f), label=m.name)
        ax.set_xlabel("Area  A  (unit-die areas)")
        ax.set_ylabel("Yield  Y(A)")
        ax.set_title("Empirical yield vs. fitted analytical models")
        ax.legend()
        fig.tight_layout()
        out = self.cfg.output_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
