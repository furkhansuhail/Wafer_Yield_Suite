"""
eda.py
======

Module 2 of the SECOM yield-prediction pipeline.

Responsibilities
----------------
Describe the dataset so a human (and the later modules) understand its
shape and pathologies *before* any modelling. Specifically:

* Class balance (how severe is the ~14:1 imbalance here?).
* Missingness profile (per-feature and overall; which sensors are mostly empty?).
* Dead / low-variance features (constant or near-constant columns carry no signal).
* Feature -> target association (which sensors actually move with pass/fail?).
* Temporal drift (does the failure rate wander over the production window?).

Outputs
-------
* An :class:`EDAReport` dataclass holding the computed tables, so the findings
  are available programmatically.
* PNG plots and a ``eda_report.md`` written to ``out_dir``.

Important boundary
------------------
EDA here is **descriptive and computed on the full dataset** purely for human
understanding. It does NOT decide the preprocessing that the model uses. The
actual drop thresholds, imputation, and scaling get *re-fit on the training
fold only* inside the model_trainer stage — otherwise statistics leak from
test into train. Treat the lists below (dead features, high-missing features)
as guidance for choosing thresholds, not as a fitted transformer.

CLI
---
    python -m secom_pipeline.eda --data-dir data --out-dir eda_report
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless; never tries to open a window
import matplotlib.pyplot as plt  # noqa: E402

from yield_excursion_fail_prediction.src.data_downloader import SecomData, load_secom  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #
@dataclass
class EDAReport:
    n_samples: int
    n_features: int
    class_balance: dict
    missingness: pd.DataFrame          # index=feature, cols: missing_count, missing_rate
    variance: pd.DataFrame             # index=feature, cols: variance, n_unique, is_dead
    associations: pd.DataFrame         # index=feature, cols: point_biserial, abs_corr, mutual_info
    temporal: pd.DataFrame | None      # failure rate binned over time (if timestamps present)
    artifacts: dict = field(default_factory=dict)  # name -> saved file path

    # ---- convenience views -------------------------------------------------
    def dead_features(self) -> list[str]:
        return self.variance.index[self.variance["is_dead"]].tolist()

    def high_missing_features(self, threshold: float = 0.5) -> list[str]:
        return self.missingness.index[
            self.missingness["missing_rate"] > threshold
        ].tolist()

    def top_associated(self, n: int = 20) -> pd.DataFrame:
        return self.associations.sort_values("abs_corr", ascending=False).head(n)


# --------------------------------------------------------------------------- #
# Individual analyses
# --------------------------------------------------------------------------- #
def class_balance(y: pd.Series) -> dict:
    n_fail = int(y.sum())
    n_pass = int(len(y) - n_fail)
    return {
        "n_pass": n_pass,
        "n_fail": n_fail,
        "fail_rate": n_fail / len(y) if len(y) else float("nan"),
        "imbalance_ratio": (n_pass / n_fail) if n_fail else float("nan"),
    }


def missingness_report(X: pd.DataFrame) -> pd.DataFrame:
    counts = X.isna().sum()
    rates = counts / len(X)
    return pd.DataFrame(
        {"missing_count": counts.astype(int), "missing_rate": rates}
    ).sort_values("missing_rate", ascending=False)


def variance_report(X: pd.DataFrame, var_threshold: float = 1e-8) -> pd.DataFrame:
    variances = X.var(numeric_only=True)
    n_unique = X.nunique(dropna=True)
    is_dead = (variances <= var_threshold) | (n_unique <= 1)
    return pd.DataFrame(
        {"variance": variances, "n_unique": n_unique.astype(int), "is_dead": is_dead}
    ).sort_values("variance")


def target_association(
    X: pd.DataFrame, y: pd.Series, compute_mutual_info: bool = True
) -> pd.DataFrame:
    """Per-feature association with the binary target.

    * point-biserial correlation == Pearson corr of a continuous feature with a
      0/1 target. ``DataFrame.corrwith`` handles NaNs pairwise, so this needs
      no imputation.
    * mutual information is optional and needs a NaN-free copy, so we median-fill
      a *temporary* copy only for this ranking (never persisted).
    """
    # Constant (dead) columns have zero std, so their correlation is undefined
    # and numpy would emit a divide warning. Silence it; those become NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = X.corrwith(y.astype(float))
    out = pd.DataFrame({"point_biserial": corr})
    out["abs_corr"] = out["point_biserial"].abs()

    if compute_mutual_info:
        try:
            from sklearn.feature_selection import mutual_info_classif

            # Median-fill a throwaway copy purely so MI can be computed.
            X_filled = X.fillna(X.median(numeric_only=True))
            # Drop any columns still all-NaN (median undefined) before MI.
            valid = X_filled.columns[~X_filled.isna().all()]
            mi = pd.Series(
                mutual_info_classif(
                    X_filled[valid].fillna(0.0), y, discrete_features=False, random_state=0
                ),
                index=valid,
            )
            out["mutual_info"] = mi
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mutual information skipped: %s", exc)
            out["mutual_info"] = np.nan
    else:
        out["mutual_info"] = np.nan

    return out.sort_values("abs_corr", ascending=False)


def temporal_failure_rate(
    timestamps: pd.Series, y: pd.Series, freq: str = "D"
) -> pd.DataFrame | None:
    """Bin the failure rate over time to expose drift. Returns None if no valid stamps."""
    if timestamps is None or timestamps.isna().all():
        return None
    df = pd.DataFrame({"ts": timestamps, "fail": y}).dropna(subset=["ts"])
    if df.empty:
        return None
    grouped = (
        df.set_index("ts")
        .resample(freq)["fail"]
        .agg(n="count", fails="sum")
    )
    grouped["fail_rate"] = grouped["fails"] / grouped["n"].replace(0, np.nan)
    return grouped


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plot_class_balance(cb: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["pass", "fail"], [cb["n_pass"], cb["n_fail"]], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("count")
    ax.set_title(
        f"Class balance  (ratio {cb['imbalance_ratio']:.1f}:1, "
        f"fail rate {cb['fail_rate']*100:.1f}%)"
    )
    for i, v in enumerate([cb["n_pass"], cb["n_fail"]]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_missingness(miss: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(miss["missing_rate"] * 100, bins=30, color="#55A868", edgecolor="white")
    ax1.set_xlabel("missing rate per feature (%)")
    ax1.set_ylabel("number of features")
    ax1.set_title("Distribution of feature missingness")

    top = miss.head(20)
    ax2.barh(range(len(top)), top["missing_rate"] * 100, color="#8172B3")
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top.index, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("missing rate (%)")
    ax2.set_title("Top-20 most-missing features")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_top_associations(assoc: pd.DataFrame, path: Path, n: int = 20) -> None:
    top = assoc.sort_values("abs_corr", ascending=False).head(n)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top["point_biserial"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(top)), top["point_biserial"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("point-biserial correlation with fail")
    ax.set_title(f"Top-{n} features by |correlation| with target")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_temporal(temporal: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(temporal.index, temporal["fail_rate"] * 100, marker="o", ms=3, color="#C44E52")
    ax.set_ylabel("failure rate (%)")
    ax.set_xlabel("date")
    ax.set_title("Failure rate over the production window (drift check)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(
    data: SecomData,
    out_dir: str | Path = "eda_report",
    var_threshold: float = 1e-8,
    make_plots: bool = True,
) -> EDAReport:
    """Run the full EDA suite, save artifacts, and return an :class:`EDAReport`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cb = class_balance(data.y)
    miss = missingness_report(data.X)
    var = variance_report(data.X, var_threshold=var_threshold)
    assoc = target_association(data.X, data.y)
    temporal = temporal_failure_rate(data.timestamps, data.y)

    artifacts: dict = {}
    if make_plots:
        _plot_class_balance(cb, out_dir / "class_balance.png")
        artifacts["class_balance_plot"] = str(out_dir / "class_balance.png")
        _plot_missingness(miss, out_dir / "missingness.png")
        artifacts["missingness_plot"] = str(out_dir / "missingness.png")
        _plot_top_associations(assoc, out_dir / "top_features.png")
        artifacts["top_features_plot"] = str(out_dir / "top_features.png")
        if temporal is not None:
            _plot_temporal(temporal, out_dir / "temporal.png")
            artifacts["temporal_plot"] = str(out_dir / "temporal.png")

    report = EDAReport(
        n_samples=data.n_samples,
        n_features=data.n_features,
        class_balance=cb,
        missingness=miss,
        variance=var,
        associations=assoc,
        temporal=temporal,
        artifacts=artifacts,
    )

    md_path = out_dir / "eda_report.md"
    md_path.write_text(_render_markdown(report))
    artifacts["markdown"] = str(md_path)
    logger.info("EDA written to %s", out_dir)
    return report


def _render_markdown(r: EDAReport) -> str:
    cb = r.class_balance
    dead = r.dead_features()
    high_miss = r.high_missing_features(0.5)
    lines = [
        "# SECOM — Exploratory Data Analysis",
        "",
        f"- Samples: **{r.n_samples}**",
        f"- Features: **{r.n_features}**",
        f"- Pass / Fail: **{cb['n_pass']} / {cb['n_fail']}** "
        f"(imbalance ratio **{cb['imbalance_ratio']:.1f}:1**, "
        f"fail rate **{cb['fail_rate']*100:.1f}%**)",
        f"- Dead / constant features: **{len(dead)}**",
        f"- Features >50% missing: **{len(high_miss)}**",
        f"- Mean missing per cell: **{r.missingness['missing_rate'].mean()*100:.2f}%**",
        "",
        "## Top 15 features by |correlation| with failure",
        "",
        r.top_associated(15)[["point_biserial", "mutual_info"]]
        .round(4)
        .to_markdown(),
        "",
        "## Modelling implications",
        "- Severe imbalance -> use PR-AUC / recall / G-mean, never raw accuracy.",
        "- Many dead and high-missing columns -> variance filter + missingness "
        "drop belong in the pipeline (fit on train fold only).",
        "- Correlations are weak and spread out -> expect feature selection to "
        "help, and no single sensor to dominate.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="Run EDA on the SECOM dataset.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="eda_report")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    data = load_secom(data_dir=args.data_dir)
    report = run_eda(data, out_dir=args.out_dir, make_plots=not args.no_plots)

    print(data.summary())
    print(f"\nDead features: {len(report.dead_features())}")
    print(f">50% missing features: {len(report.high_missing_features(0.5))}")
    print("\nTop 10 features by |corr| with failure:")
    print(report.top_associated(10)[["point_biserial", "mutual_info"]].round(4).to_string())
    print(f"\nArtifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
