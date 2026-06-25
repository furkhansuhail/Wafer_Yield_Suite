"""
model_trainer.py
================

Module 4 of the SECOM yield-prediction pipeline — the integration point.

It assembles a single **leak-free** imblearn ``Pipeline`` in this fixed order::

    MissingnessThreshold   (drop columns that are mostly empty)
        -> SimpleImputer   (median-fill the rest)
        -> VarianceThreshold (drop dead / constant columns)
        -> StandardScaler  (sensors have wildly different units)
        -> SelectKBest     (optional feature selection)
        -> <resampler>     (from module 3; only during fit; omitted if None)
        -> <estimator>     (logreg | balanced_rf | hist_gb | xgboost)

Why this order
--------------
* Cleaning is *before* resampling, so SMOTE never interpolates between noisy or
  unimputed points (the noise-amplification trap).
* VarianceThreshold is *before* StandardScaler — after scaling everything has
  unit variance, so the dead-column filter must run first.
* Every transformer is a pipeline step, so all of them — imputation medians,
  scaler statistics, selected features, AND resampling — are fit on the
  training fold only. Nothing leaks from validation/test back into fitting.

The estimator picks up its imbalance handling from the ``BalancingPlan``
(module 3): ``class_weight`` for cost-sensitive strategies, ``scale_pos_weight``
for XGBoost. ``balanced_rf`` handles imbalance internally and is meant to be
paired with the ``none`` strategy.

This module builds, cross-validates, and fits. Detailed held-out evaluation,
threshold tuning, and plots live in module 5 (evaluator).

CLI
---
    python -m secom_pipeline.model_trainer --estimator logreg --strategy smote_enn
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score

from yield_excursion_fail_prediction.src.data_downloader import SecomData, load_secom
from yield_excursion_fail_prediction.src.imbalance_balancer import BalancingPlan, build_balancing_plan

logger = logging.getLogger(__name__)

ESTIMATORS = ("logreg", "balanced_rf", "hist_gb", "xgboost")


# --------------------------------------------------------------------------- #
# Custom transformer: drop columns whose missing-rate exceeds a threshold
# --------------------------------------------------------------------------- #
class MissingnessThreshold(BaseEstimator, TransformerMixin):
    """Drop columns whose fraction of missing values (fit on train) exceeds
    ``threshold``. sklearn has no built-in for this, so we provide one that is
    pipeline-compatible and preserves feature names.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = self._to_frame(X)
        rates = X.isna().mean(axis=0).to_numpy()
        self.keep_mask_ = rates <= self.threshold
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        # Guard against dropping literally everything on pathological folds.
        if not self.keep_mask_.any():
            self.keep_mask_ = np.ones_like(self.keep_mask_, dtype=bool)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._to_frame(X)
        return X.iloc[:, self.keep_mask_].to_numpy()

    def get_feature_names_out(self, input_features=None):
        names = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else self.feature_names_in_
        )
        return names[self.keep_mask_]

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(
            X, columns=[f"feature_{i:03d}" for i in range(np.asarray(X).shape[1])]
        )


class AddMissingIndicators(BaseEstimator, TransformerMixin):
    """Append binary "was this sensor missing?" columns to the feature matrix.

    On SECOM the *fact* that a sensor failed to report is itself predictive —
    often more so than its (noisy) value. This runs FIRST, before any column is
    dropped or imputed, so indicators are created even for high-missing columns
    that ``MissingnessThreshold`` later removes: the missingness signal survives
    after the noisy values are gone.

    Only columns whose training missing-rate exceeds ``threshold`` get an
    indicator (keeps the count focused). Original columns pass through unchanged
    (NaNs intact) so the downstream imputer still works.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = MissingnessThreshold._to_frame(X)
        rates = X.isna().mean(axis=0)
        # Indicator only where there is meaningful, non-constant missingness.
        self.indicator_cols_ = [
            c for c in X.columns if self.threshold < rates[c] < 1.0
        ]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = MissingnessThreshold._to_frame(X)
        if not self.indicator_cols_:
            return X.to_numpy()
        ind = X[self.indicator_cols_].isna().astype(np.float64).to_numpy()
        return np.hstack([X.to_numpy(), ind])

    def get_feature_names_out(self, input_features=None):
        base = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else self.feature_names_in_
        )
        ind_names = np.asarray(
            [f"{c}__missing" for c in self.indicator_cols_], dtype=object
        )
        return np.concatenate([base, ind_names])


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass
class CVResults:
    estimator: str
    strategy: str
    metrics_mean: dict
    metrics_std: dict
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = f"[{self.estimator} + {self.strategy}]"
        body = "  ".join(
            f"{k}={self.metrics_mean[k]:.3f}±{self.metrics_std[k]:.3f}"
            for k in self.metrics_mean
        )
        return f"{head}  {body}"


# --------------------------------------------------------------------------- #
# Scorers tuned for severe imbalance
# --------------------------------------------------------------------------- #
def default_scorers() -> dict:
    """Metrics appropriate for a ~14:1 problem (accuracy deliberately excluded)."""
    return {
        "pr_auc": "average_precision",          # primary
        "roc_auc": "roc_auc",
        "recall_fail": "recall",                # recall of the positive (fail) class
        "f1_fail": "f1",
        "balanced_acc": "balanced_accuracy",
        "g_mean": make_scorer(geometric_mean_score),
        "mcc": make_scorer(matthews_corrcoef),
    }


# --------------------------------------------------------------------------- #
# Estimator factory (applies imbalance params from the plan)
# --------------------------------------------------------------------------- #
def build_estimator(
    name: str, plan: BalancingPlan, random_state: int = 42
) -> BaseEstimator:
    name = name.lower()
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Valid: {ESTIMATORS}")

    cw = plan.class_weight if plan.use_class_weight else None

    if name == "logreg":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight=cw,
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state,
        )

    if name == "balanced_rf":
        from imblearn.ensemble import BalancedRandomForestClassifier

        if plan.name != "none":
            logger.warning(
                "balanced_rf already balances internally; pairing it with "
                "strategy='%s' double-handles imbalance. Consider strategy='none'.",
                plan.name,
            )
        # Explicit params silence 0.14 deprecation warnings.
        return BalancedRandomForestClassifier(
            n_estimators=300,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1,
        )

    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        # HGB handles NaNs natively and supports class_weight in this sklearn.
        return HistGradientBoostingClassifier(
            class_weight=cw,
            learning_rate=0.1,
            max_iter=300,
            l2_regularization=1.0,
            random_state=random_state,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "xgboost is not installed. `pip install xgboost` or pick another estimator."
            ) from exc
        # XGBoost uses scale_pos_weight rather than class_weight.
        spw = plan.scale_pos_weight if plan.scale_pos_weight is not None else 1.0
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Pipeline builder
# --------------------------------------------------------------------------- #
def build_pipeline(
    strategy: str = "smote_enn",
    estimator: str = "logreg",
    y=None,
    missing_threshold: float = 0.5,
    var_threshold: float = 1e-8,
    k_features: int | str = "all",
    feature_score: str = "mutual_info",
    missing_indicators: bool = True,
    indicator_threshold: float = 0.05,
    random_state: int = 42,
) -> ImbPipeline:
    """Assemble the full leak-free pipeline for one (strategy, estimator) choice."""
    plan = build_balancing_plan(strategy, y=y, random_state=random_state)

    score_func = mutual_info_classif if feature_score == "mutual_info" else f_classif

    steps: list[tuple[str, Any]] = []
    if missing_indicators:
        # FIRST, before anything drops/imputes — preserves high-missing signal.
        steps.append(("indicators", AddMissingIndicators(threshold=indicator_threshold)))
    steps += [
        ("missing", MissingnessThreshold(threshold=missing_threshold)),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=var_threshold)),
        ("scale", StandardScaler()),
    ]
    if k_features != "all":
        steps.append(("select", SelectKBest(score_func=score_func, k=k_features)))

    if plan.sampler is not None:
        steps.append(("resample", plan.sampler))

    steps.append(("clf", build_estimator(estimator, plan, random_state=random_state)))
    return ImbPipeline(steps)


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (opt-in)
# --------------------------------------------------------------------------- #
def _param_distributions(estimator: str) -> dict:
    """Small, sensible search spaces per estimator (keys target the 'clf' step)."""
    if estimator == "logreg":
        return {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    if estimator == "balanced_rf":
        return {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 6, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2", 0.3],
        }
    if estimator == "hist_gb":
        return {
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_iter": [200, 300, 500],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__l2_regularization": [0.0, 1.0, 10.0],
        }
    if estimator == "xgboost":
        return {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [3, 4, 6],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.9],
            "clf__min_child_weight": [1, 3, 5],
        }
    return {}


def tune_hyperparameters(
    estimator: str,
    strategy: str,
    X,
    y,
    k_features: int | str = "all",
    missing_indicators: bool = True,
    n_iter: int = 25,
    cv: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
):
    """RandomizedSearchCV over the estimator's hyperparameters, inside the full
    leak-free pipeline. Returns (best_pipeline, best_params, best_score).

    The search is scored on PR-AUC by default and refits the best pipeline on all
    of ``X``. Resampling/cleaning stay inside each CV fold, so no leakage.
    """
    from sklearn.model_selection import RandomizedSearchCV

    base = build_pipeline(
        strategy=strategy, estimator=estimator, y=y, k_features=k_features,
        missing_indicators=missing_indicators, random_state=random_state,
    )
    dist = _param_distributions(estimator)
    if not dist:
        logger.info("No tuning grid for %s; fitting base pipeline.", estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base.fit(X, y)
        return base, {}, float("nan")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base, dist, n_iter=n_iter, scoring=scoring, cv=skf,
        random_state=random_state, n_jobs=-1, refit=True, error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    logger.info(
        "Tuned %s: best %s=%.3f params=%s",
        estimator, scoring, search.best_score_, search.best_params_,
    )
    return search.best_estimator_, search.best_params_, float(search.best_score_)


# --------------------------------------------------------------------------- #
# Train / cross-validate
# --------------------------------------------------------------------------- #
def make_train_test_split(
    data: SecomData, test_size: float = 0.2, random_state: int = 42
):
    """Stratified split that preserves the rare-class proportion."""
    return train_test_split(
        data.X,
        data.y,
        test_size=test_size,
        stratify=data.y,
        random_state=random_state,
    )


def cross_validate_pipeline(
    pipe: ImbPipeline,
    X,
    y,
    estimator_name: str,
    strategy_name: str,
    cv: int = 5,
    random_state: int = 42,
) -> CVResults:
    """Stratified k-fold CV with the imbalance-aware scorer suite."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorers = default_scorers()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quiet convergence/UndefinedMetric chatter
        res = cross_validate(
            pipe, X, y, cv=skf, scoring=scorers, n_jobs=-1, error_score="raise"
        )
    means = {k: float(np.mean(res[f"test_{k}"])) for k in scorers}
    stds = {k: float(np.std(res[f"test_{k}"])) for k in scorers}
    return CVResults(
        estimator=estimator_name,
        strategy=strategy_name,
        metrics_mean=means,
        metrics_std=stds,
        config={"cv": cv, "random_state": random_state},
    )


def train(pipe: ImbPipeline, X_train, y_train) -> ImbPipeline:
    """Fit the pipeline on training data and return it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
    return pipe


def compare_configs(
    data: SecomData,
    configs: list[tuple[str, str]],
    cv: int = 5,
    k_features: int | str = "all",
    missing_indicators: bool = True,
    random_state: int = 42,
) -> list[CVResults]:
    """Cross-validate several (estimator, strategy) pairs on the training split.

    Uses only the *training* split for selection, so the test split stays
    untouched for the evaluator. Returns results sorted by PR-AUC (desc).
    """
    X_train, _, y_train, _ = make_train_test_split(data, random_state=random_state)
    results = []
    for estimator, strategy in configs:
        pipe = build_pipeline(
            strategy=strategy,
            estimator=estimator,
            y=y_train,
            k_features=k_features,
            missing_indicators=missing_indicators,
            random_state=random_state,
        )
        r = cross_validate_pipeline(
            pipe, X_train, y_train, estimator, strategy, cv=cv, random_state=random_state
        )
        logger.info("%s", r.summary())
        results.append(r)
    results.sort(key=lambda r: r.metrics_mean["pr_auc"], reverse=True)
    return results


def get_selected_feature_names(pipe: ImbPipeline, input_features) -> np.ndarray:
    """Walk the preprocessing steps and return the feature names reaching the
    estimator. Stops at the resampler/estimator (which don't alter columns)."""
    names = np.asarray(input_features, dtype=object)
    for step_name, step in pipe.steps:
        if step_name in ("resample", "clf"):
            break
        if hasattr(step, "get_feature_names_out"):
            try:
                names = step.get_feature_names_out(names)
            except Exception:  # noqa: BLE001
                # Fall back to a generic count if a step can't map names.
                n = getattr(step, "n_features_out_", len(names))
                names = np.asarray([f"f{i}" for i in range(n)], dtype=object)
    return names


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="Train a SECOM yield-prediction model.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--estimator", default="logreg", choices=ESTIMATORS)
    p.add_argument("--strategy", default="smote_enn")
    p.add_argument("--k-features", default="all")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    data = load_secom(data_dir=args.data_dir)
    X_train, X_test, y_train, y_test = make_train_test_split(
        data, random_state=args.random_state
    )
    k = args.k_features if args.k_features == "all" else int(args.k_features)
    pipe = build_pipeline(
        strategy=args.strategy,
        estimator=args.estimator,
        y=y_train,
        k_features=k,
        random_state=args.random_state,
    )
    cv_res = cross_validate_pipeline(
        pipe, X_train, y_train, args.estimator, args.strategy, cv=args.cv,
        random_state=args.random_state,
    )
    print("Cross-validation on the training split:")
    print(cv_res.summary())
    train(pipe, X_train, y_train)
    print(f"\nFitted on {len(X_train)} train rows; {len(X_test)} rows held out for the evaluator.")


if __name__ == "__main__":
    main()
