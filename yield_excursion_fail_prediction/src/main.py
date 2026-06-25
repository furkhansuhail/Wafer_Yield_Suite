"""
main.py
=======

Orchestrator for the SECOM yield-prediction pipeline. Runs all five stages in
order and writes their artifacts:

    download  ->  EDA  ->  [optional model comparison]  ->  train  ->  evaluate

The work is split into :func:`run_pipeline` (takes an already-loaded
``SecomData`` so it can be tested/reused without a download) and :func:`main`
(handles the CLI and the download).

Examples
--------
Full run with defaults (logreg + smote_enn), tuning to catch >=80% of failures::

    python -m secom_pipeline.main --recall-target 0.8

Compare several models first, then train+evaluate the best by PR-AUC::

    python -m secom_pipeline.main --compare --recall-target 0.8

Pick a specific config explicitly::

    python -m secom_pipeline.main --estimator xgboost --strategy none --k-features 100
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from yield_excursion_fail_prediction.src.data_downloader import SecomData, load_secom
from yield_excursion_fail_prediction.src.eda import run_eda
from yield_excursion_fail_prediction.src.evaluator import evaluate, fit_select_threshold_evaluate
from yield_excursion_fail_prediction.src.predictor import Predictor

from yield_excursion_fail_prediction.src.model_trainer import (
    build_pipeline,
    compare_configs,
    cross_validate_pipeline,
    make_train_test_split,
    train,
)

logger = logging.getLogger(__name__)

# One config per available model — used by default so every estimator is run
# and reported. Strategies follow the balancer guidance (balanced_rf/xgboost
# balance internally -> 'none'; linear/HGB use class weights).
ALL_MODELS = [
    ("logreg", "class_weight"),
    ("balanced_rf", "none"),
    ("hist_gb", "class_weight"),
    ("xgboost", "none"),
]

# Larger sweep (adds resampling variants) available via --grid.
DEFAULT_COMPARE_CONFIGS = [
    ("logreg", "class_weight"),
    ("logreg", "smote_enn"),
    ("balanced_rf", "none"),
    ("hist_gb", "class_weight"),
    ("xgboost", "none"),
]


@dataclass
class PipelineResult:
    """Everything a run produces."""
    pipe: object
    eval_report: object
    comparison_df: Optional[pd.DataFrame]
    predictor: Optional[Predictor]
    model_path: Optional[Path]
    predictions: Optional[pd.DataFrame] = None


def build_holdout_comparison(
    results,
    X_train,
    y_train,
    X_test,
    y_test,
    out_dir: Path,
    k_features: int | str = "all",
    missing_indicators: bool = True,
    objective: str = "f1",
    recall_target: float | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train every compared model, pick each one's threshold honestly (OOF on
    train), and evaluate on the held-out split. Returns a tidy side-by-side table.

    NOTE on methodology: model *selection* is still done on train-split CV (the
    caller picks ``results[0]``), and each model's threshold is chosen from
    out-of-fold training predictions — never from the test set. The held-out
    columns are reported for transparency, not used to choose the winner.
    """
    cmp_dir = out_dir / "comparison"
    rows = []
    for r in results:
        _pipe, rep, thr = fit_select_threshold_evaluate(
            r.estimator, r.strategy, X_train, y_train, X_test, y_test,
            out_dir=cmp_dir / f"{r.estimator}_{r.strategy}",
            k_features=k_features, missing_indicators=missing_indicators,
            objective=objective, recall_target=recall_target, cv=cv,
            tune=False, random_state=random_state,
            make_plots=False, compute_importance=False,
        )
        rows.append({
            "estimator": r.estimator,
            "strategy": r.strategy,
            "cv_pr_auc": round(r.metrics_mean["pr_auc"], 3),
            "test_pr_auc": round(rep.pr_auc, 3),
            "test_roc_auc": round(rep.roc_auc, 3),
            "test_recall": round(rep.metrics_tuned["recall"], 3),
            "test_precision": round(rep.metrics_tuned["precision"], 3),
            "test_f1": round(rep.metrics_tuned["f1"], 3),
            "test_g_mean": round(rep.metrics_tuned["g_mean"], 3),
        })

    df = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False, ignore_index=True)
    # Mark the selected model (best train-CV PR-AUC).
    df.insert(0, "selected", ["*" if i == 0 else "" for i in range(len(df))])

    cmp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cmp_dir / "comparison.csv", index=False)
    (cmp_dir / "comparison.md").write_text(
        "# Model comparison (held-out)\n\n"
        "`*` = selected by train-split CV PR-AUC. Held-out columns are at the "
        "tuned threshold and are reported for transparency, not used for selection.\n\n"
        + df.to_markdown(index=False)
    )
    return df


def run_pipeline(
    data: SecomData,
    out_dir: str | Path = "secom_output",
    estimator: str = "logreg",
    strategy: str = "smote_enn",
    k_features: int | str = "all",
    compare: bool = False,
    model_configs: list | None = None,
    cv: int = 5,
    objective: str = "f1",
    recall_target: float | None = None,
    missing_indicators: bool = True,
    tune: bool = False,
    n_iter: int = 25,
    random_state: int = 42,
    run_eda_stage: bool = True,
    save_model: bool = True,
    model_out: str | Path | None = None,
    predict_input: str | Path | None = None,
) -> PipelineResult:
    """Execute the pipeline on a loaded dataset.

    When ``compare=True`` every config in ``model_configs`` (default
    :data:`ALL_MODELS`) is cross-validated and evaluated on the held-out split,
    the best (by train-CV PR-AUC) is fully evaluated, and — if ``save_model`` —
    persisted as a :class:`~secom_pipeline.predictor.Predictor` bundling its
    threshold. The decision threshold is always chosen from out-of-fold training
    predictions (never the test set). ``missing_indicators`` adds "was-missing"
    features; ``tune`` runs a hyperparameter search on the selected model.
    ``predict_input`` is then scored with the saved model.

    Returns a :class:`PipelineResult`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 2: EDA -----------------------------------------------------
    if run_eda_stage:
        logger.info("Stage: EDA")
        eda_report = run_eda(data, out_dir=out_dir / "eda_report")
        logger.info(
            "EDA: %d dead, %d features >50%% missing, imbalance %.1f:1",
            len(eda_report.dead_features()),
            len(eda_report.high_missing_features(0.5)),
            eda_report.class_balance["imbalance_ratio"],
        )

    # ---- Split (test split is sacred from here on) ------------------------
    X_train, X_test, y_train, y_test = make_train_test_split(
        data, random_state=random_state
    )

    # ---- Stage 4a: optional model comparison on the TRAIN split -----------
    comparison_df = None
    if compare:
        configs = model_configs if model_configs is not None else ALL_MODELS
        logger.info("Stage: comparing %d models (train-split CV)", len(configs))
        results = compare_configs(
            data, configs, cv=cv, k_features=k_features,
            missing_indicators=missing_indicators, random_state=random_state,
        )
        best = results[0]
        estimator, strategy = best.estimator, best.strategy
        logger.info("Best by train-CV PR-AUC: %s", best.summary())
        # Evaluate every model on the held-out split (threshold chosen OOF on train).
        logger.info("Stage: held-out evaluation of all %d models", len(results))
        comparison_df = build_holdout_comparison(
            results, X_train, y_train, X_test, y_test, out_dir=out_dir,
            k_features=k_features, missing_indicators=missing_indicators,
            objective=objective, recall_target=recall_target, cv=cv,
            random_state=random_state,
        )

    # ---- Stage 4b/5: train chosen model, pick threshold OOF, evaluate on test
    logger.info(
        "Stage: training [%s + %s]%s", estimator, strategy,
        " with hyperparameter tuning" if tune else "",
    )
    pipe, eval_report, threshold = fit_select_threshold_evaluate(
        estimator, strategy, X_train, y_train, X_test, y_test,
        out_dir=out_dir / "eval_report",
        k_features=k_features, missing_indicators=missing_indicators,
        objective=objective, recall_target=recall_target, cv=cv,
        tune=tune, n_iter=n_iter, random_state=random_state,
        make_plots=True, compute_importance=True,
    )

    # ---- Stage 6: persist the deployable model (pipeline + tuned threshold) -
    predictor = None
    model_path = None
    if save_model:
        predictor = Predictor.from_pipeline(
            pipe,
            threshold=eval_report.tuned_threshold,
            feature_names=list(data.X.columns),
            metadata={
                "estimator": estimator,
                "strategy": strategy,
                "test_pr_auc": round(eval_report.pr_auc, 4),
                "test_recall": round(eval_report.metrics_tuned["recall"], 4),
                "test_precision": round(eval_report.metrics_tuned["precision"], 4),
                "tuning_objective": eval_report.tuning_objective,
                "n_train_rows": int(len(X_train)),
            },
        )
        model_path = Path(model_out) if model_out else (out_dir / "model.joblib")
        predictor.save(model_path)

    # ---- Stage 7: optional prediction on new units ------------------------
    predictions = None
    if predict_input is not None and predictor is not None:
        from .predictor import _read_input

        logger.info("Stage: predicting on %s", predict_input)
        X_new = _read_input(Path(predict_input))
        predictions = predictor.predict_frame(X_new)
        pred_path = out_dir / "predictions.csv"
        predictions.to_csv(pred_path, index=False)
        logger.info("Wrote %d predictions -> %s", len(predictions), pred_path)

    return PipelineResult(
        pipe=pipe,
        eval_report=eval_report,
        comparison_df=comparison_df,
        predictor=predictor,
        model_path=model_path,
        predictions=predictions,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full SECOM pipeline end-to-end.")
    p.add_argument("--data-dir", default="data", help="Cache dir for the dataset")
    p.add_argument("--out-dir", default="secom_output", help="Where artifacts are written")
    p.add_argument("--estimator", default=None,
                   choices=["logreg", "balanced_rf", "hist_gb", "xgboost"],
                   help="Run only this model. If omitted, ALL models are run and compared.")
    p.add_argument("--strategy", default="class_weight",
                   help="Imbalance strategy when --estimator is given")
    p.add_argument("--grid", action="store_true",
                   help="Use the larger sweep (adds resampling variants) instead of one-per-model")
    p.add_argument("--k-features", default="all", help="'all' or an integer")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--objective", default="f1", choices=["f1", "g_mean"])
    p.add_argument("--recall-target", type=float, default=None,
                   help="Tune the threshold to catch at least this fraction of failures")
    p.add_argument("--tune", action="store_true",
                   help="Hyperparameter search (RandomizedSearchCV) on the selected model")
    p.add_argument("--n-iter", type=int, default=25, help="Search iterations when --tune")
    p.add_argument("--no-missing-indicators", action="store_true",
                   help="Disable 'was-missing' indicator features")
    p.add_argument("--no-eda", action="store_true", help="Skip the EDA stage")
    p.add_argument("--no-save-model", action="store_true", help="Don't persist the trained model")
    p.add_argument("--model-out", default=None, help="Path for the saved model (default: <out-dir>/model.joblib)")
    p.add_argument("--predict", default=None,
                   help="CSV or SECOM .data file of new units to score with the trained model")
    p.add_argument("--random-state", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_arg_parser().parse_args(argv)

    logger.info("Stage: download/load")
    data = load_secom(data_dir=args.data_dir)
    logger.info("%s", data.summary())

    # Default behaviour: run ALL models and compare. Pass --estimator to run one.
    run_all = args.estimator is None
    model_configs = DEFAULT_COMPARE_CONFIGS if args.grid else ALL_MODELS

    result = run_pipeline(
        data,
        out_dir=args.out_dir,
        estimator=args.estimator or "logreg",
        strategy=args.strategy,
        k_features=args.k_features,
        compare=run_all,
        model_configs=model_configs,
        cv=args.cv,
        objective=args.objective,
        recall_target=args.recall_target,
        missing_indicators=not args.no_missing_indicators,
        tune=args.tune,
        n_iter=args.n_iter,
        random_state=args.random_state,
        run_eda_stage=not args.no_eda,
        save_model=not args.no_save_model,
        model_out=args.model_out,
        predict_input=args.predict,
    )

    if result.comparison_df is not None:
        print("\n" + "=" * 78)
        print("MODEL COMPARISON — held-out test split (all available models)")
        print("=" * 78)
        print(result.comparison_df.to_string(index=False))
        print("\n  '*' = model selected by train-split CV PR-AUC (test not used for selection)")

    print("\n" + "=" * 78)
    print("SELECTED MODEL — held-out result")
    print("=" * 78)
    print(result.eval_report.summary())

    if result.model_path is not None:
        print(f"\nSaved model: {result.model_path}")
        print(f"  {result.predictor.describe()}")
        print("  Predict on new units with:")
        print(f"    python -m secom_pipeline.predictor --model {result.model_path} "
              f"--input new_units.csv --output predictions.csv")

    if result.predictions is not None:
        n_fail = int(result.predictions["decision"].sum())
        print(f"\nPredictions on '{args.predict}': "
              f"{n_fail} fail / {len(result.predictions) - n_fail} pass "
              f"-> {args.out_dir}/predictions.csv")

    print(f"\nAll artifacts under: {args.out_dir}/")
    print("  eda_report/        EDA plots + eda_report.md")
    print("  eval_report/       PR/ROC/threshold/confusion plots + evaluation_report.md")
    if result.comparison_df is not None:
        print("  comparison/        comparison.csv + comparison.md + per-model reports")
    if result.model_path is not None:
        print("  model.joblib       deployable pipeline + tuned threshold")


if __name__ == "__main__":
    main()
