"""
evaluator.py
============

Module 5 of the SECOM yield-prediction pipeline — held-out evaluation.

Takes a *fitted* pipeline (from module 4) and an untouched test split, and
produces the honest picture:

* Threshold-independent metrics: PR-AUC (primary for imbalance), ROC-AUC.
* Threshold tuning: the default 0.5 cutoff is rarely right on a 14:1 problem.
  We sweep the decision threshold and pick the operating point that best meets
  the goal — maximise F1, maximise G-mean, or hit a minimum failure-recall
  target (e.g. "catch >= 80% of failing units") at the best available precision.
* Confusion matrices and per-class metrics at BOTH the default and tuned
  thresholds, so the cost of the trade-off is explicit.
* PR curve, ROC curve, threshold sweep, and feature importance plots.
* Feature importance: native (logreg coefficients / tree importances) when the
  estimator exposes it, otherwise permutation importance on the full pipeline
  (works for any model, e.g. HistGradientBoosting).

Everything is computed on data the pipeline never saw during fitting or
threshold selection-on-train, so the numbers are a fair estimate of field
performance.

CLI
---
    python -m secom_pipeline.evaluator --estimator logreg --strategy smote_enn \
        --recall-target 0.8
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from imblearn.metrics import geometric_mean_score  # noqa: E402

from yield_excursion_fail_prediction.src.data_downloader import load_secom  # noqa: E402
from yield_excursion_fail_prediction.src.model_trainer import (  # noqa: E402
    build_pipeline,
    get_selected_feature_names,
    make_train_test_split,
    train,
    tune_hyperparameters,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationReport:
    pr_auc: float
    roc_auc: float
    default_threshold: float
    tuned_threshold: float
    tuning_objective: str
    metrics_default: dict           # precision/recall/f1/g_mean at 0.5
    metrics_tuned: dict             # ...at the tuned threshold
    confusion_default: np.ndarray   # 2x2 at 0.5
    confusion_tuned: np.ndarray     # 2x2 at tuned threshold
    importances: pd.DataFrame       # feature, importance, std, method
    artifacts: dict = field(default_factory=dict)

    def summary(self) -> str:
        d, t = self.metrics_default, self.metrics_tuned
        return (
            f"PR-AUC={self.pr_auc:.3f}  ROC-AUC={self.roc_auc:.3f}\n"
            f"  default (thr=0.50): recall={d['recall']:.3f} "
            f"precision={d['precision']:.3f} f1={d['f1']:.3f} g_mean={d['g_mean']:.3f}\n"
            f"  tuned   (thr={self.tuned_threshold:.2f}): recall={t['recall']:.3f} "
            f"precision={t['precision']:.3f} f1={t['f1']:.3f} g_mean={t['g_mean']:.3f}"
        )


# --------------------------------------------------------------------------- #
# Scores & metrics
# --------------------------------------------------------------------------- #
def _positive_scores(pipe, X) -> np.ndarray:
    """Continuous score for the positive (fail) class. Prefers predict_proba."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(X)
        # squash to (0,1) so a 0.5 'default' is meaningful
        return 1.0 / (1.0 + np.exp(-s))
    raise AttributeError("Estimator exposes neither predict_proba nor decision_function.")


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "g_mean": float(geometric_mean_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Threshold tuning
# --------------------------------------------------------------------------- #
def tune_threshold(
    y_true,
    scores,
    objective: str = "f1",
    recall_target: float | None = None,
) -> float:
    """Choose an operating threshold.

    * ``recall_target`` set -> among thresholds achieving at least that failure
      recall, return the one with the highest precision (the strictest threshold
      that still catches enough failures).
    * else ``objective='f1'``    -> threshold maximising F1.
    * else ``objective='g_mean'``-> threshold maximising sqrt(TPR*(1-FPR)).
    """
    y_true = np.asarray(y_true)

    if recall_target is not None:
        prec, rec, thr = precision_recall_curve(y_true, scores)
        # prec/rec have length len(thr)+1; align by dropping the last point.
        prec, rec = prec[:-1], rec[:-1]
        ok = rec >= recall_target
        if not ok.any():
            logger.warning(
                "No threshold reaches recall>=%.2f; falling back to max-recall point.",
                recall_target,
            )
            return float(thr[int(np.argmax(rec))])
        # among feasible, pick highest precision (ties -> highest threshold)
        feasible_idx = np.where(ok)[0]
        best = feasible_idx[np.argmax(prec[feasible_idx])]
        return float(thr[best])

    if objective == "g_mean":
        fpr, tpr, thr = roc_curve(y_true, scores)
        g = np.sqrt(tpr * (1 - fpr))
        # roc_curve's first threshold is +inf; ignore it
        valid = np.isfinite(thr)
        idx = np.argmax(np.where(valid, g, -1))
        return float(thr[idx])

    # default: maximise F1
    prec, rec, thr = precision_recall_curve(y_true, scores)
    prec, rec = prec[:-1], rec[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(thr[int(np.argmax(f1))])


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def feature_importance(
    pipe, X_test, y_test, input_features, n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """Native importances if the estimator exposes them, else permutation."""
    clf = pipe.named_steps["clf"]
    reaching = get_selected_feature_names(pipe, input_features)

    # 1) native coefficients (linear models)
    if hasattr(clf, "coef_"):
        imp = np.abs(np.ravel(clf.coef_))
        if len(imp) == len(reaching):
            return pd.DataFrame(
                {"feature": reaching, "importance": imp, "std": 0.0, "method": "coef"}
            ).sort_values("importance", ascending=False, ignore_index=True)

    # 2) native tree importances
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
        if len(imp) == len(reaching):
            return pd.DataFrame(
                {"feature": reaching, "importance": imp, "std": 0.0, "method": "tree"}
            ).sort_values("importance", ascending=False, ignore_index=True)

    # 3) permutation importance on the whole pipeline (original feature space)
    logger.info("Using permutation importance (estimator exposes no native scores).")
    r = permutation_importance(
        pipe, X_test, y_test, scoring="average_precision",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    feats = np.asarray(input_features, dtype=object)
    return pd.DataFrame(
        {"feature": feats, "importance": r.importances_mean,
         "std": r.importances_std, "method": "permutation"}
    ).sort_values("importance", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_pr(y_true, scores, default_thr, tuned_thr, path):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, color="#4C72B0", label=f"PR (AP={ap:.3f})")
    baseline = np.mean(y_true)
    ax.axhline(baseline, ls="--", color="grey", lw=1, label=f"no-skill ({baseline:.3f})")
    for t, name, c in [(default_thr, "default 0.5", "#999999"), (tuned_thr, "tuned", "#C44E52")]:
        idx = np.argmin(np.abs(thr - t)) if len(thr) else 0
        if len(thr):
            ax.scatter(rec[idx], prec[idx], color=c, zorder=5, label=f"{name} (thr={t:.2f})")
    ax.set_xlabel("recall (failures caught)")
    ax.set_ylabel("precision")
    ax.set_title("Precision–Recall curve")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_roc(y_true, scores, path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#55A868", label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_threshold_sweep(y_true, scores, tuned_thr, path):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    prec, rec = prec[:-1], rec[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr, prec, label="precision", color="#4C72B0")
    ax.plot(thr, rec, label="recall", color="#C44E52")
    ax.plot(thr, f1, label="F1", color="#8172B3")
    ax.axvline(tuned_thr, ls="--", color="k", lw=1, label=f"tuned={tuned_thr:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title("Metrics vs decision threshold"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_confusions(cm_default, cm_tuned, tuned_thr, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, cm, title in [
        (axes[0], cm_default, "threshold = 0.50"),
        (axes[1], cm_tuned, f"threshold = {tuned_thr:.2f}"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred pass", "pred fail"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true pass", "true fail"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(title)
    fig.suptitle("Confusion matrices")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_importance(imp: pd.DataFrame, path, n=20):
    top = imp.head(n)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(top)), top["importance"], xerr=top["std"], color="#4C72B0")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"importance ({top['method'].iloc[0]})")
    ax.set_title(f"Top-{n} features")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(
    pipe,
    X_test,
    y_test,
    out_dir: str | Path = "eval_report",
    objective: str = "f1",
    recall_target: float | None = None,
    threshold: float | None = None,
    make_plots: bool = True,
    compute_importance: bool = True,
) -> EvaluationReport:
    """Full held-out evaluation of a fitted pipeline.

    If ``threshold`` is given, it is used as-is (it should have been chosen on a
    validation split / out-of-fold predictions, NOT on this test set). If it is
    ``None``, the threshold is tuned on the test set — convenient but mildly
    optimistic, so a warning is logged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = _positive_scores(pipe, X_test)
    pr_auc = float(average_precision_score(y_test, scores))
    roc_auc = float(roc_auc_score(y_test, scores))

    default_thr = 0.5
    if threshold is None:
        logger.warning(
            "No frozen threshold supplied; tuning on the test set (optimistic). "
            "Prefer fit_select_threshold_evaluate() for an honest operating point."
        )
        tuned_thr = tune_threshold(y_test, scores, objective=objective, recall_target=recall_target)
        objective_label = f"recall>={recall_target}" if recall_target else objective
    else:
        tuned_thr = float(threshold)
        objective_label = "frozen (validation/OOF)"

    m_default = metrics_at_threshold(y_test, scores, default_thr)
    m_tuned = metrics_at_threshold(y_test, scores, tuned_thr)
    cm_default = confusion_matrix(y_test, (scores >= default_thr).astype(int))
    cm_tuned = confusion_matrix(y_test, (scores >= tuned_thr).astype(int))

    importances = pd.DataFrame(columns=["feature", "importance", "std", "method"])
    if compute_importance:
        importances = feature_importance(pipe, X_test, y_test, X_test.columns)

    artifacts: dict = {}
    if make_plots:
        _plot_pr(y_test, scores, default_thr, tuned_thr, out_dir / "pr_curve.png")
        _plot_roc(y_test, scores, out_dir / "roc_curve.png")
        _plot_threshold_sweep(y_test, scores, tuned_thr, out_dir / "threshold_sweep.png")
        _plot_confusions(cm_default, cm_tuned, tuned_thr, out_dir / "confusion.png")
        artifacts.update(
            pr_curve=str(out_dir / "pr_curve.png"),
            roc_curve=str(out_dir / "roc_curve.png"),
            threshold_sweep=str(out_dir / "threshold_sweep.png"),
            confusion=str(out_dir / "confusion.png"),
        )
        if compute_importance and not importances.empty:
            _plot_importance(importances, out_dir / "feature_importance.png")
            artifacts["feature_importance"] = str(out_dir / "feature_importance.png")

    report = EvaluationReport(
        pr_auc=pr_auc, roc_auc=roc_auc,
        default_threshold=default_thr, tuned_threshold=tuned_thr,
        tuning_objective=objective_label,
        metrics_default=m_default, metrics_tuned=m_tuned,
        confusion_default=cm_default, confusion_tuned=cm_tuned,
        importances=importances, artifacts=artifacts,
    )

    md = out_dir / "evaluation_report.md"
    md.write_text(_render_markdown(report))
    artifacts["markdown"] = str(md)
    report.artifacts = artifacts
    logger.info("Evaluation written to %s", out_dir)
    return report


def select_threshold_oof(
    estimator: str,
    strategy: str,
    X_train,
    y_train,
    k_features="all",
    missing_indicators: bool = True,
    objective: str = "f1",
    recall_target: float | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> float:
    """Pick a decision threshold from OUT-OF-FOLD predictions on the training set.

    cross_val_predict yields a held-out probability for every training row, so the
    threshold is chosen on data the (fold) model didn't see — no peeking at the
    test set. This is the honest alternative to tuning the threshold on test.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    pipe = build_pipeline(
        strategy=strategy, estimator=estimator, y=y_train,
        k_features=k_features, missing_indicators=missing_indicators,
        random_state=random_state,
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        oof = cross_val_predict(
            pipe, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
    return tune_threshold(y_train, oof, objective=objective, recall_target=recall_target)


def fit_select_threshold_evaluate(
    estimator: str,
    strategy: str,
    X_train,
    y_train,
    X_test,
    y_test,
    out_dir: str | Path = "eval_report",
    k_features="all",
    missing_indicators: bool = True,
    objective: str = "f1",
    recall_target: float | None = None,
    cv: int = 5,
    tune: bool = False,
    n_iter: int = 25,
    random_state: int = 42,
    make_plots: bool = True,
    compute_importance: bool = True,
):
    """Train, choose a threshold honestly (OOF on train), then evaluate on test.

    The decision threshold is selected from out-of-fold predictions on the
    training split, frozen, and only THEN applied to the untouched test split.
    Optionally tunes hyperparameters first.

    Returns ``(fitted_pipe, eval_report, threshold)``.
    """
    thr = select_threshold_oof(
        estimator, strategy, X_train, y_train, k_features=k_features,
        missing_indicators=missing_indicators, objective=objective,
        recall_target=recall_target, cv=cv, random_state=random_state,
    )

    if tune:
        pipe, _params, _score = tune_hyperparameters(
            estimator, strategy, X_train, y_train, k_features=k_features,
            missing_indicators=missing_indicators, n_iter=n_iter, cv=cv,
            random_state=random_state,
        )
    else:
        pipe = build_pipeline(
            strategy=strategy, estimator=estimator, y=y_train,
            k_features=k_features, missing_indicators=missing_indicators,
            random_state=random_state,
        )
        train(pipe, X_train, y_train)

    report = evaluate(
        pipe, X_test, y_test, out_dir=out_dir, threshold=thr,
        make_plots=make_plots, compute_importance=compute_importance,
    )
    return pipe, report, thr


def _render_markdown(r: EvaluationReport) -> str:
    cm_d, cm_t = r.confusion_default, r.confusion_tuned
    lines = [
        "# SECOM — Held-out Evaluation",
        "",
        f"- PR-AUC (avg precision): **{r.pr_auc:.3f}**",
        f"- ROC-AUC: **{r.roc_auc:.3f}**",
        f"- Tuning objective: **{r.tuning_objective}** -> threshold **{r.tuned_threshold:.3f}**",
        "",
        "## Operating points",
        "",
        "| metric | default (0.50) | tuned |",
        "|---|---|---|",
        f"| recall (failures caught) | {r.metrics_default['recall']:.3f} | {r.metrics_tuned['recall']:.3f} |",
        f"| precision | {r.metrics_default['precision']:.3f} | {r.metrics_tuned['precision']:.3f} |",
        f"| F1 | {r.metrics_default['f1']:.3f} | {r.metrics_tuned['f1']:.3f} |",
        f"| G-mean | {r.metrics_default['g_mean']:.3f} | {r.metrics_tuned['g_mean']:.3f} |",
        "",
        f"Confusion @0.50  [[TN={cm_d[0,0]}, FP={cm_d[0,1]}], [FN={cm_d[1,0]}, TP={cm_d[1,1]}]]",
        "",
        f"Confusion @{r.tuned_threshold:.2f}  [[TN={cm_t[0,0]}, FP={cm_t[0,1]}], [FN={cm_t[1,0]}, TP={cm_t[1,1]}]]",
        "",
        "## Top 15 features",
        "",
        (r.importances.head(15).to_markdown(index=False) if not r.importances.empty else "_n/a_"),
        "",
        "_Note: on this imbalanced problem, PR-AUC and failure-recall matter more "
        "than accuracy. The tuned threshold trades precision for catching more "
        "failing units — set `recall_target` to encode the cost of a missed defect._",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI (ties all five modules together for a runnable demo)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="Evaluate a SECOM model on a held-out split.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--estimator", default="logreg")
    p.add_argument("--strategy", default="smote_enn")
    p.add_argument("--k-features", default="all")
    p.add_argument("--objective", default="f1", choices=["f1", "g_mean"])
    p.add_argument("--recall-target", type=float, default=None)
    p.add_argument("--out-dir", default="eval_report")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    data = load_secom(data_dir=args.data_dir)
    X_train, X_test, y_train, y_test = make_train_test_split(data, random_state=args.random_state)
    k = args.k_features if args.k_features == "all" else int(args.k_features)
    pipe = build_pipeline(
        strategy=args.strategy, estimator=args.estimator, y=y_train,
        k_features=k, random_state=args.random_state,
    )
    train(pipe, X_train, y_train)
    report = evaluate(
        pipe, X_test, y_test, out_dir=args.out_dir,
        objective=args.objective, recall_target=args.recall_target,
    )
    print(report.summary())
    print(f"\nArtifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
