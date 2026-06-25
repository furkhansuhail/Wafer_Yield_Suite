# SECOM Yield-Prediction Pipeline

A modular, leak-free machine-learning pipeline for predicting pass/fail yield on
the [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) semiconductor
manufacturing dataset (1,567 production units × 590 sensor features, ~14:1
class imbalance, heavy missingness and noise).

## Design principles

- **Leak-free.** All cleaning (drop high-missing / dead columns, impute, scale,
  feature-select) and all resampling happen *inside* the cross-validation folds
  and are fit on the training data only. The test split is untouched until final
  evaluation.
- **Imbalance-aware.** Accuracy is never a headline metric. Selection and
  reporting use PR-AUC, failure-recall, F1, G-mean, and MCC.
- **Noise-aware ordering.** Cleaning runs *before* any SMOTE so synthetic points
  are never interpolated from noisy/unimputed data; `smote_enn` / `smote_tomek`
  additionally clean *after* synthesis.
- **Modular.** Each stage is an independent, importable module with its own CLI.

## Modules

| Module | Responsibility | Entry point |
|---|---|---|
| `data_downloader.py` | Fetch (ucimlrepo → URL fallback), cache, parse; remap labels to `0=pass / 1=fail` | `load_secom()` |
| `eda.py` | Class balance, missingness, dead features, target association, temporal drift; plots + report | `run_eda()` |
| `imbalance_balancer.py` | Resampling + cost-sensitive strategies as a `BalancingPlan` | `build_balancing_plan()` |
| `model_trainer.py` | Assemble the leak-free imblearn pipeline; stratified CV; 4 estimators | `build_pipeline()`, `compare_configs()` |
| `evaluator.py` | Held-out metrics, threshold tuning, PR/ROC/confusion plots, feature importance | `evaluate()`, `tune_threshold()` |
| `main.py` | Orchestrate all five stages | `run_pipeline()` |

## Install

```bash
pip install -r requirements.txt
```

## Quick start

End-to-end, tuning the threshold to catch at least 80% of failing units:

```bash
python -m secom_pipeline.main --recall-target 0.8
```

Compare a shortlist of models first, then train + evaluate the best:

```bash
python -m secom_pipeline.main --compare --recall-target 0.8
```

Programmatic use:

```python
from secom_pipeline.data_downloader import load_secom
from secom_pipeline.model_trainer import build_pipeline, make_train_test_split, train
from secom_pipeline.evaluator import evaluate

data = load_secom("data")
Xtr, Xte, ytr, yte = make_train_test_split(data)
pipe = build_pipeline(strategy="smote_enn", estimator="logreg", y=ytr, k_features=50)
train(pipe, Xtr, ytr)
report = evaluate(pipe, Xte, yte, recall_target=0.8, out_dir="eval_report")
print(report.summary())
```

## Strategies & estimators

- **Imbalance strategies:** `none`, `class_weight`, `random_over`, `random_under`,
  `smote`, `borderline_smote`, `adasyn`, `smote_enn`, `smote_tomek`.
  Recommended for SECOM: `class_weight`, `smote_enn`, `smote_tomek`.
- **Estimators:** `logreg`, `balanced_rf`, `hist_gb`, `xgboost`.
  (`balanced_rf` and `xgboost` handle imbalance internally → pair with `none`.)

## Output

```
secom_output/
  eda_report/    class_balance.png, missingness.png, top_features.png, temporal.png, eda_report.md
  eval_report/   pr_curve.png, roc_curve.png, threshold_sweep.png, confusion.png,
                 feature_importance.png, evaluation_report.md
```
