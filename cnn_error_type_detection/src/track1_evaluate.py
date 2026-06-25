"""
Track 1 — Module 5: Backend-agnostic evaluation.

Responsibility
--------------
Load a trained model (Keras .keras OR PyTorch TorchScript .pt), score the
held-out test split, and emit the metrics that actually reveal whether the
model beat the class imbalance:
  * per-class precision / recall / F1  (NOT just accuracy — accuracy lies when
    one class is ~80% of the data)
  * macro-F1 and weighted-F1
  * per-class PR-AUC (one-vs-rest average precision)
  * confusion matrix

Backend dispatch
----------------
Decided by file extension — the evaluator imports NEITHER model class:
  .keras / .h5  -> tensorflow.keras.models.load_model ; X as (N,H,W,1)
  .pt  / .pth   -> torch.jit.load                     ; X as (N,1,H,W)

Outputs
-------
  reports/eval_metrics.json
  reports/eval_confusion_matrix.png
  reports/eval_pr_curves.png

Usage
-----
    # Point at a model file...
    python track1_evaluate.py --model models/keras/best_model.keras --data-dir data/processed
    python track1_evaluate.py --model models/torch/best_model.pt     --data-dir data/processed
    # ...or at the directory (it finds best_model.* automatically):
    python track1_evaluate.py --model models/torch --data-dir data/processed

    python track1_evaluate.py --self-test     # no real model/data needed

Testable individually
---------------------
`--self-test` validates the metric math on synthetic predictions with KNOWN
outcomes (perfect + one-error cases), checks backend detection, confirms the
plots/JSON are written, and — since PyTorch is the lighter dep — runs a real
end-to-end load+predict through a tiny TorchScript model when torch is present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

KERAS_EXTS = {".keras", ".h5"}
TORCH_EXTS = {".pt", ".pth"}


# --------------------------------------------------------------------------- #
# Model resolution + backend dispatch
# --------------------------------------------------------------------------- #

def resolve_model_path(model: str) -> Path:
    """Accept a file or a directory; if a dir, find best_model.{keras,pt}."""
    p = Path(model)
    if p.is_dir():
        for name in ("best_model.keras", "best_model.pt", "best_model.h5"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"No best_model.* found in {p}")
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return p


def detect_backend(model_path: Path) -> str:
    ext = Path(model_path).suffix.lower()
    if ext in KERAS_EXTS:
        return "keras"
    if ext in TORCH_EXTS:
        return "torch"
    raise ValueError(f"Cannot infer backend from extension '{ext}'. "
                     f"Use --backend keras|torch.")


def predict(model_path: Path, X: np.ndarray, backend: str) -> np.ndarray:
    """
    Run inference. X is (N,H,W). Adds the per-backend channel axis and returns
    class probabilities (N, n_classes).
    """
    if backend == "keras":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return np.asarray(model.predict(X[..., np.newaxis], verbose=0))
    if backend == "torch":
        import torch
        model = torch.jit.load(str(model_path))
        model.eval()
        xb = torch.from_numpy(X.astype("float32")).unsqueeze(1)   # (N,1,H,W)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()
    raise ValueError(f"Unknown backend: {backend}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    """Per-class P/R/F1 + macro/weighted F1 + per-class PR-AUC + confusion matrix."""
    n_classes = len(classes)
    y_pred = probs.argmax(axis=1)
    labels = list(range(n_classes))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0)

    # One-vs-rest PR-AUC (average precision). Undefined for classes absent in y_true.
    y_onehot = np.eye(n_classes)[y_true]
    pr_auc = {}
    for i, cls in enumerate(classes):
        if y_onehot[:, i].sum() == 0:
            pr_auc[cls] = None
        else:
            pr_auc[cls] = float(average_precision_score(y_onehot[:, i], probs[:, i]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for cls in classes:
        r = report[cls]
        per_class[cls] = {
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
            "pr_auc": (round(pr_auc[cls], 4) if pr_auc[cls] is not None else None),
        }

    return {
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_confusion(cm, classes, out: Path) -> Path:
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row != 0)  # row-normalized
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.2 * len(classes) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_pr_curves(y_true, probs, classes, out: Path) -> Path:
    from sklearn.metrics import precision_recall_curve
    y_onehot = np.eye(len(classes))[y_true]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(classes):
        if y_onehot[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_onehot[:, i], probs[:, i])
        ap = average_precision_score(y_onehot[:, i], probs[:, i])
        ax.plot(rec, prec, label=f"{cls} (AP={ap:.2f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Per-class precision-recall"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def load_classes(model_path: Path, data_dir: Path) -> list[str]:
    sib = model_path.parent / "classes.json"
    if sib.exists():
        return json.loads(sib.read_text())
    alt = Path(data_dir) / "label_classes.json"
    if alt.exists():
        return json.loads(alt.read_text())
    raise FileNotFoundError("No classes.json (next to model) or label_classes.json (data-dir).")


def run(model: str, data_dir: Path, reports_dir: Path, backend: str | None = None) -> dict:
    data_dir, reports_dir = Path(data_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(model)
    backend = backend or detect_backend(model_path)
    classes = load_classes(model_path, data_dir)
    log.info("Evaluating %s (backend=%s, %d classes)", model_path, backend, len(classes))

    d = np.load(data_dir / "test.npz")
    X, y_true = d["X"], d["y"]
    probs = predict(model_path, X, backend)

    metrics = compute_metrics(y_true, probs, classes)
    (reports_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(metrics["confusion_matrix"], classes,
                   reports_dir / "eval_confusion_matrix.png")
    plot_pr_curves(y_true, probs, classes, reports_dir / "eval_pr_curves.png")

    log.info("accuracy=%.3f  macro_f1=%.3f  weighted_f1=%.3f",
             metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
    worst = min(metrics["per_class"].items(), key=lambda kv: kv[1]["f1"])
    log.info("weakest class: %s (F1=%.3f) — watch this one", worst[0], worst[1]["f1"])
    log.info("Artifacts -> %s/", reports_dir)
    return metrics


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _onehot_probs(y, n_classes):
    return np.eye(n_classes)[y].astype("float32")


def self_test() -> bool:
    print("=" * 60)
    print("SELF-TEST: track1_evaluate")
    print("=" * 60)
    passed = True
    classes = ["A", "B", "C"]

    # 1. backend detection by extension
    t1 = (detect_backend(Path("m/best_model.keras")) == "keras"
          and detect_backend(Path("m/best_model.pt")) == "torch")
    print(f"[{'PASS' if t1 else 'FAIL'}] backend detected from extension")
    passed &= t1

    # 2. perfect predictions -> accuracy 1, macro_f1 1, PR-AUC 1
    y = np.array([0, 0, 1, 1, 2, 2])
    m = compute_metrics(y, _onehot_probs(y, 3), classes)
    t2 = (m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
          and all(m["per_class"][c]["pr_auc"] == 1.0 for c in classes))
    print(f"[{'PASS' if t2 else 'FAIL'}] perfect preds -> acc=1, macro_f1=1, PR-AUC=1")
    passed &= t2

    # 3. one confusion: a true-A predicted as B. Check the confusion matrix cell.
    probs = _onehot_probs(y, 3).copy()
    probs[0] = [0.1, 0.9, 0.0]                 # sample 0 (true A) -> predicted B
    m2 = compute_metrics(y, probs, classes)
    cm = np.array(m2["confusion_matrix"])
    t3 = cm[0, 1] == 1 and cm[0, 0] == 1 and m2["accuracy"] < 1.0
    print(f"[{'PASS' if t3 else 'FAIL'}] one A->B error reflected in confusion matrix")
    passed &= t3

    # 4. recall for class A drops to 0.5 (1 of 2 correct), others stay 1.0
    t4 = (abs(m2["per_class"]["A"]["recall"] - 0.5) < 1e-6
          and m2["per_class"]["C"]["recall"] == 1.0)
    print(f"[{'PASS' if t4 else 'FAIL'}] per-class recall correct "
          f"(A={m2['per_class']['A']['recall']})")
    passed &= t4

    # 5. plots + json written
    with tempfile.TemporaryDirectory() as tmp:
        rep = Path(tmp) / "reports"; rep.mkdir()
        plot_confusion(m2["confusion_matrix"], classes, rep / "cm.png")
        plot_pr_curves(y, probs, classes, rep / "pr.png")
        (rep / "eval_metrics.json").write_text(json.dumps(m2))
        t5 = all((rep / f).exists() for f in ["cm.png", "pr.png", "eval_metrics.json"])
    print(f"[{'PASS' if t5 else 'FAIL'}] confusion + PR plots + metrics json written")
    passed &= t5

    # 6. real end-to-end through the torch path (torch is the lighter dep here)
    try:
        import torch
        import torch.nn as nn
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "processed"; data.mkdir()
            mdir = tmp / "models"; mdir.mkdir()
            # tiny dataset + trivial scripted classifier
            X = np.random.default_rng(0).random((12, 8, 8)).astype("float32")
            yv = np.array([0, 1, 2] * 4, dtype="int64")
            np.savez_compressed(data / "test.npz", X=X, y=yv)
            (mdir / "classes.json").write_text(json.dumps(classes))

            net = nn.Sequential(nn.Flatten(), nn.Linear(64, 3))
            scripted = torch.jit.trace(net.eval(), torch.zeros(1, 1, 8, 8).flatten(1))
            # wrap so it accepts (N,1,8,8) like the real model
            class Wrap(nn.Module):
                def __init__(self, lin): super().__init__(); self.lin = lin
                def forward(self, x): return self.lin(x.flatten(1))
            wrapped = torch.jit.script(Wrap(net.eval()))
            wrapped.save(str(mdir / "best_model.pt"))

            metrics = run(str(mdir), data, tmp / "reports")
            t6 = ("accuracy" in metrics and 0.0 <= metrics["accuracy"] <= 1.0
                  and (tmp / "reports" / "eval_metrics.json").exists())
        print(f"[{'PASS' if t6 else 'FAIL'}] real torch load+predict+report end-to-end")
        passed &= t6
    except ImportError:
        print("[SKIP] torch not installed — end-to-end load test skipped")

    print("-" * 60)
    print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
    print("=" * 60)
    return passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wafer-map evaluation (Track 1, Module 5)")
    ap.add_argument("--model", help="Model file or directory (models/keras | models/torch)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--backend", choices=["keras", "torch"],
                    help="Override auto-detection")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if not args.model:
        ap.error("--model is required (or use --self-test)")
    run(args.model, Path(args.data_dir), Path(args.reports_dir), backend=args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
