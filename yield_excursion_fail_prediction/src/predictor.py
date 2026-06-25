"""
predictor.py
============

Inference module for the SECOM yield-prediction pipeline.

Once a pipeline is trained and a decision threshold chosen (modules 4 & 5), this
module persists the two together and serves predictions on new production units.
A saved model is a single file containing:

* the fitted preprocessing+estimator pipeline,
* the tuned decision threshold (NOT 0.5 — that choice is part of the model),
* metadata (estimator/strategy, training metrics, feature names, timestamp).

Keeping the threshold *with* the pipeline matters: a yield model that catches
80% of failures at threshold 0.3 is a different deployed system than the same
pipeline at 0.5, and inference must use the threshold the model was tuned for.

Usage
-----
Programmatic::

    from secom_pipeline.predictor import Predictor
    Predictor.from_pipeline(pipe, threshold=0.31, metadata={...}).save("model.joblib")

    pred = Predictor.load("model.joblib")
    pred.predict_frame(new_X)        # DataFrame: fail_proba, decision, label

CLI::

    python -m secom_pipeline.predictor --model model.joblib --input new_units.csv \
        --output predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LABELS = {0: "pass", 1: "fail"}


def _positive_scores(pipe, X) -> np.ndarray:
    """Continuous failure score. Prefers predict_proba; squashes decision_function."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return 1.0 / (1.0 + np.exp(-pipe.decision_function(X)))
    raise AttributeError("Pipeline exposes neither predict_proba nor decision_function.")


@dataclass
class Predictor:
    """A deployable model: fitted pipeline + tuned threshold + metadata."""

    pipeline: object
    threshold: float = 0.5
    feature_names: list[str] | None = None
    metadata: dict = field(default_factory=dict)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_pipeline(
        cls, pipeline, threshold: float = 0.5, feature_names=None, metadata=None
    ) -> "Predictor":
        return cls(
            pipeline=pipeline,
            threshold=float(threshold),
            feature_names=list(feature_names) if feature_names is not None else None,
            metadata=metadata or {},
        )

    # ---- inference --------------------------------------------------------
    def _align(self, X) -> pd.DataFrame:
        """Coerce input to the DataFrame schema the pipeline was trained on.

        Accepts a DataFrame (used as-is if columns match) or array-like (columns
        assigned positionally as feature_000..). If trained feature names are
        known, reorder/validate against them.
        """
        if not isinstance(X, pd.DataFrame):
            X = np.asarray(X)
            cols = (
                self.feature_names
                if self.feature_names is not None
                else [f"feature_{i:03d}" for i in range(X.shape[1])]
            )
            X = pd.DataFrame(X, columns=cols)

        if self.feature_names is not None:
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                # If columns look positional/unnamed, assign by order instead of failing.
                if X.shape[1] == len(self.feature_names):
                    logger.warning(
                        "Input columns don't match training names; assigning by position."
                    )
                    X = X.copy()
                    X.columns = self.feature_names
                else:
                    raise ValueError(
                        f"Input is missing {len(missing)} expected feature(s), "
                        f"e.g. {missing[:5]}"
                    )
            X = X[self.feature_names]
        return X

    def predict_proba(self, X) -> np.ndarray:
        """Probability that each unit will FAIL in-house testing."""
        return _positive_scores(self.pipeline, self._align(X))

    def predict(self, X) -> np.ndarray:
        """0 = predicted pass, 1 = predicted fail, using the tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_labels(self, X) -> np.ndarray:
        return np.array([_LABELS[i] for i in self.predict(X)], dtype=object)

    def predict_frame(self, X) -> pd.DataFrame:
        """Tidy predictions: failure probability, 0/1 decision, and label."""
        proba = self.predict_proba(X)
        decision = (proba >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "fail_proba": np.round(proba, 4),
                "decision": decision,
                "label": [_LABELS[i] for i in decision],
            }
        )

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "metadata": {**self.metadata, "saved_at": datetime.now().isoformat(timespec="seconds")},
        }
        joblib.dump(payload, path)
        logger.info("Saved model -> %s (threshold=%.3f)", path, self.threshold)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        payload = joblib.load(Path(path))
        return cls(
            pipeline=payload["pipeline"],
            threshold=payload.get("threshold", 0.5),
            feature_names=payload.get("feature_names"),
            metadata=payload.get("metadata", {}),
        )

    def describe(self) -> str:
        md = self.metadata
        bits = [f"threshold={self.threshold:.3f}"]
        if "estimator" in md:
            bits.append(f"model={md['estimator']}+{md.get('strategy','?')}")
        if "test_pr_auc" in md:
            bits.append(f"test_pr_auc={md['test_pr_auc']}")
        if self.feature_names:
            bits.append(f"n_features_in={len(self.feature_names)}")
        return " | ".join(bits)


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #
def save_predictor(pipeline, threshold, path, feature_names=None, metadata=None) -> Path:
    return Predictor.from_pipeline(
        pipeline, threshold=threshold, feature_names=feature_names, metadata=metadata
    ).save(path)


def load_predictor(path) -> Predictor:
    return Predictor.load(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_input(path: Path) -> pd.DataFrame:
    """Read new units. Supports SECOM-style whitespace .data or a CSV."""
    if path.suffix.lower() in (".data", ".txt"):
        df = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN", "nan"], engine="python")
        df.columns = [f"feature_{i:03d}" for i in range(df.shape[1])]
        return df
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Predict yield pass/fail with a saved SECOM model.")
    p.add_argument("--model", required=True, help="Path to a saved .joblib model")
    p.add_argument("--input", required=True, help="CSV or SECOM .data file of new units")
    p.add_argument("--output", default="predictions.csv", help="Where to write predictions")
    args = p.parse_args(argv)

    predictor = Predictor.load(args.model)
    logger.info("Loaded model: %s", predictor.describe())
    X = _read_input(Path(args.input))
    preds = predictor.predict_frame(X)
    preds.to_csv(args.output, index=False)

    n_fail = int(preds["decision"].sum())
    print(f"Predicted {len(preds)} units: {n_fail} fail, {len(preds) - n_fail} pass "
          f"(threshold={predictor.threshold:.3f})")
    print(f"Written -> {args.output}")
    print(preds.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
