"""persistence.py — save and load fitted yield models.

A fitted model is just its name plus a handful of float parameters, so the
whole trained bundle serializes cleanly to JSON. No pickle, no version traps:
the file is human-readable and you can eyeball D0/alpha directly.

Bundle layout:
{
  "schema_version": 1,
  "best_model": "negative_binomial",
  "ref_die_area_mm2": 1.0,
  "models": { "poisson": {"params": {...}, "success": true, "scores": {...}}, ... },
  "metadata": { ... }
}
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import datetime as _dt

from .models.base import FitResult


SCHEMA_VERSION = 1


@dataclass
class ModelBundle:
    """Everything needed to make predictions later, plus provenance."""
    best_model: str
    models: dict[str, dict]          # name -> {params, success, scores}
    ref_die_area_mm2: float = 1.0
    metadata: dict | None = None

    def fit_result(self, name: str) -> FitResult:
        """Reconstruct a FitResult for the named model."""
        entry = self.models[name]
        return FitResult(
            model_name=name,
            params={k: float(v) for k, v in entry["params"].items()},
            covariance=None,
            success=bool(entry.get("success", True)),
            message="loaded from disk",
        )


class ModelStore:
    """Reads/writes a ModelBundle to a JSON file."""

    @staticmethod
    def save(
        path: str | Path,
        fits: dict[str, FitResult],
        best_model: str,
        leaderboard=None,
        ref_die_area_mm2: float = 1.0,
        metadata: dict | None = None,
    ) -> Path:
        scores_by_model = {}
        if leaderboard is not None and not leaderboard.empty:
            for row in leaderboard.to_dict(orient="records"):
                scores_by_model[row["model_name"]] = {
                    k: row[k]
                    for k in ("rmse", "weighted_rmse", "r2", "aic", "bic")
                    if k in row
                }

        models_blob = {
            name: {
                "params": fit.params,
                "success": fit.success,
                "scores": scores_by_model.get(name, {}),
            }
            for name, fit in fits.items()
        }
        meta = dict(metadata or {})
        meta.setdefault("trained_at", _dt.datetime.now().isoformat(timespec="seconds"))

        bundle = {
            "schema_version": SCHEMA_VERSION,
            "best_model": best_model,
            "ref_die_area_mm2": ref_die_area_mm2,
            "models": models_blob,
            "metadata": meta,
        }
        path = Path(path)
        path.write_text(json.dumps(bundle, indent=2, default=str))
        return path

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model file at {path}. Run `main.py train` first."
            )
        blob = json.loads(path.read_text())
        if blob.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {blob.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}. Retrain."
            )
        return ModelBundle(
            best_model=blob["best_model"],
            models=blob["models"],
            ref_die_area_mm2=blob.get("ref_die_area_mm2", 1.0),
            metadata=blob.get("metadata"),
        )
