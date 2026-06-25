"""models/base.py — common interface for all analytical yield models.

Every model expresses yield as Y = f(A; theta), where A is chip area in
unit-die areas and theta are the fitted parameters (always including D0,
the defect density per unit-die area).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FitResult:
    """Container for a fitted model's parameters and provenance."""
    model_name: str
    params: dict[str, float]
    covariance: np.ndarray | None = None
    success: bool = True
    message: str = ""

    @property
    def d0(self) -> float:
        return self.params["D0"]


class YieldModel(ABC):
    """Abstract analytical yield model Y(A; theta)."""

    name: str = "base"
    param_names: tuple[str, ...] = ("D0",)

    # -- the model equation --------------------------------------------------
    @staticmethod
    @abstractmethod
    def yield_fn(area: np.ndarray, *params: float) -> np.ndarray:
        """Vectorized Y(A) for given parameters. Implemented per model."""

    # -- fitting hooks (consumed by the trainer) -----------------------------
    @abstractmethod
    def initial_guess(self, cfg) -> list[float]:
        """Starting parameter vector for nonlinear least squares."""

    def bounds(self) -> tuple[list[float], list[float]]:
        """Lower/upper bounds for each parameter. Default: strictly positive."""
        lo = [1e-9] * len(self.param_names)
        hi = [np.inf] * len(self.param_names)
        return lo, hi

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def pack(self, popt: np.ndarray) -> dict[str, float]:
        """Map a raw parameter vector to a named dict."""
        return {k: float(v) for k, v in zip(self.param_names, popt)}

    def predict(self, area, result: FitResult) -> np.ndarray:
        """Evaluate the fitted model at given area(s)."""
        theta = [result.params[k] for k in self.param_names]
        return self.yield_fn(np.asarray(area, dtype=float), *theta)
