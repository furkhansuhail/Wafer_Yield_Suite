"""Model registry. Import MODELS to iterate over all three yield models."""
from .base import YieldModel, FitResult
from .poisson import PoissonYield
from .murphy import MurphyYield
from .negbinom import NegativeBinomialYield

# canonical ordering: increasing flexibility / clustering capability
MODEL_CLASSES = (PoissonYield, MurphyYield, NegativeBinomialYield)


def build_models() -> list[YieldModel]:
    """Fresh instances of every registered yield model."""
    return [cls() for cls in MODEL_CLASSES]


def model_by_name(name: str) -> YieldModel:
    """Return a fresh model instance for a registered model name."""
    for cls in MODEL_CLASSES:
        if cls.name == name:
            return cls()
    raise KeyError(f"unknown model '{name}'; known: "
                   f"{[c.name for c in MODEL_CLASSES]}")


__all__ = [
    "YieldModel",
    "FitResult",
    "PoissonYield",
    "MurphyYield",
    "NegativeBinomialYield",
    "MODEL_CLASSES",
    "build_models",
    "model_by_name",
]
