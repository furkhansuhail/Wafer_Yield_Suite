"""
imbalance_balancer.py
=====================

Module 3 of the SECOM yield-prediction pipeline.

Responsibilities
----------------
Provide the *imbalance-handling* component of the pipeline as interchangeable,
named strategies. Two families are supported:

1. **Resampling** (changes the training distribution):
     random_over, random_under, smote, borderline_smote, adasyn,
     smote_enn, smote_tomek
2. **Cost-sensitive** (leaves the data alone, re-weights the loss instead):
     class_weight   (and a scale_pos_weight value for XGBoost-style learners)

Plus the do-nothing baseline ``none``.

Why a "plan" object
-------------------
A balancing choice affects *two* places in the pipeline: it may insert a
resampling step, AND it may need the final estimator to carry class weights.
So instead of returning a bare sampler, :func:`build_balancing_plan` returns a
:class:`BalancingPlan` that tells the model_trainer everything it needs:
the sampler (or ``None``), whether to set ``class_weight`` on the estimator,
and a ``scale_pos_weight`` value.

Noise + imbalance interaction (the important bit)
------------------------------------------------
On a noisy, missing-heavy set like SECOM the *order* matters:

    impute/scale/feature-select  ->  RESAMPLE  ->  estimator

Cleaning must come **before** any SMOTE, or SMOTE interpolates between noisy
points and amplifies the noise. ``smote_enn`` and ``smote_tomek`` go one step
further: they oversample, then immediately delete the ambiguous points that
synthesis created near the class boundary. That post-cleaning is why they are
the recommended resamplers for SECOM. The cleaning steps themselves live in
the model_trainer (so they fit on the train fold only); this module owns only
the resample/reweight decision.

CLI
---
    python -m secom_pipeline.imbalance_balancer --demo
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Strategy names grouped by family.
RESAMPLING_STRATEGIES = (
    "random_over",
    "random_under",
    "smote",
    "borderline_smote",
    "adasyn",
    "smote_enn",
    "smote_tomek",
)
COST_SENSITIVE_STRATEGIES = ("class_weight",)
ALL_STRATEGIES = ("none",) + RESAMPLING_STRATEGIES + COST_SENSITIVE_STRATEGIES

# Human-readable notes, surfaced by describe_strategies().
_DESCRIPTIONS = {
    "none": "No handling. Baseline only — expect failures to be ignored.",
    "class_weight": "Cost-sensitive. No synthetic data, so noise is not amplified. Strong default.",
    "random_over": "Duplicate minority rows. Cheap; can overfit the few failures.",
    "random_under": "Drop majority rows. Wastes data you can't spare from ~1.5k rows.",
    "smote": "Interpolate new minority points. Needs clean input or it amplifies noise.",
    "borderline_smote": "SMOTE focused near the decision boundary.",
    "adasyn": "SMOTE variant that synthesises more where the minority is hardest.",
    "smote_enn": "SMOTE then Edited-Nearest-Neighbours cleaning. Recommended for SECOM.",
    "smote_tomek": "SMOTE then Tomek-link removal. Recommended for SECOM.",
}

# Strategies considered most robust on this dataset (for guidance / defaults).
RECOMMENDED_FOR_SECOM = ("class_weight", "smote_enn", "smote_tomek")


@dataclass
class BalancingPlan:
    """Everything the trainer needs to apply one balancing strategy.

    Attributes
    ----------
    name : str
        The chosen strategy name.
    sampler : object | None
        An imblearn sampler to insert before the estimator, or ``None`` when the
        strategy uses cost-sensitivity (or is the no-op baseline).
    use_class_weight : bool
        If True, the estimator should be constructed with ``class_weight``.
    class_weight : dict | str | None
        Value to pass to the estimator's ``class_weight`` argument when
        ``use_class_weight`` is True (a ``{0: w0, 1: w1}`` dict, or "balanced").
    scale_pos_weight : float | None
        Convenience value for XGBoost-style estimators (= n_negative / n_positive).
        Always computed when ``y`` is supplied, regardless of strategy, so the
        trainer can use it if the estimator is gradient-boosted.
    """

    name: str
    sampler: Any | None
    use_class_weight: bool
    class_weight: Any | None
    scale_pos_weight: float | None

    def describe(self) -> str:
        bits = [f"strategy={self.name}"]
        bits.append(f"sampler={'yes' if self.sampler is not None else 'no'}")
        bits.append(f"class_weight={self.class_weight if self.use_class_weight else 'no'}")
        if self.scale_pos_weight is not None:
            bits.append(f"scale_pos_weight={self.scale_pos_weight:.2f}")
        return " | ".join(bits)


# --------------------------------------------------------------------------- #
# Low-level resampler factory
# --------------------------------------------------------------------------- #
def get_resampler(
    name: str,
    random_state: int = 42,
    sampling_strategy: str | float | dict = "auto",
    k_neighbors: int = 5,
    **kwargs,
):
    """Return an imblearn sampler for ``name`` (or ``None`` for non-resampling).

    Parameters
    ----------
    name : str
        One of ``ALL_STRATEGIES``.
    random_state : int
        Reproducibility seed for stochastic samplers.
    sampling_strategy : str | float | dict
        Passed through to the sampler. "auto" balances to the majority count.
    k_neighbors : int
        Neighbourhood size for SMOTE-family samplers. Must be < the number of
        minority samples in the fold it is fitted on (SECOM train folds have
        plenty), otherwise reduce it.
    """
    name = name.lower()
    if name in ("none",) + COST_SENSITIVE_STRATEGIES:
        return None

    if name not in RESAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Valid: {ALL_STRATEGIES}"
        )

    # Lazy imports keep import cost low and isolate the optional dependency.
    from imblearn.over_sampling import (
        ADASYN,
        SMOTE,
        BorderlineSMOTE,
        RandomOverSampler,
    )
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek

    common = {"sampling_strategy": sampling_strategy}
    smote_common = {**common, "random_state": random_state, "k_neighbors": k_neighbors}

    if name == "random_over":
        return RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    if name == "random_under":
        return RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    if name == "smote":
        return SMOTE(**smote_common, **kwargs)
    if name == "borderline_smote":
        return BorderlineSMOTE(**smote_common, **kwargs)
    if name == "adasyn":
        # ADASYN uses n_neighbors rather than k_neighbors.
        return ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=k_neighbors,
            **kwargs,
        )
    if name == "smote_enn":
        inner_smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        return SMOTEENN(random_state=random_state, smote=inner_smote, **kwargs)
    if name == "smote_tomek":
        inner_smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        return SMOTETomek(random_state=random_state, smote=inner_smote, **kwargs)

    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Cost-sensitive helpers
# --------------------------------------------------------------------------- #
def compute_class_weight_dict(y) -> dict:
    """Balanced class weights as a ``{class: weight}`` dict.

    weight_c = n_samples / (n_classes * count_c). This is sklearn's "balanced"
    rule, returned explicitly so it can be logged/inspected.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    return {int(c): float(n / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}


def compute_scale_pos_weight(y) -> float:
    """n_negative / n_positive — the XGBoost-recommended imbalance weight."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# --------------------------------------------------------------------------- #
# High-level plan builder
# --------------------------------------------------------------------------- #
def build_balancing_plan(
    strategy: str = "smote_enn",
    y=None,
    random_state: int = 42,
    **resampler_kwargs,
) -> BalancingPlan:
    """Build a :class:`BalancingPlan` for ``strategy``.

    ``y`` is optional but, when given, lets us compute explicit class weights
    and a scale_pos_weight value. For the ``class_weight`` strategy without
    ``y`` we fall back to the string ``"balanced"`` (sklearn understands it).
    """
    strategy = strategy.lower()
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {ALL_STRATEGIES}")

    sampler = get_resampler(strategy, random_state=random_state, **resampler_kwargs)

    use_cw = strategy in COST_SENSITIVE_STRATEGIES
    if use_cw:
        class_weight: Any = compute_class_weight_dict(y) if y is not None else "balanced"
    else:
        class_weight = None

    spw = compute_scale_pos_weight(y) if y is not None else None

    plan = BalancingPlan(
        name=strategy,
        sampler=sampler,
        use_class_weight=use_cw,
        class_weight=class_weight,
        scale_pos_weight=spw,
    )
    logger.info("Balancing plan: %s", plan.describe())
    return plan


def describe_strategies() -> dict[str, str]:
    """Return {strategy_name: one-line description}."""
    return dict(_DESCRIPTIONS)


# --------------------------------------------------------------------------- #
# CLI / demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    """Show each strategy's effect on a small imbalanced synthetic set."""
    from collections import Counter
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=6,
        weights=[0.93, 0.07],  # ~13:1, SECOM-like
        random_state=0,
    )
    print(f"Original distribution: {dict(Counter(y))}\n")
    print(f"{'strategy':<18}{'after resampling':<28}{'notes'}")
    print("-" * 90)
    for name in ALL_STRATEGIES:
        plan = build_balancing_plan(name, y=y)
        if plan.sampler is not None:
            Xr, yr = plan.sampler.fit_resample(X, y)
            dist = str(dict(Counter(yr)))
        elif plan.use_class_weight:
            dist = f"(unchanged) class_weight={ {k: round(v,2) for k,v in plan.class_weight.items()} }"
        else:
            dist = "(unchanged)"
        print(f"{name:<18}{dist:<28}{_DESCRIPTIONS[name][:42]}")
    print(f"\nRecommended for SECOM: {RECOMMENDED_FOR_SECOM}")
    print(f"scale_pos_weight for this data: {compute_scale_pos_weight(y):.2f}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description="Imbalance-handling strategies for SECOM.")
    p.add_argument("--demo", action="store_true", help="Show each strategy's effect")
    p.add_argument("--list", action="store_true", help="List strategies and descriptions")
    args = p.parse_args(argv)
    if args.list:
        for k, v in describe_strategies().items():
            print(f"{k:<18} {v}")
    else:
        _demo()


if __name__ == "__main__":
    main()
