"""main.py — train once, predict many times.

Two commands:

  TRAIN   fit all three models on WM-811K, rank them, and save to a JSON file.
      python -m wm811k_yield.main train --data LSWMD.pkl --save models.json

  PREDICT load the saved file and turn an area A into a yield Y.
      python -m wm811k_yield.main predict --load models.json --area 4
      python -m wm811k_yield.main predict --load models.json --area 1 4 9 16
      python -m wm811k_yield.main predict --load models.json --target-yield 0.9

`--area` gives forward prediction (area -> yield) for every model, marking the
best-fitting one. `--target-yield` gives the inverse (largest area that still
meets the target) using the best model.
"""
from __future__ import annotations
import argparse
import numpy as np

from .config import Config
from .data_loader import WM811KLoader
from .features import WindowExtractor, YieldCurve
from .models import build_models, model_by_name
from .trainer import YieldModelTrainer
from .evaluator import ModelEvaluator
from .predictor import YieldPredictor
from .persistence import ModelStore, ModelBundle


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(cfg: Config, save_path: str, ref_die_area_mm2: float) -> ModelBundle:
    """Load -> extract curve -> fit all -> rank -> persist."""
    loader = WM811KLoader(cfg)
    df = loader.load()
    print(f"[train] {len(df)} wafers after filtering")

    curve = WindowExtractor(cfg).extract(loader.wafer_maps())
    print("[train] empirical yield curve:")
    print(curve.as_frame().to_string(index=False))

    models = build_models()
    fits = YieldModelTrainer(cfg).fit_all(models, curve)
    board = ModelEvaluator(models).leaderboard(fits, curve)
    print("\n[train] leaderboard (lower AIC = better fit):")
    print(board.to_string(index=False))

    best = board.iloc[0]["model_name"]
    path = ModelStore.save(
        save_path, fits, best_model=best, leaderboard=board,
        ref_die_area_mm2=ref_die_area_mm2,
        metadata={
            "n_wafers": int(len(df)),
            "failure_types": list(cfg.failure_types) if cfg.failure_types else "all",
            "window_sizes": list(cfg.window_sizes),
        },
    )
    print(f"\n[train] best model = {best}; saved -> {path}")
    return ModelStore.load(path)


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
def _predictor_for(bundle: ModelBundle, name: str) -> YieldPredictor:
    model = model_by_name(name)
    return YieldPredictor(model, bundle.fit_result(name))


def predict_yield(bundle: ModelBundle, areas: list[float]) -> None:
    """Forward: area A -> yield Y, for every model, marking the best."""
    areas = np.asarray(areas, dtype=float)
    names = list(bundle.models.keys())
    best = bundle.best_model

    header = "area".rjust(8) + "".join(n.rjust(22) for n in names)
    print(header)
    print("-" * len(header))
    preds = {n: _predictor_for(bundle, n).predict_yield(areas) for n in names}
    for i, a in enumerate(areas):
        row = f"{a:8.2f}"
        for n in names:
            tag = " *" if n == best else "  "
            row += f"{preds[n][i]*100:18.2f}%{tag}"
        print(row)
    print(f"\n( * = best-fitting model: {best} )")
    print("Yield Y is the fraction of dies of area A expected defect-free.")


def predict_max_area(bundle: ModelBundle, target: float) -> None:
    """Inverse: largest area meeting a target yield, using the best model."""
    pred = _predictor_for(bundle, bundle.best_model)
    a_max = pred.max_area_for_yield(target)
    print(f"Best model ({bundle.best_model}): to keep yield >= {target:.0%}, "
          f"max die area = {a_max:.3f} unit-die areas "
          f"(={a_max * bundle.ref_die_area_mm2:.3f} mm^2 at the trained reference).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wm811k_yield.main",
                                description="Train yield models; predict yield from area.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fit all models on WM-811K and save them")
    t.add_argument("--data", default="LSWMD.pkl")
    t.add_argument("--save", default="models.json")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--failure-types", nargs="*", default=None)
    t.add_argument("--max-wafers", type=int, default=None)
    t.add_argument("--ref-die-area-mm2", type=float, default=1.0,
                   help="physical area of one die, for mm^2 reporting")

    q = sub.add_parser("predict", help="load saved models and predict yield")
    q.add_argument("--load", default="models.json")
    q.add_argument("--area", nargs="*", type=float, default=None,
                   help="one or more areas (unit-die areas) -> yield")
    q.add_argument("--target-yield", type=float, default=None,
                   help="inverse: largest area meeting this yield (0-1)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        cfg = Config(
            data_path=args.data,
            output_dir=args.out,
            failure_types=tuple(args.failure_types) if args.failure_types else None,
            max_wafers=args.max_wafers,
        )
        train(cfg, args.save, args.ref_die_area_mm2)

    elif args.command == "predict":
        bundle = ModelStore.load(args.load)
        if args.area:
            predict_yield(bundle, args.area)
        if args.target_yield is not None:
            predict_max_area(bundle, args.target_yield)
        if not args.area and args.target_yield is None:
            raise SystemExit("give --area A [A ...] and/or --target-yield Y")


if __name__ == "__main__":
    main()
