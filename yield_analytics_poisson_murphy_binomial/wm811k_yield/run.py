"""run.py — end-to-end orchestration of the WM-811K yield pipeline.

Usage:
    python -m wm811k_yield.run --data LSWMD.pkl --failure-types Random
    python -m wm811k_yield.run --data LSWMD.pkl          # all wafers

Stages: load -> EDA -> extract curve -> fit 3 models -> evaluate -> predict.
Each stage is a separate module; this file only wires them together.
"""
from __future__ import annotations
import argparse
import json

from .config import Config
from .data_loader import WM811KLoader
from .features import WindowExtractor
from .eda import YieldEDA
from .models import build_models
from .trainer import YieldModelTrainer
from .evaluator import ModelEvaluator
from .predictor import YieldPredictor


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="WM-811K analytical yield modeling")
    p.add_argument("--data", default="LSWMD.pkl", help="path to WM-811K pickle")
    p.add_argument("--out", default="artifacts", help="output directory")
    p.add_argument("--failure-types", nargs="*", default=None,
                   help="subset of failure types, e.g. Random Center")
    p.add_argument("--max-wafers", type=int, default=None)
    args = p.parse_args()
    return Config(
        data_path=args.data,
        output_dir=args.out,
        failure_types=tuple(args.failure_types) if args.failure_types else None,
        max_wafers=args.max_wafers,
    )


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or parse_args()

    # 1. load --------------------------------------------------------------
    loader = WM811KLoader(cfg)
    df = loader.load()
    print(f"[load] {len(df)} wafers after filtering")

    # 2. EDA ---------------------------------------------------------------
    eda = YieldEDA(cfg)
    print("[eda] wafer summary:\n", eda.wafer_summary(df))
    print("[eda] overdispersion:", eda.overdispersion(df))
    eda.plot_distributions(df)

    # 3. feature extraction (yield-vs-area curve) --------------------------
    curve = WindowExtractor(cfg).extract(loader.wafer_maps())
    print("[features] yield curve:\n", curve.as_frame())

    # 4. train all three models -------------------------------------------
    models = build_models()
    fits = YieldModelTrainer(cfg).fit_all(models, curve)
    for name, fit in fits.items():
        print(f"[train] {name}: success={fit.success} params={fit.params}")

    # 5. evaluate / rank ---------------------------------------------------
    evaluator = ModelEvaluator(models)
    board = evaluator.leaderboard(fits, curve)
    print("[eval] leaderboard:\n", board)
    eda.plot_curve_with_fits(curve, models, fits)

    # 6. predict with the winning model -----------------------------------
    best_name = board.iloc[0]["model_name"]
    best_model = next(m for m in models if m.name == best_name)
    predictor = YieldPredictor(best_model, fits[best_name])
    demo = {
        "best_model": best_name,
        "yield_at_A=4": predictor.predict_from_physical(4.0),
        "max_area_for_90pct_yield": predictor.max_area_for_yield(0.90),
    }
    print("[predict]", demo)

    # persist a small JSON summary ----------------------------------------
    summary = {
        "n_wafers": int(len(df)),
        "overdispersion": eda.overdispersion(df),
        "leaderboard": board.to_dict(orient="records"),
        "prediction_demo": demo,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                            default=str))
    return summary


if __name__ == "__main__":
    main()
