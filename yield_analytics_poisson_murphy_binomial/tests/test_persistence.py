"""test_persistence.py — verify train -> save -> load -> predict round-trips."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from wm811k_yield.config import Config
from wm811k_yield.features import YieldCurve
from wm811k_yield.models import build_models, model_by_name
from wm811k_yield.trainer import YieldModelTrainer
from wm811k_yield.evaluator import ModelEvaluator
from wm811k_yield.predictor import YieldPredictor
from wm811k_yield.persistence import ModelStore
from wm811k_yield import main as cli


def make_curve(D0=0.06, alpha=1.8):
    area = np.array([1, 4, 9, 16, 25, 36, 49, 64], dtype=float)
    y = (1 + D0 * area / alpha) ** (-alpha)
    n = np.full_like(area, 4000, dtype=int)
    return YieldCurve(area=area, window=np.sqrt(area).astype(int),
                      yield_frac=y, n_valid=n, n_yield=(y * n).astype(int))


def test_roundtrip(tmp="models_test.json"):
    cfg = Config(output_dir="artifacts_test")
    curve = make_curve()
    models = build_models()
    fits = YieldModelTrainer(cfg).fit_all(models, curve)
    board = ModelEvaluator(models).leaderboard(fits, curve)
    best = board.iloc[0]["model_name"]

    ModelStore.save(tmp, fits, best_model=best, leaderboard=board,
                    ref_die_area_mm2=2.0, metadata={"note": "synthetic"})
    bundle = ModelStore.load(tmp)

    # 1. best model preserved
    assert bundle.best_model == best
    print(f"best model persisted: {bundle.best_model}")

    # 2. prediction identical before vs after the round-trip
    for name in bundle.models:
        live = YieldPredictor(model_by_name(name), fits[name])
        loaded = YieldPredictor(model_by_name(name), bundle.fit_result(name))
        a = np.array([1.0, 4.0, 16.0])
        assert np.allclose(live.predict_yield(a), loaded.predict_yield(a))
    print("predictions match before/after save+load for all models")

    # 3. exercise the CLI-facing helpers
    print("\n-- forward: area -> yield --")
    cli.predict_yield(bundle, [1, 4, 9, 16])
    print("\n-- inverse: max area for target yield --")
    cli.predict_max_area(bundle, 0.90)


if __name__ == "__main__":
    test_roundtrip()
    print("\nPersistence round-trip OK.")
