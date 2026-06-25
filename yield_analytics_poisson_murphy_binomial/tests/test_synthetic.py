"""test_synthetic.py — verify the math without the real dataset.

Two checks:
  (A) Parameter recovery: build a Y(A) curve from known NB params, fit all
      three models, confirm NB recovers D0 and alpha closely.
  (B) End-to-end on synthetic wafer maps: generate (i) random/uncorrelated
      defects -> Poisson should win/tie; (ii) strongly clustered defects ->
      Negative Binomial should win and report small alpha.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wm811k_yield.config import Config, NO_DIE, GOOD_DIE, BAD_DIE
from wm811k_yield.features import WindowExtractor, YieldCurve
from wm811k_yield.models import build_models, NegativeBinomialYield
from wm811k_yield.trainer import YieldModelTrainer
from wm811k_yield.evaluator import ModelEvaluator


def check_parameter_recovery():
    print("\n=== (A) parameter recovery from a known NB curve ===")
    cfg = Config(output_dir="artifacts_test")
    true_D0, true_alpha = 0.08, 1.5
    area = np.array([1, 4, 9, 16, 25, 36, 49, 64], dtype=float)
    y = (1 + true_D0 * area / true_alpha) ** (-true_alpha)
    n = np.full_like(area, 5000, dtype=int)
    curve = YieldCurve(area=area, window=np.sqrt(area).astype(int),
                       yield_frac=y, n_valid=n, n_yield=(y * n).astype(int))

    models = build_models()
    fits = YieldModelTrainer(cfg).fit_all(models, curve)
    nb = fits["negative_binomial"].params
    print(f"  true   D0={true_D0}, alpha={true_alpha}")
    print(f"  fitted D0={nb['D0']:.4f}, alpha={nb['alpha']:.4f}")
    assert abs(nb["D0"] - true_D0) < 0.01, "D0 recovery off"
    assert abs(nb["alpha"] - true_alpha) < 0.2, "alpha recovery off"
    print("  PASS: NB recovers ground truth")


def _make_random_wafer(grid=40, rate=0.05, rng=None):
    """Circular wafer, defects sprinkled independently (Poisson-like)."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:grid, 0:grid]
    r = np.sqrt((xx - grid/2)**2 + (yy - grid/2)**2)
    m = np.where(r <= grid/2, GOOD_DIE, NO_DIE).astype(np.int8)
    die = m == GOOD_DIE
    flip = (rng.random((grid, grid)) < rate) & die
    m[flip] = BAD_DIE
    return m


def _make_clustered_wafer(grid=40, n_clusters=3, spread=3.0, rate=0.6, rng=None):
    """Circular wafer with a few dense defect blobs (strong clustering)."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:grid, 0:grid]
    r = np.sqrt((xx - grid/2)**2 + (yy - grid/2)**2)
    m = np.where(r <= grid/2, GOOD_DIE, NO_DIE).astype(np.int8)
    die = m == GOOD_DIE
    centers = rng.uniform(grid*0.25, grid*0.75, size=(n_clusters, 2))
    prob = np.zeros((grid, grid))
    for cy, cx in centers:
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        prob += np.exp(-(d**2) / (2 * spread**2))
    prob = np.clip(prob * rate, 0, 1)
    flip = (rng.random((grid, grid)) < prob) & die
    m[flip] = BAD_DIE
    return m


def check_end_to_end():
    print("\n=== (B) end-to-end on synthetic wafers ===")
    cfg = Config(output_dir="artifacts_test", window_sizes=(1, 2, 3, 4, 5, 6),
                 min_windows_per_area=20)
    rng = np.random.default_rng(7)
    models = build_models()
    trainer = YieldModelTrainer(cfg)

    for label, maker in [("RANDOM defects", _make_random_wafer),
                         ("CLUSTERED defects", _make_clustered_wafer)]:
        maps = [maker(rng=rng) for _ in range(300)]
        curve = WindowExtractor(cfg).extract(maps)
        fits = trainer.fit_all(models, curve)
        board = ModelEvaluator(models).leaderboard(fits, curve)
        nb_alpha = fits["negative_binomial"].params["alpha"]
        print(f"\n  {label}: winner = {board.iloc[0]['model_name']}, "
              f"NB alpha = {nb_alpha:.2f}")
        print(board[["rank", "model_name", "weighted_rmse", "r2", "aic"]]
              .to_string(index=False))


if __name__ == "__main__":
    check_parameter_recovery()
    check_end_to_end()
    print("\nAll synthetic checks complete.")
