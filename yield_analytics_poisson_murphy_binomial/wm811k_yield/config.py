"""Central configuration for the WM-811K yield-modeling pipeline.

All tunables live here so the rest of the package stays declarative.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ---- Wafer-map cell encoding (WM-811K convention) -------------------------
# 0 = outside wafer / no die, 1 = good die, 2 = defective (failed) die
NO_DIE = 0
GOOD_DIE = 1
BAD_DIE = 2


@dataclass
class Config:
    # --- I/O ---------------------------------------------------------------
    data_path: Path = Path("LSWMD.pkl")          # raw WM-811K pickle
    output_dir: Path = Path("artifacts")          # plots, fitted params, curves

    # --- Window / area-scaling extraction ----------------------------------
    # Window edge lengths in *dies*. Area A = w**2 in unit-die areas.
    window_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    stride: int | None = None     # None -> non-overlapping tiling (stride=w)
    min_windows_per_area: int = 30  # drop area points with too few samples

    # --- Wafer filtering ----------------------------------------------------
    min_die_count: int = 100      # ignore tiny maps with unstable statistics
    failure_types: tuple[str, ...] | None = None  # e.g. ("Random",) to subset
    max_wafers: int | None = None  # cap for quick runs; None = use all

    # --- Model fitting ------------------------------------------------------
    # Initial guesses / bounds for nonlinear least squares.
    d0_init: float = 0.05         # defects per unit-die area
    alpha_init: float = 2.0       # NB clustering parameter
    fit_maxfev: int = 20000

    # --- Reproducibility ----------------------------------------------------
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
