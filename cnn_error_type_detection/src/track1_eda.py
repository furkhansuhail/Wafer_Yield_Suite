"""
Track 1 — Module 2: Exploratory Data Analysis for WM-811K.

Responsibility
--------------
Read the raw pickle (read-only) and surface the facts that drive every
downstream decision:
  * class imbalance       -> informs class weights / oversampling (Module 4)
  * labeled vs unlabeled  -> how much data Module 3 can actually keep
  * wafer-dimension spread -> sets the resize/pad target (Module 3)
Outputs plots + a machine-readable summary to `reports/`.

Data quirks handled here
------------------------
  * `failureType` and `trianTestLabel` cells are nested numpy arrays
    (e.g. array([['Center']])), empty arrays for unlabeled rows, or plain
    strings depending on the mirror. `_normalize_label` flattens all of these.
  * 'none' is a real labeled class (inspected, no pattern) and is NOT the
    same as 'Unlabeled' (never inspected / empty array). We keep them distinct.
  * `waferMap` cells are 2-D arrays with values {0=background, 1=pass, 2=fail}.

Usage
-----
    python track1_eda.py --raw-dir data/raw --reports-dir reports
    python track1_eda.py --pkl data/raw/LSWMD.pkl --reports-dir reports
    python track1_eda.py --self-test          # no real data needed

Testable individually
---------------------
`--self-test` builds a synthetic LSWMD-shaped DataFrame with KNOWN class
counts and wafer dimensions, runs the full pipeline into a temp dir, and
asserts the summary numbers and output files are correct.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render to files, never to a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eda")

PICKLE_NAME = "LSWMD.pkl"
LABEL_COL = "failureType"
SPLIT_COL = "trianTestLabel"      # sic — dataset misspelling
MAP_COL = "waferMap"
UNLABELED = "Unlabeled"           # our token for empty/missing failureType


# --------------------------------------------------------------------------- #
# Loading + label normalization
# --------------------------------------------------------------------------- #

def resolve_pickle(raw_dir: str | None, pkl: str | None) -> Path:
    """Figure out where the pickle is, from either --pkl or --raw-dir."""
    if pkl:
        return Path(pkl)
    if raw_dir:
        return Path(raw_dir) / PICKLE_NAME
    raise ValueError("Provide either --pkl or --raw-dir")


def read_wm811k_pickle(path):
    """
    Read LSWMD.pkl, including the legacy pickle written by very old pandas
    UNDER PYTHON 2. A normal read is tried first; only if it fails do we:
      1. install module shims (old 'pandas.indexes' path + removed
         Int64Index/Float64Index classes), and
      2. fall back to raw pickle.load with encoding='latin1', which decodes
         the Python-2 byte strings that break Python 3's default ASCII unpickler.
    Modern pickles take the fast path and are completely unaffected.
    """
    import pickle
    import sys
    import types
    try:
        return pd.read_pickle(path)
    except (ModuleNotFoundError, AttributeError, UnicodeDecodeError):
        import pandas.core.indexes as _new_idx
        import pandas.core.indexes.base as _idx_base
        sys.modules.setdefault("pandas.indexes", _new_idx)
        sys.modules.setdefault("pandas.indexes.base", _idx_base)
        sys.modules.setdefault("pandas.core.index", _new_idx)
        _numeric = types.ModuleType("pandas_compat_numeric")
        for _n in ("Int64Index", "Float64Index", "UInt64Index"):
            setattr(_numeric, _n, getattr(pd, _n, pd.Index))
        sys.modules.setdefault("pandas.indexes.numeric", _numeric)
        sys.modules.setdefault("pandas.core.indexes.numeric", _numeric)
        # Decode Python-2 byte strings with latin1, preferring pandas' own
        # compatibility unpickler (it knows the full history of internal class
        # renames); fall back to plain pickle.load if it's unavailable.
        try:
            from pandas.compat.pickle_compat import Unpickler as _PdUnpickler
        except Exception:
            _PdUnpickler = None
        with open(path, "rb") as fh:
            if _PdUnpickler is not None:
                try:
                    return _PdUnpickler(fh, encoding="latin1").load()
                except TypeError:
                    fh.seek(0)
            return pickle.load(fh, encoding="latin1")


def load_dataframe(pkl_path: Path) -> pd.DataFrame:
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")
    log.info("Loading %s ...", pkl_path)
    df = read_wm811k_pickle(pkl_path)
    log.info("Loaded %s rows, %s columns", f"{len(df):,}", len(df.columns))
    return df


def _normalize_label(val) -> str:
    """
    Flatten a failureType / split cell to a single clean string.

    Handles array([['Center']]) -> 'Center', [] -> 'Unlabeled',
    'Center' -> 'Center', np.str_ -> str.
    """
    if val is None:
        return UNLABELED
    arr = np.asarray(val, dtype=object).ravel()
    if arr.size == 0:
        return UNLABELED
    s = str(arr[0]).strip()
    return s if s else UNLABELED


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add flat string columns `label` and `split` derived from the nested cells."""
    out = df.copy()
    if LABEL_COL in out.columns:
        out["label"] = out[LABEL_COL].map(_normalize_label)
    else:
        out["label"] = UNLABELED
    if SPLIT_COL in out.columns:
        out["split"] = out[SPLIT_COL].map(_normalize_label)
    else:
        out["split"] = UNLABELED
    return out


# --------------------------------------------------------------------------- #
# Dataset preview (human-readable head)
# --------------------------------------------------------------------------- #

def _compact_cell(val):
    """
    Render one cell compactly for preview.

    The `waferMap` cells are large 2-D arrays; printing them in full makes
    `head()` unreadable, so anything sizeable collapses to a `<ndarray RxC
    dtype>` tag. Small nested cells (e.g. array([['Center']]) or the empty
    arrays of unlabeled rows) are left as-is, since seeing their raw shape is
    actually informative.
    """
    if isinstance(val, np.ndarray):
        if val.size > 20:
            return f"<ndarray {'x'.join(map(str, val.shape))} {val.dtype}>"
        return val
    if isinstance(val, list):
        arr = np.asarray(val, dtype=object)
        if arr.size > 20:
            return f"<list {'x'.join(map(str, arr.shape))}>"
    return val


def preview_dataframe(df: pd.DataFrame, n: int = 5, save_to: Path | None = None) -> str:
    """
    Show the freshly loaded data as a dataset — like `df.head()`, but readable.

    Builds a display copy where each cell is passed through `_compact_cell`
    (so wafer maps show as a shape tag instead of dumping the whole array),
    then assembles a text block with the frame shape, column list, dtypes, and
    the first `n` rows. The block is logged and, if `save_to` is given, also
    written there as plain text. Returns the text block.
    """
    head = df.head(n).copy()
    for col in head.columns:
        head[col] = head[col].map(_compact_cell)

    lines = [
        f"shape  : {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"columns: {list(df.columns)}",
        "",
        "dtypes:",
        df.dtypes.to_string(),
        "",
        f"head({n}):",
    ]
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 40):
        lines.append(head.to_string())
    text = "\n".join(lines)

    log.info("Dataset preview\n%s", text)
    if save_to is not None:
        Path(save_to).write_text(text)
    return text


# --------------------------------------------------------------------------- #
# Profiling pieces
# --------------------------------------------------------------------------- #

def class_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Counts per failure-type label (including 'none' and 'Unlabeled')."""
    return dict(Counter(df["label"]).most_common())


def wafer_dimensions(df: pd.DataFrame) -> list[tuple[int, int]]:
    """(rows, cols) shape of each wafer map."""
    dims = []
    for m in df[MAP_COL]:
        arr = np.asarray(m)
        if arr.ndim == 2:
            dims.append((arr.shape[0], arr.shape[1]))
    return dims


def summary_stats(df: pd.DataFrame) -> dict:
    """Everything a human (or Module 3) needs at a glance."""
    dist = class_distribution(df)
    n = len(df)
    labeled = sum(v for k, v in dist.items() if k != UNLABELED)
    dims = wafer_dimensions(df)
    uniq_dims = sorted(set(dims))
    return {
        "n_total": n,
        "n_labeled": labeled,
        "n_unlabeled": dist.get(UNLABELED, 0),
        "labeled_fraction": round(labeled / n, 4) if n else 0.0,
        "n_classes_labeled": len([k for k in dist if k != UNLABELED]),
        "class_distribution": dist,
        "n_unique_wafer_dims": len(uniq_dims),
        "wafer_dim_min": list(min(uniq_dims)) if uniq_dims else None,
        "wafer_dim_max": list(max(uniq_dims)) if uniq_dims else None,
        "imbalance_ratio": (
            round(max(v for k, v in dist.items() if k != UNLABELED)
                  / min(v for k, v in dist.items() if k != UNLABELED), 1)
            if labeled and len([k for k in dist if k != UNLABELED]) > 1 else None
        ),
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_class_distribution(df: pd.DataFrame, out: Path) -> Path:
    dist = class_distribution(df)
    labels, counts = list(dist.keys()), list(dist.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, counts)
    ax.set_yscale("log")  # log scale or the rare classes vanish
    ax.set_ylabel("count (log scale)")
    ax.set_title("WM-811K class distribution")
    ax.tick_params(axis="x", rotation=45)
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_wafer_dimensions(df: pd.DataFrame, out: Path) -> Path:
    dims = wafer_dimensions(df)
    areas = [r * c for r, c in dims]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(areas, bins=min(50, max(5, len(set(areas)))))
    ax.set_xlabel("wafer map area (rows x cols)")
    ax.set_ylabel("count")
    ax.set_title(f"Wafer map size spread — {len(set(dims))} unique shapes")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_sample_maps(df: pd.DataFrame, out: Path, per_class: int = 3) -> Path:
    """Grid of sample wafer maps, one row per labeled class."""
    classes = [c for c in class_distribution(df) if c != UNLABELED]
    if not classes:
        classes = [UNLABELED]
    fig, axes = plt.subplots(
        len(classes), per_class,
        figsize=(per_class * 2.2, len(classes) * 2.2),
        squeeze=False,
    )
    for r, cls in enumerate(classes):
        subset = df[df["label"] == cls]
        for c in range(per_class):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(subset):
                ax.imshow(np.asarray(subset.iloc[c][MAP_COL]), cmap="viridis")
            if c == 0:
                ax.set_title(cls, loc="left", fontsize=9)
    fig.suptitle("Sample wafer maps per class")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(pkl_path: Path, reports_dir: Path, preview_rows: int = 5) -> dict:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_labels(load_dataframe(pkl_path))

    # View the dataset as soon as it's loaded: a readable head() that collapses
    # the big waferMap arrays so the columns are actually legible.
    preview_dataframe(df, n=preview_rows, save_to=reports_dir / "eda_head_preview.txt")

    stats = summary_stats(df)
    (reports_dir / "eda_summary.json").write_text(json.dumps(stats, indent=2))

    plot_class_distribution(df, reports_dir / "eda_class_distribution.png")
    plot_wafer_dimensions(df, reports_dir / "eda_wafer_dimensions.png")
    plot_sample_maps(df, reports_dir / "eda_sample_maps.png")

    log.info("EDA done. %s labeled / %s total (%.1f%%), %s classes, imbalance ~%sx",
             f"{stats['n_labeled']:,}", f"{stats['n_total']:,}",
             100 * stats["labeled_fraction"], stats["n_classes_labeled"],
             stats["imbalance_ratio"])
    log.info("Artifacts written to %s/", reports_dir)
    return stats


# --------------------------------------------------------------------------- #
# Self-test (synthetic data, no real pickle)
# --------------------------------------------------------------------------- #

def _make_synthetic_df() -> pd.DataFrame:
    """
    Build a DataFrame mimicking LSWMD's nested structure with KNOWN counts:
      Center x5, Scratch x2, none x4, Unlabeled(empty) x3  => 14 rows, 3 labeled classes.
    Two distinct wafer-map shapes to exercise the dimension logic.
    """
    rng = np.random.default_rng(0)

    def fmap(shape):
        return rng.integers(0, 3, size=shape, dtype=np.uint8)

    rows = []
    plan = [("Center", 5, (26, 26)),
            ("Scratch", 2, (40, 35)),
            ("none", 4, (26, 26)),
            (None, 3, (26, 26))]   # None -> empty failureType -> Unlabeled
    for cls, k, shape in plan:
        for _ in range(k):
            ft = np.array([], dtype=object) if cls is None else np.array([[cls]], dtype=object)
            rows.append({
                MAP_COL: fmap(shape),
                "dieSize": shape[0] * shape[1],
                "lotName": "lotX",
                "waferIndex": 0,
                SPLIT_COL: np.array([["Training"]], dtype=object),
                LABEL_COL: ft,
            })
    return pd.DataFrame(rows)


def self_test() -> bool:
    print("=" * 60)
    print("SELF-TEST: track1_eda")
    print("=" * 60)
    passed = True
    df = normalize_labels(_make_synthetic_df())

    # 1. label normalization unwraps nested arrays + empties
    dist = class_distribution(df)
    t1 = (dist.get("Center") == 5 and dist.get("Scratch") == 2
          and dist.get("none") == 4 and dist.get(UNLABELED) == 3)
    print(f"[{'PASS' if t1 else 'FAIL'}] class counts correct: {dist}")
    passed &= t1

    # 2. 'none' and 'Unlabeled' stay distinct
    t2 = "none" in dist and UNLABELED in dist and dist["none"] != dist[UNLABELED]
    print(f"[{'PASS' if t2 else 'FAIL'}] 'none' kept distinct from 'Unlabeled'")
    passed &= t2

    # 3. summary stats math
    s = summary_stats(df)
    t3 = (s["n_total"] == 14 and s["n_labeled"] == 11 and s["n_unlabeled"] == 3
          and s["n_classes_labeled"] == 3 and s["n_unique_wafer_dims"] == 2)
    print(f"[{'PASS' if t3 else 'FAIL'}] summary: total={s['n_total']} "
          f"labeled={s['n_labeled']} classes={s['n_classes_labeled']} "
          f"dims={s['n_unique_wafer_dims']}")
    passed &= t3

    # 4. imbalance ratio = max/min labeled class = 5/2 = 2.5
    t4 = s["imbalance_ratio"] == 2.5
    print(f"[{'PASS' if t4 else 'FAIL'}] imbalance ratio = {s['imbalance_ratio']} (expect 2.5)")
    passed &= t4

    # 5. all plot + json artifacts get written
    with tempfile.TemporaryDirectory() as tmp:
        rep = Path(tmp) / "reports"
        rep.mkdir()
        plot_class_distribution(df, rep / "c.png")
        plot_wafer_dimensions(df, rep / "d.png")
        plot_sample_maps(df, rep / "s.png")
        (rep / "eda_summary.json").write_text(json.dumps(s))
        t5 = all((rep / f).exists() for f in ["c.png", "d.png", "s.png", "eda_summary.json"])
    print(f"[{'PASS' if t5 else 'FAIL'}] all artifacts written")
    passed &= t5

    # 6. preview renders, collapses the big waferMap, keeps labels visible,
    #    and writes its text file
    with tempfile.TemporaryDirectory() as tmp:
        prev_path = Path(tmp) / "eda_head_preview.txt"
        text = preview_dataframe(df, n=3, save_to=prev_path)
        t6 = (
            prev_path.exists()
            and "<ndarray" in text            # wafer map collapsed
            and "Center" in text              # nested label still visible
            and "head(3):" in text            # header rendered with n
        )
    print(f"[{'PASS' if t6 else 'FAIL'}] dataset preview renders + writes file")
    passed &= t6

    print("-" * 60)
    print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
    print("=" * 60)
    return passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WM-811K EDA (Track 1, Module 2)")
    ap.add_argument("--raw-dir", help="Dir holding LSWMD.pkl")
    ap.add_argument("--pkl", help="Explicit path to LSWMD.pkl")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--head", type=int, default=5,
                    help="Rows to show in the dataset preview (default 5)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    pkl = resolve_pickle(args.raw_dir, args.pkl)
    run(pkl, Path(args.reports_dir), preview_rows=args.head)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# """
# Track 1 — Module 2: Exploratory Data Analysis for WM-811K.
#
# Responsibility
# --------------
# Read the raw pickle (read-only) and surface the facts that drive every
# downstream decision:
#   * class imbalance       -> informs class weights / oversampling (Module 4)
#   * labeled vs unlabeled  -> how much data Module 3 can actually keep
#   * wafer-dimension spread -> sets the resize/pad target (Module 3)
# Outputs plots + a machine-readable summary to `reports/`.
#
# Data quirks handled here
# ------------------------
#   * `failureType` and `trianTestLabel` cells are nested numpy arrays
#     (e.g. array([['Center']])), empty arrays for unlabeled rows, or plain
#     strings depending on the mirror. `_normalize_label` flattens all of these.
#   * 'none' is a real labeled class (inspected, no pattern) and is NOT the
#     same as 'Unlabeled' (never inspected / empty array). We keep them distinct.
#   * `waferMap` cells are 2-D arrays with values {0=background, 1=pass, 2=fail}.
#
# Usage
# -----
#     python track1_eda.py --raw-dir data/raw --reports-dir reports
#     python track1_eda.py --pkl data/raw/LSWMD.pkl --reports-dir reports
#     python track1_eda.py --self-test          # no real data needed
#
# Testable individually
# ---------------------
# `--self-test` builds a synthetic LSWMD-shaped DataFrame with KNOWN class
# counts and wafer dimensions, runs the full pipeline into a temp dir, and
# asserts the summary numbers and output files are correct.
# """
#
# from __future__ import annotations
#
# import argparse
# import json
# import logging
# import sys
# import tempfile
# from collections import Counter
# from pathlib import Path
#
# import matplotlib
# matplotlib.use("Agg")  # headless: render to files, never to a screen
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-7s | %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger("eda")
#
# PICKLE_NAME = "LSWMD.pkl"
# LABEL_COL = "failureType"
# SPLIT_COL = "trianTestLabel"      # sic — dataset misspelling
# MAP_COL = "waferMap"
# UNLABELED = "Unlabeled"           # our token for empty/missing failureType
#
#
# # --------------------------------------------------------------------------- #
# # Loading + label normalization
# # --------------------------------------------------------------------------- #
#
# def resolve_pickle(raw_dir: str | None, pkl: str | None) -> Path:
#     """Figure out where the pickle is, from either --pkl or --raw-dir."""
#     if pkl:
#         return Path(pkl)
#     if raw_dir:
#         return Path(raw_dir) / PICKLE_NAME
#     raise ValueError("Provide either --pkl or --raw-dir")
#
#
# def read_wm811k_pickle(path):
#     """
#     Read LSWMD.pkl, including the legacy pickle written by very old pandas
#     UNDER PYTHON 2. A normal read is tried first; only if it fails do we:
#       1. install module shims (old 'pandas.indexes' path + removed
#          Int64Index/Float64Index classes), and
#       2. fall back to raw pickle.load with encoding='latin1', which decodes
#          the Python-2 byte strings that break Python 3's default ASCII unpickler.
#     Modern pickles take the fast path and are completely unaffected.
#     """
#     import pickle
#     import sys
#     import types
#     try:
#         return pd.read_pickle(path)
#     except (ModuleNotFoundError, AttributeError, UnicodeDecodeError):
#         import pandas.core.indexes as _new_idx
#         import pandas.core.indexes.base as _idx_base
#         sys.modules.setdefault("pandas.indexes", _new_idx)
#         sys.modules.setdefault("pandas.indexes.base", _idx_base)
#         sys.modules.setdefault("pandas.core.index", _new_idx)
#         _numeric = types.ModuleType("pandas_compat_numeric")
#         for _n in ("Int64Index", "Float64Index", "UInt64Index"):
#             setattr(_numeric, _n, getattr(pd, _n, pd.Index))
#         sys.modules.setdefault("pandas.indexes.numeric", _numeric)
#         sys.modules.setdefault("pandas.core.indexes.numeric", _numeric)
#         # Decode Python-2 byte strings with latin1, preferring pandas' own
#         # compatibility unpickler (it knows the full history of internal class
#         # renames); fall back to plain pickle.load if it's unavailable.
#         try:
#             from pandas.compat.pickle_compat import Unpickler as _PdUnpickler
#         except Exception:
#             _PdUnpickler = None
#         with open(path, "rb") as fh:
#             if _PdUnpickler is not None:
#                 try:
#                     return _PdUnpickler(fh, encoding="latin1").load()
#                 except TypeError:
#                     fh.seek(0)
#             return pickle.load(fh, encoding="latin1")
#
#
# def load_dataframe(pkl_path: Path) -> pd.DataFrame:
#     pkl_path = Path(pkl_path)
#     if not pkl_path.exists():
#         raise FileNotFoundError(f"Pickle not found: {pkl_path}")
#     log.info("Loading %s ...", pkl_path)
#     df = read_wm811k_pickle(pkl_path)
#     log.info("Loaded %s rows, %s columns", f"{len(df):,}", len(df.columns))
#     return df
#
#
# def _normalize_label(val) -> str:
#     """
#     Flatten a failureType / split cell to a single clean string.
#
#     Handles array([['Center']]) -> 'Center', [] -> 'Unlabeled',
#     'Center' -> 'Center', np.str_ -> str.
#     """
#     if val is None:
#         return UNLABELED
#     arr = np.asarray(val, dtype=object).ravel()
#     if arr.size == 0:
#         return UNLABELED
#     s = str(arr[0]).strip()
#     return s if s else UNLABELED
#
#
# def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """Add flat string columns `label` and `split` derived from the nested cells."""
#     out = df.copy()
#     if LABEL_COL in out.columns:
#         out["label"] = out[LABEL_COL].map(_normalize_label)
#     else:
#         out["label"] = UNLABELED
#     if SPLIT_COL in out.columns:
#         out["split"] = out[SPLIT_COL].map(_normalize_label)
#     else:
#         out["split"] = UNLABELED
#     return out
#
#
# # --------------------------------------------------------------------------- #
# # Profiling pieces
# # --------------------------------------------------------------------------- #
#
# def class_distribution(df: pd.DataFrame) -> dict[str, int]:
#     """Counts per failure-type label (including 'none' and 'Unlabeled')."""
#     return dict(Counter(df["label"]).most_common())
#
#
# def wafer_dimensions(df: pd.DataFrame) -> list[tuple[int, int]]:
#     """(rows, cols) shape of each wafer map."""
#     dims = []
#     for m in df[MAP_COL]:
#         arr = np.asarray(m)
#         if arr.ndim == 2:
#             dims.append((arr.shape[0], arr.shape[1]))
#     return dims
#
#
# def summary_stats(df: pd.DataFrame) -> dict:
#     """Everything a human (or Module 3) needs at a glance."""
#     dist = class_distribution(df)
#     n = len(df)
#     labeled = sum(v for k, v in dist.items() if k != UNLABELED)
#     dims = wafer_dimensions(df)
#     uniq_dims = sorted(set(dims))
#     return {
#         "n_total": n,
#         "n_labeled": labeled,
#         "n_unlabeled": dist.get(UNLABELED, 0),
#         "labeled_fraction": round(labeled / n, 4) if n else 0.0,
#         "n_classes_labeled": len([k for k in dist if k != UNLABELED]),
#         "class_distribution": dist,
#         "n_unique_wafer_dims": len(uniq_dims),
#         "wafer_dim_min": list(min(uniq_dims)) if uniq_dims else None,
#         "wafer_dim_max": list(max(uniq_dims)) if uniq_dims else None,
#         "imbalance_ratio": (
#             round(max(v for k, v in dist.items() if k != UNLABELED)
#                   / min(v for k, v in dist.items() if k != UNLABELED), 1)
#             if labeled and len([k for k in dist if k != UNLABELED]) > 1 else None
#         ),
#     }
#
#
# # --------------------------------------------------------------------------- #
# # Plotting
# # --------------------------------------------------------------------------- #
#
# def plot_class_distribution(df: pd.DataFrame, out: Path) -> Path:
#     dist = class_distribution(df)
#     labels, counts = list(dist.keys()), list(dist.values())
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.bar(labels, counts)
#     ax.set_yscale("log")  # log scale or the rare classes vanish
#     ax.set_ylabel("count (log scale)")
#     ax.set_title("WM-811K class distribution")
#     ax.tick_params(axis="x", rotation=45)
#     for i, c in enumerate(counts):
#         ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
#     fig.tight_layout()
#     fig.savefig(out, dpi=120)
#     plt.close(fig)
#     return out
#
#
# def plot_wafer_dimensions(df: pd.DataFrame, out: Path) -> Path:
#     dims = wafer_dimensions(df)
#     areas = [r * c for r, c in dims]
#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.hist(areas, bins=min(50, max(5, len(set(areas)))))
#     ax.set_xlabel("wafer map area (rows x cols)")
#     ax.set_ylabel("count")
#     ax.set_title(f"Wafer map size spread — {len(set(dims))} unique shapes")
#     fig.tight_layout()
#     fig.savefig(out, dpi=120)
#     plt.close(fig)
#     return out
#
#
# def plot_sample_maps(df: pd.DataFrame, out: Path, per_class: int = 3) -> Path:
#     """Grid of sample wafer maps, one row per labeled class."""
#     classes = [c for c in class_distribution(df) if c != UNLABELED]
#     if not classes:
#         classes = [UNLABELED]
#     fig, axes = plt.subplots(
#         len(classes), per_class,
#         figsize=(per_class * 2.2, len(classes) * 2.2),
#         squeeze=False,
#     )
#     for r, cls in enumerate(classes):
#         subset = df[df["label"] == cls]
#         for c in range(per_class):
#             ax = axes[r][c]
#             ax.axis("off")
#             if c < len(subset):
#                 ax.imshow(np.asarray(subset.iloc[c][MAP_COL]), cmap="viridis")
#             if c == 0:
#                 ax.set_title(cls, loc="left", fontsize=9)
#     fig.suptitle("Sample wafer maps per class")
#     fig.tight_layout()
#     fig.savefig(out, dpi=120)
#     plt.close(fig)
#     return out
#
#
# # --------------------------------------------------------------------------- #
# # Orchestration
# # --------------------------------------------------------------------------- #
#
# def run(pkl_path: Path, reports_dir: Path) -> dict:
#     reports_dir = Path(reports_dir)
#     reports_dir.mkdir(parents=True, exist_ok=True)
#
#     df = normalize_labels(load_dataframe(pkl_path))
#
#     stats = summary_stats(df)
#     (reports_dir / "eda_summary.json").write_text(json.dumps(stats, indent=2))
#
#     plot_class_distribution(df, reports_dir / "eda_class_distribution.png")
#     plot_wafer_dimensions(df, reports_dir / "eda_wafer_dimensions.png")
#     plot_sample_maps(df, reports_dir / "eda_sample_maps.png")
#
#     log.info("EDA done. %s labeled / %s total (%.1f%%), %s classes, imbalance ~%sx",
#              f"{stats['n_labeled']:,}", f"{stats['n_total']:,}",
#              100 * stats["labeled_fraction"], stats["n_classes_labeled"],
#              stats["imbalance_ratio"])
#     log.info("Artifacts written to %s/", reports_dir)
#     return stats
#
#
# # --------------------------------------------------------------------------- #
# # Self-test (synthetic data, no real pickle)
# # --------------------------------------------------------------------------- #
#
# def _make_synthetic_df() -> pd.DataFrame:
#     """
#     Build a DataFrame mimicking LSWMD's nested structure with KNOWN counts:
#       Center x5, Scratch x2, none x4, Unlabeled(empty) x3  => 14 rows, 3 labeled classes.
#     Two distinct wafer-map shapes to exercise the dimension logic.
#     """
#     rng = np.random.default_rng(0)
#
#     def fmap(shape):
#         return rng.integers(0, 3, size=shape, dtype=np.uint8)
#
#     rows = []
#     plan = [("Center", 5, (26, 26)),
#             ("Scratch", 2, (40, 35)),
#             ("none", 4, (26, 26)),
#             (None, 3, (26, 26))]   # None -> empty failureType -> Unlabeled
#     for cls, k, shape in plan:
#         for _ in range(k):
#             ft = np.array([], dtype=object) if cls is None else np.array([[cls]], dtype=object)
#             rows.append({
#                 MAP_COL: fmap(shape),
#                 "dieSize": shape[0] * shape[1],
#                 "lotName": "lotX",
#                 "waferIndex": 0,
#                 SPLIT_COL: np.array([["Training"]], dtype=object),
#                 LABEL_COL: ft,
#             })
#     return pd.DataFrame(rows)
#
#
# def self_test() -> bool:
#     print("=" * 60)
#     print("SELF-TEST: track1_eda")
#     print("=" * 60)
#     passed = True
#     df = normalize_labels(_make_synthetic_df())
#
#     # 1. label normalization unwraps nested arrays + empties
#     dist = class_distribution(df)
#     t1 = (dist.get("Center") == 5 and dist.get("Scratch") == 2
#           and dist.get("none") == 4 and dist.get(UNLABELED) == 3)
#     print(f"[{'PASS' if t1 else 'FAIL'}] class counts correct: {dist}")
#     passed &= t1
#
#     # 2. 'none' and 'Unlabeled' stay distinct
#     t2 = "none" in dist and UNLABELED in dist and dist["none"] != dist[UNLABELED]
#     print(f"[{'PASS' if t2 else 'FAIL'}] 'none' kept distinct from 'Unlabeled'")
#     passed &= t2
#
#     # 3. summary stats math
#     s = summary_stats(df)
#     t3 = (s["n_total"] == 14 and s["n_labeled"] == 11 and s["n_unlabeled"] == 3
#           and s["n_classes_labeled"] == 3 and s["n_unique_wafer_dims"] == 2)
#     print(f"[{'PASS' if t3 else 'FAIL'}] summary: total={s['n_total']} "
#           f"labeled={s['n_labeled']} classes={s['n_classes_labeled']} "
#           f"dims={s['n_unique_wafer_dims']}")
#     passed &= t3
#
#     # 4. imbalance ratio = max/min labeled class = 5/2 = 2.5
#     t4 = s["imbalance_ratio"] == 2.5
#     print(f"[{'PASS' if t4 else 'FAIL'}] imbalance ratio = {s['imbalance_ratio']} (expect 2.5)")
#     passed &= t4
#
#     # 5. all plot + json artifacts get written
#     with tempfile.TemporaryDirectory() as tmp:
#         rep = Path(tmp) / "reports"
#         rep.mkdir()
#         plot_class_distribution(df, rep / "c.png")
#         plot_wafer_dimensions(df, rep / "d.png")
#         plot_sample_maps(df, rep / "s.png")
#         (rep / "eda_summary.json").write_text(json.dumps(s))
#         t5 = all((rep / f).exists() for f in ["c.png", "d.png", "s.png", "eda_summary.json"])
#     print(f"[{'PASS' if t5 else 'FAIL'}] all artifacts written")
#     passed &= t5
#
#     print("-" * 60)
#     print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
#     print("=" * 60)
#     return passed
#
#
# # --------------------------------------------------------------------------- #
# # CLI
# # --------------------------------------------------------------------------- #
#
# def main(argv: list[str] | None = None) -> int:
#     ap = argparse.ArgumentParser(description="WM-811K EDA (Track 1, Module 2)")
#     ap.add_argument("--raw-dir", help="Dir holding LSWMD.pkl")
#     ap.add_argument("--pkl", help="Explicit path to LSWMD.pkl")
#     ap.add_argument("--reports-dir", default="reports")
#     ap.add_argument("--self-test", action="store_true")
#     args = ap.parse_args(argv)
#
#     if args.self_test:
#         return 0 if self_test() else 1
#
#     pkl = resolve_pickle(args.raw_dir, args.pkl)
#     run(pkl, Path(args.reports_dir))
#     return 0
#
#
# if __name__ == "__main__":
#     sys.exit(main())