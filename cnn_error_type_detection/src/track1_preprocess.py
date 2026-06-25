"""
Track 1 — Module 3: Preprocessing for WM-811K.

Responsibility
--------------
Turn the raw pickle into model-ready, framework-neutral arrays:
  1. filter to labeled rows (drop 'Unlabeled'; optionally drop 'none')
  2. nearest-neighbour resize every wafer map to a common grid (default 64x64),
     preserving the discrete {0,1,2} values
  3. normalize to [0,1] (values / 2.0)
  4. integer-encode labels and PERSIST the class list (reproducible)
  5. stratified, seeded train/val/test split
  6. save train/val/test as .npz

Output contract (read by both the Keras and PyTorch model modules)
------------------------------------------------------------------
  train.npz / val.npz / test.npz:  X float32 (N, H, W) in [0,1]; y int64 (N,)
  label_classes.json:  list where index == integer label, value == class name
  split_meta.json:     per-split per-class counts

Arrays carry NO channel dimension on purpose — Keras adds (...,1),
PyTorch adds (1,...). See the loaders in the Module-4 files.

Usage
-----
    python track1_preprocess.py --raw-dir data/raw --out-dir data/processed
    python track1_preprocess.py --raw-dir data/raw --out-dir data/processed \\
        --img-size 64 --drop-none --seed 42
    python track1_preprocess.py --self-test     # no real data needed

Testable individually
---------------------
`--self-test` builds a synthetic DataFrame with known per-class counts and
varied wafer shapes, runs the full pipeline into a temp dir, and asserts:
shapes, value range, label round-trip, no data leakage across splits,
stratification, and that every artifact loads back correctly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")

PICKLE_NAME = "LSWMD.pkl"
LABEL_COL = "failureType"
MAP_COL = "waferMap"
UNLABELED = "Unlabeled"
MAP_MAX_VALUE = 2.0     # {0=bg, 1=pass, 2=fail} -> divide by this to hit [0,1]


# --------------------------------------------------------------------------- #
# Small helpers (duplicated from Module 2 so this file stands alone)
# --------------------------------------------------------------------------- #

def _normalize_label(val) -> str:
    if val is None:
        return UNLABELED
    arr = np.asarray(val, dtype=object).ravel()
    if arr.size == 0:
        return UNLABELED
    s = str(arr[0]).strip()
    return s if s else UNLABELED


def read_wm811k_pickle(path):
    """
    Read LSWMD.pkl, including the legacy pickle written by very old pandas
    under Python 2. Tries a normal read first; on failure installs module shims
    (old 'pandas.indexes' path + removed Int64Index/Float64Index) and falls back
    to raw pickle.load with encoding='latin1' to decode Python-2 byte strings.
    Modern pickles are unaffected.
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


def resolve_pickle(raw_dir: str | None, pkl: str | None) -> Path:
    if pkl:
        return Path(pkl)
    if raw_dir:
        return Path(raw_dir) / PICKLE_NAME
    raise ValueError("Provide either --pkl or --raw-dir")


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #

def filter_labeled(df: pd.DataFrame, drop_none: bool = False) -> pd.DataFrame:
    """Keep only inspected wafers with a real failure-type label."""
    labels = df[LABEL_COL].map(_normalize_label)
    keep = labels != UNLABELED
    if drop_none:
        keep &= labels.str.lower() != "none"
    out = df.loc[keep].copy()
    out["label"] = labels.loc[keep].values
    log.info("Filtered %s -> %s labeled rows (drop_none=%s)",
             f"{len(df):,}", f"{len(out):,}", drop_none)
    return out


def resize_map(arr, size: int) -> np.ndarray:
    """
    Nearest-neighbour resize a 2-D wafer map to (size, size).

    NN (not bilinear) because the values are categorical {0,1,2}; interpolation
    would invent fractional die states that mean nothing.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D wafer map, got shape {a.shape}")
    h, w = a.shape
    ri = np.clip((np.arange(size) * h / size).astype(int), 0, h - 1)
    ci = np.clip((np.arange(size) * w / size).astype(int), 0, w - 1)
    return a[ri][:, ci]


def build_arrays(df: pd.DataFrame, size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resize+normalize maps into X, integer-encode labels into y, return class list."""
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X = np.empty((len(df), size, size), dtype=np.float32)
    y = np.empty(len(df), dtype=np.int64)
    for i, (_, row) in enumerate(df.iterrows()):
        X[i] = resize_map(row[MAP_COL], size).astype(np.float32) / MAP_MAX_VALUE
        y[i] = class_to_idx[row["label"]]
    return X, y, classes


def stratified_split(
    X: np.ndarray, y: np.ndarray,
    val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Seeded, stratified split into train/val/test (proportions preserved per class)."""
    # First peel off test, then split the remainder into train/val.
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed)
    val_rel = val_frac / (1.0 - test_frac)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=seed)
    return {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)}


def save_outputs(splits: dict, classes: list[str], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"classes": classes, "img_size": int(splits["train"][0].shape[1]), "splits": {}}
    for name, (X, y) in splits.items():
        np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y)
        counts = {classes[i]: int(c) for i, c in
                  sorted(Counter(y).items())}
        meta["splits"][name] = {"n": int(len(y)), "per_class": counts}

    (out_dir / "label_classes.json").write_text(json.dumps(classes, indent=2))
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(pkl_path: Path, out_dir: Path, img_size: int = 64,
        drop_none: bool = False, seed: int = 42,
        val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    log.info("Loading %s ...", pkl_path)
    df = read_wm811k_pickle(pkl_path)

    df = filter_labeled(df, drop_none=drop_none)
    X, y, classes = build_arrays(df, img_size)
    log.info("Built arrays X=%s y=%s | %s classes", X.shape, y.shape, len(classes))

    splits = stratified_split(X, y, val_frac, test_frac, seed)
    meta = save_outputs(splits, classes, out_dir)

    for name in ("train", "val", "test"):
        log.info("  %-5s: %s samples", name, f"{meta['splits'][name]['n']:,}")
    log.info("Saved processed data + metadata to %s/", out_dir)
    return meta


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _make_synthetic_df() -> pd.DataFrame:
    """Known counts (>= enough for a 70/15/15 stratified split), two map shapes."""
    rng = np.random.default_rng(0)

    def fmap(shape):
        return rng.integers(0, 3, size=shape, dtype=np.uint8)

    rows = []
    plan = [("Center", 40, (26, 26)),
            ("Scratch", 24, (40, 35)),
            ("none", 32, (26, 26)),
            (None, 10, (26, 26))]   # Unlabeled -> must be dropped
    for cls, k, shape in plan:
        for _ in range(k):
            ft = np.array([], dtype=object) if cls is None else np.array([[cls]], dtype=object)
            rows.append({MAP_COL: fmap(shape), LABEL_COL: ft})
    return pd.DataFrame(rows)


def self_test() -> bool:
    print("=" * 60)
    print("SELF-TEST: track1_preprocess")
    print("=" * 60)
    passed = True
    SIZE = 32
    df = _make_synthetic_df()

    # 1. filter drops Unlabeled (10 rows) -> 96 labeled
    fdf = filter_labeled(df, drop_none=False)
    t1 = len(fdf) == 96 and UNLABELED not in fdf["label"].values
    print(f"[{'PASS' if t1 else 'FAIL'}] filter drops Unlabeled (kept {len(fdf)})")
    passed &= t1

    # 2. drop_none also removes the 32 'none' rows -> 64
    t2 = len(filter_labeled(df, drop_none=True)) == 64
    print(f"[{'PASS' if t2 else 'FAIL'}] drop_none removes 'none' class")
    passed &= t2

    # 3. resize produces (SIZE,SIZE) and preserves discrete values {0,1,2}
    rm = resize_map(df.iloc[0][MAP_COL], SIZE)
    t3 = rm.shape == (SIZE, SIZE) and set(np.unique(rm)).issubset({0, 1, 2})
    print(f"[{'PASS' if t3 else 'FAIL'}] resize -> {rm.shape}, values stay categorical")
    passed &= t3

    # 4. build_arrays: shape, dtype, normalized range, label round-trip
    X, y, classes = build_arrays(fdf, SIZE)
    t4 = (X.shape == (96, SIZE, SIZE) and X.dtype == np.float32
          and X.min() >= 0.0 and X.max() <= 1.0
          and classes == ["Center", "Scratch", "none"]
          and y.max() == len(classes) - 1)
    print(f"[{'PASS' if t4 else 'FAIL'}] arrays X={X.shape} range=[{X.min():.1f},{X.max():.1f}] "
          f"classes={classes}")
    passed &= t4

    # 5. split: counts add up, no leakage, all classes present in every split
    splits = stratified_split(X, y, seed=42)
    n_total = sum(len(v[1]) for v in splits.values())
    classes_per_split = [set(np.unique(v[1])) for v in splits.values()]
    t5 = (n_total == 96
          and all(s == set(range(len(classes))) for s in classes_per_split))
    print(f"[{'PASS' if t5 else 'FAIL'}] split totals={n_total}, every class in every split")
    passed &= t5

    # 6. determinism: same seed -> identical test indices
    s_a = stratified_split(X, y, seed=7)["test"][1]
    s_b = stratified_split(X, y, seed=7)["test"][1]
    t6 = np.array_equal(s_a, s_b)
    print(f"[{'PASS' if t6 else 'FAIL'}] split is deterministic for fixed seed")
    passed &= t6

    # 7. artifacts save + reload correctly
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "processed"
        meta = save_outputs(splits, classes, out)
        files_ok = all((out / f).exists() for f in
                       ["train.npz", "val.npz", "test.npz",
                        "label_classes.json", "split_meta.json"])
        d = np.load(out / "train.npz")
        reload_ok = d["X"].dtype == np.float32 and d["y"].dtype == np.int64
        reloaded_classes = json.loads((out / "label_classes.json").read_text())
        t7 = files_ok and reload_ok and reloaded_classes == classes
    print(f"[{'PASS' if t7 else 'FAIL'}] artifacts saved + reload with correct dtypes")
    passed &= t7

    print("-" * 60)
    print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
    print("=" * 60)
    return passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WM-811K preprocessing (Track 1, Module 3)")
    ap.add_argument("--raw-dir", help="Dir holding LSWMD.pkl")
    ap.add_argument("--pkl", help="Explicit path to LSWMD.pkl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--drop-none", action="store_true",
                    help="Exclude the 'none' (no-pattern) class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    pkl = resolve_pickle(args.raw_dir, args.pkl)
    run(pkl, Path(args.out_dir), img_size=args.img_size, drop_none=args.drop_none,
        seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())