"""
Track 1 — Module 4a (KERAS path): Wafer-map CNN + training.

Responsibility
--------------
Read the framework-neutral .npz from Module 3, train a small CNN to classify
wafer-map defect patterns, and save the best model + training history.

Mirrors the PyTorch path (Module 4b) exactly:
  * same architecture (4 conv blocks, BN, SpatialDropout, GAP, softmax)
  * same imbalance handling (class weights from the training split)
  * same augmentation (random 90-degree rotations + flips — exact, no interp)
  * same output contract for evaluate.py (Module 5)

Channel convention
------------------
Keras is channels-LAST, so load_split adds a trailing axis: (N,H,W) -> (N,H,W,1).

Outputs
-------
  models/keras/best_model.keras   best-val-loss checkpoint
  models/keras/history.json       per-epoch metrics
  models/keras/classes.json       class list (copied from processed dir)

Usage
-----
    python track1_model_keras.py --data-dir data/processed --out-dir models/keras
    python track1_model_keras.py --data-dir data/processed --epochs 30 --batch 256
    python track1_model_keras.py --self-test     # trains 2 epochs on synthetic data

Testable individually
---------------------
`--self-test` fabricates tiny synthetic splits, runs the FULL train() for a
couple of epochs, and asserts: model builds with the right output shape, loss
is finite and decreases, a checkpoint is written, and it reloads + predicts.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model_keras")


def _require_tf():
    """Lazy import so --help works even without TensorFlow installed."""
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "TensorFlow is required for the Keras path.\n"
            "  pip install tensorflow        (or tensorflow-cpu)\n"
            f"Import error: {exc}"
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split(data_dir: Path, name: str):
    """Load a split and add the channels-last axis: (N,H,W) -> (N,H,W,1)."""
    d = np.load(Path(data_dir) / f"{name}.npz")
    X = d["X"].astype("float32")[..., np.newaxis]   # channels-last
    y = d["y"].astype("int64")
    return X, y


def class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """Inverse-frequency weights, normalized to mean 1.0 — counters imbalance."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0                      # avoid div-by-zero
    w = counts.sum() / (n_classes * counts)
    w = w / w.mean()
    return {i: float(w[i]) for i in range(n_classes)}


def make_dataset(tf, X, y, batch: int, training: bool):
    """tf.data pipeline. Training set is shuffled + augmented (rot90 + flips)."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 4096))

        def aug(img, label):
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)   # 0/90/180/270 deg
            img = tf.image.rot90(img, k)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            return img, label

        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(tf, input_shape, n_classes: int, dropout: float = 0.2):
    """4 conv blocks (Conv-BN-ReLU-Pool) + SpatialDropout, GAP, softmax head."""
    L = tf.keras.layers
    model = tf.keras.Sequential(name="wafer_cnn")
    model.add(L.Input(shape=input_shape))
    for filters in (32, 64, 128, 128):
        model.add(L.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(L.BatchNormalization())
        model.add(L.ReLU())
        model.add(L.MaxPooling2D(2))
        model.add(L.SpatialDropout2D(dropout))
    model.add(L.GlobalAveragePooling2D())
    model.add(L.Dense(128, activation="relu"))
    model.add(L.Dropout(0.3))
    model.add(L.Dense(n_classes, activation="softmax"))
    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(data_dir: Path, out_dir: Path, epochs: int = 30, batch: int = 256,
          lr: float = 1e-3, dropout: float = 0.2, seed: int = 42) -> dict:
    tf = _require_tf()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = json.loads((data_dir / "label_classes.json").read_text())
    n_classes = len(classes)

    X_tr, y_tr = load_split(data_dir, "train")
    X_va, y_va = load_split(data_dir, "val")
    input_shape = X_tr.shape[1:]
    log.info("train=%s val=%s input=%s classes=%d",
             X_tr.shape, X_va.shape, input_shape, n_classes)

    cw = class_weights(y_tr, n_classes)
    log.info("class weights: %s", {classes[i]: round(v, 2) for i, v in cw.items()})

    model = build_model(tf, input_shape, n_classes, dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = out_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    hist = model.fit(
        make_dataset(tf, X_tr, y_tr, batch, training=True),
        validation_data=make_dataset(tf, X_va, y_va, batch, training=False),
        epochs=epochs, class_weight=cw, callbacks=callbacks, verbose=2,
    )

    if not ckpt.exists():                       # e.g. epochs too few to improve
        model.save(ckpt)
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in hist.history.items()}, indent=2))
    shutil.copy2(data_dir / "label_classes.json", out_dir / "classes.json")

    log.info("Saved model -> %s", ckpt)
    return hist.history


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _make_synthetic_processed(dir_: Path, n_classes=3, size=16, per=24, seed=0):
    """Write tiny train/val/test .npz + label_classes.json. Classes are
    linearly separable-ish (per-class bias) so loss can actually drop."""
    rng = np.random.default_rng(seed)
    dir_.mkdir(parents=True, exist_ok=True)
    for split, k in [("train", per), ("val", per // 2), ("test", per // 2)]:
        Xs, ys = [], []
        for c in range(n_classes):
            base = rng.random((k, size, size)).astype("float32") * 0.3
            base += c / n_classes                      # class-dependent signal
            Xs.append(np.clip(base, 0, 1)); ys.append(np.full(k, c))
        X = np.concatenate(Xs); y = np.concatenate(ys).astype("int64")
        np.savez_compressed(dir_ / f"{split}.npz", X=X, y=y)
    (dir_ / "label_classes.json").write_text(
        json.dumps([f"cls{i}" for i in range(n_classes)]))


def self_test() -> bool:
    print("=" * 60)
    print("SELF-TEST: track1_model_keras")
    print("=" * 60)
    tf = _require_tf()
    passed = True

    # 1. class weights: rarer class gets higher weight, mean ~1
    cw = class_weights(np.array([0, 0, 0, 0, 1]), n_classes=2)
    t1 = cw[1] > cw[0] and abs(np.mean(list(cw.values())) - 1.0) < 1e-6
    print(f"[{'PASS' if t1 else 'FAIL'}] class weights up-weight rare class {cw}")
    passed &= t1

    # 2. model builds with correct output shape
    m = build_model(tf, (16, 16, 1), n_classes=3)
    out = m(np.zeros((2, 16, 16, 1), dtype="float32"))
    t2 = tuple(out.shape) == (2, 3)
    print(f"[{'PASS' if t2 else 'FAIL'}] model output shape {tuple(out.shape)} == (2, 3)")
    passed &= t2

    # 3. full train() runs, writes checkpoint, loss is finite & drops
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "processed"
        out = Path(tmp) / "models"
        _make_synthetic_processed(data)
        hist = train(data, out, epochs=4, batch=8, lr=1e-3, seed=0)
        losses = hist["loss"]
        t3 = (out / "best_model.keras").exists() and np.isfinite(losses).all()
        print(f"[{'PASS' if t3 else 'FAIL'}] train ran, checkpoint saved, losses finite")
        passed &= t3

        t4 = losses[-1] < losses[0]
        print(f"[{'PASS' if t4 else 'FAIL'}] loss decreased {losses[0]:.3f} -> {losses[-1]:.3f}")
        passed &= t4

        # 5. reload + predict
        reloaded = tf.keras.models.load_model(out / "best_model.keras")
        d = np.load(data / "test.npz")
        pred = reloaded.predict(d["X"][..., None], verbose=0)
        t5 = pred.shape == (len(d["y"]), 3) and np.allclose(pred.sum(1), 1, atol=1e-4)
        print(f"[{'PASS' if t5 else 'FAIL'}] reloaded model predicts valid probabilities")
        passed &= t5

    print("-" * 60)
    print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
    print("=" * 60)
    return passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wafer-map CNN — Keras (Track 1, Module 4a)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="models/keras")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    train(Path(args.data_dir), Path(args.out_dir), epochs=args.epochs,
          batch=args.batch, lr=args.lr, dropout=args.dropout, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
