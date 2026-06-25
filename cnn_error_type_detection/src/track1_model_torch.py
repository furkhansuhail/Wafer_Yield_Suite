"""
Track 1 — Module 4b (PYTORCH path): Wafer-map CNN + training.

Responsibility
--------------
Read the framework-neutral .npz from Module 3, train a small CNN to classify
wafer-map defect patterns, and save the best model + training history.

Mirrors the Keras path (Module 4a) exactly:
  * same architecture (4 conv blocks, BN, SpatialDropout, GAP, head)
  * same imbalance handling (inverse-frequency class weights)
  * same augmentation (random 90-degree rotations + flips)
  * same output contract for evaluate.py (Module 5)

Channel convention
------------------
PyTorch is channels-FIRST, so load_split adds a leading axis: (N,H,W) -> (N,1,H,W).

Portability
-----------
The best model is saved as TorchScript (best_model.pt). evaluate.py can then
load it with torch.jit.load without importing this file's WaferCNN class.

Outputs
-------
  models/torch/best_model.pt    TorchScript, best val-loss
  models/torch/history.json     per-epoch metrics
  models/torch/classes.json     class list (copied from processed dir)

Usage
-----
    python track1_model_torch.py --data-dir data/processed --out-dir models/torch
    python track1_model_torch.py --data-dir data/processed --epochs 30 --batch 256
    python track1_model_torch.py --self-test     # trains a few epochs on synthetic data

Testable individually
---------------------
`--self-test` fabricates tiny synthetic splits, runs the FULL train() for a
few epochs, and asserts: model builds with the right output shape, loss is
finite and decreases, a TorchScript checkpoint is written, and it reloads +
predicts valid probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
log = logging.getLogger("model_torch")


def _require_torch():
    """Lazy import so --help works even without PyTorch installed."""
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyTorch is required for the Torch path.\n"
            "  pip install torch        (CPU build is fine)\n"
            f"Import error: {exc}"
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split(data_dir: Path, name: str):
    """Load a split as channels-first tensors: X (N,1,H,W) float32, y (N,) int64."""
    torch = _require_torch()
    d = np.load(Path(data_dir) / f"{name}.npz")
    X = torch.from_numpy(d["X"].astype("float32")).unsqueeze(1)   # (N,1,H,W)
    y = torch.from_numpy(d["y"].astype("int64"))
    return X, y


def class_weights(y, n_classes: int):
    """Inverse-frequency weights, normalized to mean 1.0. Returns a 1-D tensor."""
    torch = _require_torch()
    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    counts = np.bincount(y_np, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _build_dataset(X, y, training: bool, seed: int = 0):
    """Dataset that applies rot90 + flip augmentation per-sample when training."""
    torch = _require_torch()
    from torch.utils.data import Dataset

    class WaferDataset(Dataset):
        def __init__(self):
            self.X, self.y, self.training = X, y, training
            self.rng = random.Random(seed)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x = self.X[i]                       # (1,H,W)
            if self.training:
                x = torch.rot90(x, self.rng.randint(0, 3), dims=[1, 2])
                if self.rng.random() < 0.5:
                    x = torch.flip(x, dims=[2])
                if self.rng.random() < 0.5:
                    x = torch.flip(x, dims=[1])
            return x, self.y[i]

    return WaferDataset()


# --------------------------------------------------------------------------- #
# Model — mirrors the Keras architecture
# --------------------------------------------------------------------------- #

def build_model(input_shape, n_classes: int, dropout: float = 0.2):
    """4 conv blocks (Conv-BN-ReLU-Pool) + SpatialDropout (Dropout2d), GAP, head.

    Returns logits (no softmax) — CrossEntropyLoss applies log-softmax internally.
    """
    torch = _require_torch()
    import torch.nn as nn

    class WaferCNN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks, in_ch = [], input_shape[0]
            for filters in (32, 64, 128, 128):
                blocks += [
                    nn.Conv2d(in_ch, filters, 3, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout),       # spatial dropout
                ]
                in_ch = filters
            self.features = nn.Sequential(*blocks)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.head(self.gap(self.features(x)))

    return WaferCNN()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _run_epoch(torch, model, loader, criterion, optimizer=None):
    """One pass. optimizer=None -> eval mode. Returns (mean_loss, accuracy)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            if train_mode:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train(data_dir: Path, out_dir: Path, epochs: int = 30, batch: int = 256,
          lr: float = 1e-3, dropout: float = 0.2, seed: int = 42,
          patience: int = 6) -> dict:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = json.loads((data_dir / "label_classes.json").read_text())
    n_classes = len(classes)

    X_tr, y_tr = load_split(data_dir, "train")
    X_va, y_va = load_split(data_dir, "val")
    input_shape = tuple(X_tr.shape[1:])          # (1,H,W)
    log.info("train=%s val=%s input=%s classes=%d",
             tuple(X_tr.shape), tuple(X_va.shape), input_shape, n_classes)

    cw = class_weights(y_tr, n_classes)
    log.info("class weights: %s", {classes[i]: round(float(cw[i]), 2) for i in range(n_classes)})

    model = build_model(input_shape, n_classes, dropout)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-6)

    tr_loader = DataLoader(_build_dataset(X_tr, y_tr, True, seed), batch_size=batch, shuffle=True)
    va_loader = DataLoader(_build_dataset(X_va, y_va, False), batch_size=batch, shuffle=False)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(torch, model, tr_loader, criterion, optimizer)
        va_loss, va_acc = _run_epoch(torch, model, va_loader, criterion)
        scheduler.step(va_loss)
        history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_accuracy"].append(va_acc)
        log.info("epoch %2d/%d | loss %.4f acc %.3f | val_loss %.4f val_acc %.3f",
                 epoch, epochs, tr_loss, tr_acc, va_loss, va_acc)

        if va_loss < best_val - 1e-5:
            best_val, no_improve = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:                   # restore best weights
        model.load_state_dict(best_state)

    # Save as TorchScript for backend-agnostic loading in evaluate.py.
    model.eval()
    example = torch.zeros((1, *input_shape), dtype=torch.float32)
    scripted = torch.jit.trace(model, example)
    ckpt = out_dir / "best_model.pt"
    scripted.save(str(ckpt))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    shutil.copy2(data_dir / "label_classes.json", out_dir / "classes.json")
    log.info("Saved TorchScript model -> %s", ckpt)
    return history


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _make_synthetic_processed(dir_: Path, n_classes=3, size=16, per=24, seed=0):
    """Tiny train/val/test .npz with a per-class signal so loss can actually drop."""
    rng = np.random.default_rng(seed)
    dir_.mkdir(parents=True, exist_ok=True)
    for split, k in [("train", per), ("val", per // 2), ("test", per // 2)]:
        Xs, ys = [], []
        for c in range(n_classes):
            base = rng.random((k, size, size)).astype("float32") * 0.3 + c / n_classes
            Xs.append(np.clip(base, 0, 1)); ys.append(np.full(k, c))
        X = np.concatenate(Xs); y = np.concatenate(ys).astype("int64")
        np.savez_compressed(dir_ / f"{split}.npz", X=X, y=y)
    (dir_ / "label_classes.json").write_text(
        json.dumps([f"cls{i}" for i in range(n_classes)]))


def self_test() -> bool:
    print("=" * 60)
    print("SELF-TEST: track1_model_torch")
    print("=" * 60)
    torch = _require_torch()
    passed = True

    # 1. class weights: rarer class higher, mean ~1
    cw = class_weights(torch.tensor([0, 0, 0, 0, 1]), n_classes=2)
    t1 = cw[1] > cw[0] and abs(float(cw.mean()) - 1.0) < 1e-6
    print(f"[{'PASS' if t1 else 'FAIL'}] class weights up-weight rare class "
          f"{[round(float(x), 2) for x in cw]}")
    passed &= t1

    # 2. model output shape (logits)
    m = build_model((1, 16, 16), n_classes=3)
    out = m(torch.zeros((2, 1, 16, 16)))
    t2 = tuple(out.shape) == (2, 3)
    print(f"[{'PASS' if t2 else 'FAIL'}] model output shape {tuple(out.shape)} == (2, 3)")
    passed &= t2

    # 3+4. full train, checkpoint saved, finite + decreasing loss
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "processed"
        out_dir = Path(tmp) / "models"
        _make_synthetic_processed(data)
        hist = train(data, out_dir, epochs=6, batch=8, lr=1e-3, seed=0, patience=99)
        losses = hist["loss"]
        t3 = (out_dir / "best_model.pt").exists() and np.isfinite(losses).all()
        print(f"[{'PASS' if t3 else 'FAIL'}] train ran, TorchScript saved, losses finite")
        passed &= t3
        t4 = losses[-1] < losses[0]
        print(f"[{'PASS' if t4 else 'FAIL'}] loss decreased {losses[0]:.3f} -> {losses[-1]:.3f}")
        passed &= t4

        # 5. reload via torch.jit.load (no class import) + valid probabilities
        reloaded = torch.jit.load(str(out_dir / "best_model.pt"))
        reloaded.eval()
        d = np.load(data / "test.npz")
        xb = torch.from_numpy(d["X"].astype("float32")).unsqueeze(1)
        with torch.no_grad():
            probs = torch.softmax(reloaded(xb), dim=1)
        t5 = (tuple(probs.shape) == (len(d["y"]), 3)
              and torch.allclose(probs.sum(1), torch.ones(len(d["y"])), atol=1e-4))
        print(f"[{'PASS' if t5 else 'FAIL'}] reloaded TorchScript predicts valid probabilities")
        passed &= t5

    print("-" * 60)
    print("RESULT:", "ALL PASSED ✅" if passed else "SOME FAILED ❌")
    print("=" * 60)
    return passed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wafer-map CNN — PyTorch (Track 1, Module 4b)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="models/torch")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    train(Path(args.data_dir), Path(args.out_dir), epochs=args.epochs,
          batch=args.batch, lr=args.lr, dropout=args.dropout,
          seed=args.seed, patience=args.patience)
    return 0


if __name__ == "__main__":
    sys.exit(main())
