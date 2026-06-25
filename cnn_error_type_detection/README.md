# Wafer Map Classifier — Track 1

A modular CNN pipeline that classifies **defect patterns on WM-811K wafer maps**
(Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none).
Each stage is a standalone, individually-testable module. The model stage ships
in **two interchangeable backends — Keras and PyTorch** — that share the same
preprocessed data and the same evaluator.

> This is a self-contained project. The SEM defect-detection work (Track 2) is
> intentionally kept in a separate repository — the two share no code.

---

## What this answers

WM-811K wafer maps are low-resolution grids where each die is pass/fail. The
*spatial pattern* of failures points to a systemic process issue. This pipeline
learns to name that pattern. The central challenge is **severe class imbalance**
(`none` dominates; `Near-full` is rare), so the project is built around
imbalance-aware training and **per-class** evaluation rather than raw accuracy.

---

## Project structure

```
wafer-map-classifier/
├── config.yaml                 # central settings (mirror of the CLI flags)
├── requirements.txt            # core deps + your chosen backend
├── README.md
│
├── run_pipeline.py             # orchestrator: runs all stages in order
├── track1_downloader.py        # Module 1: fetch + verify LSWMD.pkl
├── track1_eda.py               # Module 2: imbalance + dimension profiling
├── track1_preprocess.py        # Module 3: filter / resize / encode / split
├── track1_model_keras.py       # Module 4a: Keras CNN + training
├── track1_model_torch.py       # Module 4b: PyTorch CNN + training
├── track1_evaluate.py          # Module 5: backend-agnostic metrics
│
├── data/
│   ├── raw/                     # LSWMD.pkl                 (git-ignored)
│   └── processed/              # train/val/test .npz + json (git-ignored)
├── models/
│   ├── keras/                   # best_model.keras + history (git-ignored)
│   └── torch/                   # best_model.pt + history    (git-ignored)
└── reports/                     # EDA + evaluation plots / json
```

Pipeline flow:

```
downloader ──▶ eda ──▶ preprocess ──▶ ┬─ model_keras ─┐
                                      └─ model_torch ─┴─▶ evaluate
```

---

## Install

Python 3.10+ recommended.

```bash
python -m venv .venv && source .venv/bin/activate
# Edit requirements.txt first: uncomment ONE backend (tensorflow OR torch)
pip install -r requirements.txt
```

For the downloader's Kaggle path you also need API credentials at
`~/.kaggle/kaggle.json` (from your Kaggle account → Settings → API). If you'd
rather download `LSWMD.pkl` by hand, skip Kaggle and use `--from-file`.

---

## Run the full pipeline

### One command (recommended)

`run_pipeline.py` runs every stage in order, driven by `config.yaml`:

```bash
python run_pipeline.py                       # full pipeline, settings from config.yaml
python run_pipeline.py --backend keras       # override the backend
python run_pipeline.py --epochs 50 --batch 512
python run_pipeline.py --skip download       # pickle already on disk
python run_pipeline.py --only train evaluate # re-run just the tail
python run_pipeline.py --dry-run             # print commands, run nothing
python run_pipeline.py --self-test           # run every stage's self-test in order
```

It launches each stage as a subprocess, so only the backend you select ever
loads a framework, and it stops at the first failing stage (use `--keep-going`
to continue). The stages are: `download eda preprocess train evaluate`.

### Manual, stage by stage

Settings below match the defaults in `config.yaml`; change them there for
reference and pass the same values on the command line.

```bash
# 1. Get the data (Kaggle API) ...
python track1_downloader.py  --raw-dir data/raw
#    ... or register a manual download:
# python track1_downloader.py --raw-dir data/raw --from-file ~/Downloads/LSWMD.pkl

# 2. Explore (writes plots + eda_summary.json to reports/)
python track1_eda.py         --raw-dir data/raw --reports-dir reports

# 3. Preprocess -> data/processed/{train,val,test}.npz + label_classes.json
python track1_preprocess.py  --raw-dir data/raw --out-dir data/processed --img-size 64

# 4. Train — pick ONE backend:
python track1_model_torch.py --data-dir data/processed --out-dir models/torch --epochs 30 --batch 256
# python track1_model_keras.py --data-dir data/processed --out-dir models/keras --epochs 30 --batch 256

# 5. Evaluate (auto-detects backend from the model file)
python track1_evaluate.py    --model models/torch --data-dir data/processed --reports-dir reports
```

Useful preprocessing variant — the common 8-defect-classes-only setup:

```bash
python track1_preprocess.py --raw-dir data/raw --out-dir data/processed --drop-none
```

---

## Testing

Every module carries a `--self-test` that runs on synthetic data — no real
dataset, no trained model required. Run them all:

```bash
for m in downloader eda preprocess model_keras model_torch evaluate; do
  echo "== $m =="
  python track1_$m.py --self-test || echo "FAILED: $m"
done
```

`model_keras` requires TensorFlow installed; `model_torch` and the end-to-end
portion of `evaluate` require PyTorch. The non-model modules need only the core
dependencies.

---

## Outputs

| Stage | Writes |
|-------|--------|
| Module 1 | `data/raw/LSWMD.pkl` |
| Module 2 | `reports/eda_{class_distribution,wafer_dimensions,sample_maps}.png`, `reports/eda_summary.json` |
| Module 3 | `data/processed/{train,val,test}.npz`, `label_classes.json`, `split_meta.json` |
| Module 4 | `models/<backend>/best_model.{keras,pt}`, `history.json`, `classes.json` |
| Module 5 | `reports/eval_metrics.json`, `reports/eval_confusion_matrix.png`, `reports/eval_pr_curves.png` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pandas.indexes'` when reading LSWMD.pkl.**
The public WM-811K pickle was created with a ~2016 pandas, where index classes
lived under `pandas.indexes` and included `Int64Index`/`Float64Index` (renamed in
pandas 0.20, removed in pandas 2.0). The loaders handle this automatically: a
normal read is tried first, and only on failure are compatibility shims installed
and the read retried. If you somehow still hit it, you can re-save a clean copy
once from a Python REPL:

```python
from track1_eda import read_wm811k_pickle
read_wm811k_pickle("data/raw/LSWMD.pkl").to_pickle("data/raw/LSWMD.pkl")
```

In the rare case the legacy frame can't be revived on a very new pandas, do the
one-time re-save above inside a throwaway `python==3.10` env with `pandas==1.5.*`,
then use the resulting file normally.

**`UnicodeDecodeError` (cp1252) during the Kaggle download on Windows.** Harmless
output-decoding noise from the Kaggle CLI; the download still completes. Fixed by
decoding subprocess output as UTF-8.

## Key design notes

- **Framework-neutral data.** Module 3 saves arrays as `(N, H, W)` with no
  channel axis. Keras adds channels-last `(N,H,W,1)`; PyTorch adds channels-first
  `(N,1,H,W)`. One `.npz`, both backends.
- **Nearest-neighbour resize.** Map values are categorical `{0,1,2}`
  (background/pass/fail); interpolation would invent meaningless fractional die
  states, so resizing uses NN only.
- **Imbalance handling.** Inverse-frequency class weights (normalized to mean 1)
  during training; **macro-F1 and per-class F1** at evaluation, because overall
  accuracy is misleading when one class is ~80% of the data.
- **Backend-agnostic evaluation.** The PyTorch model is saved as TorchScript so
  the evaluator loads it without importing the model class, and dispatches purely
  on file extension.

---

## Suggested `.gitignore`

```
data/raw/
data/processed/
models/
.venv/
__pycache__/
*.pyc
```
