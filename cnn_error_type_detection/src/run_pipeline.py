"""
Track 1 — Pipeline orchestrator.

Runs the full wafer-map pipeline in order, driven by config.yaml:

    download -> eda -> preprocess -> train(backend) -> evaluate

Each stage is launched as a SUBPROCESS (sys.executable + the module's own CLI),
so the orchestrator never imports TensorFlow or PyTorch itself — only the stage
you actually run loads a framework. This keeps backend isolation clean and means
a crash in one stage can't take down the runner.

Usage
-----
    # Run everything using settings from config.yaml:
    python run_pipeline.py

    # Override the backend / a few common knobs:
    python run_pipeline.py --backend keras --epochs 50

    # Re-run just the tail end (data already prepared):
    python run_pipeline.py --only train evaluate

    # Skip the (slow) download because the pickle is already on disk:
    python run_pipeline.py --skip download

    # See the exact commands without executing anything:
    python run_pipeline.py --dry-run

    # Verify every stage wires up correctly (runs each module's --self-test):
    python run_pipeline.py --self-test

Testable individually
---------------------
`--self-test` invokes each stage module with its own `--self-test` flag, in
order, and reports a per-stage pass/fail plus an overall result. No dataset or
trained model is required. For the train stage it tests only the SELECTED
backend (matching what a real run would execute).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# Module filename for each stage (resolved relative to this script).
MODULES = {
    "download": "track1_downloader.py",
    "eda": "track1_eda.py",
    "preprocess": "track1_preprocess.py",
    "train_keras": "track1_model_keras.py",
    "train_torch": "track1_model_torch.py",
    "evaluate": "track1_evaluate.py",
}

# Logical stage order (train resolves to a backend-specific module at build time).
STAGE_ORDER = ["download", "eda", "preprocess", "train", "evaluate"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def _module(name: str) -> str:
    return str(HERE / MODULES[name])


def build_commands(cfg: dict, args) -> list[tuple[str, list[str]]]:
    """Return [(stage_name, command_list), ...] for the real pipeline run."""
    paths = cfg.get("paths", {})
    raw = paths.get("raw_dir", "data/raw")
    processed = paths.get("processed_dir", "data/processed")
    reports = paths.get("reports_dir", "reports")
    models_dir = paths.get("models_dir", "models")

    pre = cfg.get("preprocess", {})
    tr = cfg.get("train", {})
    backend = (args.backend or tr.get("backend", "torch")).lower()
    if backend not in ("keras", "torch"):
        raise ValueError(f"backend must be keras|torch, got '{backend}'")

    epochs = args.epochs if args.epochs is not None else tr.get("epochs", 30)
    batch = args.batch if args.batch is not None else tr.get("batch", 256)
    lr = tr.get("lr", 1e-3)
    dropout = tr.get("dropout", 0.2)
    img_size = args.img_size if args.img_size is not None else pre.get("img_size", 64)
    drop_none = args.drop_none or pre.get("drop_none", False)

    py = sys.executable
    model_out = f"{models_dir}/{backend}"

    cmds: dict[str, list[str]] = {}

    dl = [py, _module("download"), "--raw-dir", raw]
    if args.from_file:
        dl += ["--from-file", args.from_file]
    cmds["download"] = dl

    cmds["eda"] = [py, _module("eda"), "--raw-dir", raw, "--reports-dir", reports]

    pp = [py, _module("preprocess"), "--raw-dir", raw, "--out-dir", processed,
          "--img-size", str(img_size)]
    if drop_none:
        pp += ["--drop-none"]
    cmds["preprocess"] = pp

    train_mod = _module("train_keras" if backend == "keras" else "train_torch")
    cmds["train"] = [py, train_mod, "--data-dir", processed, "--out-dir", model_out,
                     "--epochs", str(epochs), "--batch", str(batch),
                     "--lr", str(lr), "--dropout", str(dropout)]

    cmds["evaluate"] = [py, _module("evaluate"), "--model", model_out,
                        "--data-dir", processed, "--reports-dir", reports,
                        "--backend", backend]

    selected = _select_stages(args)
    return [(s, cmds[s]) for s in STAGE_ORDER if s in selected]


def build_selftest_commands(cfg: dict, args) -> list[tuple[str, list[str]]]:
    """Self-test variant: each stage module invoked with --self-test."""
    tr = cfg.get("train", {})
    backend = (args.backend or tr.get("backend", "torch")).lower()
    py = sys.executable

    mapping = {
        "download": "download",
        "eda": "eda",
        "preprocess": "preprocess",
        "train": "train_keras" if backend == "keras" else "train_torch",
        "evaluate": "evaluate",
    }
    selected = _select_stages(args)
    return [(s, [py, _module(mapping[s]), "--self-test"])
            for s in STAGE_ORDER if s in selected]


def _select_stages(args) -> set[str]:
    if args.only:
        unknown = set(args.only) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"--only got unknown stages: {sorted(unknown)}")
        return set(args.only)
    return set(STAGE_ORDER) - set(args.skip or [])


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run(commands: list[tuple[str, list[str]]], dry_run: bool, keep_going: bool) -> int:
    if not commands:
        print("Nothing to run (check --only / --skip).")
        return 0

    plan = " -> ".join(name for name, _ in commands)
    _print_header(f"PIPELINE PLAN: {plan}")
    for name, cmd in commands:
        print(f"  [{name:9}] {' '.join(cmd)}")

    if dry_run:
        print("\n--dry-run set: no commands executed.")
        return 0

    failures = []
    for i, (name, cmd) in enumerate(commands, 1):
        _print_header(f"STAGE {i}/{len(commands)}: {name}")
        t0 = time.time()
        result = subprocess.run(cmd)
        dt = time.time() - t0
        if result.returncode == 0:
            print(f"\n  ✅ {name} finished in {dt:.1f}s")
        else:
            print(f"\n  ❌ {name} FAILED (exit {result.returncode}) after {dt:.1f}s")
            failures.append(name)
            if not keep_going:
                print("\nStopping (use --keep-going to continue past failures).")
                break

    _print_header("PIPELINE SUMMARY")
    if failures:
        print(f"  FAILED stages: {failures}")
        return 1
    print(f"  All {len(commands)} stage(s) completed successfully ✅")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track 1 pipeline orchestrator")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--backend", choices=["keras", "torch"],
                    help="Override config train.backend")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--drop-none", action="store_true")
    ap.add_argument("--from-file", help="Pass a manual LSWMD.pkl to the downloader")
    ap.add_argument("--skip", nargs="*", default=[], metavar="STAGE",
                    help=f"Skip stages. Choices: {STAGE_ORDER}")
    ap.add_argument("--only", nargs="*", metavar="STAGE",
                    help=f"Run only these stages. Choices: {STAGE_ORDER}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue past a failing stage")
    ap.add_argument("--self-test", action="store_true",
                    help="Run each stage's --self-test instead of the real pipeline")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    commands = (build_selftest_commands(cfg, args) if args.self_test
                else build_commands(cfg, args))
    return run(commands, dry_run=args.dry_run, keep_going=args.self_test or args.keep_going)


if __name__ == "__main__":
    sys.exit(main())
