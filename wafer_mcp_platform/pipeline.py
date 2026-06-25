"""
pipeline.py
===========

The whole-pipeline orchestration layer: **download → train → status**, with a
small on-disk *job ledger* so any front-end can launch long-running work and
watch it progress.

Why this exists
---------------
The MCP `train` tool fits a model in one blocking call. That's fine for an agent,
but a UI that "controls the whole pipeline" needs three more things:

  1. an explicit **download** step it can trigger and *confirm* ("is the data
     here yet?"),
  2. the ability to start training **in the background** and report that it is
     *running* (not just freeze until it's done), and
  3. a durable record so it can tell the user **when training completes** — even
     across Streamlit reruns.

This module provides all three, independent of MCP. Both the Streamlit dashboard
(in-process background threads) and the MCP server (synchronous tools) call the
same functions, so there is one source of truth.

Job ledger
----------
Every download/train is a job written to ``_workspace/jobs/<job_id>.json``::

    {
      "job_id": "train_secom_20260625_120000_ab12cd",
      "kind":   "train" | "download",
      "target": "secom" | "yield_curve" | "wafer_cnn" | "wm811k",
      "state":  "queued" | "running" | "completed" | "failed",
      "messages": [{"t": "...", "text": "Fitting pipeline…"}, ...],
      "result":   {...} | null,
      "error":    null | "RuntimeError: ...",
      "created_at": "...", "started_at": "...", "finished_at": "..."
    }

Front-ends poll ``read_job(job_id)`` / ``list_jobs()`` to render live status.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import platform_config as cfg  # noqa: E402
from shared import data_registry as registry  # noqa: E402

JOBS_DIR = cfg.WORKSPACE / "jobs"

VALID_DATASETS = ("secom", "wm811k")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _add_paths(*paths: Path) -> None:
    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _jobs_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _write(job: dict) -> None:
    job["updated_at"] = _now()
    _job_path(job["job_id"]).write_text(json.dumps(job, indent=2, default=str))


def _msg(job: dict, text: str) -> None:
    job["messages"].append({"t": _now(), "text": text})
    _write(job)


# --------------------------------------------------------------------------- #
# ledger read API (used by the dashboard and the MCP job tools)
# --------------------------------------------------------------------------- #
def read_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def list_jobs(limit: int = 50) -> list[dict]:
    """Most-recently-updated jobs first."""
    files = sorted(_jobs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict] = []
    for f in files[:limit]:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out


def active_job_for(target: str) -> dict | None:
    """Return a queued/running job for this target, if one exists (dedupe)."""
    for j in list_jobs():
        if j.get("target") == target and j.get("state") in ("queued", "running"):
            return j
    return None


def _new_job(kind: str, target: str, params: dict) -> dict:
    job_id = f"{kind}_{target}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    job = {
        "job_id": job_id,
        "kind": kind,
        "target": target,
        "params": params,
        "state": "queued",
        "messages": [],
        "result": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "finished_at": None,
    }
    _write(job)
    return job


# --------------------------------------------------------------------------- #
# the actual work (synchronous; each takes the job dict to log progress)
# --------------------------------------------------------------------------- #
def _do_download(job: dict, dataset: str, allow_download: bool) -> dict:
    cfg.ensure_dirs()
    if dataset == "secom":
        _msg(job, "Resolving SECOM dataset (cache-or-fetch via its loader)…")
        registry.ensure_secom()
    elif dataset == "wm811k":
        _msg(job, f"Resolving WM-811K pickle (allow_download={allow_download})…")
        registry.ensure_wm811k(allow_download=allow_download)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use one of {VALID_DATASETS}.")
    status = registry.dataset_status().get(dataset, {})
    cached = status.get("cached")
    _msg(job, "Dataset is cached and ready." if cached else "Loader ran but cache flag is false — check logs.")
    return {"dataset": dataset, "cached": bool(cached), "status": status}


def _do_register(job: dict, dataset: str, file_path: str) -> dict:
    cfg.ensure_dirs()
    if dataset != "wm811k":
        raise ValueError("register currently supports only 'wm811k'.")
    _msg(job, f"Registering existing pickle from {file_path}…")
    path = registry.register_path(file_path)  # copies into the canonical location
    status = registry.dataset_status().get("wm811k", {})
    _msg(job, f"Registered -> {path}")
    return {"dataset": "wm811k", "cached": bool(status.get("cached")),
            "path": str(path), "status": status}


def _do_train(job: dict, domain: str, params: dict) -> dict:
    cfg.ensure_dirs()
    if domain == cfg.DOMAIN_SECOM:
        return _train_secom(job, params)
    if domain == cfg.DOMAIN_YIELD:
        return _train_yield(job, params)
    if domain == cfg.DOMAIN_WAFER_CNN:
        return _train_wafer_cnn(job, params)
    raise KeyError(f"Unknown domain '{domain}'. Use one of {cfg.ALL_DOMAINS}.")


def _train_secom(job: dict, params: dict) -> dict:
    # The SECOM project uses absolute `yield_excursion_fail_prediction.src.*`
    # imports, so the *parent* of that project (PROJECTS_ROOT, the suite root)
    # must be on sys.path — not just the project dir. In Docker the working dir
    # is /app/wafer_mcp_platform, so PROJECTS_ROOT (/app) isn't implicitly there.
    _add_paths(cfg.PROJECTS_ROOT, cfg.SECOM_PROJECT, cfg.SECOM_PROJECT / "src")
    _msg(job, "Ensuring SECOM data is available (download-once)…")
    data = registry.ensure_secom()
    _msg(job, "Data ready. Importing the SECOM training pipeline…")
    try:
        from main import run_pipeline  # type: ignore
    except Exception:
        from yield_excursion_fail_prediction.src.main import run_pipeline  # type: ignore

    estimator = params.get("estimator", "logreg")
    strategy = params.get("strategy", "class_weight")
    recall_target = params.get("recall_target", 0.8)
    compare = params.get("compare", False)
    _msg(job, f"Fitting SECOM model (estimator={estimator}, strategy={strategy}, "
              f"recall_target={recall_target}, compare={compare})…")
    run_pipeline(
        data,
        out_dir=str(cfg.REPORTS_ROOT / "secom"),
        estimator=estimator, strategy=strategy, compare=compare,
        recall_target=recall_target,
        save_model=True, model_out=str(cfg.SECOM_MODEL_PATH),
    )
    _msg(job, f"Saved model artifact -> {cfg.SECOM_MODEL_PATH}")
    return {"status": "trained", "domain": cfg.DOMAIN_SECOM,
            "model_path": str(cfg.SECOM_MODEL_PATH)}


def _train_yield(job: dict, params: dict) -> dict:
    _add_paths(cfg.YIELD_PROJECT)
    allow_download = params.get("allow_download", False)
    _msg(job, f"Ensuring WM-811K pickle (allow_download={allow_download})…")
    pickle_path = registry.ensure_wm811k(allow_download=allow_download)
    _msg(job, "Data ready. Importing the yield-curve modules…")
    from wm811k_yield.config import Config  # type: ignore
    from wm811k_yield.data_loader import WM811KLoader  # type: ignore
    from wm811k_yield.features import WindowExtractor  # type: ignore
    from wm811k_yield.models import build_models  # type: ignore
    from wm811k_yield.trainer import YieldModelTrainer  # type: ignore
    from wm811k_yield.evaluator import ModelEvaluator  # type: ignore
    from wm811k_yield.persistence import ModelStore  # type: ignore

    conf = Config(data_path=pickle_path, output_dir=cfg.REPORTS_ROOT / "yield_curve")
    _msg(job, "Loading wafer maps…")
    loader = WM811KLoader(conf)
    loader.load()
    _msg(job, "Extracting the yield-vs-area curve…")
    curve = WindowExtractor(conf).extract(loader.wafer_maps())
    _msg(job, "Fitting Poisson / Murphy / Negative-Binomial models…")
    models = build_models()
    fits = YieldModelTrainer(conf).fit_all(models, curve)
    _msg(job, "Scoring the leaderboard…")
    board = ModelEvaluator(models).leaderboard(fits, curve)
    best = board.iloc[0]["model_name"] if board is not None and not board.empty else "poisson"
    ModelStore.save(cfg.YIELD_MODEL_PATH, fits, best_model=best, leaderboard=board)
    _msg(job, f"Best model = {best}. Saved bundle -> {cfg.YIELD_MODEL_PATH}")
    return {"status": "trained", "domain": cfg.DOMAIN_YIELD,
            "best_model": best, "model_path": str(cfg.YIELD_MODEL_PATH)}


def _train_wafer_cnn(job: dict, params: dict) -> dict:
    """Train the wafer-map CNN in-process on the shared WM-811K data.

    Reuses the CNN project's OWN preprocessing (resize + /2.0 normalize + class
    persistence) and its torch trainer, so the artifact is preprocessed exactly
    the way the serving adapter expects. Defaults are tuned to finish on CPU; the
    caller can raise epochs / img_size for a stronger model.
    """
    import json
    _add_paths(cfg.CNN_PROJECT, cfg.CNN_PROJECT / "src")
    allow_download = params.get("allow_download", False)
    img_size = int(params.get("img_size", 48))
    epochs = int(params.get("epochs", 8))
    batch = int(params.get("batch", 128))
    drop_none = bool(params.get("drop_none", False))

    _msg(job, f"Ensuring WM-811K pickle (allow_download={allow_download})…")
    pickle_path = registry.ensure_wm811k(allow_download=allow_download)

    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Wafer-CNN training needs PyTorch, which isn't installed. Install a "
            "backend (`pip install torch`) or, for the Keras backend, tensorflow. "
            f"(import error: {exc})"
        ) from exc

    import track1_preprocess as prep  # type: ignore
    import track1_model_torch as cnn  # type: ignore

    processed_dir = cfg.WM811K_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    _msg(job, f"Preprocessing wafer maps -> {processed_dir} (img_size={img_size}, drop_none={drop_none})…")
    meta = prep.run(Path(pickle_path), processed_dir, img_size=img_size, drop_none=drop_none)
    n_classes = len(json.loads((processed_dir / "label_classes.json").read_text()))
    _msg(job, f"Built dataset: {meta['splits']['train']['n']} train / "
              f"{meta['splits']['val']['n']} val / {meta['splits']['test']['n']} test, "
              f"{n_classes} classes. Training CNN ({epochs} epochs)…")

    cfg.CNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = cnn.train(processed_dir, cfg.CNN_MODEL_DIR, epochs=epochs, batch=batch)

    # Record the img_size the model was traced at, so the adapter resizes to match.
    (cfg.CNN_MODEL_DIR / "train_meta.json").write_text(
        json.dumps({"img_size": img_size, "epochs": epochs, "backend": "torch",
                    "n_classes": n_classes}, indent=2))
    final_val_acc = (history.get("val_accuracy") or [None])[-1] if isinstance(history, dict) else None
    _msg(job, f"Saved CNN -> {cfg.CNN_MODEL_DIR / 'best_model.pt'} "
              f"(final val_acc={final_val_acc})")
    return {"status": "trained", "domain": cfg.DOMAIN_WAFER_CNN,
            "model_dir": str(cfg.CNN_MODEL_DIR), "img_size": img_size,
            "n_classes": n_classes, "final_val_accuracy": final_val_acc}


# --------------------------------------------------------------------------- #
# runner: wrap a worker with state transitions + error capture
# --------------------------------------------------------------------------- #
def _run(job: dict, worker, *args) -> None:
    job["state"] = "running"
    job["started_at"] = _now()
    _write(job)
    try:
        result = worker(job, *args)
        job["result"] = result
        # A manual (CNN) result is "completed" but not "trained"; reflect that.
        job["state"] = "completed"
        # If this was a successful training job, drop the cached adapter so the
        # next prediction reloads the freshly-saved artifact from disk.
        if job.get("kind") == "train" and isinstance(result, dict) and \
                result.get("status") == "trained":
            try:
                from mcp_server import adapters as _adapters
                _adapters.reset_adapter(job.get("target"))
            except Exception:
                pass
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["traceback"] = traceback.format_exc()
        job["state"] = "failed"
        _msg(job, f"FAILED: {job['error']}")
    finally:
        job["finished_at"] = _now()
        _write(job)


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def start_download(dataset: str, allow_download: bool = True) -> dict:
    """Launch a download in a background thread. Returns the job immediately."""
    existing = active_job_for(dataset)
    if existing:
        return existing
    job = _new_job("download", dataset, {"allow_download": allow_download})
    threading.Thread(target=_run, args=(job, _do_download, dataset, allow_download),
                     daemon=True).start()
    return job


def start_train(domain: str, **params) -> dict:
    """Launch training in a background thread. Returns the job immediately."""
    existing = active_job_for(domain)
    if existing:
        return existing
    job = _new_job("train", domain, params)
    threading.Thread(target=_run, args=(job, _do_train, domain, params),
                     daemon=True).start()
    return job


def run_download_sync(dataset: str, allow_download: bool = True) -> dict:
    """Download synchronously (used by the MCP tool). Returns the final job."""
    job = _new_job("download", dataset, {"allow_download": allow_download})
    _run(job, _do_download, dataset, allow_download)
    return read_job(job["job_id"]) or job


def start_register(dataset: str, file_path: str) -> dict:
    """Register an already-downloaded pickle in the background (no Kaggle)."""
    existing = active_job_for(dataset)
    if existing:
        return existing
    job = _new_job("register", dataset, {"file_path": str(file_path)})
    threading.Thread(target=_run, args=(job, _do_register, dataset, file_path),
                     daemon=True).start()
    return job


def run_register_sync(dataset: str, file_path: str) -> dict:
    """Register an already-downloaded pickle synchronously (used by the MCP tool)."""
    job = _new_job("register", dataset, {"file_path": str(file_path)})
    _run(job, _do_register, dataset, file_path)
    return read_job(job["job_id"]) or job


def run_train_sync(domain: str, **params) -> dict:
    """Train synchronously (used by the MCP tool). Returns the final job."""
    job = _new_job("train", domain, params)
    _run(job, _do_train, domain, params)
    return read_job(job["job_id"]) or job
