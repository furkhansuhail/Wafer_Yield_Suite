"""
platform_api.py
===============

The single in-process service layer — the one source of truth for every
operation the platform performs. Both front-ends call THIS:

  • the Streamlit dashboard imports it directly (no subprocess, no stdio, no
    environment-propagation surprises, no per-call interpreter spawn), and
  • the MCP server (`mcp_server/server.py`) wraps each function as a tool, so
    external agents over stdio / HTTP get the exact same behaviour.

Because there is exactly one implementation, the dashboard and the MCP tools can
never drift apart, and they always read/write the same workspace. (The previous
design had the dashboard spawn the MCP server over stdio for predictions while
running downloads/training in-process; the stdio child silently lost
WAFER_PLATFORM_WORKSPACE, so it looked at an empty workspace and reported every
model as untrained even right after a successful train. Collapsing to one
in-process core removes that whole class of bug.)

Everything here returns plain JSON-able dicts/lists.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import platform_config as cfg            # noqa: E402
from shared import data_registry as registry  # noqa: E402
from mcp_server import adapters, router  # noqa: E402
import pipeline                          # noqa: E402


# --------------------------------------------------------------------------- #
# Discovery / status
# --------------------------------------------------------------------------- #
def list_models() -> dict:
    """Model families on the platform and whether each is trained."""
    cfg.ensure_dirs()
    return {
        "paths": cfg.project_paths_summary(),
        "models": {d: adapters.get_adapter(d).metadata() for d in cfg.ALL_DOMAINS},
    }


def verify_models() -> dict:
    """Verify every model family's artifact is present AND actually loadable."""
    cfg.ensure_dirs()
    expected = {
        cfg.DOMAIN_SECOM: cfg.SECOM_MODEL_PATH,
        cfg.DOMAIN_WAFER_CNN: cfg.CNN_MODEL_DIR,
        cfg.DOMAIN_YIELD: cfg.YIELD_MODEL_PATH,
    }
    report: dict[str, dict] = {}
    for domain in cfg.ALL_DOMAINS:
        adapter = adapters.get_adapter(domain)
        trained = bool(adapter.is_trained())

        if domain == cfg.DOMAIN_WAFER_CNN:
            model_file = adapter._model_file()
            artifact = str(model_file) if model_file else str(expected[domain])
            exists = model_file is not None
        else:
            path = expected[domain]
            artifact = str(path)
            exists = path.exists()

        loadable: bool | None = None
        error: str | None = None
        if trained:
            try:
                adapter._ensure_loaded()
                loadable = True
            except Exception as exc:
                loadable = False
                error = str(exc)

        report[domain] = {
            "trained": trained,
            "artifact": artifact,
            "exists": exists,
            "loadable": loadable,
            "available": bool(trained and exists and loadable),
            "error": error,
        }

    n_total = len(cfg.ALL_DOMAINS)
    n_available = sum(1 for d in report.values() if d["available"])
    return {
        "all_available": n_available == n_total,
        "n_available": n_available,
        "n_total": n_total,
        "missing": [name for name, d in report.items() if not d["available"]],
        "models": report,
    }


def dataset_status() -> dict:
    """The download-once data ledger: what is cached and who shares it."""
    return registry.dataset_status()


def get_example_data(domain: str, n: int = 3) -> dict:
    """Ready-to-send example inputs for a domain ('secom'|'wafer_cnn'|'yield_curve')."""
    if domain == cfg.DOMAIN_SECOM:
        df = registry.secom_examples(n)
        return {"domain": domain, "payload_key": "features",
                "examples": df.values.tolist(), "n_features": int(df.shape[1])}
    if domain == cfg.DOMAIN_WAFER_CNN:
        return {"domain": domain, "payload_key": "wafer_maps",
                "examples": registry.wafer_examples(n)}
    if domain == cfg.DOMAIN_YIELD:
        return {"domain": domain, "payload_key": "area",
                "examples": registry.yield_area_examples()}
    raise KeyError(f"Unknown domain '{domain}'.")


# --------------------------------------------------------------------------- #
# Per-model prediction
# --------------------------------------------------------------------------- #
def predict_secom(features: list) -> dict:
    """One row or a list of rows of ~590 sensor values -> P(fail) + decision."""
    return adapters.get_adapter(cfg.DOMAIN_SECOM).predict({"features": features})


def classify_wafer_map(wafer_map: list, img_size: int | None = None) -> dict:
    """A 2-D grid of {0:no-die,1:good,2:bad} -> defect pattern + probabilities."""
    payload: dict = {"wafer_map": wafer_map}
    if img_size is not None:
        payload["img_size"] = img_size
    return adapters.get_adapter(cfg.DOMAIN_WAFER_CNN).predict(payload)


def predict_yield(area: Any, model: str | None = None) -> dict:
    """Expected defect-free yield for a die of the given area (scalar or list)."""
    payload: dict = {"area": area}
    if model:
        payload["model"] = model
    return adapters.get_adapter(cfg.DOMAIN_YIELD).predict(payload)


def max_die_area_for_yield(target_yield: float, model: str | None = None) -> dict:
    """Largest die area whose predicted yield still meets the target (0<t<1)."""
    payload: dict = {"target_yield": target_yield}
    if model:
        payload["model"] = model
    return adapters.get_adapter(cfg.DOMAIN_YIELD).predict(payload)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def explain_routing(payload: dict) -> dict:
    """Which model the router would pick for `payload`, and why (no inference)."""
    return router.explain(payload)


def route_and_predict(payload: dict) -> dict:
    """Inspect `payload`, choose the model, run it, return decision + prediction."""
    domain, rationale = router.route(payload)
    result = adapters.get_adapter(domain).predict(payload)
    return {"routing": {"chosen_domain": domain, "rationale": rationale},
            "prediction": result}


# --------------------------------------------------------------------------- #
# Data + training (download-once ledger lives in pipeline)
# --------------------------------------------------------------------------- #
def download_dataset(dataset: str, allow_download: bool = False) -> dict:
    return pipeline.run_download_sync(dataset, allow_download=allow_download)


def register_wm811k(file_path: str) -> dict:
    return pipeline.run_register_sync("wm811k", file_path)


def job_status(job_id: str) -> dict:
    job = pipeline.read_job(job_id)
    if job is None:
        raise KeyError(f"No job '{job_id}'.")
    return job


def list_jobs(limit: int = 20) -> dict:
    return {"jobs": pipeline.list_jobs(limit=limit)}


def train(domain: str, **params) -> dict:
    """Train a model synchronously and return the unwrapped result.

    domain='secom'       -> imblearn pipeline on cached SECOM
                            (strategy, estimator, recall_target, compare)
    domain='yield_curve' -> Poisson/Murphy/NB on the shared WM-811K pickle
                            (allow_download)
    domain='wafer_cnn'   -> CNN on the shared WM-811K pickle
                            (allow_download, img_size, epochs, batch, drop_none)
    """
    if domain not in cfg.ALL_DOMAINS:
        raise KeyError(f"Unknown domain '{domain}'. Use one of {cfg.ALL_DOMAINS}.")
    job = pipeline.run_train_sync(domain, **params)
    if job.get("state") == "completed":
        return job.get("result") or {"status": "completed"}
    raise RuntimeError(job.get("error") or "training failed")


# ---- Background variants (used by the dashboard for live progress) --------- #
def start_download(dataset: str, allow_download: bool = True) -> dict:
    return pipeline.start_download(dataset, allow_download=allow_download)


def start_register(dataset: str, file_path: str) -> dict:
    return pipeline.start_register(dataset, file_path)


def start_train(domain: str, **params) -> dict:
    return pipeline.start_train(domain, **params)


def active_job_for(target: str) -> dict | None:
    return pipeline.active_job_for(target)
