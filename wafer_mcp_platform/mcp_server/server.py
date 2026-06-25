"""
mcp_server/server.py
====================

The MCP server: a THIN wrapper that exposes the in-process platform core
(`platform_api`) as MCP tools over stdio / HTTP for external agents (Claude
Desktop, other MCP clients). Every tool body is a one-liner delegating to
platform_api, so the server and the Streamlit dashboard share one
implementation and can never drift apart or read a different workspace.

Tools
-----
  list_models()                         catalogue + trained/untrained status
  verify_models()                       check every model is present AND loadable
  dataset_status()                      the download-once ledger
  download_dataset(dataset, ...)        fetch a dataset once + confirm it is cached
  register_wm811k(file_path)            register a pickle you already have (no Kaggle)
  job_status(job_id)                    state/messages/result of a download/train job
  list_jobs(limit)                      recent download/train jobs and their states
  get_example_data(domain, n)           example inputs to send to a model
  predict_secom(features)               tabular row(s) -> P(fail) + decision
  classify_wafer_map(wafer_map)         2-D grid     -> defect pattern
  predict_yield(area, model)            die area     -> expected yield
  max_die_area_for_yield(target, model) target yield -> largest die area
  explain_routing(payload)              why the router would pick a model
  route_and_predict(payload)            MCP decides the model AND runs it
  train(domain, ...)                    fetch-once data + fit, save to workspace

Run
---
  python -m mcp_server.server            # stdio (Claude Desktop / external clients)
  python -m mcp_server.server --http     # streamable HTTP on :8000
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import platform_config as cfg     # noqa: E402
import platform_api as api        # noqa: E402  (the single in-process core)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mcp_server")

# Prefer the official SDK's bundled FastMCP; fall back to the standalone package.
try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    from fastmcp import FastMCP

mcp = FastMCP("wafer-yield-platform")


# --------------------------------------------------------------------------- #
# Discovery / status
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_models() -> dict:
    """List the model families on the platform and whether each is trained."""
    return api.list_models()


@mcp.tool()
def verify_models() -> dict:
    """Verify, in one call, that every model family's trained artifact is present
    AND actually loadable (trained + exists + deserialises). The summary
    (all_available / n_available / missing) lets an agent confirm the suite is
    ready before relying on the prediction/route tools."""
    return api.verify_models()


@mcp.tool()
def dataset_status() -> dict:
    """Show the download-once data ledger: what is cached and who shares it."""
    return api.dataset_status()


@mcp.tool()
def download_dataset(dataset: str, allow_download: bool = False) -> dict:
    """Fetch a dataset once into the shared workspace and confirm it is cached.

    dataset='secom'  : resolve the SECOM cache via its loader (small).
    dataset='wm811k' : resolve the shared LSWMD.pkl (~1.5 GB, needs Kaggle creds;
                       only downloads when allow_download is true)."""
    return api.download_dataset(dataset, allow_download=allow_download)


@mcp.tool()
def register_wm811k(file_path: str) -> dict:
    """Register an LSWMD.pkl you already downloaded — no Kaggle needed. Copies it
    into the shared canonical location so every wafer-map model reuses it."""
    return api.register_wm811k(file_path)


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Current state of a download/train job (state, messages, result, error)."""
    return api.job_status(job_id)


@mcp.tool()
def list_jobs(limit: int = 20) -> dict:
    """List recent download/train jobs (most recent first) with their states."""
    return api.list_jobs(limit=limit)


@mcp.tool()
def get_example_data(domain: str, n: int = 3) -> dict:
    """Return ready-to-send example inputs for a domain
    ('secom' | 'wafer_cnn' | 'yield_curve')."""
    return api.get_example_data(domain, n=n)


# --------------------------------------------------------------------------- #
# Per-model prediction
# --------------------------------------------------------------------------- #
@mcp.tool()
def predict_secom(features: list) -> dict:
    """Predict pass/fail for SECOM unit(s). `features` is one row or a list of
    rows of ~590 numeric sensor values."""
    return api.predict_secom(features)


@mcp.tool()
def classify_wafer_map(wafer_map: list, img_size: int | None = None) -> dict:
    """Classify the defect pattern of a single wafer map: a 2-D grid of
    {0:no-die, 1:good, 2:bad}. img_size defaults to the model's training size."""
    return api.classify_wafer_map(wafer_map, img_size=img_size)


@mcp.tool()
def predict_yield(area: Any, model: str | None = None) -> dict:
    """Expected defect-free yield for a die of the given area (unit-die areas).
    `area` may be a scalar or a list. `model` optionally pins poisson/murphy/
    negative_binomial; default uses the best-fitting model."""
    return api.predict_yield(area, model=model)


@mcp.tool()
def max_die_area_for_yield(target_yield: float, model: str | None = None) -> dict:
    """Largest die area (unit-die areas) whose predicted yield still meets the
    target (0<target<1)."""
    return api.max_die_area_for_yield(target_yield, model=model)


# --------------------------------------------------------------------------- #
# Router tools — MCP decides which model
# --------------------------------------------------------------------------- #
@mcp.tool()
def explain_routing(payload: dict) -> dict:
    """Explain which model the router would pick for `payload`, and why, without
    running inference."""
    return api.explain_routing(payload)


@mcp.tool()
def route_and_predict(payload: dict) -> dict:
    """Inspect `payload`, choose the appropriate model, run it, and return both
    the routing decision and the prediction.

    payload keys the router understands:
      • {"wafer_map": [[...]]}                  -> wafer_cnn
      • {"area": A} or {"target_yield": Y}      -> yield_curve
      • {"features": [[...590...]]}             -> secom
      • {"domain": "<name>", ...}               -> explicit override
    """
    return api.route_and_predict(payload)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@mcp.tool()
def train(domain: str, strategy: str = "class_weight", estimator: str = "logreg",
         recall_target: float | None = 0.8, compare: bool = False,
         allow_download: bool = False, img_size: int = 48, epochs: int = 8,
         batch: int = 128, drop_none: bool = False) -> dict:
    """Train/fit the model for a domain using the shared (download-once) data,
    saving the artifact into the workspace so the prediction tools can load it.

    domain='secom'       : fit the imblearn pipeline on cached SECOM
                           (strategy, estimator, recall_target, compare).
    domain='yield_curve' : fit Poisson/Murphy/NB on the shared WM-811K pickle
                           (allow_download lets it Kaggle-fetch if not cached).
    domain='wafer_cnn'   : train the CNN on the shared WM-811K pickle
                           (allow_download, img_size, epochs, batch, drop_none).
    """
    if domain == cfg.DOMAIN_SECOM:
        return api.train(cfg.DOMAIN_SECOM, strategy=strategy, estimator=estimator,
                         recall_target=recall_target, compare=compare)
    if domain == cfg.DOMAIN_YIELD:
        return api.train(cfg.DOMAIN_YIELD, allow_download=allow_download)
    if domain == cfg.DOMAIN_WAFER_CNN:
        return api.train(cfg.DOMAIN_WAFER_CNN, allow_download=allow_download,
                        img_size=img_size, epochs=epochs, batch=batch,
                        drop_none=drop_none)
    raise KeyError(f"Unknown domain '{domain}'.")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    cfg.ensure_dirs()
    if "--http" in argv:
        logger.info("Starting MCP server on streamable HTTP :8000")
        mcp.run(transport="streamable-http")
    else:
        logger.info("Starting MCP server on stdio")
        mcp.run()


if __name__ == "__main__":
    main()
