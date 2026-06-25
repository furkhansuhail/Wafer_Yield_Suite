"""
platform_config.py
==================

The single source of truth for *where things live*. Every other module on the
platform imports paths from here instead of hard-coding them. This is what makes
"download the data once and reuse it everywhere" actually hold: there is exactly
one canonical location for each dataset and each trained model, and all three
underlying projects are pointed at it.

Layout (everything under PLATFORM_ROOT/_workspace by default)::

    _workspace/
      data/
        wm811k/LSWMD.pkl        <- ONE copy, shared by CNN + yield-analytics
        secom/                  <- SECOM cache (ucimlrepo / UCI flat files)
        manifest.json           <- records what has been fetched (download-once ledger)
      models/
        secom/model.joblib
        wafer_cnn/best_model.pt | best_model.keras
        yield_curve/models.json
      reports/                  <- shared output dir for EDA / eval artifacts

Override any of these with environment variables (handy for Docker / CI):
    WAFER_PLATFORM_WORKSPACE, WAFER_PROJECTS_ROOT
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
PLATFORM_ROOT = Path(__file__).resolve().parent

# Where the three original project folders live. By default they sit next to
# this platform folder (the layout produced by the delivered zip).
PROJECTS_ROOT = Path(
    os.environ.get("WAFER_PROJECTS_ROOT", PLATFORM_ROOT.parent)
).resolve()

SECOM_PROJECT = PROJECTS_ROOT / "yield_excursion_fail_prediction"
CNN_PROJECT = PROJECTS_ROOT / "cnn_error_type_detection"
YIELD_PROJECT = PROJECTS_ROOT / "yield_analytics_poisson_murphy_binomial"

# --------------------------------------------------------------------------- #
# Workspace (data + models + reports). One place, created on demand.
# --------------------------------------------------------------------------- #
WORKSPACE = Path(
    os.environ.get("WAFER_PLATFORM_WORKSPACE", PLATFORM_ROOT / "_workspace")
).resolve()

DATA_ROOT = WORKSPACE / "data"
MODEL_ROOT = WORKSPACE / "models"
REPORTS_ROOT = WORKSPACE / "reports"

# --- Datasets (canonical, shared) ------------------------------------------ #
WM811K_DIR = DATA_ROOT / "wm811k"
WM811K_PICKLE = WM811K_DIR / "LSWMD.pkl"          # the single shared copy
SECOM_DIR = DATA_ROOT / "secom"
DATA_MANIFEST = DATA_ROOT / "manifest.json"        # download-once ledger

# --- Trained model artifacts ----------------------------------------------- #
SECOM_MODEL_DIR = MODEL_ROOT / "secom"
SECOM_MODEL_PATH = SECOM_MODEL_DIR / "model.joblib"

CNN_MODEL_DIR = MODEL_ROOT / "wafer_cnn"           # holds best_model.pt/.keras
YIELD_MODEL_DIR = MODEL_ROOT / "yield_curve"
YIELD_MODEL_PATH = YIELD_MODEL_DIR / "models.json"

# --------------------------------------------------------------------------- #
# Domains: the three model families the MCP server exposes as tools.
# --------------------------------------------------------------------------- #
DOMAIN_SECOM = "secom"            # tabular sensor row -> pass/fail
DOMAIN_WAFER_CNN = "wafer_cnn"    # 2-D wafer map      -> defect-pattern class
DOMAIN_YIELD = "yield_curve"      # die area (scalar)  -> expected yield

ALL_DOMAINS = (DOMAIN_SECOM, DOMAIN_WAFER_CNN, DOMAIN_YIELD)


def ensure_dirs() -> None:
    """Create the workspace tree. Cheap and idempotent."""
    for d in (
        DATA_ROOT, MODEL_ROOT, REPORTS_ROOT, WM811K_DIR, SECOM_DIR,
        SECOM_MODEL_DIR, CNN_MODEL_DIR, YIELD_MODEL_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def project_paths_summary() -> dict:
    """Used by the dashboard sidebar / `list_models` to show what is wired up."""
    return {
        "platform_root": str(PLATFORM_ROOT),
        "projects_root": str(PROJECTS_ROOT),
        "workspace": str(WORKSPACE),
        "datasets": {
            "wm811k_pickle": str(WM811K_PICKLE),
            "secom_dir": str(SECOM_DIR),
        },
        "models": {
            DOMAIN_SECOM: str(SECOM_MODEL_PATH),
            DOMAIN_WAFER_CNN: str(CNN_MODEL_DIR),
            DOMAIN_YIELD: str(YIELD_MODEL_PATH),
        },
    }
