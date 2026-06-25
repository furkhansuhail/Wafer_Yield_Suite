# syntax=docker/dockerfile:1
#
# Wafer & Yield MCP Suite — container image
# ==========================================
# One image that bundles Python, the MCP server, the Streamlit dashboard, and the
# three model projects. The end user runs `docker compose up` and opens a browser
# — no Python, pip, npm, or venv on their side.
#
# Image size note: the base install (platform + light training extras: sklearn,
# scipy, imbalanced-learn, ucimlrepo) stays small because everything ships as
# manylinux wheels — no compiler needed. The CNN's deep-learning backend (torch /
# tensorflow, ~2 GB) is OFF by default; enable it with:  --build-arg WITH_DL=torch
FROM python:3.12-slim

# Keep Python snappy and quiet in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    # Pin the download-once workspace to a stable path we mount a volume on.
    WAFER_PLATFORM_WORKSPACE=/data

# curl is only needed for the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer (cached unless requirements change) ------------------- #
# requirements-train.txt now includes `kaggle`, so the `kaggle` CLI is baked into
# the image and the WM-811K auto-download path works. It still needs credentials
# at runtime — supply KAGGLE_USERNAME/KAGGLE_KEY (see docker-compose.yml) or mount
# a kaggle.json. Without creds, use the "register an existing LSWMD.pkl" path.
COPY wafer_mcp_platform/requirements.txt /app/wafer_mcp_platform/requirements.txt
COPY requirements-train.txt /app/requirements-train.txt
RUN pip install -r /app/wafer_mcp_platform/requirements.txt \
    && pip install -r /app/requirements-train.txt

# Optional deep-learning backend for the wafer CNN. Off by default.
#   docker build --build-arg WITH_DL=torch .
ARG WITH_DL=none
RUN if [ "$WITH_DL" = "torch" ]; then pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu ; \
    elif [ "$WITH_DL" = "tensorflow" ]; then pip install "tensorflow-cpu>=2.16" ; \
    else echo "No DL backend baked in (CNN training disabled). Set --build-arg WITH_DL=torch to enable." ; fi

# --- Application code ------------------------------------------------------- #
COPY . /app

# Persisted, download-once data + trained models live here (mount a volume).
RUN mkdir -p /data
VOLUME ["/data"]

# Streamlit serves the dashboard and calls the platform core IN-PROCESS. The MCP
# server is a separate, optional service for external agents (the `mcp-http`
# profile in docker-compose.yml) — the dashboard no longer spawns it per call.
WORKDIR /app/wafer_mcp_platform
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
