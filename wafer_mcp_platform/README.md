# Wafer & Yield MCP Platform

Ties your three standalone projects together behind **one MCP server** and **one
Streamlit dashboard**, with the dataset downloaded **once** and shared.

```
yield_excursion_fail_prediction/   (SECOM)  ─┐
cnn_error_type_detection/        (wafer CNN) ─┤   wrapped, not rewritten
yield_analytics_poisson_..._binomial/ (yield)─┘
                     │
        wafer_mcp_platform/   ◀── this folder
        ├── shared/         download-once data + model discovery
        ├── mcp_server/     models exposed as MCP tools + a router
        └── dashboard/      Streamlit = MCP client that drives it all
```

## What you asked for, and where it lives

| Request | Implementation |
|---|---|
| Turn the projects into an MCP | `mcp_server/server.py` — FastMCP server; each model family is a tool |
| Models are tools attached to the MCP | `predict_secom`, `classify_wafer_map`, `predict_yield`, `max_die_area_for_yield` |
| MCP decides which model to use | `mcp_server/router.py` + the `route_and_predict` tool |
| Streamlit app with example data → models → results | `dashboard/streamlit_app.py` (it's an MCP *client*) |
| Data downloaded once, reused for model development | `shared/data_registry.py` + `platform_config.py` |
| Streamlit is an MCP dashboard tying it together | dashboard launches the server over stdio and calls its tools |

## The three model families (now MCP tools)

| Domain | Input | Output | Tool(s) |
|---|---|---|---|
| `secom` | a row of ~590 sensor features | fail probability + pass/fail | `predict_secom` |
| `wafer_cnn` | a 2-D wafer map `{0,1,2}` | defect pattern + probabilities | `classify_wafer_map` |
| `yield_curve` | die area (scalar) | expected yield; or inverse | `predict_yield`, `max_die_area_for_yield` |

Plus the glue tools: `list_models`, `verify_models`, `dataset_status`,
`download_dataset`, `register_wm811k`, `job_status`, `list_jobs`,
`get_example_data`, `explain_routing`, `route_and_predict`, `train`.

## Download once, use everywhere

`cnn_error_type_detection` and `yield_analytics_...` both need the same ~1.5 GB
WM-811K pickle. The registry gives it a single canonical home and records it in a
manifest, so the second consumer never re-downloads:

```
_workspace/data/
  wm811k/LSWMD.pkl     ← ONE copy, shared by wafer_cnn + yield_curve
  secom/               ← SECOM cache (one fetch)
  manifest.json        ← the download-once ledger
```

Resolution order in `ensure_wm811k()`: cached → `register_path()` of a file you
already have → Kaggle download (reusing the CNN project's verified downloader).
SECOM goes through its own cached `load_secom()`.

### Where the files actually land on disk

Everything writes under the **workspace**. In Docker that's the `/data` mount,
which `docker-compose.yml` now bind-mounts to a **visible host folder**,
`./workspace` (next to the compose file). So after a download you can see the
real files on your own disk:

```
./workspace/data/
  secom/secom.data  secom_labels.data   ← SECOM (downloaded from UCI, no creds)
  wm811k/LSWMD.pkl                       ← WM-811K (shared by wafer_cnn + yield_curve)
  manifest.json                          ← download-once ledger
```

(Earlier versions used a *named* Docker volume, which lives in Docker's internal
storage and isn't browsable from your project folder — that's the usual reason a
download "succeeds" but you can't find it on disk. The bind mount fixes that.)

The Download buttons in the dashboard write here. If you'd rather put the data on
disk directly — or a download silently did nothing and you want to see why — use
the seeder, which calls the exact same code and prints the on-disk paths:

```bash
# SECOM needs no credentials; this writes ./workspace/data/secom/*.data
docker compose exec dashboard python seed_data.py --secom

# WM-811K: register a pickle you already have…
docker compose exec dashboard python seed_data.py --wm811k-file /data/incoming/LSWMD.pkl
# …or auto-download it (needs KAGGLE_USERNAME/KAGGLE_KEY or a mounted kaggle.json)
docker compose exec dashboard python seed_data.py --kaggle
```

If a download produces nothing, it's almost always one of: the container has no
outbound network (SECOM can't reach UCI), or WM-811K has no Kaggle credentials.
Both now surface a clear reason in the dashboard's **Activity** log and in the
seeder's output.

### WM-811K download failing? (especially in Docker)

The WM-811K Kaggle path needs two things:

1. the **`kaggle` package** — **now baked into the image** (it's a regular line in
   `requirements-train.txt`, which the Dockerfile installs), so the `kaggle` CLI the
   downloader shells out to is already present, and
2. **Kaggle API credentials** — supply these at runtime; they're not in the image.

So a "download 'wm811k' failed" in the dashboard is now almost always a credentials
problem. The job's **Activity** entry shows the real reason (e.g. *"kaggle CLI not
found…"* or an auth error). You have two ways forward:

**A — Register a file you already have (no Kaggle, recommended).** Download
`LSWMD.pkl` once from Kaggle on your host, make it reachable inside the container
(e.g. drop it under the mounted `/data` volume), then use the Pipeline tab's
*"Already have LSWMD.pkl? Register it"* control, or the `register_wm811k` tool, or:

```bash
# the workspace volume is mounted at /data in the container
docker cp ./LSWMD.pkl wafer-yield-dashboard:/data/data/wm811k/LSWMD.pkl
# that IS the canonical path — the platform will just pick it up
```

**B — Enable the Kaggle auto-download (just add credentials).** The CLI is already
installed; you only need to give it credentials. Easiest is environment variables
via a `.env` file next to `docker-compose.yml`:

```bash
cp .env.example .env        # then edit: KAGGLE_USERNAME=... KAGGLE_KEY=...
docker compose up           # compose injects them into the container
```

(Get the values from kaggle.com → Account → "Create New API Token", which downloads
a `kaggle.json` containing `username` and `key`.)

Prefer the token file instead of env vars? Uncomment the mount already present in
`docker-compose.yml`:

```yaml
# docker-compose.yml (dashboard service)
    volumes:
      - workspace:/data
      - ${HOME}/.kaggle:/root/.kaggle:ro      # your kaggle.json, read-only
```

Either way, then tick *"Allow Kaggle download"* before pressing Download.

## Routing: how "the MCP decides"

`route_and_predict(payload)` inspects the payload's shape — no ML needed:

- `wafer_map` / `wafer_maps` (a 2-D grid) → `wafer_cnn`
- `area` or `target_yield` (scalar) → `yield_curve`
- `features` / `rows` (~590 wide) → `secom`
- `domain` key → explicit override

`explain_routing(payload)` returns the full score breakdown and rationale, so the
decision is auditable rather than a black box.

## Optional: natural-language routing (LLM)

The router above is deterministic and needs no API key. As a **separate, opt-in**
extra, the dashboard also has an **Ask (LLM)** tab: you type a request in plain
English, an Anthropic model reads the MCP tool catalogue and picks which tool to
call (and its arguments), and the tool then runs through the normal MCP session —
the model only *decides*, it never touches data directly. Proposed calls are shown
for review (and the arguments are editable) before anything runs.

This is entirely off unless you enable it:

```bash
cd wafer_mcp_platform
cp keys.env.example keys.env          # keys.env is gitignored
# edit keys.env -> ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic                  # optional dep, only for this tab
```

`keys.env` precedence: `ANTHROPIC_API_KEY` there wins, then the same environment
variable. Pin a model with `WAFER_LLM_MODEL` (keys.env or env) if you don't want
the default. If the key or the `anthropic` package is missing, the tab simply
shows how to set it up and everything else keeps working.

## Install & run

```bash
cd wafer_mcp_platform
pip install -r requirements.txt          # MCP server + Streamlit client glue

# 1) the dashboard (it launches the server for you, over stdio)
./run_dashboard.sh        # or: streamlit run dashboard/streamlit_app.py

# 2) or run the server standalone
./run_server.sh           # stdio, for Claude Desktop / agents
./run_server.sh --http    # streamable HTTP on :8000
```

The platform **imports and runs before anything is trained** — example-data tools
fall back to synthetic wafer maps / SECOM rows, so the dashboard is always
demonstrable. Prediction tools return a clear "not trained yet" message until you
train.

### Connect from Claude Desktop / any MCP host

```json
{
  "mcpServers": {
    "wafer-yield": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/wafer_mcp_platform"
    }
  }
}
```

## Training (fills in the models)

Training needs each project's own heavy deps (sklearn/imblearn, torch/tensorflow,
scipy). Install those from the project folders when needed.

```python
# via the MCP tool (fetches data once, fits, saves into _workspace/models/)
train(domain="secom")          # fits the SECOM imblearn pipeline
train(domain="yield_curve")    # fits Poisson/Murphy/NegBinomial on shared WM-811K
train(domain="wafer_cnn")      # heavy — see below
```

You can do the same from the **dashboard**: the **Pipeline** tab is the whole flow
in one place — download a dataset and confirm it's cached, start training, and
watch a live **Activity** feed that shows each job moving through
queued → running → completed (with a toast when it finishes). Training runs in a
background thread and writes its status to a job ledger under
`_workspace/jobs/`, so the page stays responsive and survives reruns. The same
status is available to agents through the `download_dataset`, `job_status`, and
`list_jobs` tools. The **Verify models** tab — backed by `verify_models` —
confirms in one shot that every trained artifact is present *and* loadable, and
the sidebar always shows an "N / 3 models available" summary.

> Heavy training/CNN deps still need to be installed for fits to actually run
> (see `requirements-train.txt`); if they're missing, the job is recorded as
> **failed** with the import error in its Activity log rather than crashing the app.

**Training the CNN.** Because it's a long GPU job, train it from the CLI against
the *shared* pickle and drop the artifacts where the adapter looks:

```bash
PKL=_workspace/data/wm811k/LSWMD.pkl
python ../cnn_error_type_detection/src/track1_preprocess.py --raw-dir $(dirname $PKL) --out-dir _workspace/data/wm811k/processed
python ../cnn_error_type_detection/src/track1_model_torch.py --data-dir _workspace/data/wm811k/processed --out-dir _workspace/models/wafer_cnn
# adapter loads best_model.pt + classes.json from _workspace/models/wafer_cnn/
```

## Layout

```
wafer_mcp_platform/
├── platform_config.py          single source of truth for all paths
├── pipeline.py                 download/train orchestration + background-job ledger
├── llm_routing.py              OPTIONAL natural-language routing (reads keys.env)
├── keys.env.example            template -> copy to keys.env to enable the LLM tab
├── shared/
│   └── data_registry.py        download-once data layer + example data
├── mcp_server/
│   ├── adapters.py             SECOM / wafer-CNN / yield wrappers (uniform predict())
│   ├── router.py               shape-based model selection
│   └── server.py               FastMCP tools + training + entrypoint
├── dashboard/
│   └── streamlit_app.py        MCP client UI (Pipeline / Predict / Router / Ask-LLM / Verify / Raw tools)
├── requirements.txt
├── run_server.sh
└── run_dashboard.sh
```

## Design notes

- **Wrap, don't rewrite.** Adapters import each project's existing `Predictor` /
  `ModelStore` / model classes. Your pipelines are untouched.
- **Graceful degradation.** Missing heavy deps or untrained models never break
  import; tools return actionable messages.
- **One stdio session per dashboard action** keeps Streamlit's rerun model simple.
  Swap to a persistent session for a production UI.
- **MCP SDK pin.** `mcp>=1.27,<2` (v2 stabilises mid-2026). `server.py` also falls
  back to the standalone `fastmcp` package if that's what you have installed.
```
