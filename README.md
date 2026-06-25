# Wafer & Yield MCP Suite

Three semiconductor ML pipelines unified behind one **MCP server** and one
**Streamlit dashboard**, dataset downloaded **once** and shared. Pick the launch
path that matches your effort budget — Docker needs nothing but Docker installed.

```
wafer_yield_mcp_suite/
├── Dockerfile  docker-compose.yml      ◀── zero-effort run (recommended)
├── package.json  scripts/              npm wrapper (builds a Python venv)
├── wafer_mcp_platform/                 the glue: MCP server + Streamlit dashboard
├── yield_excursion_fail_prediction/    SECOM pass/fail pipeline      (unchanged)
├── cnn_error_type_detection/           WM-811K wafer-map CNN         (unchanged)
└── yield_analytics_poisson_murphy_binomial/  WM-811K yield models    (unchanged)
```

## The three models — what they answer and how they're trained

Each project answers a **different question** about the same end goal (better
yield), and the MCP router sends each input to the model whose question it
matches. Quick map:

| You're asking… | Model (domain) | Input → Output |
|---|---|---|
| *Will this unit pass or fail final test?* | SECOM (`secom`) | ~590 sensor readings → fail probability + pass/fail |
| *What defect pattern is on this wafer?* | Wafer CNN (`wafer_cnn`) | a wafer-map grid → pattern class + probabilities |
| *What yield will a die of this size get?* | Yield curve (`yield_curve`) | a die area → expected yield (and the inverse) |

---

### 1. SECOM — yield excursion / fail prediction (`secom`)

**Question it answers.** Given the ~590 process-sensor measurements taken while a
production unit was made, *will that unit fail in-house electrical testing?* This
is per-unit, binary (pass/fail), and the failing unit is the rare, costly event
you want to catch early.

**Data.** UCI SECOM: 1,567 units × 590 sensor features, with a ~14:1 pass:fail
imbalance, heavy missingness, and noisy/dead columns. Labels are remapped so the
rare failure is the positive class (`0 = pass`, `1 = fail`).

**How it's trained.**
- A **leak-free** scikit-learn / imbalanced-learn pipeline: dropping high-missing
  and dead columns, imputation, scaling, feature selection, **and** any
  resampling all happen *inside* each cross-validation fold, fit on training data
  only — the test split is never touched during fitting.
- **Imbalance-aware** throughout. Accuracy is deliberately never the headline;
  selection and reporting use PR-AUC, failure-recall, F1, G-mean, and MCC.
  Resampling options include `class_weight`, `smote`, `smote_enn`, `smote_tomek`;
  estimators include logistic regression, balanced random forest, hist gradient
  boosting, and XGBoost.
- The **decision threshold is tuned, not 0.5** — e.g. lowered to catch ≥80% of
  failures — and that threshold is saved *with* the pipeline, because the same
  model at a different threshold is a different deployed system.

**Output.** Per unit: a failure probability and a pass/fail decision at the tuned
threshold. Saved as a `.joblib` bundle (pipeline + threshold + metadata).

---

### 2. Wafer CNN — defect-pattern classification (`wafer_cnn`)

**Question it answers.** A wafer map is a grid where each die is pass/fail; the
*spatial pattern* of failures points to a systemic process problem. This model
asks: *which named defect pattern is this wafer showing?* — one of **Center,
Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch,** or **none**.

**Data.** WM-811K wafer maps (~811k maps). Cells are categorical
`{0 = no die, 1 = good, 2 = bad}`. Severe class imbalance — `none` dominates,
`Near-full` is rare.

**How it's trained.**
- A **convolutional neural network** in two interchangeable backends — **Keras**
  and **PyTorch** — sharing the same preprocessed data and the same evaluator.
- Maps are resized with **nearest-neighbour only** (cells are categorical;
  interpolation would invent meaningless fractional die states). Data is stored
  framework-neutral as `(N, H, W)`; Keras adds a trailing channel, PyTorch a
  leading one.
- **Imbalance handling:** inverse-frequency class weights (normalized to mean 1)
  during training; evaluation reports **macro-F1 and per-class F1**, not overall
  accuracy (which is misleading when one class is ~80% of the data).
- The PyTorch model is saved as **TorchScript** so the evaluator can load it
  backend-agnostically.

**Output.** The predicted pattern plus per-class probabilities.

---

### 3. Yield curve — analytical area→yield models (`yield_curve`)

**Question it answers.** *What defect-free yield should a die of area A achieve,
and how large a die can I make while still hitting a target yield?* It also
returns interpretable process analytics: defect density `D0` and the clustering
parameter `alpha`. This is a **population statistic** (a yield rate), not a
per-die prediction.

**Data.** Derived from the same WM-811K wafer maps via the **Stapper window
method**: for a `w × w` window of dies (area `A = w²`), the window "yields" only
if every die inside is good; pooling windows over many wafers gives an empirical
yield-vs-area curve `Y(A)`.

**How it's trained.**
- Three classical models are fit to that curve, differing only in how much defect
  **clustering** they allow:

  | Model | Formula | Clustering |
  |---|---|---|
  | Poisson | `Y = exp(-D0·A)` | none (uniform) |
  | Murphy | `Y = [(1−exp(−D0·A)) / (D0·A)]²` | mild |
  | Negative Binomial | `Y = (1 + D0·A/alpha)^(−alpha)` | free (`alpha`) |

  They're **nested**: as `alpha → ∞`, Negative Binomial collapses to Poisson.
- Fitting is **weighted least squares** on the empirical curve (weights from the
  binomial standard error of each area point). The three fits are ranked by
  RMSE / R² / AIC / BIC, and the best-fitting model is selected — real wafers
  cluster, so Negative Binomial (small `alpha`) usually wins.
- The fitted bundle is just model name + a few floats, saved as human-readable
  **JSON** (you can eyeball `D0` / `alpha` directly).

**Output.** Forward: yield for one or more areas. Inverse: the largest die area
that still meets a target yield.

---

**How it's wired together.** There is one in-process **platform core**
(`platform_api`) that does all the real work — list/verify, predict, route, and
train. Two thin front-ends sit on top of it: the **Streamlit dashboard** (which
calls the core directly, in-process) and the **MCP server** (which wraps each
core function as a tool over stdio/HTTP for external agents like Claude Desktop).
Because both share one implementation and one workspace, they can never disagree
about which models are trained. The core's `route_and_predict` reads the *shape*
of your input — a 2-D grid → wafer CNN, a scalar area / target → yield curve, a
~590-wide row → SECOM — and runs the right one; `explain_routing` shows the
decision. Routing is deterministic and needs no API key; the dashboard also has
an **optional** "Ask (LLM)" tab that lets an Anthropic model pick the tool from a
plain-English request (add a key to `wafer_mcp_platform/keys.env`). All three
models — including the wafer CNN — train from the dashboard's **Pipeline** tab.
Per-project deep dives live in each folder's own `README.md`.

## Option A — Docker (recommended, nothing to install but Docker)

```bash
cd wafer_yield_mcp_suite
docker compose up            # first run builds the image, then starts everything
```

Open **http://localhost:8501**. No Python, pip, npm, or venv on your side — the
image bundles all of it. Data and trained models persist in the `workspace`
volume across restarts.

For agents that connect to the MCP server directly over HTTP:

```bash
docker compose --profile agents up mcp-http      # MCP server on :8000
```

Bake in the CNN deep-learning backend (large image) when you need it:

```bash
docker compose build --build-arg WITH_DL=torch
```

### Why Docker and not Kubernetes?

This is a single dashboard with an MCP subprocess — one container is the right
unit, and `docker compose up` is the least effort possible. Kubernetes is for
scaling to many users, high availability, and multi-node orchestration; it would
*add* setup without changing what the user sees. Move to k8s only if you later
need concurrent multi-user serving or autoscaling — the same image drops into a
Deployment + Service then.

## Option B — npm (builds a local Python venv for you)

```bash
cd wafer_yield_mcp_suite
npm install        # creates ./.venv and pip-installs MCP + Streamlit
npm start          # launches the dashboard
```

Needs Node 18+ and Python 3.10+ on PATH.

## Option C — plain Python

```bash
cd wafer_mcp_platform
pip install -r requirements.txt
streamlit run dashboard/streamlit_app.py
```

## Training the models

The dashboard runs immediately with synthetic example data; train to get real
predictions.

**In Docker** (image already has the light training deps):

```bash
docker compose exec dashboard python -c \
  "import sys; sys.path.insert(0,'.'); from mcp_server.server import _train_yield; print(_train_yield())"
```

or use the dashboard's **Pipeline** tab → download the data (and confirm it's
cached) → pick a domain → start training → watch the live **Activity** feed
report when it's done. **All three models, including the wafer CNN, train from
here.** The CNN trains in-process on the shared WM-811K maps using the project's
own preprocessing, so what it's served at inference matches what it was trained
on. CNN training needs a backend (`pip install torch`, or build the image with
`--build-arg WITH_DL=torch`); without one, the other two models still work and
CNN training returns a clear "install a backend" message. After training, the
model's own tab and the **Verify** tab unlock automatically.

**With npm:**

```bash
npm run setup:train      # installs sklearn / imbalanced-learn / scipy / ucimlrepo
npm run train:yield
npm run train:secom
```

The wafer CNN is a heavy GPU job (needs the ~1.5 GB WM-811K dataset and a
torch/tensorflow backend) — see `wafer_mcp_platform/README.md`.

## npm script reference

| Command | Does |
|---|---|
| `npm install` / `npm run setup` | create venv + install MCP server & Streamlit |
| `npm run setup:train` | also install training extras |
| `npm start` / `npm run dashboard` | launch the dashboard (+ MCP server) |
| `npm run server` / `server:http` | run the MCP server alone (stdio / HTTP) |
| `npm run train:secom` / `train:yield` | fit a model |
| `npm run clean` | delete `.venv` and the workspace |

## Verified

- Docker config: installs from manylinux wheels (no compiler), pins the
  download-once workspace to the mounted `/data` volume (verified), serves
  Streamlit on `0.0.0.0:8501` with a `/_stcore/health` healthcheck.
- `npm install` → `npm start` run end-to-end on a clean machine: venv built,
  Streamlit + MCP installed, dashboard served HTTP 200.
- The train → route → predict loop validated through the MCP tools.

> Building a Python venv (Options B/C) needs a filesystem that allows symlinks;
> Docker (Option A) sidesteps that entirely.
