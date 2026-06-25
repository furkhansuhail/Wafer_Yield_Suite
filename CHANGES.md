# What changed — stability rework

This pass fixed the instability and finished wiring all three models end-to-end.
The three underlying ML projects are unchanged; everything here is in the
`wafer_mcp_platform` glue.

## 1. One in-process core, two thin front-ends (the big fix)

**Before:** the dashboard ran two different code paths against two different
folders. Download/train ran *in-process* and wrote to the workspace
(`WAFER_PLATFORM_WORKSPACE`, e.g. `/data` in Docker), while predict/verify/list
ran by spawning the MCP server over **stdio per call**. The MCP SDK's stdio
client strips the child environment down to `HOME/PATH/TERM`, so the subprocess
never received `WAFER_PLATFORM_WORKSPACE` and fell back to an empty
`_workspace/`. Result: you could download and train successfully, but the
sidebar/Verify tab (running in the blind subprocess) kept reporting **0/3
models** and the predict tabs stayed greyed out forever.

**After:** a single in-process service layer, `platform_api.py`, is the one
source of truth. The Streamlit dashboard imports it directly (no subprocess, no
stdio, no asyncio-in-Streamlit, no per-call interpreter spawn), and the MCP
server (`mcp_server/server.py`) is now a thin wrapper whose every tool delegates
to the same core. Both share one implementation and one workspace, so they can
never disagree about what's trained. This collapses the whole
environment-propagation bug class and makes the UI much faster.

## 2. The wafer CNN now actually trains from the dashboard

**Before:** `train(domain="wafer_cnn")` just returned `{"status":"manual"}` — the
CNN could never be trained or used through the UI.

**After:** `pipeline._train_wafer_cnn` trains the CNN **in-process** on the
shared WM-811K maps, reusing the CNN project's *own* preprocessing and torch
trainer, then saves TorchScript + `classes.json` where the adapter looks. Needs
a backend (`pip install torch` / `--build-arg WITH_DL=torch`); without one it
returns a clear "install a backend" message instead of failing obscurely.

## 3. Fixed a silent CNN train/serve skew

**Before:** training normalized wafer maps to `[0,1]` (`/2.0`), but the serving
adapter fed raw `{0,1,2}` cells in with no normalization — the model saw values
it was never trained on and returned quietly wrong answers.

**After:** the adapter resizes + normalizes through the CNN project's own
`resize_map` + `MAP_MAX_VALUE`, so serve-time preprocessing is identical to
training. It also recovers the training `img_size` so the resize matches.

## 4. Adapters are cached, and the cache resets after training

**Before:** `get_adapter()` returned a fresh instance every call, so every
prediction re-read the model from disk and the in-adapter cache was dead code.

**After:** one cached adapter per domain (load once, reuse), with
`reset_adapter()` called automatically after a successful training job so new
artifacts are picked up immediately.

## 5. Dashboard: a capability tab per model

New per-model tabs — **SECOM**, **Wafer CNN**, **Yield** — each describing what
the model does and offering an interactive demo (score example units; classify a
wafer map with a per-class probability chart; plot the yield curve and solve the
inverse). Each tab greys out with a friendly pointer to the Pipeline tab until
its model is trained and loadable, then unlocks automatically.

## Verified

- Train → route → predict validated in-process for SECOM, yield, and the CNN
  (CNN trains on synthetic multi-class maps and classifies through the router).
- `verify_models` correctly reports availability and degrades gracefully when a
  model is untrained.
- The dashboard loads and all interactions run without error in both a trained
  workspace and an empty one (tested headless via Streamlit's `AppTest`).
- The external MCP path still works: a stdio client sees all 15 tools and gets
  correct results from `verify_models` / `predict_yield`.
