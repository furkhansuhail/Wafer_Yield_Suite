"""
dashboard/streamlit_app.py
==========================

The Streamlit front-end. It talks to the platform IN-PROCESS through
`platform_api` — the same core the MCP server wraps — so the dashboard and any
external MCP agent share one implementation and one workspace. (Earlier this
dashboard spawned the MCP server over stdio per call; that child lost the
workspace env var and reported trained models as missing. In-process removes
that entire failure mode and is much faster.)

Layout
------
  Sidebar      : data + model status at a glance
  Pipeline     : download data + train each model, with live background progress
  SECOM / Wafer CNN / Yield : one capability tab per model (greyed until trained)
  Router       : rule-based "let the platform pick the model"
  Ask (LLM)    : optional natural-language tool routing (needs an API key)
  Verify       : full trained/loadable report
  Raw tools    : call any tool by name with a JSON payload
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PLATFORM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_ROOT))

import platform_config as cfg   # noqa: E402
import platform_api as api      # noqa: E402  (the single in-process core)
import pipeline                 # noqa: E402  (background jobs + ledger)

TRAIN_TIMEOUT_SECONDS = 30 * 60

DOMAIN_LABELS = {
    cfg.DOMAIN_SECOM: "SECOM — pass/fail",
    cfg.DOMAIN_WAFER_CNN: "Wafer CNN — defect pattern",
    cfg.DOMAIN_YIELD: "Yield curve — area↔yield",
}


# --------------------------------------------------------------------------- #
# In-process tool dispatch (keeps the Raw-tools and LLM tabs generic)
# --------------------------------------------------------------------------- #
def call_tool(tool, args=None, timeout=None):
    args = dict(args or {})
    if tool is None:
        return _tool_catalogue()
    dispatch = {
        "list_models": lambda: api.list_models(),
        "verify_models": lambda: api.verify_models(),
        "dataset_status": lambda: api.dataset_status(),
        "get_example_data": lambda: api.get_example_data(args["domain"], args.get("n", 3)),
        "predict_secom": lambda: api.predict_secom(args["features"]),
        "classify_wafer_map": lambda: api.classify_wafer_map(args["wafer_map"], args.get("img_size")),
        "predict_yield": lambda: api.predict_yield(args["area"], args.get("model")),
        "max_die_area_for_yield": lambda: api.max_die_area_for_yield(args["target_yield"], args.get("model")),
        "explain_routing": lambda: api.explain_routing(args["payload"]),
        "route_and_predict": lambda: api.route_and_predict(args["payload"]),
        "download_dataset": lambda: api.download_dataset(args["dataset"], args.get("allow_download", False)),
        "register_wm811k": lambda: api.register_wm811k(args["file_path"]),
        "job_status": lambda: api.job_status(args["job_id"]),
        "list_jobs": lambda: api.list_jobs(args.get("limit", 20)),
        "train": lambda: api.train(args.pop("domain"), **args),
    }
    fn = dispatch.get(tool)
    if fn is None:
        raise KeyError(f"Unknown tool '{tool}'.")
    return fn()


@st.cache_data(show_spinner=False)
def _tool_catalogue():
    """The MCP tool catalogue (names + descriptions + schemas), read once
    in-process from the server module — used by the LLM and Raw-tools tabs."""
    import asyncio
    from mcp_server import server as mcp_server
    tools = asyncio.run(mcp_server.mcp.list_tools())
    return [{"name": t.name, "description": t.description or "",
             "input_schema": getattr(t, "inputSchema", None)} for t in tools]


@st.cache_data(show_spinner=False, ttl=5)
def _verify_cached():
    return api.verify_models()


def _availability():
    """Map domain -> availability dict from verify_models (cached briefly)."""
    try:
        return _verify_cached()["models"]
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Page + sidebar
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Wafer / Yield MCP Dashboard", layout="wide")
st.title("Wafer & Yield MCP Dashboard")
st.caption("Streamlit front-end → in-process platform core → SECOM / wafer-CNN / yield-curve models")

with st.sidebar:
    st.header("Status")
    if st.button("Refresh status"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Data (download once)")
    try:
        ds = api.dataset_status()
        for name, info in ds.items():
            if name == "manifest":
                continue
            flag = "✅ cached" if info.get("cached") else "⬇️ not fetched"
            shared = ", ".join(info.get("shared_by", []))
            st.markdown(f"**{name}** — {flag}  \nshared by: {shared or '—'}")
    except Exception as exc:
        st.warning(f"dataset_status failed: {exc}")

    st.subheader("Models")
    try:
        report = _verify_cached()
        st.metric(
            "Models available",
            f"{report['n_available']} / {report['n_total']}",
            delta=("all available" if report["all_available"]
                   else f"missing: {', '.join(report['missing'])}"),
            delta_color=("normal" if report["all_available"] else "inverse"),
        )
        for dom, md in report["models"].items():
            if md["available"]:
                icon = "🟢 available"
            elif md["trained"] and not md["loadable"]:
                icon = "🟠 present, not loadable"
            elif md["trained"]:
                icon = "🟡 trained"
            else:
                icon = "⚪ untrained"
            st.markdown(f"**{dom}** — {icon}")
    except Exception as exc:
        st.warning(f"verify_models failed: {exc}")


# --------------------------------------------------------------------------- #
# Small helpers for the model tabs
# --------------------------------------------------------------------------- #
def _unavailable_notice(domain: str, md: dict | None):
    """Render a friendly 'not ready yet' block for a model tab. Returns True if
    the model is NOT available (caller should stop rendering the interactive UI)."""
    if md and md.get("available"):
        return False
    st.info(
        f"**{DOMAIN_LABELS.get(domain, domain)}** isn't ready yet. "
        "Head to the **Pipeline** tab to download the data and train it — this "
        "tab unlocks automatically once the model is trained and loadable."
    )
    if md and md.get("trained") and not md.get("loadable"):
        st.error(f"The artifact is present but failed to load: {md.get('error')}")
    return True


def _is_failure(res) -> bool:
    """True if a tool returned the structured failure envelope (ok == False)."""
    return isinstance(res, dict) and res.get("ok") is False and "error_code" in res


def render_failure(res) -> bool:
    """Render the structured failure envelope produced by error_advisor.advise()
    (error_code / title / explanation / how_to_fix). Returns True if `res` was a
    failure (so the caller should stop before rendering success output)."""
    if not _is_failure(res):
        return False
    st.error(f"**{res.get('title', 'Operation failed')}**")
    explanation = res.get("explanation")
    if explanation:
        # The LLM-written (or built-in) plain-language explanation.
        st.write(explanation)
        if res.get("llm_used"):
            st.caption("Explanation written by the LLM (API key detected).")
        elif res.get("llm_error"):
            st.caption(f"LLM explanation unavailable ({res['llm_error']}); showing built-in guidance.")
    fixes = res.get("how_to_fix") or []
    if fixes:
        st.markdown("**How to fix it:**")
        for i, step in enumerate(fixes, 1):
            st.markdown(f"{i}. {step}")
    with st.expander("Technical details"):
        st.json({k: v for k, v in res.items() if k != "explanation"})
    return True


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
(tab_pipeline, tab_secom, tab_cnn, tab_yield,
 tab_router, tab_llm, tab_verify, tab_tools) = st.tabs(
    ["Pipeline", "SECOM", "Wafer CNN", "Yield",
     "Router (rules)", "Ask (LLM)", "Verify", "Raw tools"]
)

avail = _availability()


# ======================  TAB: Pipeline  ==================================== #
with tab_pipeline:
    st.subheader("Pipeline control")
    st.caption(
        "Download data once, train models, and watch live progress. Heavy work "
        "runs in a background thread and reports status below, so the page stays "
        "responsive and tells you when each step finishes."
    )

    # ---------------- 1) Data ---------------- #
    st.markdown("### 1 · Data")
    try:
        ds = api.dataset_status()
    except Exception as exc:
        ds = {}
        st.warning(f"dataset_status failed: {exc}")

    d1, d2 = st.columns(2)
    for col, name, label, big in (
        (d1, "secom", "SECOM (tabular)", False),
        (d2, "wm811k", "WM-811K (wafer maps)", True),
    ):
        with col:
            info = ds.get(name, {}) if isinstance(ds, dict) else {}
            st.markdown(f"**{label}**")
            if info.get("cached"):
                mb = (info.get("bytes") or 0) / 1e6
                st.success(f"✅ cached{f' ({mb:.0f} MB)' if mb else ''}")
            else:
                st.info("⬇️ not fetched yet")
            allow_dl = True
            if big:
                allow_dl = st.checkbox(
                    "Allow Kaggle download (~1.5 GB, needs kaggle.json)",
                    value=False, key=f"dl_allow_{name}")
            if st.button(f"Download {name}", key=f"dl_btn_{name}"):
                job = api.start_download(name, allow_download=allow_dl)
                st.session_state["watch_job"] = job["job_id"]
                st.toast(f"Download started for {name}", icon="⬇️")
                st.rerun()
            if big:
                with st.expander("Already have LSWMD.pkl? Register it (no Kaggle)"):
                    pkl_path = st.text_input("Path to LSWMD.pkl", key="wm_register_path",
                                             placeholder="/data/incoming/LSWMD.pkl")
                    if st.button("Register file", key="wm_register_btn"):
                        if not pkl_path.strip():
                            st.warning("Enter a path first.")
                        else:
                            job = api.start_register("wm811k", pkl_path.strip())
                            st.session_state["watch_job"] = job["job_id"]
                            st.toast("Registering WM-811K pickle…", icon="📦")
                            st.rerun()

    st.divider()

    # ---------------- 2) Train ---------------- #
    st.markdown("### 2 · Train")
    train_domain = st.selectbox("Domain to train", list(cfg.ALL_DOMAINS),
                                format_func=lambda d: DOMAIN_LABELS.get(d, d),
                                key="train_domain")

    if train_domain == cfg.DOMAIN_SECOM:
        c1, c2 = st.columns(2)
        with c1:
            estimator = st.selectbox("Estimator",
                                     ["logreg", "balanced_rf", "hist_gb", "xgboost"],
                                     key="secom_estimator")
            strategy = st.selectbox("Imbalance strategy",
                                    ["class_weight", "smote", "smote_enn", "undersample", "none"],
                                    key="secom_strategy")
        with c2:
            recall_target = st.slider("Recall target", 0.0, 1.0, 0.8, 0.05, key="secom_recall")
            compare = st.checkbox("Compare estimators/strategies first", value=False, key="secom_compare")
        train_params = {"estimator": estimator, "strategy": strategy,
                        "recall_target": recall_target, "compare": compare}
    elif train_domain == cfg.DOMAIN_YIELD:
        allow_download = st.checkbox("Allow WM-811K download if not cached",
                                     value=False, key="yield_dl")
        train_params = {"allow_download": allow_download}
    else:  # wafer_cnn — now trains in-process
        st.caption("Trains a CNN on the shared WM-811K maps, in-process. Defaults "
                   "are tuned to finish on CPU; raise epochs / image size for a "
                   "stronger model (needs PyTorch installed).")
        c1, c2, c3 = st.columns(3)
        with c1:
            cnn_img = st.select_slider("Image size", [32, 40, 48, 64], value=48, key="cnn_img")
        with c2:
            cnn_epochs = st.slider("Epochs", 2, 40, 8, key="cnn_epochs")
        with c3:
            cnn_drop_none = st.checkbox("Drop 'none' class", value=False, key="cnn_dropnone")
        cnn_allow = st.checkbox("Allow WM-811K download if not cached", value=False, key="cnn_dl")
        train_params = {"img_size": cnn_img, "epochs": cnn_epochs,
                        "drop_none": cnn_drop_none, "allow_download": cnn_allow}

    running = api.active_job_for(train_domain)
    if running:
        st.warning(f"A {running['kind']} job for '{train_domain}' is already "
                   f"{running['state']} (job {running['job_id']}).")
    if st.button("🚀 Start training", key="do_train", type="primary", disabled=bool(running)):
        job = api.start_train(train_domain, **train_params)
        st.session_state["watch_job"] = job["job_id"]
        st.toast(f"Training started for {train_domain}", icon="🚀")
        st.rerun()

    st.divider()

    # ---------------- 3) Activity ---------------- #
    st.markdown("### 3 · Activity")
    top = st.columns([1, 1, 2])
    with top[0]:
        if st.button("🔄 Refresh"):
            st.rerun()
    with top[1]:
        auto = st.checkbox("Auto-refresh", value=True, key="auto_refresh",
                           help="Refresh every ~2s while a job is running.")

    jobs = api.list_jobs(limit=20)["jobs"]
    any_running = any(j.get("state") in ("queued", "running") for j in jobs)

    seen = st.session_state.setdefault("seen_done", set())
    for j in jobs:
        if j.get("state") in ("completed", "failed") and j["job_id"] not in seen:
            seen.add(j["job_id"])
            if j["state"] == "completed":
                st.toast(f"✅ {j['kind']} '{j['target']}' completed.", icon="✅")
                if j["kind"] == "train":
                    st.cache_data.clear()  # refresh sidebar / model-tab availability
            else:
                st.toast(f"❌ {j['kind']} '{j['target']}' failed.", icon="❌")

    if not jobs:
        st.caption("No jobs yet. Start a download or training above.")
    else:
        _badge = {"queued": "🕒 queued", "running": "🔵 running",
                  "completed": "🟢 completed", "failed": "🔴 failed"}
        for j in jobs:
            state = j.get("state", "?")
            head = f"{_badge.get(state, state)} — **{j['kind']}** · `{j['target']}`  ·  {j['job_id']}"
            with st.expander(head, expanded=(state in ("queued", "running"))):
                for m in (j.get("messages") or [])[-12:]:
                    st.text(f"{m['t'].split('T')[-1]}  {m['text']}")
                if j.get("result"):
                    st.json(j["result"])
                if j.get("failure"):
                    # A known, explainable failure (data not downloaded / model
                    # not trained): present it clearly, optionally LLM-enriched.
                    advised = j["failure"]
                    try:
                        import error_advisor
                        advised = error_advisor.advise(j["failure"])
                    except Exception:
                        advised = {**j["failure"], "ok": False}
                    render_failure(advised)
                elif j.get("error"):
                    st.error(j["error"])

    if auto and any_running:
        import time as _time
        _time.sleep(2)
        st.rerun()


# ======================  TAB: SECOM  ======================================= #
with tab_secom:
    st.subheader("SECOM — yield excursion / fail prediction")
    st.markdown(
        "Given ~590 process-sensor readings from a production unit, predict whether "
        "it will **fail** final electrical test. The rare failure is the positive "
        "class; the model uses a **tuned decision threshold** (not 0.5) saved with "
        "the pipeline, and reports a failure probability per unit."
    )
    md = avail.get(cfg.DOMAIN_SECOM)
    if not _unavailable_notice(cfg.DOMAIN_SECOM, md):
        try:
            meta = api.list_models()["models"][cfg.DOMAIN_SECOM]
            cols = st.columns(3)
            cols[0].metric("Decision threshold", f"{meta.get('threshold', '?')}")
            cols[1].metric("Status", "available")
            if meta.get("summary"):
                cols[2].caption(meta["summary"])
        except Exception:
            pass

        st.markdown("#### Try it")
        n = st.slider("How many example units to score", 1, 5, 3, key="secom_n")
        if st.button("Load example units & predict", key="secom_go", type="primary"):
            ex = api.get_example_data(cfg.DOMAIN_SECOM, n)["examples"]
            res = api.predict_secom(ex)
            st.session_state["secom_res"] = res
        res = st.session_state.get("secom_res")
        if res and not render_failure(res):
            c = st.columns(3)
            c[0].metric("Units scored", res["n_units"])
            c[1].metric("Predicted fail", res["n_predicted_fail"])
            c[2].metric("Threshold", res["threshold"])
            import pandas as pd
            df = pd.DataFrame(res["predictions"])
            st.dataframe(df, hide_index=True, width="stretch")
            st.bar_chart(df["fail_proba"])

        with st.expander("Paste your own row(s) of 590 features (JSON)"):
            txt = st.text_area("features = one row or a list of rows", "[]",
                               height=120, key="secom_paste")
            if st.button("Predict pasted", key="secom_paste_btn"):
                try:
                    feats = json.loads(txt)
                    res2 = api.predict_secom(feats)
                    if not render_failure(res2):
                        st.json(res2)
                except Exception as exc:
                    st.error(exc)


# ======================  TAB: Wafer CNN  =================================== #
with tab_cnn:
    st.subheader("Wafer CNN — defect-pattern classification")
    st.markdown(
        "A wafer map is a grid of dies (`0` no-die, `1` good, `2` bad). The spatial "
        "pattern of failures points to a systemic process problem. This CNN names "
        "the pattern — Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, "
        "Scratch, or none — with per-class probabilities."
    )
    md = avail.get(cfg.DOMAIN_WAFER_CNN)
    if not _unavailable_notice(cfg.DOMAIN_WAFER_CNN, md):
        st.markdown("#### Try it")
        if st.button("Load an example wafer map", key="cnn_load"):
            st.session_state["cnn_examples"] = api.get_example_data(cfg.DOMAIN_WAFER_CNN, 4)["examples"]
            st.session_state.pop("cnn_res", None)
        examples = st.session_state.get("cnn_examples")
        if examples:
            import numpy as np
            idx = st.number_input("Example #", 0, len(examples) - 1, 0, key="cnn_idx")
            grid = np.array(examples[int(idx)])
            left, right = st.columns([1, 1])
            with left:
                st.caption(f"Wafer map ({grid.shape[0]}×{grid.shape[1]}) — 0/1/2")
                st.dataframe(grid, height=260)
            with right:
                if st.button("Classify this map", key="cnn_go", type="primary"):
                    st.session_state["cnn_res"] = api.classify_wafer_map(examples[int(idx)])
                res = st.session_state.get("cnn_res")
                if res and not render_failure(res):
                    r0 = res["results"][0]
                    st.metric("Predicted pattern", r0["pattern"], f"conf {r0['confidence']}")
                    classes = res.get("classes") or [f"class_{i}" for i in range(len(r0["probabilities"]))]
                    import pandas as pd
                    probs = pd.DataFrame({"pattern": classes, "probability": r0["probabilities"]})
                    st.bar_chart(probs.set_index("pattern"))


# ======================  TAB: Yield  ====================================== #
with tab_yield:
    st.subheader("Yield curve — analytical area↔yield models")
    st.markdown(
        "Fits Poisson, Murphy and Negative-Binomial models to the empirical "
        "yield-vs-area curve from WM-811K. **Forward:** expected defect-free yield "
        "for a die of area A. **Inverse:** the largest die area that still meets a "
        "target yield. Also reports defect density `D0` and clustering `alpha`."
    )
    md = avail.get(cfg.DOMAIN_YIELD)
    if not _unavailable_notice(cfg.DOMAIN_YIELD, md):
        try:
            meta = api.list_models()["models"][cfg.DOMAIN_YIELD]
            st.caption(f"Best-fitting model: **{meta.get('best_model','?')}**")
            with st.expander("Fitted parameters per model"):
                st.json(meta.get("models", {}))
            model_choice = st.selectbox(
                "Model", ["(best)"] + list((meta.get("models") or {}).keys()), key="yield_model")
            model_arg = None if model_choice == "(best)" else model_choice
        except Exception:
            model_arg = None

        st.markdown("#### Forward — yield as a function of die area")
        amax = st.slider("Max area to plot (unit-die areas)", 2, 60, 25, key="yield_amax")
        areas = list(range(1, amax + 1))
        try:
            fwd = api.predict_yield(areas, model=model_arg)
            if not render_failure(fwd):
                import pandas as pd
                curve = pd.DataFrame(fwd["yield_by_area"]).set_index("area")
                st.line_chart(curve)
                st.caption(f"model = {fwd['model']} · params = {fwd.get('params', {})}")
        except Exception as exc:
            st.error(exc)

        st.markdown("#### Inverse — largest die for a target yield")
        ty = st.slider("Target yield", 0.50, 0.99, 0.90, 0.01, key="yield_target")
        if st.button("Compute max die area", key="yield_inv", type="primary"):
            try:
                inv = api.max_die_area_for_yield(ty, model=model_arg)
                if not render_failure(inv):
                    st.metric("Max die area (unit-die areas)",
                              inv["max_area_unit_dies"] if inv["max_area_unit_dies"] is not None else "∞")
            except Exception as exc:
                st.error(exc)


# ======================  TAB: Router  ===================================== #
with tab_router:
    st.subheader("Let the platform choose the model")
    st.caption("Edit the payload; the router inspects its shape and dispatches.")
    preset = st.selectbox("Preset payload", ["die area", "target yield", "wafer map", "secom row"])

    def _example(domain):
        try:
            return api.get_example_data(domain, 1)["examples"][0]
        except Exception:
            return None

    presets = {
        "die area": {"area": [1, 4, 9, 16]},
        "target yield": {"target_yield": 0.9},
        "wafer map": {"wafer_map": _example(cfg.DOMAIN_WAFER_CNN)},
        "secom row": {"features": [_example(cfg.DOMAIN_SECOM)]},
    }
    payload_text = st.text_area("Payload (JSON)", json.dumps(presets[preset]), height=160)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Explain routing only"):
            try:
                st.json(api.explain_routing(json.loads(payload_text)))
            except Exception as exc:
                st.error(exc)
    with col_b:
        if st.button("Route AND predict"):
            try:
                res = api.route_and_predict(json.loads(payload_text))
                if not render_failure(res):
                    st.info(f"Platform chose: **{res['routing']['chosen_domain']}** — "
                            f"{res['routing']['rationale']}")
                    st.json(res["prediction"])
            except Exception as exc:
                st.error(f"{exc}")


# ======================  TAB: Ask (LLM)  ================================== #
with tab_llm:
    st.subheader("Ask in plain English — an LLM picks the tool (optional)")
    st.caption(
        "Opt-in. Sends your request plus the tool catalogue to an Anthropic model, "
        "which chooses a tool and arguments; the tool then runs through the same "
        "in-process core. Requires an API key in keys.env."
    )
    import llm_routing

    api_key = llm_routing.load_api_key()
    if not llm_routing.anthropic_installed():
        st.warning("The `anthropic` package isn't installed, so LLM routing is off.")
        st.code("pip install anthropic", language="bash")
    elif not api_key:
        st.warning("No API key found — LLM routing is off. Add one to enable this tab.")
        with st.expander("Set up keys.env"):
            st.code("cd wafer_mcp_platform\ncp keys.env.example keys.env\n"
                    "# then edit keys.env:\nANTHROPIC_API_KEY=sk-ant-...", language="bash")
    else:
        st.success("LLM routing enabled — key loaded.")
        model = st.text_input("Model", value=llm_routing.get_model())
        user_text = st.text_area(
            "What do you want to do?",
            placeholder='e.g. "What\'s the largest die area that still hits 92% yield?"',
            height=100, key="llm_user_text_input")

        if st.button("Ask the LLM to pick a tool", key="llm_pick", type="primary"):
            if not user_text.strip():
                st.warning("Type a request first.")
            else:
                try:
                    with st.spinner("Asking the model…"):
                        choice = llm_routing.choose_tool(user_text, _tool_catalogue(),
                                                         api_key=api_key, model=model)
                    st.session_state["llm_choice"] = choice
                    st.session_state["llm_user_text"] = user_text
                    st.session_state.pop("llm_result", None)
                except Exception as exc:
                    st.error(f"LLM routing failed: {exc}")

        choice = st.session_state.get("llm_choice")
        if choice:
            if choice.get("text"):
                st.markdown(f"**Model said:** {choice['text']}")
            if choice.get("tool"):
                st.info(f"Proposed tool: **{choice['tool']}**")
                if choice["tool"] == "train":
                    st.warning("This is a training tool — it writes a model artifact. "
                               "Review the arguments before running.")
                args_text = st.text_area("Proposed arguments (editable JSON)",
                                         json.dumps(choice["arguments"], indent=2),
                                         height=140, key="llm_args")
                if st.button("Run selected tool", key="llm_run"):
                    try:
                        args = json.loads(args_text)
                    except json.JSONDecodeError as exc:
                        st.error(f"Arguments aren't valid JSON: {exc}")
                    else:
                        try:
                            with st.spinner(f"Running {choice['tool']}…"):
                                result = call_tool(choice["tool"], args)
                            st.session_state["llm_result"] = result
                            st.success("Tool result:")
                            st.json(result)
                            if choice["tool"] == "train":
                                st.cache_data.clear()
                        except Exception as exc:
                            st.error(f"Tool execution failed: {exc}")

                if st.session_state.get("llm_result") is not None:
                    if st.button("Explain result with the LLM", key="llm_explain"):
                        try:
                            with st.spinner("Summarising…"):
                                summary = llm_routing.summarize_result(
                                    st.session_state.get("llm_user_text", user_text),
                                    choice["tool"], st.session_state["llm_result"],
                                    api_key=api_key, model=model)
                            st.markdown(summary)
                        except Exception as exc:
                            st.error(f"Summary failed: {exc}")
            else:
                st.caption("The model didn't select a tool — try rephrasing.")


# ======================  TAB: Verify  ===================================== #
with tab_verify:
    st.subheader("Verify trained models are available")
    st.caption("For each model: is it trained, is its artifact on disk, and does it "
               "actually load? 'Available' means all three are true.")
    if st.button("🔍 Verify all models", key="do_verify", type="primary"):
        st.cache_data.clear()
        st.session_state["verify_report"] = api.verify_models()

    report = st.session_state.get("verify_report")
    if report is None:
        try:
            report = api.verify_models()
        except Exception as exc:
            st.error(f"verify_models failed: {exc}")
            report = None

    if report is not None:
        if report["all_available"]:
            st.success(f"✅ All {report['n_total']} models are trained and available.")
        else:
            st.warning(f"{report['n_available']} / {report['n_total']} models available. "
                       f"Missing or not loadable: {', '.join(report['missing'])}. "
                       "Train them from the Pipeline tab.")
        rows = []
        for dom, m in report["models"].items():
            rows.append({
                "domain": dom,
                "trained": "✅" if m["trained"] else "—",
                "artifact exists": "✅" if m["exists"] else "—",
                "loadable": "✅" if m["loadable"] else ("—" if m["loadable"] is None else "❌"),
                "available": "🟢" if m["available"] else "🔴",
                "artifact path": m["artifact"],
                "error": m["error"] or "",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Raw verify_models response"):
            st.json(report)


# ======================  TAB: Raw tools  ================================== #
with tab_tools:
    st.subheader("Call any tool")
    st.caption("The same tools the MCP server exposes to external agents, run here in-process.")
    names = [t["name"] for t in _tool_catalogue()]
    tool = st.selectbox("Tool", names)
    args_text = st.text_area("Arguments (JSON)", "{}", height=120)
    if st.button("Call"):
        try:
            out = call_tool(tool, json.loads(args_text))
            if not render_failure(out):
                st.json(out)
        except Exception as exc:
            st.error(exc)
