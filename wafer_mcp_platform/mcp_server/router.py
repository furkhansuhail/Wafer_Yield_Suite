"""
mcp_server/router.py
====================

The "MCP decides which model" logic.

Given a payload with no explicit domain, infer which of the three model families
should handle it, purely from the *shape and keys* of the input. This is what
lets the MCP server expose a single `route_and_predict` tool that an agent (or
the Streamlit dashboard) can call without knowing which model is appropriate.

Routing rules (first match wins), each with a short rationale string so the
decision is auditable rather than magic:

    has 'wafer_map' / 'wafer_maps'         -> wafer_cnn   (a 2-D defect grid)
    has 'target_yield' or 'area'           -> yield_curve (area<->yield question)
    has 'features' / 'rows' (~590 wide)    -> secom       (tabular sensor row)

`route(payload)` returns (domain, rationale). `explain(payload)` returns the full
scoring breakdown, which the dashboard shows so the user can see *why* a model
was chosen.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import platform_config as cfg  # noqa: E402


def _looks_2d(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple))


def score(payload: dict) -> dict[str, tuple[int, str]]:
    """Return a {domain: (score, reason)} breakdown for transparency."""
    out: dict[str, tuple[int, str]] = {d: (0, "no matching signal") for d in cfg.ALL_DOMAINS}

    # --- wafer CNN -------------------------------------------------------- #
    if "wafer_map" in payload or "wafer_maps" in payload:
        wm = payload.get("wafer_map") or (payload.get("wafer_maps") or [None])[0]
        if _looks_2d(wm):
            out[cfg.DOMAIN_WAFER_CNN] = (10, "payload carries a 2-D wafer-map grid")
        else:
            out[cfg.DOMAIN_WAFER_CNN] = (4, "wafer_map key present but not clearly 2-D")

    # --- yield curve ------------------------------------------------------ #
    if "target_yield" in payload:
        out[cfg.DOMAIN_YIELD] = (10, "asks for max area at a target yield (inverse)")
    elif "area" in payload and "features" not in payload and "rows" not in payload:
        out[cfg.DOMAIN_YIELD] = (9, "asks for yield as a function of die area (forward)")

    # --- SECOM ------------------------------------------------------------ #
    if "features" in payload or "rows" in payload:
        width = _payload_width(payload)
        if width is not None and width >= 100:
            out[cfg.DOMAIN_SECOM] = (10, f"tabular row with {width} numeric features (~SECOM 590)")
        else:
            out[cfg.DOMAIN_SECOM] = (6, "tabular features/rows present (narrow)")

    return out


def _payload_width(payload: dict):
    try:
        if "features" in payload:
            f = payload["features"]
            row = f[0] if (isinstance(f, (list, tuple)) and f and isinstance(f[0], (list, tuple))) else f
            return len(row)
        if "rows" in payload and payload["rows"]:
            return len(payload["rows"][0])
    except Exception:
        return None
    return None


def route(payload: dict) -> tuple[str, str]:
    """Pick the best domain. Returns (domain, rationale). Honors an explicit
    'domain' override if the caller already knows what they want."""
    if payload.get("domain") in cfg.ALL_DOMAINS:
        return payload["domain"], "explicit domain override in payload"

    breakdown = score(payload)
    domain, (best, reason) = max(breakdown.items(), key=lambda kv: kv[1][0])
    if best == 0:
        raise ValueError(
            "Could not route this payload to any model. Provide one of: "
            "'wafer_map' (2-D grid), 'area'/'target_yield' (scalar), or "
            "'features'/'rows' (tabular). Or set 'domain' explicitly."
        )
    return domain, reason


def explain(payload: dict) -> dict:
    breakdown = score(payload)
    chosen, reason = route(payload)
    return {
        "chosen_domain": chosen,
        "rationale": reason,
        "scores": {d: {"score": s, "reason": r} for d, (s, r) in breakdown.items()},
    }
