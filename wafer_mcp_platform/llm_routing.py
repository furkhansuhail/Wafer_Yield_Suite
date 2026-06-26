"""
llm_routing.py
==============

OPTIONAL, opt-in natural-language routing via an LLM.

The platform's default router (``mcp_server/router.py``) is deterministic: it
picks a model purely from the *shape and keys* of a payload, needs no API key,
and is always on. This module is a separate, additive feature that the dashboard
exposes in its own "Ask (LLM)" tab. When (and only when) an Anthropic API key is
available, the user can describe what they want in plain English; an Anthropic
model reads the MCP tool catalogue and decides which tool to call with which
arguments. The chosen tool is then executed through the normal MCP session — the
LLM only *decides*; it never touches data or models directly.

Nothing here runs unless a key is present, so the rest of the platform is
unaffected if you never use it.

Key handling
------------
The API key is read, in order of precedence, from:

  1. ``keys.env`` at the platform root — simple ``KEY=VALUE`` lines.
  2. the ``ANTHROPIC_API_KEY`` environment variable.

``keys.env`` is gitignored. Copy ``keys.env.example`` to ``keys.env`` and paste
your key. The key is only ever passed to the Anthropic SDK; it is never logged,
printed, or placed in a URL.

You can also pin the routing model with ``WAFER_LLM_MODEL`` (in keys.env or the
environment); otherwise ``DEFAULT_MODEL`` is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# A current Anthropic model id. Override per-deployment via WAFER_LLM_MODEL
# (model availability depends on your account); the dashboard also lets you edit
# it at runtime without touching code.
DEFAULT_MODEL = "claude-sonnet-4-6"

KEYS_ENV_NAME = "keys.env"
_PLATFORM_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# keys.env parsing
# --------------------------------------------------------------------------- #
def _default_env_path() -> Path:
    return _PLATFORM_ROOT / KEYS_ENV_NAME


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (no external dotenv dependency)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def load_api_key(env_path: Path | None = None) -> str | None:
    """Return the Anthropic API key from keys.env, then the environment, else None."""
    vals = _parse_env_file(env_path or _default_env_path())
    return vals.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or None


def get_model(env_path: Path | None = None) -> str:
    """Return the routing model id (keys.env > env var > DEFAULT_MODEL)."""
    vals = _parse_env_file(env_path or _default_env_path())
    return vals.get("WAFER_LLM_MODEL") or os.environ.get("WAFER_LLM_MODEL") or DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Availability checks (so the UI can degrade gracefully)
# --------------------------------------------------------------------------- #
def anthropic_installed() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def llm_available(env_path: Path | None = None) -> bool:
    """True only when the SDK is installed AND a key is configured."""
    return anthropic_installed() and bool(load_api_key(env_path))


# --------------------------------------------------------------------------- #
# Tool schema translation + routing
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are the routing brain for a wafer / yield-analytics MCP platform. The "
    "user describes, in plain language, what they want. Choose the single most "
    "appropriate tool and fill its arguments from the request. If the request is "
    "ambiguous or no tool fits, do NOT call a tool — instead ask a brief "
    "clarifying question or explain what is missing. Prefer read-only tools; only "
    "choose the `train` tool when the user clearly asks to train or (re)fit a model."
)


def _to_anthropic_tools(mcp_tools: list[dict]) -> list[dict]:
    """Map MCP tool dicts ({name, description, input_schema}) to the Anthropic
    tools format ({name, description, input_schema})."""
    tools: list[dict] = []
    for t in mcp_tools:
        schema = t.get("input_schema") or {"type": "object", "properties": {}}
        tools.append({
            "name": t["name"],
            "description": (t.get("description") or "").strip()[:1024],
            "input_schema": schema,
        })
    return tools


def choose_tool(
    user_text: str,
    mcp_tools: list[dict],
    *,
    api_key: str,
    model: str | None = None,
    system: str | None = None,
) -> dict:
    """Ask the model which tool to call for ``user_text``.

    Returns ``{"tool": name|None, "arguments": {...}, "text": <assistant text>,
    "stop_reason": ..., "model": ...}``. ``tool`` is None when the model chose
    not to call anything (e.g. asked a clarifying question).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    model = model or DEFAULT_MODEL
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system or SYSTEM_PROMPT,
        tools=_to_anthropic_tools(mcp_tools),
        messages=[{"role": "user", "content": user_text}],
    )

    tool_name: str | None = None
    arguments: dict = {}
    text_parts: list[str] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            tool_name = block.name
            arguments = dict(block.input or {})
        elif btype == "text":
            text_parts.append(block.text)

    return {
        "tool": tool_name,
        "arguments": arguments,
        "text": "\n".join(text_parts).strip(),
        "stop_reason": resp.stop_reason,
        "model": model,
    }


def summarize_result(
    user_text: str,
    tool_name: str,
    result: dict,
    *,
    api_key: str,
    model: str | None = None,
) -> str:
    """Optional second pass: have the model explain a tool result in plain words.

    The result JSON is truncated before sending so a huge payload can't blow up
    the request. The system prompt forbids inventing numbers.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    model = model or DEFAULT_MODEL
    payload = json.dumps(result, indent=2, default=str)[:6000]
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=("Explain the tool result to the user concisely and accurately. "
                "Use only the numbers present in the result; never invent values."),
        messages=[{
            "role": "user",
            "content": (f"My request: {user_text}\n\n"
                        f"Tool `{tool_name}` returned:\n{payload}\n\n"
                        "Summarise the answer in a few sentences."),
        }],
    )
    return "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()


def explain_failure(
    failure: dict,
    *,
    api_key: str,
    model: str | None = None,
) -> str:
    """Turn a structured failure into a short, friendly, accurate explanation.

    ``failure`` is the dict produced by ``errors.PlatformError.to_dict()`` (it
    has error_code / title / what_happened / how_to_fix / domain / dataset). The
    system prompt pins the model to ONLY the facts in that dict — it must not
    invent paths, tools, or credentials — so the LLM explanation can never give
    advice that contradicts the deterministic remedy steps already attached.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    model = model or DEFAULT_MODEL
    payload = json.dumps(failure, indent=2, default=str)[:4000]
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=(
            "You explain why an operation on a wafer/yield ML platform failed, "
            "in plain language a non-expert can act on. You are given a "
            "structured failure description. Rules: (1) Use ONLY the facts in it "
            "— never invent file paths, tool names, datasets, or credentials. "
            "(2) Say plainly what went wrong, then give the exact next step(s) "
            "from 'how_to_fix', in order. (3) Be concise: 2–4 sentences, no "
            "preamble, no apology."
        ),
        messages=[{
            "role": "user",
            "content": f"Structured failure:\n{payload}\n\n"
                       "Explain what went wrong and what to do next.",
        }],
    )
    return "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
