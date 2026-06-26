"""
error_advisor.py
================

Turns a failure into a clear, user-facing response — and, *only when an API key
is configured*, enriches it with a plain-language LLM explanation.

The exact behaviour you asked for
---------------------------------
                          ┌─ API key found?  ──► hand the structured failure to
   an operation fails ──► │                      the LLM, return its explanation
                          └─ no API key      ──► return the built-in plain-English
                                                 message ("model not trained",
                                                 "data not downloaded") + the fix

Either way the caller gets the SAME structured fields (error_code, title,
how_to_fix, ...); the only thing the key toggles is whether a friendlier
``explanation`` is written by the LLM or taken from the local template. The LLM
is never required — without a key the platform still tells the user exactly what
went wrong and how to fix it.

This module is import-light: ``anthropic`` is only touched when a key is present
and the LLM path is actually taken, so importing it costs nothing on the no-key
path.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import errors  # noqa: E402
import llm_routing  # noqa: E402


def _fallback_message(failure: dict) -> str:
    """The plain, no-LLM explanation. Always available, always accurate."""
    lines = [failure.get("what_happened", failure.get("title", "The operation failed."))]
    fixes = failure.get("how_to_fix") or []
    if fixes:
        lines.append("")
        lines.append("How to fix it:")
        lines.extend(f"  {i}. {step}" for i, step in enumerate(fixes, 1))
    return "\n".join(lines)


def advise(
    failure: dict,
    *,
    use_llm: bool | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Build the user-facing failure response.

    Parameters
    ----------
    failure : dict
        Output of ``errors.PlatformError.to_dict()`` (or any dict with the same
        keys: error_code, title, what_happened, how_to_fix, ...).
    use_llm : bool | None
        Force the LLM path on/off. ``None`` (default) means "auto": use the LLM
        when, and only when, a key is configured and the SDK is installed.
    api_key, model : str | None
        Optional overrides; by default the key/model come from keys.env or env.

    Returns
    -------
    dict
        The ``failure`` fields, plus:
          ok            : always False (this is a failure response)
          explanation   : the human-readable text (LLM or template)
          llm_used      : whether the LLM actually wrote the explanation
          llm_error     : present only if the LLM path was attempted but failed
    """
    out = dict(failure)
    out["ok"] = False

    key = api_key or llm_routing.load_api_key()
    # "auto": the flag you described — check for a key, use the LLM if present.
    want_llm = llm_routing.llm_available() if use_llm is None else bool(use_llm)

    if want_llm and key:
        try:
            out["explanation"] = llm_routing.explain_failure(
                failure, api_key=key, model=model
            )
            out["llm_used"] = True
        except Exception as exc:  # network/SDK/rate-limit — never hide the failure
            out["explanation"] = _fallback_message(failure)
            out["llm_used"] = False
            out["llm_error"] = f"{type(exc).__name__}: {exc}"
    else:
        # No key (or LLM disabled): fall back to the clear built-in message.
        out["explanation"] = _fallback_message(failure)
        out["llm_used"] = False

    return out


def advise_for_exception(
    exc: Exception,
    *,
    use_llm: bool | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict | None:
    """Convenience wrapper: if ``exc`` is one of the platform's explainable
    failures, return an advised response dict; otherwise return None so the
    caller can re-raise genuine/unexpected errors unchanged."""
    if isinstance(exc, errors.PlatformError):
        return advise(exc.to_dict(), use_llm=use_llm, api_key=api_key, model=model)
    return None
