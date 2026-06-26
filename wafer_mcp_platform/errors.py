"""
errors.py
=========

Typed, self-describing failures for the two conditions a user hits most often:

  • a model hasn't been trained yet, and they tried to predict with it, and
  • a dataset hasn't been downloaded yet, and something tried to read it.

Why typed errors (instead of bare RuntimeError / FileNotFoundError)
-------------------------------------------------------------------
The detection points (the adapters' `_ensure_loaded`, the data registry's
`ensure_wm811k`) already produce clear English. But the *presentation* layer —
and especially the optional LLM advisor — needs to know reliably WHICH kind of
failure happened, for which domain/dataset, without string-sniffing the message.
Each exception below therefore carries:

    .code      a stable machine code (MODEL_NOT_TRAINED / DATA_NOT_DOWNLOADED)
    .domain    the model family, when relevant (secom / wafer_cnn / yield_curve)
    .dataset   the dataset, when relevant (wm811k / secom)
    .title     a one-line human summary
    .remedy    an ordered list of concrete, copy-pasteable next steps

Backward compatibility
-----------------------
`ModelNotTrainedError` still subclasses RuntimeError and
`DataNotDownloadedError` still subclasses FileNotFoundError, so any existing
`except RuntimeError` / `except FileNotFoundError` keeps working unchanged. The
only thing that's new is the extra structure hanging off the exception.
"""
from __future__ import annotations

from typing import Any


class PlatformError(Exception):
    """Base class for the platform's user-facing, explainable failures."""

    code: str = "PLATFORM_ERROR"

    def __init__(
        self,
        message: str,
        *,
        domain: str | None = None,
        dataset: str | None = None,
        title: str | None = None,
        remedy: list[str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.domain = domain
        self.dataset = dataset
        self.title = title or message.split("\n", 1)[0]
        self.remedy = remedy or []
        self.extra = extra

    def to_dict(self) -> dict:
        """A plain, JSON-able description of the failure (no LLM involved)."""
        return {
            "error_code": self.code,
            "title": self.title,
            "domain": self.domain,
            "dataset": self.dataset,
            "what_happened": self.message,
            "how_to_fix": list(self.remedy),
            **self.extra,
        }


class ModelNotTrainedError(PlatformError, RuntimeError):
    """Raised when a prediction is requested but the model isn't trained yet."""

    code = "MODEL_NOT_TRAINED"


class DataNotDownloadedError(PlatformError, FileNotFoundError):
    """Raised when required data isn't present on disk yet."""

    code = "DATA_NOT_DOWNLOADED"


# --------------------------------------------------------------------------- #
# Remedy templates — one source of truth for "what should the user do now".
# Both the plain fallback message AND the LLM prompt are built from these, so
# the advice can never drift between the two paths.
# --------------------------------------------------------------------------- #
_TRAIN_TOOL_HINT = {
    "secom": "train(domain='secom')",
    "wafer_cnn": "train(domain='wafer_cnn', allow_download=True)",
    "yield_curve": "train(domain='yield_curve', allow_download=True)",
}

_DATA_FOR_DOMAIN = {
    "secom": "secom",
    "wafer_cnn": "wm811k",
    "yield_curve": "wm811k",
}


def model_not_trained(domain: str, expected_path: str | None = None) -> ModelNotTrainedError:
    """Build a fully-populated 'model not trained' error for a domain."""
    train_hint = _TRAIN_TOOL_HINT.get(domain, f"train(domain='{domain}')")
    needs_data = _DATA_FOR_DOMAIN.get(domain)
    remedy = [
        f"Train this model first by calling the `train` tool: {train_hint}.",
        "Then re-run your prediction.",
    ]
    if needs_data == "wm811k":
        remedy.insert(
            0,
            "This model is trained on the WM-811K wafer-map data. If that data "
            "isn't downloaded yet, train with allow_download=True (Kaggle creds) "
            "or register a local copy via the `register_wm811k` tool first.",
        )
    msg = f"The '{domain}' model has not been trained yet, so it cannot make predictions."
    if expected_path:
        msg += f" (No saved artifact was found at {expected_path}.)"
    return ModelNotTrainedError(
        msg,
        domain=domain,
        dataset=needs_data,
        title=f"{domain} model is not trained yet",
        remedy=remedy,
        expected_path=expected_path,
    )


def data_not_downloaded(
    dataset: str,
    expected_path: str | None = None,
    detail: str | None = None,
) -> DataNotDownloadedError:
    """Build a fully-populated 'data not downloaded' error for a dataset."""
    if dataset == "wm811k":
        remedy = [
            "Already have the LSWMD.pkl file? Register it once with the "
            "`register_wm811k` tool (no Kaggle needed): register_wm811k('/path/LSWMD.pkl').",
            "Want it fetched automatically? Put valid Kaggle credentials at "
            "~/.kaggle/kaggle.json, then call download_dataset('wm811k', allow_download=True).",
        ]
    elif dataset == "secom":
        remedy = [
            "Fetch the SECOM cache once with download_dataset('secom').",
            "It is small and downloads automatically; no credentials are required.",
        ]
    else:
        remedy = [f"Download the '{dataset}' dataset before using it."]
    msg = f"The '{dataset}' dataset has not been downloaded yet, so it isn't available on disk."
    if expected_path:
        msg += f" (Expected at {expected_path}.)"
    if detail:
        msg += f" Underlying detail: {detail}"
    return DataNotDownloadedError(
        msg,
        dataset=dataset,
        title=f"{dataset} dataset is not downloaded yet",
        remedy=remedy,
        expected_path=expected_path,
    )
