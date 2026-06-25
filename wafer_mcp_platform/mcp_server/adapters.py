"""
mcp_server/adapters.py
======================

Three model families, three very different prediction signatures. The MCP server
should not care about those differences, so each family is wrapped in an adapter
with one shared shape:

    adapter.domain         -> str
    adapter.is_trained()   -> bool
    adapter.metadata()     -> dict
    adapter.predict(payload) -> dict   (a plain JSON-able result)

Adapters import the underlying project lazily and tolerate missing heavy deps
(sklearn / torch / tensorflow), so the platform and dashboard still import and
run even before anything has been trained.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import platform_config as cfg  # noqa: E402

logger = logging.getLogger("adapters")


def _add_paths(*paths: Path) -> None:
    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# --------------------------------------------------------------------------- #
# SECOM: tabular sensor row -> P(fail)
# --------------------------------------------------------------------------- #
class SecomAdapter:
    domain = cfg.DOMAIN_SECOM
    description = (
        "SECOM yield excursion / fail prediction. Input: one or more rows of "
        "~590 numeric sensor features. Output: failure probability + pass/fail "
        "decision at the model's tuned threshold."
    )

    def __init__(self):
        self._predictor = None

    def is_trained(self) -> bool:
        return cfg.SECOM_MODEL_PATH.exists()

    def _ensure_loaded(self):
        if self._predictor is not None:
            return
        if not self.is_trained():
            raise RuntimeError(
                f"SECOM model not trained yet (expected {cfg.SECOM_MODEL_PATH}). "
                "Run the `train` tool with domain='secom' first."
            )
        _add_paths(cfg.PROJECTS_ROOT, cfg.SECOM_PROJECT, cfg.SECOM_PROJECT / "src")
        try:
            from predictor import Predictor  # type: ignore
        except Exception:
            from yield_excursion_fail_prediction.src.predictor import Predictor  # type: ignore
        self._predictor = Predictor.load(cfg.SECOM_MODEL_PATH)

    def metadata(self) -> dict:
        md = {"domain": self.domain, "trained": self.is_trained()}
        if self.is_trained():
            try:
                self._ensure_loaded()
                md["summary"] = self._predictor.describe()
                md["threshold"] = self._predictor.threshold
            except Exception as exc:  # pragma: no cover
                md["error"] = str(exc)
        return md

    def predict(self, payload: dict) -> dict:
        """payload = {"features": [[...590...], ...]} or {"rows": <records>}."""
        import pandas as pd
        self._ensure_loaded()
        if "features" in payload:
            import numpy as np
            arr = np.atleast_2d(np.asarray(payload["features"], dtype=float))
            X = pd.DataFrame(arr)
        elif "rows" in payload:
            X = pd.DataFrame(payload["rows"])
        else:
            raise ValueError("SECOM payload needs 'features' (array) or 'rows' (records).")
        frame = self._predictor.predict_frame(X)
        return {
            "domain": self.domain,
            "n_units": int(len(frame)),
            "n_predicted_fail": int(frame["decision"].sum()),
            "threshold": float(self._predictor.threshold),
            "predictions": frame.to_dict(orient="records"),
        }


# --------------------------------------------------------------------------- #
# Wafer CNN: 2-D wafer map -> defect-pattern class
# --------------------------------------------------------------------------- #
class WaferCnnAdapter:
    domain = cfg.DOMAIN_WAFER_CNN
    description = (
        "WM-811K wafer-map defect-pattern classifier (Center, Donut, Edge-Loc, "
        "Edge-Ring, Loc, Near-full, Random, Scratch, none). Input: a 2-D wafer "
        "map grid of {0:no-die, 1:good, 2:bad}. Output: predicted pattern + "
        "per-class probabilities."
    )

    def __init__(self):
        self._model = None
        self._classes = None
        self._backend = None
        self._img_size = None

    def _model_file(self):
        for name in ("best_model.pt", "best_model.keras"):
            p = cfg.CNN_MODEL_DIR / name
            if p.exists():
                return p
        return None

    def is_trained(self) -> bool:
        return self._model_file() is not None

    def metadata(self) -> dict:
        f = self._model_file()
        return {
            "domain": self.domain,
            "trained": f is not None,
            "model_file": str(f) if f else None,
            "backend": "torch" if (f and f.suffix == ".pt") else
                       ("keras" if f else None),
        }

    def _ensure_loaded(self):
        if self._model is not None:
            return
        f = self._model_file()
        if f is None:
            raise RuntimeError(
                "Wafer CNN not trained yet. Run the `train` tool with "
                "domain='wafer_cnn' (needs the shared WM-811K data + a backend)."
            )
        import json
        classes_file = cfg.CNN_MODEL_DIR / "classes.json"
        self._classes = json.loads(classes_file.read_text()) if classes_file.exists() else None
        # Recover the img_size the model was traced at, so serve-time resize matches.
        meta_file = cfg.CNN_MODEL_DIR / "history.json"
        try:
            train_meta = json.loads((cfg.CNN_MODEL_DIR / "train_meta.json").read_text())
            self._img_size = int(train_meta.get("img_size")) or None
        except Exception:
            self._img_size = None
        if f.suffix == ".pt":
            import torch  # noqa
            self._model = torch.jit.load(str(f)).eval()
            self._backend = "torch"
        else:
            _add_paths(cfg.CNN_PROJECT / "src")
            import tensorflow as tf  # noqa
            self._model = tf.keras.models.load_model(str(f))
            self._backend = "keras"

    @staticmethod
    def _preprocess_fns():
        """Return (resize_fn, map_max) matching how the model was TRAINED.

        Prefer the CNN project's own preprocessing so train and serve are
        guaranteed identical; fall back to a local copy if it can't be imported.
        """
        try:
            _add_paths(cfg.CNN_PROJECT / "src")
            from track1_preprocess import resize_map, MAP_MAX_VALUE  # type: ignore
            return resize_map, float(MAP_MAX_VALUE)
        except Exception:
            return WaferCnnAdapter._resize_nn_local, 2.0

    def predict(self, payload: dict) -> dict:
        """payload = {"wafer_map": [[...]]} or {"wafer_maps": [[[...]]]}, img_size optional."""
        import numpy as np
        self._ensure_loaded()
        maps = payload.get("wafer_maps")
        if maps is None:
            single = payload.get("wafer_map")
            if single is None:
                raise ValueError("Wafer payload needs 'wafer_map' or 'wafer_maps'.")
            maps = [single]
        size = int(payload.get("img_size", self._img_size or 64))
        # Preprocess EXACTLY as training did (resize then normalize to [0,1]),
        # otherwise the model sees {0,1,2} at serve time but was trained on
        # {0,0.5,1.0} — a silent train/serve skew. Reuse the project's own code.
        resize_nn, map_max = self._preprocess_fns()
        batch = np.stack([
            resize_nn(np.asarray(m), size).astype(np.float32) / map_max
            for m in maps
        ])

        if self._backend == "torch":
            import torch
            with torch.no_grad():
                logits = self._model(torch.from_numpy(batch[:, None, :, :]))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            probs = self._model.predict(batch[..., None], verbose=0)

        out = []
        for p in probs:
            idx = int(np.argmax(p))
            label = self._classes[idx] if self._classes else f"class_{idx}"
            out.append({"pattern": label,
                        "confidence": round(float(p[idx]), 4),
                        "probabilities": [round(float(x), 4) for x in p]})
        return {"domain": self.domain, "classes": self._classes, "results": out}

    @staticmethod
    def _resize_nn_local(m, size: int):
        """Fallback nearest-neighbour resize, mirroring the CNN project's
        resize_map (categorical cells — never interpolate). Used only if the
        project's own function can't be imported."""
        import numpy as np
        a = np.asarray(m)
        h, w = a.shape
        ri = np.clip((np.arange(size) * h / size).astype(int), 0, h - 1)
        ci = np.clip((np.arange(size) * w / size).astype(int), 0, w - 1)
        return a[ri][:, ci]


# --------------------------------------------------------------------------- #
# Yield curve: die area -> expected yield (Poisson / Murphy / Negative Binomial)
# --------------------------------------------------------------------------- #
class YieldCurveAdapter:
    domain = cfg.DOMAIN_YIELD
    description = (
        "Analytical yield models (Poisson, Murphy, Negative Binomial) fitted on "
        "WM-811K. Forward: area A -> expected defect-free yield. Inverse: target "
        "yield -> largest die area that still meets it. Also reports D0 / alpha."
    )

    def __init__(self):
        self._bundle = None

    def is_trained(self) -> bool:
        return cfg.YIELD_MODEL_PATH.exists()

    def _ensure_loaded(self):
        if self._bundle is not None:
            return
        if not self.is_trained():
            raise RuntimeError(
                f"Yield models not fitted yet (expected {cfg.YIELD_MODEL_PATH}). "
                "Run the `train` tool with domain='yield_curve'."
            )
        _add_paths(cfg.YIELD_PROJECT)
        from wm811k_yield.persistence import ModelStore  # type: ignore
        self._bundle = ModelStore.load(cfg.YIELD_MODEL_PATH)

    def metadata(self) -> dict:
        md = {"domain": self.domain, "trained": self.is_trained()}
        if self.is_trained():
            try:
                self._ensure_loaded()
                md["best_model"] = self._bundle.best_model
                md["models"] = {k: v.get("params", {}) for k, v in self._bundle.models.items()}
            except Exception as exc:  # pragma: no cover
                md["error"] = str(exc)
        return md

    def _predictor(self, model_name: str):
        _add_paths(cfg.YIELD_PROJECT)
        from wm811k_yield.models import model_by_name  # type: ignore
        from wm811k_yield.predictor import YieldPredictor  # type: ignore
        model = model_by_name(model_name)
        fit = self._bundle.fit_result(model_name)
        return YieldPredictor(model, fit)

    def predict(self, payload: dict) -> dict:
        """payload = {"area": A | [A,...]} (forward) or {"target_yield": Y} (inverse).
        Optional 'model' selects one of poisson/murphy/negative_binomial; default = best."""
        self._ensure_loaded()
        model_name = payload.get("model", self._bundle.best_model)
        pred = self._predictor(model_name)

        if "target_yield" in payload:
            area = pred.max_area_for_yield(float(payload["target_yield"]))
            return {"domain": self.domain, "model": model_name,
                    "target_yield": float(payload["target_yield"]),
                    "max_area_unit_dies": (None if area == float("inf") else round(area, 4))}

        if "area" in payload:
            import numpy as np
            areas = np.atleast_1d(np.asarray(payload["area"], dtype=float))
            ys = pred.predict_yield(areas)
            return {"domain": self.domain, "model": model_name,
                    "params": pred.params,
                    "yield_by_area": [{"area": float(a), "yield": round(float(y), 4)}
                                      for a, y in zip(areas, ys)]}

        raise ValueError("Yield payload needs 'area' (forward) or 'target_yield' (inverse).")


ADAPTERS = {
    cfg.DOMAIN_SECOM: SecomAdapter,
    cfg.DOMAIN_WAFER_CNN: WaferCnnAdapter,
    cfg.DOMAIN_YIELD: YieldCurveAdapter,
}

# Cache one adapter instance per domain so a loaded model (joblib / JSON / torch)
# is reused across calls instead of being re-read from disk every prediction.
_INSTANCES: dict = {}


def get_adapter(domain: str):
    if domain not in ADAPTERS:
        raise KeyError(f"Unknown domain '{domain}'. Choose from {list(ADAPTERS)}.")
    inst = _INSTANCES.get(domain)
    if inst is None:
        inst = ADAPTERS[domain]()
        _INSTANCES[domain] = inst
    return inst


def reset_adapter(domain: str | None = None) -> None:
    """Drop cached adapter(s) so the next call reloads from disk. Call this after
    training so freshly-saved artifacts are picked up immediately."""
    if domain is None:
        _INSTANCES.clear()
    else:
        _INSTANCES.pop(domain, None)
