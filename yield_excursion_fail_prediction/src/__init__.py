"""SECOM yield-prediction pipeline: modular, leak-free, imbalance-aware."""
from .data_downloader import load_secom, SecomData
from .eda import run_eda, EDAReport
from .imbalance_balancer import build_balancing_plan, BalancingPlan
from .model_trainer import (
    build_pipeline, compare_configs, make_train_test_split, train,
    tune_hyperparameters, AddMissingIndicators,
)
from .evaluator import (
    evaluate, tune_threshold, EvaluationReport,
    fit_select_threshold_evaluate, select_threshold_oof,
)
from .predictor import Predictor, load_predictor, save_predictor
from .main import run_pipeline, PipelineResult, ALL_MODELS

__version__ = "1.2.0"
__all__ = [
    "load_secom", "SecomData", "run_eda", "EDAReport",
    "build_balancing_plan", "BalancingPlan", "build_pipeline", "compare_configs",
    "make_train_test_split", "train", "tune_hyperparameters", "AddMissingIndicators",
    "evaluate", "tune_threshold", "EvaluationReport",
    "fit_select_threshold_evaluate", "select_threshold_oof",
    "Predictor", "load_predictor", "save_predictor",
    "run_pipeline", "PipelineResult", "ALL_MODELS",
]
