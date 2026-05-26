"""Centralized paths for the app-local MASI backend."""

from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = APP_ROOT / "data"
ML_DIR = APP_ROOT / "ml"
LOGS_DIR = APP_ROOT / "logs"

MASTER_DATASET_PATH = DATA_DIR / "final" / "master_dataset.csv"
FORECAST_DIR = DATA_DIR / "forecasts"
FORECAST_LOG_PATH = FORECAST_DIR / "forecast_log.csv"
REPORT_DIR = DATA_DIR / "reports"
PLOTS_DIR = REPORT_DIR / "plots"
TEST_PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
TRAINING_HISTORY_PATH = REPORT_DIR / "training_history.json"
TRAINING_DIAGNOSTICS_PATH = REPORT_DIR / "training_diagnostics.json"
STAT_BACKTEST_PATH = REPORT_DIR / "statistical_backtest.json"
ECON_BACKTEST_PATH = REPORT_DIR / "economic_backtest.json"

ARTIFACT_DIR = ML_DIR / "artifacts"
VAR_SCALER_PATH = ARTIFACT_DIR / "var_feature_scaler.joblib"
RETURN_SCALER_PATH = ARTIFACT_DIR / "return_feature_scaler.joblib"
RETURN_CALIBRATION_PATH = ARTIFACT_DIR / "return_calibration.joblib"
RETURN_SELECTOR_PATH = ARTIFACT_DIR / "return_selector.joblib"
RETURN_TABULAR_PATH = ARTIFACT_DIR / "return_tabular.joblib"
RETURN_EGARCH_AR1_PATH = ARTIFACT_DIR / "return_egarch_ar1.joblib"
RETURN_DIRECTIONAL_PATH = ARTIFACT_DIR / "return_directional_egarch.joblib"
VAR_MODEL_PATH = ARTIFACT_DIR / "var_lstm.pt"
RETURN_MODEL_PATH = ARTIFACT_DIR / "return_lstm.pt"
ES_MODEL_PATH = ARTIFACT_DIR / "es_ridge.joblib"
HMM_MODEL_PATH = ARTIFACT_DIR / "hmm_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


def ensure_backend_dirs() -> None:
    for path in (FORECAST_DIR, REPORT_DIR, PLOTS_DIR, ARTIFACT_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
