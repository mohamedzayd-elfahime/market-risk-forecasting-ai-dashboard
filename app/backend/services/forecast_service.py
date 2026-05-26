"""High-level service for training, inference, logging, reporting and plotting."""

from __future__ import annotations

import json
import pandas as pd

from backend.core.config import DEFAULT_CONFIG, ForecastConfig
from backend.core.paths import (
    ECON_BACKTEST_PATH,
    FORECAST_LOG_PATH,
    MASTER_DATASET_PATH,
    METADATA_PATH,
    PLOTS_DIR,
    REPORT_DIR,
    STAT_BACKTEST_PATH,
    TRAINING_DIAGNOSTICS_PATH,
    TEST_PREDICTIONS_PATH,
    TRAINING_HISTORY_PATH,
    ensure_backend_dirs,
)
from ml.inference.forecast_engine import make_forecast
from ml.training.train_pipeline import train_forecast_models
from ml.utils.data_loading import dataset_version, load_master_dataset
from ml.utils.forecast_log import append_forecasts
from ml.utils.backtesting import compute_wealth_curves, summarize_economic_backtest, summarize_test_predictions
from ml.utils.diagnostics import summarize_loss_history
from ml.utils.plotting import plot_latest_forecast
from ml.utils.reporting import write_forecast_report


def run_forecast_pipeline(config: ForecastConfig = DEFAULT_CONFIG, train: bool = True) -> dict[str, object]:
    ensure_backend_dirs()
    master_df = load_master_dataset(MASTER_DATASET_PATH)
    data_version = dataset_version(MASTER_DATASET_PATH)

    training_outputs = None
    statistical_backtest = None
    economic_backtest = None
    if train:
        training_outputs = train_forecast_models(master_df, data_version, config)
        test_predictions = training_outputs["test_predictions"].copy()
        test_predictions = compute_wealth_curves(test_predictions)
        test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

        history_payload = {
            "var_history": training_outputs["var_history"],
            "return_history": training_outputs["return_history"],
        }
        TRAINING_HISTORY_PATH.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
        
    # Always update summaries if test predictions exist
    if TEST_PREDICTIONS_PATH.exists():
        test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
        # Ensure wealth curves are present/updated
        test_predictions = compute_wealth_curves(test_predictions)
        test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
        
        statistical_backtest = summarize_test_predictions(test_predictions, alpha=config.alpha)
        economic_backtest = summarize_economic_backtest(test_predictions)
        
        STAT_BACKTEST_PATH.write_text(json.dumps(statistical_backtest, indent=2), encoding="utf-8")
        ECON_BACKTEST_PATH.write_text(json.dumps(economic_backtest, indent=2), encoding="utf-8")

        if train and training_outputs:
            diagnostics_payload = {
                "var_model": summarize_loss_history(training_outputs["var_history"]),
                "return_model": summarize_loss_history(training_outputs["return_history"]),
                "es_model": {
                    "method": "two_step_ridge_after_var",
                    "ridge_alpha": config.es_ridge_alpha,
                    "n_train_violations_used": training_outputs["es_bundle"].get("n_violations_used"),
                    "shortfall_floor": training_outputs["es_bundle"].get("shortfall_floor"),
                    "test_es_never_above_var": bool((test_predictions["es_pred"] <= test_predictions["var_pred"]).all()),
                    "n_es_tail_observations": statistical_backtest.get("n_es_tail_observations"),
                    "es_tail_calibration_stat": statistical_backtest.get("es_tail_calibration_stat"),
                    "es_tail_calibration_p_value": statistical_backtest.get("es_tail_calibration_p_value"),
                    "es_tail_residual_mean": statistical_backtest.get("es_tail_residual_mean"),
                },
            }
            TRAINING_DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")

    forecast_df = make_forecast(master_df, config, data_version)
    forecast_log = append_forecasts(FORECAST_LOG_PATH, forecast_df)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    plot_path = plot_latest_forecast(master_df, forecast_df, PLOTS_DIR / "latest_price_var_es_forecast.png")
    report_path = write_forecast_report(REPORT_DIR / "latest_forecast_report.md", forecast_df, metadata, plot_path)

    return {
        "forecast": forecast_df,
        "forecast_log": forecast_log,
        "report_path": report_path,
        "plot_path": plot_path,
        "training_outputs": training_outputs,
        "statistical_backtest": statistical_backtest,
        "economic_backtest": economic_backtest,
    }
