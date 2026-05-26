"""Run the full MASI workflow: data update, training, forecast, report and plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.core.config import DEFAULT_CONFIG
from backend.core.paths import FORECAST_LOG_PATH
from backend.services.backtest_service import run_progressive_backtest
from backend.services.forecast_service import run_forecast_pipeline
from pipelines.data_pipeline.run_masi_update import run_masi_update
from jobs.hyperparameter_config import build_config_from_best_hyperparameters


def _assert_forecast_log_writable() -> None:
    if not FORECAST_LOG_PATH.exists():
        return
    try:
        with FORECAST_LOG_PATH.open("a", encoding="utf-8"):
            pass
    except PermissionError as exc:
        raise PermissionError(
            f"Forecast log is open or locked: {FORECAST_LOG_PATH}. "
            "Close it in Excel before running the automated pipeline."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MASI data and forecast pipeline.")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Update MASI data, then reuse existing ML artifacts for inference only.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip enrichment/backtesting of existing forecast log after the data update.",
    )
    parser.add_argument(
        "--use-best-params",
        action="store_true",
        help="Train with the accepted Optuna VaR/ES hyperparameters instead of DEFAULT_CONFIG.",
    )
    args = parser.parse_args()
    _assert_forecast_log_writable()
    config = build_config_from_best_hyperparameters() if args.use_best_params else DEFAULT_CONFIG

    print("Step 1/3 - Updating MASI data...")
    try:
        data_result = run_masi_update(archive_raw=True)
        data_summary = data_result["summary"]
    except FileNotFoundError as exc:
        print(f"No new MASI raw file to process: {exc}")
        print("Continuing with the current master dataset.")
        data_summary = {
            "imported_rows": 0,
            "new_dates_added": 0,
            "final_shape": "unchanged",
            "final_path": APP_ROOT / "data" / "final" / "master_dataset.csv",
        }

    backtest_summary = None
    if args.skip_backtest:
        print("\nStep 2/3 - Skipping forecast log enrichment/backtesting.")
    elif FORECAST_LOG_PATH.exists():
        print("\nStep 2/3 - Enriching forecast log and running progressive backtest...")
        backtest_result = run_progressive_backtest(config=config)
        backtest_summary = backtest_result["summary"]
    else:
        print("\nStep 2/3 - No forecast log found yet; skipping backtest enrichment.")

    print("\nStep 3/3 - Running MASI forecast engine...")
    forecast_result = run_forecast_pipeline(config=config, train=not args.no_train)
    forecast = forecast_result["forecast"]

    print("\nFull MASI pipeline completed successfully.")
    print(f"Rows imported: {data_summary['imported_rows']}")
    print(f"New dates added: {data_summary['new_dates_added']}")
    print(f"Final dataset shape: {data_summary['final_shape']}")
    print(f"Final dataset path: {data_summary['final_path']}")
    if backtest_summary is not None:
        print("\nProgressive backtest summary:")
        print(backtest_summary)
    print("\nLatest forecasts:")
    print(forecast.to_string(index=False))
    print(f"\nForecast log: {APP_ROOT / 'data' / 'forecasts' / 'forecast_log.csv'}")
    print(f"Report: {forecast_result['report_path']}")
    print(f"Plot: {forecast_result['plot_path']}")


if __name__ == "__main__":
    main()
