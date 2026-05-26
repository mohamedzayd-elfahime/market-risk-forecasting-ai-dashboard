"""Run the MASI forecasting pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import DEFAULT_CONFIG
from backend.services.forecast_service import run_forecast_pipeline
from jobs.hyperparameter_config import (
    BEST_RETURN_PARAMS_PATH,
    build_config_from_best_hyperparameters,
    build_config_from_best_return_hyperparameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MASI 1-day forecast pipeline.")
    parser.add_argument("--no-train", action="store_true", help="Reuse existing artifacts and run inference only.")
    parser.add_argument(
        "--use-best-params",
        action="store_true",
        help="Train with the latest Optuna best_hyperparameters.json instead of DEFAULT_CONFIG.",
    )
    parser.add_argument(
        "--use-best-return-params",
        action="store_true",
        help="Also train with the latest return-only Optuna hyperparameters.",
    )
    args = parser.parse_args()
    if (args.use_best_params or args.use_best_return_params) and args.no_train:
        parser.error("Optuna parameter overrides require training; remove --no-train.")

    config = DEFAULT_CONFIG
    if args.use_best_params:
        config = build_config_from_best_hyperparameters(
            return_path=BEST_RETURN_PARAMS_PATH if args.use_best_return_params else None
        )
    elif args.use_best_return_params:
        config = build_config_from_best_return_hyperparameters(config)

    if args.use_best_params:
        print("Using latest Optuna best hyperparameters for training/inference.")
        print(
            "Config override: "
            f"seq_len={config.seq_len}, "
            f"lstm_hidden_1={config.lstm_hidden_1}, "
            f"lstm_hidden_2={config.lstm_hidden_2}, "
            f"dense_hidden={config.dense_hidden}, "
            f"dropout={config.dropout}, "
            f"lr={config.lr}, "
            f"batch_size={config.batch_size}, "
            f"es_ridge_alpha={config.es_ridge_alpha}"
        )
    if args.use_best_return_params:
        print("Using latest return-only Optuna hyperparameters for the return LSTM.")
        print(
            "Return config override: "
            f"return_lstm_hidden_1={config.return_lstm_hidden_1 or config.lstm_hidden_1}, "
            f"return_lstm_hidden_2={config.return_lstm_hidden_2 or config.lstm_hidden_2}, "
            f"return_dense_hidden={config.return_dense_hidden or config.dense_hidden}, "
            f"return_dropout={config.return_dropout if config.return_dropout is not None else config.dropout}, "
            f"return_lr={config.return_lr if config.return_lr is not None else config.lr}, "
            f"return_batch_size={config.return_batch_size or config.batch_size}"
        )

    result = run_forecast_pipeline(config=config, train=not args.no_train)
    forecast = result["forecast"]
    print("MASI forecast pipeline completed.")
    print(forecast.to_string(index=False))
    print(f"Forecast log: {Path('data/forecasts/forecast_log.csv').resolve()}")
    print(f"Report: {result['report_path']}")
    print(f"Plot: {result['plot_path']}")


if __name__ == "__main__":
    main()
