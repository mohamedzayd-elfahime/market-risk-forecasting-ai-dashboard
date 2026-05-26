# Analysis Utilities

Reusable notebook logic lives here so the notebooks stay readable on GitHub.

## Modules

- `garch_benchmark_utils.py`: rolling split preparation, EGARCH/GJR-GARCH fitting, and parametric VaR/ES utilities.
- `risk_backtesting.py`: Kupiec, Christoffersen, and ES calibration backtests.
- `economic_evaluation.py`: strategy returns, wealth curves, drawdown, performance ratios, and VaR-budget allocation rules.
- `lstm_var_es_pipeline.py`: feature engineering, chronological splits, LSTM VaR training, Ridge ES estimation, and forecast backtesting helpers.

The notebooks import these modules while keeping final result cells in place for direct inspection.
