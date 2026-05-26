# Baseline Modeling Notebooks

This folder contains econometric baseline experiments for MASI downside-risk forecasting.

## Main Benchmark Notebook

`01_garch_var_es_benchmark_backtesting.ipynb`

This is the GitHub-ready benchmark notebook. It:

- prepares a strict chronological rolling-window split;
- screens EGARCH and GJR-GARCH specifications;
- computes one-step-ahead VaR and Expected Shortfall;
- runs VaR/ES statistical backtests;
- evaluates economic allocation metrics.

Older baseline notebooks are kept for traceability but are not the primary reading path.
