# Notebooks

The notebooks are organized by research stage. The cleaned GitHub-ready workflow is:

1. `01_statistical_validation/01_masi_return_statistical_investigation.ipynb`
2. `01_statistical_validation/02_multivariate_predictor_eda_for_hybrid_risk_models.ipynb`
3. `02_baseline_modeling/01_garch_var_es_benchmark_backtesting.ipynb`

Each notebook keeps result-producing cells visible for GitHub rendering. Reusable functions live in `../src/analysis/` inside the `research/` area, so the notebooks stay focused on interpretation and outputs without mixing with the production dashboard code.

## Folder Map

- `01_statistical_validation/`: return diagnostics, exploratory analysis, and multivariate predictor validation.
- `02_baseline_modeling/`: econometric baseline and benchmark modeling.
- `03_deep_learning_risk/`: deep-learning risk forecasting experiments.
