# MASI Backend Forecasting

This backend is MASI-only. It uses `app/data/final/master_dataset.csv` as the canonical input.

## Scientific Scope

- Main validated engine: 1-day-ahead forecasts.
- Risk threshold: `alpha = 0.05`.
- Dynamic rolling window: the backend selects the most recent `4000 + test_window` valid observations.
- The first 4000 observations of that rolling window are the learning block; the last test observations are the test block.
- `test_window` is derived from the current validated ratio: `786 / 4000`, so the current dynamic test window is about 786 observations.
- 10-day and 25-day outputs are operational extensions of the 1-day forecast, not separately learned multi-horizon models.
- VaR and ES use `sqrt(h)` scaling; return uses `h` scaling.

## Anti-Leakage Rules

- Chronological split only.
- Feature scaler is fit on train only.
- Validation is inside the train window.
- Test data never affects train, scaler, VaR model, ES model, or HMM fit.
- LSTM sequences use only past observations.
- Forecasts are logged ex ante with `run_date` and `target_date`; realized returns are filled later.
- HMM is fit only on train and used for current regime detection, not unvalidated future regime forecasting.

## Run

From `app/`:

```powershell
..\.venv\Scripts\python.exe .\jobs\run_forecast_pipeline.py
```

To reuse existing artifacts and run inference only:

```powershell
..\.venv\Scripts\python.exe .\jobs\run_forecast_pipeline.py --no-train
```

To enrich the forecast log after realized returns become available:

```powershell
..\.venv\Scripts\python.exe .\jobs\run_backtest_from_log.py
```

## Outputs

- `app/data/forecasts/forecast_log.csv`
- `app/data/reports/latest_forecast_report.md`
- `app/data/reports/plots/latest_price_var_es_forecast.png`
- `app/ml/artifacts/feature_scaler.joblib`
- `app/ml/artifacts/var_lstm.pt`
- `app/ml/artifacts/return_lstm.pt`
- `app/ml/artifacts/es_ridge.joblib`
- `app/ml/artifacts/hmm_model.joblib`
- `app/ml/artifacts/model_metadata.json`

## Price Plot Method

The model forecasts returns. For the plot, the price forecast is reconstructed as:

```text
forecast_price_proxy = last_observed_price * exp(return_forecast)
```

VaR and ES are displayed on the same price scale using the same transformation.
