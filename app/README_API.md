# MASI FastAPI Layer

This API is a read-only layer over the existing MASI backend outputs. It does not retrain models and does not modify the forecasting engine.

## Run

From `app/`:

```powershell
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Routes

- `GET /health`
- `GET /forecast/latest`
- `GET /forecast/latest?horizon=1`
- `GET /forecast/history?limit=50`
- `GET /forecast/history?horizon=1&limit=10`
- `GET /report/latest`
- `GET /plot/latest`
- `GET /backtest/latest`

## Files Read

- `app/data/forecasts/forecast_log.csv`
- `app/data/reports/latest_forecast_report.md`
- `app/data/reports/statistical_backtest.json`
- `app/data/reports/economic_backtest.json`
- `app/data/reports/plots/latest_price_var_es_forecast.png`

## Notes

- The API exposes existing pipeline outputs only.
- No database is used.
- Forecast-log realization columns stay empty until the target date exists in the master dataset and the full pipeline enriches the log.
