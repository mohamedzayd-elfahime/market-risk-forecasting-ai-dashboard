# MASI Forecast Report

Model: masi_lstm_var_es_hmm
Version: 0.1.0
Alpha: 0.05
Data version: ab17add4c863

10-day and 25-day forecasts are operational mathematical extensions of the 1-day model, not separately learned multi-horizon models.
VaR and ES use sqrt(h) scaling; return uses h scaling.

## Economic Backtest Interpretation

The economic backtest evaluates whether the risk forecasts have historical decision-usefulness under a simulated risk-managed rule.
Economic metrics should be interpreted as a simulated wealth comparison, historical maximum drawdown, and historical Sharpe ratio.
The simulated risk-managed rule is used for historical evaluation only and does not constitute investment advice.

## Latest Forecasts

```text
           run_date target_date           model_name  horizon  alpha  mean_forecast  volatility_forecast  return_forecast  var_forecast  es_forecast            regime model_version data_version  realized_return
2026-05-24T11:45:16  2026-05-25 masi_lstm_var_es_hmm        1   0.05       0.001092             0.008869         0.003231     -0.014491    -0.019790 medium_volatility         0.1.0 ab17add4c863              NaN
2026-05-24T11:45:16  2026-06-05 masi_lstm_var_es_hmm       10   0.05       0.010915             0.028045         0.010915     -0.045826    -0.062582 medium_volatility         0.1.0 ab17add4c863              NaN
2026-05-24T11:45:16  2026-06-26 masi_lstm_var_es_hmm       25   0.05       0.027288             0.044343         0.027288     -0.072457    -0.098951 medium_volatility         0.1.0 ab17add4c863              NaN
```

Plot: `app/data/reports/plots/latest_price_var_es_forecast.png`
