# Statistical Validation Notebooks

This folder contains notebooks used to validate the MASI dataset before risk modeling.

## Recommended Order

1. `01_masi_return_statistical_investigation.ipynb`
   - Statistical characterization of MASI returns.
   - Covers log returns, distribution diagnostics, stationarity tests, volatility clustering, ARCH-LM, Ljung-Box, and leverage asymmetry.

2. `02_multivariate_predictor_eda_for_hybrid_risk_models.ipynb`
   - Multivariate predictor analysis for hybrid risk forecasting.
   - Covers feature relationships, LSTM VaR plus Ridge ES modeling, HMM regime diagnostics, and economic evaluation.

`02_exploratory_data_analysys.ipynb` is kept as an earlier exploratory notebook.
