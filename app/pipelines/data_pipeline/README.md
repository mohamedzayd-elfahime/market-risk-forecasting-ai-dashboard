# Data Pipeline

Purpose: turn uploaded or source MASI files into the canonical application dataset.

Responsibilities:

- parse Investing.com MASI files
- clean and consolidate MASI history
- transform price data into model-ready features
- compute causal EGARCH features
- write outputs to `app/data/intermediate/` and `app/data/final/`

This package owns data preparation logic. CLI scripts in `app/jobs/` only call it.
