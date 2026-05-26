"""Enrich and backtest the MASI forecast log."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.backtest_service import run_progressive_backtest


def main() -> None:
    result = run_progressive_backtest()
    print("Progressive backtest completed.")
    print(result["summary"])
    print(f"Updated forecast log: {Path('data/forecasts/forecast_log.csv').resolve()}")


if __name__ == "__main__":
    main()
