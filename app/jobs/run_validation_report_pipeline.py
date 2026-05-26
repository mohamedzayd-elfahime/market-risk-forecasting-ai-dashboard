"""Run validation outputs and regenerate the latest report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.backtest_service import run_progressive_backtest
from backend.services.report_service import regenerate_latest_report


def main() -> None:
    backtest_result = run_progressive_backtest()
    report_path = regenerate_latest_report()

    print("Validation and report pipeline completed.")
    print("\nBacktest summary:")
    print(backtest_result["summary"])
    print(f"\nReport regenerated: {report_path}")


if __name__ == "__main__":
    main()
