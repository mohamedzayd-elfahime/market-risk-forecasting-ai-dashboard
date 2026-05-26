"""Regenerate the latest MASI forecast report from the forecast log."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.report_service import regenerate_latest_report


def main() -> None:
    report_path = regenerate_latest_report()
    print(f"Report regenerated: {report_path}")


if __name__ == "__main__":
    main()
