"""CLI entry point for the MASI data update pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.data_pipeline.run_masi_update import main, run_masi_update


if __name__ == "__main__":
    main()
