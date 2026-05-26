"""Backward-compatible entry point for the MASI-only pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.data_pipeline.run_masi_update import main, run_masi_update


if __name__ == "__main__":
    main()
