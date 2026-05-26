"""Centralized paths for the MASI-only data pipeline."""

from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = APP_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
ARCHIVE_DIR = DATA_DIR / "archive"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
LOGS_DIR = APP_ROOT / "logs"

HISTORICAL_RAW_MASI_PATH = RAW_DIR / "MASI.csv"
MASI_HISTORY_PATH = INTERMEDIATE_DIR / "masi_history.csv"
MASI_CLEANED_PATH = INTERMEDIATE_DIR / "masi_cleaned.csv"
MASI_TRANSFORMED_PATH = INTERMEDIATE_DIR / "masi_transformed.csv"
MASTER_DATASET_PATH = FINAL_DIR / "master_dataset.csv"

RAW_FILE_PATTERNS = ("*.csv", "*.txt", "*.xlsx", "*.xls")


def ensure_pipeline_dirs() -> None:
    for directory in (RAW_DIR, ARCHIVE_DIR, INTERMEDIATE_DIR, FINAL_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
