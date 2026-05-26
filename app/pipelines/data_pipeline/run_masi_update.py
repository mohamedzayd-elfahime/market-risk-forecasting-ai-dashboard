"""Run the MASI-only incremental update pipeline."""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.data_pipeline.cleaning import DataCleaner
from pipelines.data_pipeline.paths import (
    ARCHIVE_DIR,
    HISTORICAL_RAW_MASI_PATH,
    LOGS_DIR,
    MASI_CLEANED_PATH,
    MASI_HISTORY_PATH,
    MASI_TRANSFORMED_PATH,
    MASTER_DATASET_PATH,
    RAW_DIR,
    RAW_FILE_PATTERNS,
    ensure_pipeline_dirs,
)
from pipelines.data_pipeline.transforming import DataTransformer
from ml.utils.egarch_features import compute_egarch_features


logger = logging.getLogger("masi_update")


def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"masi_update_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _find_raw_masi_files() -> list[Path]:
    files: list[Path] = []
    for pattern in RAW_FILE_PATTERNS:
        files.extend(RAW_DIR.glob(pattern))

    ignored = {HISTORICAL_RAW_MASI_PATH.resolve()}
    return sorted(
        (path for path in files if path.resolve() not in ignored and path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )


def _load_existing_history(cleaner: DataCleaner) -> pd.DataFrame:
    if MASI_HISTORY_PATH.exists():
        logger.info("Loading MASI history from %s", MASI_HISTORY_PATH)
        return cleaner.clean_masi_history(MASI_HISTORY_PATH)

    if HISTORICAL_RAW_MASI_PATH.exists():
        logger.info("Bootstrapping MASI history from %s", HISTORICAL_RAW_MASI_PATH)
        return cleaner.clean_MASI(HISTORICAL_RAW_MASI_PATH)

    raise FileNotFoundError(
        "No MASI history found. Expected one of: "
        f"{MASI_HISTORY_PATH}, {HISTORICAL_RAW_MASI_PATH}"
    )


def _merge_incremental_history(history: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    existing_dates = set(pd.to_datetime(history["date"]).dt.normalize())
    incoming_dates = set(pd.to_datetime(incoming["date"]).dt.normalize())

    merged = pd.concat([history, incoming], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)

    new_dates_added = len(incoming_dates.difference(existing_dates))
    return merged, new_dates_added


def _standardize_raw_investing_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(col).strip() for col in df.columns]

    required_columns = ["Date", "Dernier", "Ouv.", "Plus Haut", "Plus Bas", "Vol.", "Variation %"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing MASI raw columns in {path}: "
            + ", ".join(missing_columns)
        )

    raw = df[required_columns].copy()
    raw["_date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["_date"])
    return raw


def _update_historical_raw_masi(incoming_files: list[Path]) -> None:
    raw_frames = []
    if HISTORICAL_RAW_MASI_PATH.exists():
        raw_frames.append(_standardize_raw_investing_frame(HISTORICAL_RAW_MASI_PATH))

    for incoming_file in incoming_files:
        raw_frames.append(_standardize_raw_investing_frame(incoming_file))

    if not raw_frames:
        return

    historical_raw = pd.concat(raw_frames, ignore_index=True)
    historical_raw = historical_raw.drop_duplicates(subset=["_date"], keep="last")
    historical_raw = historical_raw.sort_values("_date", ascending=False)
    historical_raw = historical_raw.drop(columns=["_date"])
    historical_raw.to_csv(HISTORICAL_RAW_MASI_PATH, index=False)


def _archive_processed_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"{path.stem}_{timestamp}{path.suffix}"
    shutil.move(str(path), archive_path)
    return archive_path


def _nan_count(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def run_masi_update(archive_raw: bool = True) -> dict[str, object]:
    ensure_pipeline_dirs()
    _setup_logging()

    cleaner = DataCleaner()
    transformer = DataTransformer()

    raw_files = _find_raw_masi_files()
    if not raw_files:
        raise FileNotFoundError(f"No new MASI raw file found in {RAW_DIR}")

    history = _load_existing_history(cleaner)
    imported_rows = 0
    new_dates_added = 0
    processed_files: list[Path] = []

    for raw_file in raw_files:
        logger.info("Cleaning incoming MASI raw file: %s", raw_file)
        incoming = cleaner.clean_MASI(raw_file)
        imported_rows += len(incoming)
        history, added = _merge_incremental_history(history, incoming)
        new_dates_added += added
        processed_files.append(raw_file)

    outputs = transformer.run_pipeline(history)
    logger.info("Computing causal EGARCH features for the forecast master dataset")
    outputs["master"] = compute_egarch_features(outputs["master"])

    MASI_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    outputs["cleaned"].to_csv(MASI_CLEANED_PATH, index=False)
    outputs["cleaned"].to_csv(MASI_HISTORY_PATH, index=False)
    outputs["transformed"].to_csv(MASI_TRANSFORMED_PATH, index=False)
    outputs["master"].to_csv(MASTER_DATASET_PATH, index=False)

    _update_historical_raw_masi(processed_files)

    archived_files: list[Path] = []
    if archive_raw:
        for raw_file in processed_files:
            archived_files.append(_archive_processed_file(raw_file))

    final_dataset = outputs["master"]
    summary = {
        "imported_rows": imported_rows,
        "new_dates_added": new_dates_added,
        "date_min": final_dataset["date"].min(),
        "date_max": final_dataset["date"].max(),
        "final_shape": final_dataset.shape,
        "nan_count": _nan_count(final_dataset),
        "final_path": MASTER_DATASET_PATH,
        "processed_files": processed_files,
        "archived_files": archived_files,
    }

    print("\nMASI update completed successfully")
    print(f"Rows imported: {summary['imported_rows']}")
    print(f"New dates added: {summary['new_dates_added']}")
    print(f"Final date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"Final shape: {summary['final_shape']}")
    print(f"NaN count: {summary['nan_count']}")
    print(f"Final dataset saved to: {summary['final_path']}")

    return {"outputs": outputs, "summary": summary}


def main() -> dict[str, object]:
    return run_masi_update(archive_raw=True)


if __name__ == "__main__":
    main()
