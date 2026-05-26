"""Dataset-specific transformations plus synchronized merge pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.data.cleaning import DataCleaner


class DataTransformer:
    """Transform each cleaned dataset independently, then merge them."""

    CORE_WINDOWS = (5, 10, 21)
    MODEL_WINDOWS = (5, 20)
    MODEL_LAGS = (1, 2, 5)
    MODEL_SERIES_ORDER = (
        "masi",
        "atw",
        "bcp",
        "iam",
        "lhm",
        "mng",
        "brent",
        "gold",
        "eur_mad",
        "gpr_index",
    )

    @staticmethod
    def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        valid = (numerator > 0) & (denominator > 0)
        result = pd.Series(np.nan, index=numerator.index, dtype=float)
        result.loc[valid] = np.log(numerator.loc[valid] / denominator.loc[valid])
        return result

    @staticmethod
    def _safe_pct_change(series: pd.Series) -> pd.Series:
        shifted = series.shift(1)
        valid = (series > 0) & (shifted > 0)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        result.loc[valid] = (series.loc[valid] / shifted.loc[valid]) - 1.0
        return result

    def transform_market_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        transformed = df.copy().sort_values("date").reset_index(drop=True)

        transformed["simple_return"] = self._safe_pct_change(transformed["close"])
        transformed["log_return"] = self._safe_log_ratio(transformed["close"], transformed["close"].shift(1))
        transformed["log_open_close"] = self._safe_log_ratio(transformed["close"], transformed["open"])
        transformed["log_high_low"] = self._safe_log_ratio(transformed["high"], transformed["low"])
        transformed["price_range_pct"] = np.where(
            transformed["close"] > 0,
            (transformed["high"] - transformed["low"]) / transformed["close"],
            np.nan,
        )
        transformed["close_to_open_ratio"] = np.where(
            transformed["open"] > 0,
            transformed["close"] / transformed["open"],
            np.nan,
        )
        transformed["volume_log_change"] = self._safe_log_ratio(
            transformed["volume"].replace(0, np.nan),
            transformed["volume"].shift(1).replace(0, np.nan),
        )

        for window in self.CORE_WINDOWS:
            transformed[f"log_return_mean_{window}"] = transformed["log_return"].rolling(window).mean()
            transformed[f"log_return_std_{window}"] = transformed["log_return"].rolling(window).std()
            transformed[f"volume_ma_{window}"] = transformed["volume"].rolling(window).mean()

        transformed["dataset"] = dataset_name
        return transformed

    def transform_bam_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy().sort_values("date").reset_index(drop=True)
        rate_cols = ["policy_rate", "reserve_requirement_ratio", "reserve_remuneration"]

        for col in rate_cols:
            transformed[f"{col}_diff"] = transformed[col].diff()
            transformed[f"{col}_change_flag"] = transformed[f"{col}_diff"].fillna(0).ne(0).astype(int)

        transformed["dataset"] = "BAM"
        return transformed

    def transform_all_datasets(self, cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        transformed_data: Dict[str, pd.DataFrame] = {}
        for dataset_name, df in cleaned_data.items():
            if dataset_name == "BAM":
                transformed_data[dataset_name] = self.transform_bam_dataset(df)
            else:
                transformed_data[dataset_name] = self.transform_market_dataset(df, dataset_name)
        return transformed_data

    @staticmethod
    def _prefix_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        rename_map = {
            col: f"{dataset_name.lower()}_{col}"
            for col in df.columns
            if col not in {"date", "dataset"}
        }
        prefixed = df.drop(columns=["dataset"], errors="ignore").rename(columns=rename_map)
        return prefixed

    def build_synchronized_dataset(self, transformed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if "MASI" not in transformed_data:
            raise ValueError("MASI dataset is required as the synchronization backbone.")

        combined = self._prefix_columns(transformed_data["MASI"], "MASI")

        for dataset_name, df in transformed_data.items():
            if dataset_name == "MASI":
                continue

            incoming = self._prefix_columns(df, dataset_name)
            if dataset_name == "BAM":
                incoming = incoming.sort_values("date")
                combined = pd.merge_asof(
                    combined.sort_values("date"),
                    incoming,
                    on="date",
                    direction="backward",
                )
                bam_cols = [col for col in combined.columns if col.startswith("bam_")]
                combined[bam_cols] = combined[bam_cols].ffill()
            else:
                combined = combined.merge(incoming, on="date", how="left")

        return combined.sort_values("date").reset_index(drop=True)

    def build_master_dataset(self, combined: pd.DataFrame) -> pd.DataFrame:
        """Create the final modeling table with only date and MASI log return."""
        master = combined.sort_values("date").copy()

        required_columns = ["date", "masi_log_return"]
        missing_columns = [col for col in required_columns if col not in master.columns]
        if missing_columns:
            raise ValueError(
                "Missing required columns for the final dataset: "
                + ", ".join(missing_columns)
            )

        model = master[required_columns].copy()
        model = model.dropna().reset_index(drop=True)
        return model

    def run_pipeline(self, raw_data_path: Path) -> Dict[str, object]:
        cleaner = DataCleaner()
        cleaned = cleaner.clean_all_datasets(raw_data_path)
        transformed = self.transform_all_datasets(cleaned)
        combined = self.build_synchronized_dataset(transformed)
        master = self.build_master_dataset(combined)
        return {
            "cleaned": cleaned,
            "transformed": transformed,
            "combined": combined,
            "master": master,
        }
