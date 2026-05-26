"""MASI-only transformations for the forecasting dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataTransformer:
    """Transform the consolidated MASI history into a model-ready dataset."""

    CORE_WINDOWS = (5, 10, 21)

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

    def transform_masi_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
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

        transformed["dataset"] = "MASI"
        return transformed

    @staticmethod
    def build_master_dataset(transformed: pd.DataFrame) -> pd.DataFrame:
        """Create a MASI-only final dataset without external variables."""
        master = transformed.sort_values("date").copy()
        master = master.drop_duplicates(subset=["date"], keep="last")

        if "log_return" not in master.columns:
            raise ValueError("Missing required MASI transformation column: log_return")

        master = master.rename(
            columns={
                "open": "masi_open",
                "high": "masi_high",
                "low": "masi_low",
                "close": "masi_close",
                "volume": "masi_volume",
                "change_pct": "masi_change_pct",
                "simple_return": "masi_simple_return",
                "log_return": "masi_log_return",
                "log_open_close": "masi_log_open_close",
                "log_high_low": "masi_log_high_low",
                "price_range_pct": "masi_price_range_pct",
                "close_to_open_ratio": "masi_close_to_open_ratio",
                "volume_log_change": "masi_volume_log_change",
                "log_return_mean_5": "masi_log_return_mean_5",
                "log_return_std_5": "masi_log_return_std_5",
                "volume_ma_5": "masi_volume_ma_5",
                "log_return_mean_10": "masi_log_return_mean_10",
                "log_return_std_10": "masi_log_return_std_10",
                "volume_ma_10": "masi_volume_ma_10",
                "log_return_mean_21": "masi_log_return_mean_21",
                "log_return_std_21": "masi_log_return_std_21",
                "volume_ma_21": "masi_volume_ma_21",
            }
        )
        master = master.drop(
            columns=[
                "dataset",
                "masi_volume",
                "masi_volume_log_change",
                "masi_volume_ma_5",
                "masi_volume_ma_10",
                "masi_volume_ma_21",
            ],
            errors="ignore",
        )
        return master.reset_index(drop=True)

    def run_pipeline(self, masi_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
        transformed = self.transform_masi_dataset(masi_history)
        master = self.build_master_dataset(transformed)
        return {
            "cleaned": masi_history,
            "transformed": transformed,
            "master": master,
        }
