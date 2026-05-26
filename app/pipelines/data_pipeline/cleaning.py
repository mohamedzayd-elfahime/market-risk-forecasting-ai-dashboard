"""MASI-only cleaning utilities for Investing.com historical files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class DataCleaner:
    """Clean MASI raw files before history consolidation and transformation."""

    MASI_DATASET_NAME = "MASI"

    @staticmethod
    def _clean_columns(columns: Iterable[str]) -> list[str]:
        return [str(col).strip() for col in columns]

    @staticmethod
    def _parse_comma_number(value: object) -> float:
        """Parse Investing.com values such as 17.243,58."""
        if pd.isna(value) or value == "":
            return np.nan
        cleaned = str(value).strip().replace(".", "").replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    @staticmethod
    def _parse_volume_with_suffix(value: object) -> float:
        """Parse values like 461.93K, 240,23K or 1,10M."""
        if pd.isna(value) or value == "":
            return np.nan

        raw_value = str(value).strip()
        cleaned = raw_value
        multiplier = 1
        if cleaned.endswith("K"):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]

        cleaned = cleaned.replace(".", "").replace(",", ".")
        if "." in raw_value and "," not in raw_value:
            cleaned = raw_value.rstrip("KM")
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return np.nan

    @staticmethod
    def _parse_percent(value: object) -> float:
        if pd.isna(value) or value == "":
            return np.nan
        raw_value = str(value).strip()
        cleaned = raw_value.rstrip("%").replace(".", "").replace(",", ".")
        if "." in raw_value and "," not in raw_value:
            cleaned = raw_value.rstrip("%")
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    @staticmethod
    def _finalize_masi_frame(df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        output["date"] = pd.to_datetime(output["date"], errors="coerce")

        numeric_cols = [col for col in output.columns if col not in {"date", "dataset"}]
        for col in numeric_cols:
            output[col] = pd.to_numeric(output[col], errors="coerce")

        output = output.dropna(subset=["date"])
        output = output.drop_duplicates(subset=["date"], keep="last")
        output = output.sort_values("date").reset_index(drop=True)
        output["dataset"] = DataCleaner.MASI_DATASET_NAME
        return output

    def clean_MASI(self, filepath: Path) -> pd.DataFrame:
        """Clean a MASI CSV or Excel file exported from Investing.com."""
        filepath = Path(filepath)
        if filepath.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        df.columns = self._clean_columns(df.columns)

        required_columns = {"Date", "Dernier", "Ouv.", "Plus Haut", "Plus Bas", "Vol.", "Variation %"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Missing MASI Investing.com columns in {filepath}: "
                + ", ".join(missing_columns)
            )

        cleaned = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce"),
                "open": df["Ouv."].apply(self._parse_comma_number),
                "high": df["Plus Haut"].apply(self._parse_comma_number),
                "low": df["Plus Bas"].apply(self._parse_comma_number),
                "close": df["Dernier"].apply(self._parse_comma_number),
                "volume": df["Vol."].apply(self._parse_volume_with_suffix),
                "change_pct": df["Variation %"].apply(self._parse_percent),
            }
        )

        if cleaned["volume"].isna().all():
            cleaned["volume"] = np.nan

        return self._finalize_masi_frame(cleaned)

    def clean_masi_history(self, filepath: Path) -> pd.DataFrame:
        """Load an existing cleaned MASI history or clean a raw Investing.com file."""
        filepath = Path(filepath)
        df = pd.read_csv(filepath)
        df.columns = self._clean_columns(df.columns)

        if "date" in df.columns:
            return self._finalize_masi_frame(df)

        return self.clean_MASI(filepath)
