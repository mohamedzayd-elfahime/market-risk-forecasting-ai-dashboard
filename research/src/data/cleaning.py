"""Dataset-specific cleaning utilities."""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


MAIN_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class DataCleaner:
    """Clean each raw dataset independently before any cross-dataset merge."""

    CSV_FILE_MAP = {
        "MASI": "MASI.csv",
        "Brent": "Brent.csv",
        "EUR_MAD": "EUR_MAD.csv",
        "gold": "gold.csv",
        "ATW": "ATW -equities.csv",
        "BCP": "BCP -equities.csv",
        "IAM": "IAM -equities.csv",
        "LHM": "LHM -equities.csv",
        "MNG": "MNG -equities.csv",
        "TQM": "TQM -equities.csv",
    }

    @staticmethod
    def _clean_columns(columns: Iterable[str]) -> list[str]:
        return [str(col).strip() for col in columns]

    @staticmethod
    def _parse_comma_number(value: object) -> float:
        """Parse values like 17.243,58 or 240,23K after suffix removal."""
        if pd.isna(value) or value == "":
            return np.nan
        cleaned = str(value).strip().replace(".", "").replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    @staticmethod
    def _parse_dot_number(value: object) -> float:
        """Parse values already using decimal dots."""
        if pd.isna(value) or value == "":
            return np.nan
        cleaned = str(value).strip().replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    @staticmethod
    def _parse_volume_with_suffix(value: object) -> float:
        """Parse values like 461.93K, 240,23K or 1,10M."""
        if pd.isna(value) or value == "":
            return np.nan

        cleaned = str(value).strip()
        multiplier = 1
        if cleaned.endswith("K"):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]

        cleaned = cleaned.replace(".", "").replace(",", ".")
        if "." in str(value) and "," not in str(value):
            cleaned = str(value).strip().rstrip("KM")
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return np.nan

    @staticmethod
    def _parse_percent(value: object) -> float:
        if pd.isna(value) or value == "":
            return np.nan
        cleaned = str(value).strip().rstrip("%").replace(".", "").replace(",", ".")
        if "." in str(value) and "," not in str(value):
            cleaned = str(value).strip().rstrip("%")
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    @staticmethod
    def _finalize_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        output = df.copy()
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        numeric_cols = [col for col in output.columns if col != "date"]
        for col in numeric_cols:
            output[col] = pd.to_numeric(output[col], errors="coerce")

        output = output.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        output = output.sort_values("date").reset_index(drop=True)
        output["dataset"] = dataset_name
        return output

    def _clean_investing_csv(self, filepath: Path, dataset_name: str, date_format: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df.columns = self._clean_columns(df.columns)

        european_format = dataset_name not in {"Brent"}
        price_parser = self._parse_comma_number if european_format else self._parse_dot_number

        cleaned = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], format=date_format, errors="coerce"),
                "open": df["Ouv." if european_format else "Open"].apply(price_parser),
                "high": df["Plus Haut" if european_format else "High"].apply(price_parser),
                "low": df["Plus Bas" if european_format else "Low"].apply(price_parser),
                "close": df["Dernier" if european_format else "Price"].apply(price_parser),
                "volume": df["Vol."].apply(self._parse_volume_with_suffix),
                "change_pct": df["Variation %" if european_format else "Change %"].apply(self._parse_percent),
            }
        )

        if dataset_name in {"MASI", "EUR_MAD"}:
            cleaned["volume"] = np.nan

        return self._finalize_frame(cleaned, dataset_name)

    def clean_MASI(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="MASI", date_format="%d/%m/%Y")

    def clean_Brent(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="Brent", date_format="%m/%d/%Y")

    def clean_EUR_MAD(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="EUR_MAD", date_format="%d/%m/%Y")

    def clean_gold(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="gold", date_format="%d/%m/%Y")

    def clean_ATW(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="ATW", date_format="%d/%m/%Y")

    def clean_BCP(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="BCP", date_format="%d/%m/%Y")

    def clean_IAM(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="IAM", date_format="%d/%m/%Y")

    def clean_LHM(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="LHM", date_format="%d/%m/%Y")

    def clean_MNG(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="MNG", date_format="%d/%m/%Y")

    def clean_TQM(self, filepath: Path) -> pd.DataFrame:
        return self._clean_investing_csv(filepath, dataset_name="TQM", date_format="%d/%m/%Y")

    def clean_BAM(self, filepath: Path) -> pd.DataFrame:
        """Read BAM daily series directly from XLSX XML to avoid extra dependencies."""
        with zipfile.ZipFile(filepath) as archive:
            sheet_xml = archive.read("xl/worksheets/sheet1.xml")

        root = ET.fromstring(sheet_xml)
        rows = []
        for row in root.findall(".//main:sheetData/main:row", MAIN_NS):
            values = []
            for cell in row.findall("main:c", MAIN_NS):
                inline_text = cell.find("main:is/main:t", MAIN_NS)
                value = inline_text.text if inline_text is not None else ""
                values.append(value)
            if values:
                rows.append(values)

        if not rows:
            raise ValueError(f"No rows found in BAM workbook: {filepath}")

        header = self._clean_columns(rows[0])
        df = pd.DataFrame(rows[1:], columns=header)

        cleaned = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], errors="coerce"),
                "policy_rate": df["Taux directeur"].apply(self._parse_percent),
                "reserve_requirement_ratio": df["Ratio r\u00e9serve obligatoire"].apply(self._parse_percent),
                "reserve_remuneration": df["R\u00e9mun\u00e9ration r\u00e9serve"].apply(self._parse_percent),
            }
        )
        return self._finalize_frame(cleaned, "BAM")

    def clean_GPR(self, filepath: Path) -> pd.DataFrame:
        """Clean GPR index when an XLS engine is available."""
        df = pd.read_excel(filepath)
        df.columns = self._clean_columns(df.columns)

        cleaned = pd.DataFrame(
            {
                "date": pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d", errors="coerce"),
                "open": pd.to_numeric(df["GPRD"], errors="coerce"),
            }
        )
        cleaned["high"] = cleaned["open"]
        cleaned["low"] = cleaned["open"]
        cleaned["close"] = cleaned["open"]
        cleaned["volume"] = np.nan
        cleaned["change_pct"] = cleaned["close"].pct_change() * 100
        return self._finalize_frame(cleaned, "GPR")

    def clean_all_datasets(self, raw_data_path: Path) -> Dict[str, pd.DataFrame]:
        """Load and clean every dataset independently."""
        raw_data_path = Path(raw_data_path)
        cleaned_data: Dict[str, pd.DataFrame] = {}

        for dataset_name, filename in self.CSV_FILE_MAP.items():
            filepath = raw_data_path / filename
            if filepath.exists():
                cleaned_data[dataset_name] = getattr(self, f"clean_{dataset_name}")(filepath)

        bam_filepath = raw_data_path / "BAM.xlsx"
        if bam_filepath.exists():
            cleaned_data["BAM"] = self.clean_BAM(bam_filepath)

        gpr_filepath = raw_data_path / "GPR-index.xls"
        if gpr_filepath.exists():
            try:
                cleaned_data["GPR"] = self.clean_GPR(gpr_filepath)
            except ImportError as exc:
                warnings.warn(
                    "Skipping GPR cleaning because the Excel reader dependency is missing "
                    f"({exc}). Install xlrd to include GPR-index.xls."
                )

        return cleaned_data
