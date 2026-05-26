"""Execute the end-to-end data preparation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.data.transforming import DataTransformer


def _write_outputs(outputs: dict, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = processed_dir / "cleaned"
    transformed_dir = processed_dir / "transformed"
    final_dir = processed_dir / "final"

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    transformed_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, frame in outputs["cleaned"].items():
        frame.to_csv(cleaned_dir / f"{dataset_name.lower()}_cleaned.csv", index=False)

    for dataset_name, frame in outputs["transformed"].items():
        frame.to_csv(transformed_dir / f"{dataset_name.lower()}_transformed.csv", index=False)

    outputs["combined"].to_csv(final_dir / "combined_synchronized_dataset.csv", index=False)
    outputs["master"].to_csv(final_dir / "master_dataset.csv", index=False)


def main(output_dir: Optional[Path] = None) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "data" / "raw"
    pipeline = DataTransformer()
    outputs = pipeline.run_pipeline(raw_data_path)
    target_dir = output_dir or (project_root / "data" / "processed")

    try:
        _write_outputs(outputs, target_dir)
    except PermissionError:
        if output_dir is not None:
            raise
        fallback_dir = project_root
        _write_outputs(outputs, fallback_dir)

    return outputs


if __name__ == "__main__":
    main()
