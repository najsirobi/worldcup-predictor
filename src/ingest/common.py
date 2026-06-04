"""Common utilities for data ingestion."""
import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_data_dir(subdir: str = "") -> Path:
    """Get data directory path."""
    base = Path(__file__).parent.parent.parent / "data"
    if subdir:
        return base / subdir
    return base


def load_csv_file(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load CSV file with standard error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load CSV {filepath}: {e}")


def load_parquet_file(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load Parquet file with standard error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    try:
        return pd.read_parquet(filepath, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load Parquet {filepath}: {e}")


def load_excel_file(filepath: Path, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
    """Load Excel file with standard error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    try:
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load Excel {filepath}: {e}")


def select_dataset_csv(
    raw_dir: Path,
    preferred_names: Optional[list[str]] = None,
    pick: str = "first",
) -> Path:
    """Deterministically select one CSV from a dataset directory.

    ``Path.glob`` returns files in arbitrary filesystem order, so selecting
    ``list(raw_dir.glob("*.csv"))[0]`` can silently load the wrong file when a
    dataset ships several CSVs with different schemas (e.g. ``goalscorers.csv``
    instead of ``results.csv``). This helper prefers an explicitly named file
    and otherwise falls back to a deterministic, sorted choice while warning.

    Args:
        raw_dir: directory to search for ``*.csv`` files.
        preferred_names: file names to look for, in priority order.
        pick: ``"first"`` or ``"last"`` of the sorted list when no preferred
            name matches (use ``"last"`` to pick the latest dated snapshot).
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    if preferred_names:
        by_name = {f.name: f for f in csv_files}
        for name in preferred_names:
            if name in by_name:
                return by_name[name]
        if len(csv_files) > 1:
            logger.warning(
                "None of the preferred files %s found in %s; falling back to "
                "%s of %d CSV file(s): %s",
                preferred_names, raw_dir, pick, len(csv_files),
                [f.name for f in csv_files],
            )

    chosen = csv_files[-1] if pick == "last" else csv_files[0]
    if len(csv_files) > 1 and not preferred_names:
        logger.warning(
            "Multiple CSV files found in %s: %s. Selecting '%s' (%s). "
            "Pass preferred_names to disambiguate.",
            raw_dir, [f.name for f in csv_files], chosen.name, pick,
        )
    return chosen


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], name: str = "dataframe") -> None:
    """Validate that required columns exist in dataframe."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def save_csv_file(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """Save dataframe to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved CSV: {filepath}")


def save_parquet_file(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """Save dataframe to Parquet."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=index)
    logger.info(f"Saved Parquet: {filepath}")
