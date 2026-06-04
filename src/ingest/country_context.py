"""Load country context data (World Bank, manual imports)."""
import logging
import json
from pathlib import Path

import pandas as pd

from .common import get_data_dir, load_csv_file, validate_required_columns

logger = logging.getLogger(__name__)


def load_world_bank_country_context() -> pd.DataFrame:
    """Load cleaned World Bank country context data."""
    interim_path = get_data_dir("interim") / "world_bank_country_context.csv"
    if not interim_path.exists():
        logger.warning(f"World Bank country context not found at {interim_path}")
        return pd.DataFrame()

    df = load_csv_file(interim_path)
    logger.info(f"Loaded World Bank context: {len(df)} rows, {len(df.columns)} columns")

    return df


def load_world_bank_raw_responses() -> dict:
    """Load raw World Bank API responses (JSON)."""
    raw_dir = get_data_dir("raw/world_bank") / "country_context_raw"
    if not raw_dir.exists():
        logger.warning(f"World Bank raw directory not found at {raw_dir}")
        return {}

    responses = {}
    json_files = list(raw_dir.glob("*.json"))
    for json_file in json_files:
        with open(json_file) as f:
            indicator = json_file.stem
            responses[indicator] = json.load(f)
            logger.info(f"Loaded World Bank response for {indicator}")

    return responses


def load_google_trends_football_interest() -> pd.DataFrame:
    """Load Google Trends football interest data (manual template)."""
    manual_path = get_data_dir("raw/manual") / "google_trends_football_interest_manual.csv"
    if not manual_path.exists():
        logger.warning(f"Google Trends data not found at {manual_path}")
        return pd.DataFrame()

    df = load_csv_file(manual_path)
    logger.info(f"Loaded Google Trends context: {len(df)} rows")

    return df


def load_fifa_professional_football_landscape() -> pd.DataFrame:
    """Load FIFA professional football landscape data (manual)."""
    manual_path = get_data_dir("raw/manual") / "fifa_professional_football_landscape_manual.csv"
    if not manual_path.exists():
        logger.warning(f"FIFA professional football landscape not found at {manual_path}")
        return pd.DataFrame()

    df = load_csv_file(manual_path)
    logger.info(f"Loaded FIFA professional landscape: {len(df)} rows")

    return df


def load_fifa_forward_funding() -> pd.DataFrame:
    """Load FIFA Forward funding data (manual)."""
    manual_path = get_data_dir("raw/manual") / "fifa_forward_funding_manual.csv"
    if not manual_path.exists():
        logger.warning(f"FIFA Forward funding data not found at {manual_path}")
        return pd.DataFrame()

    df = load_csv_file(manual_path)
    logger.info(f"Loaded FIFA Forward funding: {len(df)} rows")

    return df


def validate_country_context_schema(df: pd.DataFrame, source_type: str) -> None:
    """Validate country context dataframe schema."""
    if df.empty:
        logger.warning(f"Country context dataframe ({source_type}) is empty")
        return

    # All context data should have country identifier
    if not any(col in df.columns for col in ["country", "country_code", "country_name"]):
        raise ValueError(
            f"Country context dataframe ({source_type}) must have "
            "country, country_code, or country_name column"
        )
