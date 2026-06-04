"""Load World Cup specific data."""
import logging
from pathlib import Path

import pandas as pd

from .common import get_data_dir, load_csv_file, select_dataset_csv, validate_required_columns

logger = logging.getLogger(__name__)


def load_world_cup_history() -> pd.DataFrame:
    """Load historical World Cup matches."""
    raw_dir = get_data_dir("raw/kaggle") / "world_cup_history"
    # Directory also contains world_cup.csv (tournament summary) and a FIFA
    # ranking snapshot; select the historical match file explicitly.
    matches_file = select_dataset_csv(
        raw_dir, preferred_names=["matches_1930_2022.csv", "world_cup.csv"]
    )
    df = load_csv_file(matches_file)
    df.columns = df.columns.str.lower().str.strip()

    return df


def load_world_cup_database() -> dict:
    """Load World Cup database (may contain multiple tables)."""
    raw_dir = get_data_dir("raw/kaggle") / "world_cup_database"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    # Load all CSV files into a dict
    tables = {}
    for csv_file in csv_files:
        table_name = csv_file.stem
        logger.info(f"Loading World Cup database table: {table_name}")
        tables[table_name] = load_csv_file(csv_file)

    return tables


def load_fifa_2026_fixtures() -> pd.DataFrame:
    """Load 2026 World Cup fixtures from interim or manual template."""
    # Check interim first (if parsing succeeded)
    interim_path = get_data_dir("interim") / "fifa_2026_fixtures_detected.csv"
    if interim_path.exists():
        logger.info(f"Loading parsed FIFA 2026 fixtures from {interim_path}")
        return load_csv_file(interim_path)

    # Fall back to manual template
    manual_path = get_data_dir("raw/manual") / "worldcup_2026_fixtures_manual.csv"
    if manual_path.exists():
        logger.info(f"Loading manual FIFA 2026 fixtures from {manual_path}")
        return load_csv_file(manual_path)

    raise FileNotFoundError(
        "2026 fixtures not found. Expected either:\n"
        f"  - {interim_path} (parsed from official FIFA page)\n"
        f"  - {manual_path} (manual template to be filled)"
    )


def load_fifa_2026_squads() -> pd.DataFrame:
    """Load 2026 World Cup squads."""
    interim_path = get_data_dir("interim") / "fifa_2026_squads_detected.csv"
    if interim_path.exists():
        logger.info(f"Loading parsed FIFA 2026 squads from {interim_path}")
        return load_csv_file(interim_path)

    manual_path = get_data_dir("raw/manual") / "worldcup_2026_squads_manual.csv"
    if manual_path.exists():
        logger.info(f"Loading manual FIFA 2026 squads from {manual_path}")
        return load_csv_file(manual_path)

    raise FileNotFoundError(
        "2026 squads not found. Expected either:\n"
        f"  - {interim_path} (parsed from official FIFA page)\n"
        f"  - {manual_path} (manual template to be filled)"
    )


def load_fifa_2026_teams() -> pd.DataFrame:
    """Load 2026 World Cup teams."""
    interim_path = get_data_dir("interim") / "fifa_2026_teams_detected.csv"
    if interim_path.exists():
        logger.info(f"Loading parsed FIFA 2026 teams from {interim_path}")
        return load_csv_file(interim_path)

    manual_path = get_data_dir("raw/manual") / "worldcup_2026_teams_manual.csv"
    if manual_path.exists():
        logger.info(f"Loading manual FIFA 2026 teams from {manual_path}")
        return load_csv_file(manual_path)

    raise FileNotFoundError(
        "2026 teams not found. Expected either:\n"
        f"  - {interim_path} (parsed from official FIFA page)\n"
        f"  - {manual_path} (manual template to be filled)"
    )
