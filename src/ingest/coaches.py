"""Load coach/manager data."""
import logging

import pandas as pd

from .common import get_data_dir, load_csv_file

logger = logging.getLogger(__name__)


def load_world_cup_managers() -> pd.DataFrame:
    """Load World Cup managers from World Cup database.

    This requires the world_cup_database to be downloaded first.
    """
    raw_dir = get_data_dir("raw/kaggle") / "world_cup_database"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    # Look for managers or coaches table
    csv_files = {f.stem: f for f in raw_dir.glob("*.csv")}
    possible_names = ["managers", "coaches", "manager", "coach"]

    manager_file = None
    for name in possible_names:
        if name in csv_files:
            manager_file = csv_files[name]
            break

    if manager_file is None:
        available = list(csv_files.keys())
        logger.warning(
            f"No manager/coach file found in {raw_dir}. "
            f"Available files: {available}"
        )
        return pd.DataFrame()

    logger.info(f"Loading manager data from {manager_file}")
    df = load_csv_file(manager_file)
    df.columns = df.columns.str.lower().str.strip()

    return df


def validate_coach_schema(df: pd.DataFrame) -> None:
    """Validate coach dataframe schema."""
    if df.empty:
        logger.warning("Coach dataframe is empty")
        return

    # Typical coach columns: team, name, year, dob, nationality, etc.
    expected_patterns = [
        {"team", "name"},
        {"country", "manager"},
        {"country", "coach"},
    ]

    cols_set = set(df.columns)
    if not any(pattern.issubset(cols_set) for pattern in expected_patterns):
        logger.warning(
            f"Coach dataframe schema may be unexpected. "
            f"Columns: {list(df.columns)}"
        )
