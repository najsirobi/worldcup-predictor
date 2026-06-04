"""Load international match results."""
import logging
from pathlib import Path

import pandas as pd

from .common import get_data_dir, load_csv_file, select_dataset_csv, validate_required_columns
from .team_names import create_team_name_normalizer, load_team_name_map

logger = logging.getLogger(__name__)

# Canonical column set for the standardized historical match backbone.
BACKBONE_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def load_international_results(dataset_name: str = "international_results") -> pd.DataFrame:
    """Load international match results from Kaggle dataset.

    Expected columns (may vary by source):
    - date: match date
    - home_team: home team name
    - away_team: away team name
    - home_score: goals scored by home team
    - away_score: goals scored by away team
    """
    raw_dir = get_data_dir("raw/kaggle") / dataset_name
    # This dataset also ships goalscorers.csv / shootouts.csv / former_names.csv,
    # which have different schemas; select results.csv explicitly.
    results_file = select_dataset_csv(raw_dir, preferred_names=["results.csv"])
    logger.info(f"Loading international results from {results_file}")

    df = load_csv_file(results_file)

    # Standardize column names if needed
    df.columns = df.columns.str.lower().str.strip()

    return df


def validate_match_schema(df: pd.DataFrame) -> None:
    """Validate that match dataframe has expected schema."""
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    validate_required_columns(df, required, name="match_results")

    # Type checks
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type")

    if not pd.api.types.is_numeric_dtype(df["home_score"]):
        raise ValueError("'home_score' must be numeric")

    if not pd.api.types.is_numeric_dtype(df["away_score"]):
        raise ValueError("'away_score' must be numeric")


def get_matches_before_date(df: pd.DataFrame, date) -> pd.DataFrame:
    """Get all matches strictly before a given date (for feature engineering).

    Dates are compared as real timestamps, not lexicographically: comparing
    raw strings only works for zero-padded ISO dates and silently breaks for
    any other format, so both sides are parsed with pandas first.
    """
    cutoff = pd.to_datetime(date)
    match_dates = pd.to_datetime(df["date"])
    return df[match_dates < cutoff].copy()


def get_recent_form(df: pd.DataFrame, team: str, date, periods: int = 5) -> dict:
    """Get recent match results for a team before a given date (skeleton only).

    This is a placeholder function. Actual feature engineering happens in features/ module.
    """
    matches = get_matches_before_date(df, date)
    # Order by date so "recent" means most-recent-by-date, not by row position.
    matches = matches.assign(_d=pd.to_datetime(matches["date"])).sort_values("_d")
    home_matches = matches[matches["home_team"] == team].tail(periods)
    away_matches = matches[matches["away_team"] == team].tail(periods)

    return {
        "team": team,
        "date": date,
        "recent_home_matches": len(home_matches),
        "recent_away_matches": len(away_matches),
    }


# ---------------------------------------------------------------------------
# Historical match backbone (model-ready standardized table)
# ---------------------------------------------------------------------------

def coerce_match_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the standardized columns into proper dtypes.

    - date -> datetime64 (normalized to midnight)
    - home_score / away_score -> numeric (kept float here so NaN survives until
      unplayed fixtures are filtered out)
    - neutral -> boolean
    """
    validate_required_columns(df, BACKBONE_COLUMNS, name="international_results")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        n = int(out["date"].isna().sum())
        raise ValueError(f"{n} match rows have an unparseable 'date' value")
    out["date"] = out["date"].dt.normalize()
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    if out["neutral"].dtype != bool:
        out["neutral"] = (
            out["neutral"].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        )
    return out


def filter_played_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into played matches (valid scores) and unplayed/future fixtures.

    Rows with a null home or away score are scheduled-but-unplayed fixtures
    (e.g. future World Cup matches) and have no result/target, so they are
    excluded from the historical backbone and returned separately for reporting.
    """
    unplayed_mask = df["home_score"].isna() | df["away_score"].isna()
    played = df[~unplayed_mask].copy()
    dropped = df[unplayed_mask].copy()
    return played, dropped


def add_match_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target/label columns to a played-matches table.

    Assumes scores are non-null. Casts scores to int and derives:
    result_label, home_points, away_points, goal_diff, total_goals,
    home_goals, away_goals.
    """
    out = df.copy()
    if out["home_score"].isna().any() or out["away_score"].isna().any():
        raise ValueError("add_match_targets requires non-null scores; filter unplayed matches first")
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)

    home_win = out["home_score"] > out["away_score"]
    away_win = out["home_score"] < out["away_score"]
    draw = out["home_score"] == out["away_score"]

    out["result_label"] = "draw"
    out.loc[home_win, "result_label"] = "home_win"
    out.loc[away_win, "result_label"] = "away_win"

    out["home_points"] = 1
    out.loc[home_win, "home_points"] = 3
    out.loc[away_win, "home_points"] = 0
    out["away_points"] = 1
    out.loc[away_win, "away_points"] = 3
    out.loc[home_win, "away_points"] = 0

    out["goal_diff"] = out["home_score"] - out["away_score"]
    out["total_goals"] = out["home_score"] + out["away_score"]
    out["home_goals"] = out["home_score"]
    out["away_goals"] = out["away_score"]
    return out


def validate_clean_matches(df: pd.DataFrame) -> None:
    """Validate the standardized played-matches backbone.

    Raises ValueError with a clear message on the first failed invariant.
    """
    validate_required_columns(df, BACKBONE_COLUMNS, name="matches_clean")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type")
    if df["date"].isna().any():
        raise ValueError("'date' column contains null values")

    for col in ("home_score", "away_score"):
        s = df[col]
        if s.isna().any():
            raise ValueError(f"'{col}' contains null values (unplayed fixtures must be filtered out)")
        if not (s == s.astype(int)).all():
            raise ValueError(f"'{col}' must be integer-valued")
        if (s < 0).any():
            raise ValueError(f"'{col}' must be non-negative")

    for col in ("home_team", "away_team"):
        if df[col].isna().any():
            raise ValueError(f"'{col}' contains null team names")

    same = df["home_team"] == df["away_team"]
    if same.any():
        raise ValueError(f"{int(same.sum())} rows have home_team == away_team")

    if "tournament" in df.columns and df["tournament"].isna().any():
        raise ValueError("'tournament' contains null values")


def find_unmapped_teams(df: pd.DataFrame, source: str = "international_results") -> list[str]:
    """Return team names in the backbone that have no entry in team_name_map.csv.

    Reports missing mappings rather than guessing or silently merging names.
    Returns an empty list if the mapping has no rows for ``source`` (so the
    backbone build is never blocked), but the caller should surface the count.
    """
    mapping_df = load_team_name_map()
    normalizer = create_team_name_normalizer(mapping_df)
    known = set(normalizer.get(source, {}).keys())
    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())
    return sorted(teams - known)
