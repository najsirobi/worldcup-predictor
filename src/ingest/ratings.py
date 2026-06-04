"""Load team strength ratings (Elo, FIFA ranking)."""
import logging
from pathlib import Path

import pandas as pd

from .common import get_data_dir, load_csv_file, select_dataset_csv, validate_required_columns
from .team_names import canonicalize_team_series, load_team_name_map, normalize_team_whitespace

logger = logging.getLogger(__name__)

# Source keys used in team_name_map.csv for the rating sources.
ELO_SOURCE_KEY = "international_elo"
FIFA_SOURCE_KEY = "fifa_ranking"

ELO_CLEAN_COLUMNS = [
    "rating_date", "raw_team_name", "canonical_team_name", "country_code",
    "elo_rating", "source", "source_file",
]
FIFA_CLEAN_COLUMNS = [
    "ranking_date", "raw_team_name", "canonical_team_name", "country_code",
    "fifa_rank", "fifa_points", "source", "source_file",
]


def load_international_elo() -> pd.DataFrame:
    """Load international Elo ratings dataset."""
    raw_dir = get_data_dir("raw/kaggle") / "international_elo"
    elo_file = select_dataset_csv(raw_dir, preferred_names=["eloratings.csv"])
    logger.info(f"Loading international Elo from {elo_file}")

    df = load_csv_file(elo_file)
    df.columns = df.columns.str.lower().str.strip()

    return df


def load_fifa_world_ranking() -> pd.DataFrame:
    """Load FIFA world ranking dataset."""
    raw_dir = get_data_dir("raw/kaggle") / "fifa_world_ranking"
    # Snapshots are named fifa_ranking-YYYY-MM-DD.csv; the latest snapshot
    # contains the most complete history, so pick the last sorted file.
    ranking_file = select_dataset_csv(raw_dir, pick="last")
    logger.info(f"Loading FIFA world ranking from {ranking_file}")

    df = load_csv_file(ranking_file)
    df.columns = df.columns.str.lower().str.strip()

    return df


def get_rating_asof_date(df: pd.DataFrame, team: str, date: str, rating_col: str = "rating") -> float:
    """Get team rating as of a specific date (latest strictly before).

    This enforces no-future-leakage by using only ratings available before match date.
    """
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in dataframe")

    # Resolve the team identifier column explicitly. Different sources use
    # different names (Elo: 'team', FIFA ranking: 'country_full'); guessing via
    # df.get(...) silently returns None when none match, which then raises an
    # opaque error, so be explicit instead.
    team_col = next(
        (c for c in ("team", "country", "country_full") if c in df.columns), None
    )
    if team_col is None:
        raise ValueError(
            "Rating dataframe must have a 'team', 'country', or 'country_full' "
            f"column. Found: {list(df.columns)}"
        )

    # Filter for team and dates strictly before match date
    team_ratings = df[df[team_col] == team].copy()
    if team_ratings.empty:
        raise ValueError(f"Team '{team}' not found in ratings (column '{team_col}')")

    # Ensure date column is datetime
    if "date" not in team_ratings.columns and "rank_date" in team_ratings.columns:
        team_ratings = team_ratings.rename(columns={"rank_date": "date"})

    team_ratings["date"] = pd.to_datetime(team_ratings["date"])
    date_dt = pd.to_datetime(date)

    # Get ratings strictly before match date
    prior_ratings = team_ratings[team_ratings["date"] < date_dt]
    if prior_ratings.empty:
        raise ValueError(f"No ratings found for {team} before {date}")

    # Return latest prior rating
    latest = prior_ratings.sort_values("date").iloc[-1]
    return float(latest[rating_col])


def validate_rating_schema(df: pd.DataFrame) -> None:
    """Validate rating dataframe has expected columns."""
    # Elo typically has: date, team/country, elo, elo_win_pct, elo_draw_pct, elo_loss_pct, rank
    # FIFA ranking typically has: rank_date, country, total_points, rank, rank_change
    if "date" not in df.columns and "rank_date" not in df.columns:
        raise ValueError("Rating dataframe must have 'date' or 'rank_date' column")

    if not any(col in df.columns for col in ["team", "country", "country_full"]):
        raise ValueError(
            "Rating dataframe must have 'team', 'country', or 'country_full' column"
        )


# ---------------------------------------------------------------------------
# Cleaned rating tables (canonicalized, deduplicated, validated)
# ---------------------------------------------------------------------------

def _dedupe_ratings(df: pd.DataFrame, team_col: str, date_col: str, value_col: str) -> tuple:
    """Deterministically drop duplicate (canonical_team, date) rows.

    Keeps the last row after a stable sort by (team, date, value). Returns
    (deduped_df, n_dropped) so the caller can report it.
    """
    key = [team_col, date_col]
    dup_mask = df.duplicated(key, keep=False)
    n_dups = int(dup_mask.sum())
    if n_dups == 0:
        return df.reset_index(drop=True), 0
    df_sorted = df.sort_values([team_col, date_col, value_col], kind="stable")
    deduped = df_sorted.drop_duplicates(key, keep="last").reset_index(drop=True)
    n_dropped = len(df) - len(deduped)
    logger.warning(
        "Dropped %d duplicate (%s,%s) rating rows (kept last after stable sort)",
        n_dropped, team_col, date_col,
    )
    return deduped, n_dropped


def standardize_elo(raw: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: raw Elo rows -> cleaned Elo table (testable, no file I/O).

    - parses the mixed-format date column
    - normalizes whitespace (Elo uses non-breaking spaces); not name-merging
    - drops rows with unparseable date or null rating
    - canonicalizes team names (explicit-map-or-identity)
    - deduplicates (canonical_team_name, rating_date)
    """
    out = pd.DataFrame()
    out["rating_date"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    out["raw_team_name"] = normalize_team_whitespace(raw["team"])
    out["elo_rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    canonical, _ = canonicalize_team_series(out["raw_team_name"], ELO_SOURCE_KEY, mapping_df)
    out["canonical_team_name"] = canonical
    out["country_code"] = pd.NA  # Elo dataset carries no country code
    out["source"] = "international_elo"
    out["source_file"] = "eloratings.csv"

    out = out.dropna(subset=["rating_date", "elo_rating"]).copy()
    out, _ = _dedupe_ratings(out, "canonical_team_name", "rating_date", "elo_rating")
    return out[ELO_CLEAN_COLUMNS]


def standardize_fifa(raw: pd.DataFrame, mapping_df: pd.DataFrame, source_file: str = "fifa_ranking.csv") -> pd.DataFrame:
    """Pure transform: raw FIFA-ranking rows -> cleaned FIFA table (testable)."""
    out = pd.DataFrame()
    out["ranking_date"] = pd.to_datetime(raw["rank_date"], format="mixed", errors="coerce")
    out["raw_team_name"] = normalize_team_whitespace(raw["country_full"])
    out["country_code"] = raw["country_abrv"].astype(str).str.strip()
    out["fifa_rank"] = pd.to_numeric(raw["rank"], errors="coerce")
    out["fifa_points"] = pd.to_numeric(raw["total_points"], errors="coerce")
    canonical, _ = canonicalize_team_series(out["raw_team_name"], FIFA_SOURCE_KEY, mapping_df)
    out["canonical_team_name"] = canonical
    out["source"] = "fifa_world_ranking"
    out["source_file"] = source_file

    out = out.dropna(subset=["ranking_date"]).copy()
    out, _ = _dedupe_ratings(out, "canonical_team_name", "ranking_date", "fifa_points")
    return out[FIFA_CLEAN_COLUMNS]


def clean_elo_ratings(mapping_df: pd.DataFrame = None) -> pd.DataFrame:
    """Load + standardize the Elo ratings dataset."""
    raw = load_international_elo()  # columns: date, team, rating, change
    if mapping_df is None:
        mapping_df = load_team_name_map()
    return standardize_elo(raw, mapping_df)


def clean_fifa_rankings(mapping_df: pd.DataFrame = None, source_file: str = None) -> pd.DataFrame:
    """Load + standardize the FIFA ranking dataset (latest snapshot)."""
    raw = load_fifa_world_ranking()
    if mapping_df is None:
        mapping_df = load_team_name_map()
    if source_file is None:
        raw_dir = get_data_dir("raw/kaggle") / "fifa_world_ranking"
        source_file = select_dataset_csv(raw_dir, pick="last").name
    return standardize_fifa(raw, mapping_df, source_file)


def validate_clean_ratings(df: pd.DataFrame, date_col: str, value_cols: list) -> None:
    """Validate a cleaned rating table.

    - date column is datetime, no nulls
    - value columns numeric
    - canonical_team_name present (no nulls)
    - no duplicate (canonical_team_name, date)
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"'{date_col}' must be datetime")
    if df[date_col].isna().any():
        raise ValueError(f"'{date_col}' contains nulls")
    if df["canonical_team_name"].isna().any():
        raise ValueError("'canonical_team_name' contains nulls (required for production join)")
    for c in value_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be numeric")
    if df.duplicated(["canonical_team_name", date_col]).any():
        raise ValueError(
            f"duplicate (canonical_team_name, {date_col}) rows remain; resolve before join"
        )


# ---------------------------------------------------------------------------
# As-of join (strict no-future-leakage)
# ---------------------------------------------------------------------------

def asof_rating_join(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    match_team_col: str,
    value_cols: list,
    prefix: str,
    match_date_col: str = "date",
    rating_team_col: str = "canonical_team_name",
    rating_date_col: str = "rating_date",
) -> pd.DataFrame:
    """Attach the latest rating STRICTLY BEFORE each match date.

    Uses ``merge_asof(direction="backward", allow_exact_matches=False)`` which
    enforces ``rating_date < match_date`` (a rating ON the match date is NOT
    used, because it may already include that match's result). Unmatched teams
    yield NaN value columns (reported as missing, never guessed).

    Returns a DataFrame aligned to ``matches`` index with columns
    ``{prefix}_{value_col}...`` plus ``{prefix}_rating_date``.

    Raises ValueError if ``ratings`` contains duplicate (team, date) keys, so a
    non-deterministic as-of result can never slip through.
    """
    if ratings.duplicated([rating_team_col, rating_date_col]).any():
        raise ValueError(
            f"ratings contain duplicate ({rating_team_col}, {rating_date_col}) rows; "
            "deduplicate before as-of join to keep the result deterministic"
        )

    left = matches[[match_date_col, match_team_col]].copy()
    left["_orig"] = range(len(left))
    left = left.rename(columns={match_date_col: "_date", match_team_col: "_team"})
    left["_date"] = pd.to_datetime(left["_date"])
    left = left.sort_values("_date", kind="stable")

    right = ratings[[rating_team_col, rating_date_col] + value_cols].copy()
    right = right.rename(columns={rating_team_col: "_team", rating_date_col: "_date"})
    right["_date"] = pd.to_datetime(right["_date"])
    right["_rating_date"] = right["_date"]
    right = right.sort_values("_date", kind="stable")

    merged = pd.merge_asof(
        left, right, on="_date", by="_team",
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("_orig")

    result = pd.DataFrame(index=matches.index)
    for c in value_cols:
        result[f"{prefix}_{c}"] = merged[c].values
    result[f"{prefix}_rating_date"] = merged["_rating_date"].values
    return result
