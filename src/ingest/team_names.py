"""Team name mapping and normalization."""
import logging
from pathlib import Path

import pandas as pd

from .common import get_data_dir, load_csv_file, validate_required_columns

logger = logging.getLogger(__name__)


def load_team_name_map() -> pd.DataFrame:
    """Load team name mapping from reference file."""
    mapping_path = get_data_dir("reference") / "team_name_map.csv"
    if not mapping_path.exists():
        logger.warning(f"Team name map not found at {mapping_path}. Returning empty map.")
        return pd.DataFrame(columns=["source", "raw_name", "canonical_team_name", "country_code", "notes"])

    df = load_csv_file(mapping_path)
    validate_required_columns(
        df,
        ["source", "raw_name", "canonical_team_name", "country_code"],
        name="team_name_map"
    )
    return df


def create_team_name_normalizer(mapping_df: pd.DataFrame) -> dict:
    """Create a normalizer function from mapping dataframe."""
    normalizer = {}
    for _, row in mapping_df.iterrows():
        source = row["source"]
        raw_name = row["raw_name"]
        canonical = row["canonical_team_name"]
        if source not in normalizer:
            normalizer[source] = {}
        if raw_name in normalizer[source] and normalizer[source][raw_name] != canonical:
            raise ValueError(
                f"Conflicting team-name mapping for '{raw_name}' in source "
                f"'{source}': '{normalizer[source][raw_name]}' vs '{canonical}'. "
                "Resolve the duplicate row in team_name_map.csv."
            )
        normalizer[source][raw_name] = canonical
    return normalizer


def normalize_team_whitespace(series):
    """Normalize whitespace in a team-name Series (data hygiene, not merging).

    The Elo dataset uses non-breaking spaces (U+00A0) inside multi-word names
    (e.g. "South\\xa0Africa"), which would otherwise fail to exact-match the
    backbone/FIFA names that use ordinary spaces. This only canonicalizes
    whitespace characters; it never changes the actual words.
    """
    return (
        series.astype(str)
        .str.replace(" ", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def canonicalize_team_series(series, source: str, mapping_df=None):
    """Map a Series of raw team names to canonical names for a given source.

    Resolution is **explicit-map-or-identity**: if an explicit row exists in
    team_name_map.csv for (source, raw_name) the mapped canonical is used;
    otherwise the raw name is passed through unchanged (identity).

    Identity passthrough is NOT a guess — it keeps the literal source name, so
    only names that are *byte-identical across sources* will join. Aliases
    (e.g. "Cape Verde" vs "Cabo Verde") therefore do NOT silently merge; they
    stay distinct until an explicit mapping is added. Returns (canonical_series,
    used_explicit_map_bool_series).
    """
    if mapping_df is None:
        mapping_df = load_team_name_map()
    normalizer = create_team_name_normalizer(mapping_df)
    src_map = normalizer.get(source, {})
    canonical = series.map(lambda x: src_map.get(x, x))
    used_map = series.map(lambda x: x in src_map)
    return canonical, used_map


def normalize_team_name(team_name: str, source: str, normalizer: dict) -> str:
    """Normalize a single team name using the mapping.

    If team name is not in mapping, raises ValueError (no silent normalization).
    """
    if source not in normalizer:
        raise ValueError(f"Source '{source}' not found in team name mapping. Available sources: {list(normalizer.keys())}")

    if team_name not in normalizer[source]:
        raise ValueError(
            f"Team '{team_name}' from source '{source}' not found in mapping. "
            f"Available teams for {source}: {list(normalizer[source].keys())}"
        )

    return normalizer[source][team_name]
