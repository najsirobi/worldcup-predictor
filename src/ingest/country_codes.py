"""Country code mapping and normalization."""
import logging

import pandas as pd

from .common import get_data_dir, load_csv_file, validate_required_columns

logger = logging.getLogger(__name__)

_LEGACY_EMPTY_COLUMNS = [
    "source",
    "raw_country_name",
    "canonical_country_name",
    "fifa_code",
    "iso2",
    "iso3",
    "world_bank_code",
    "notes",
]

_NEW_TO_LEGACY = {
    "team_name": "raw_country_name",
    "canonical_team": "canonical_country_name",
    "country_code_iso3": "iso3",
}


def load_country_code_map() -> pd.DataFrame:
    """Load country code mapping from reference file."""
    mapping_path = get_data_dir("reference") / "country_code_map.csv"
    if not mapping_path.exists():
        logger.warning(f"Country code map not found at {mapping_path}. Returning empty map.")
        return pd.DataFrame(columns=_LEGACY_EMPTY_COLUMNS)

    df = load_csv_file(mapping_path)
    for new_col, legacy_col in _NEW_TO_LEGACY.items():
        if legacy_col not in df.columns and new_col in df.columns:
            df[legacy_col] = df[new_col]
    if "raw_country_name" not in df.columns and "world_bank_country_name" in df.columns:
        df["raw_country_name"] = df["world_bank_country_name"]
    validate_required_columns(
        df,
        ["source", "raw_country_name", "canonical_country_name"],
        name="country_code_map"
    )
    return df


def create_country_code_normalizer(mapping_df: pd.DataFrame) -> dict:
    """Create a normalizer function from mapping dataframe."""
    normalizer = {}
    for _, row in mapping_df.iterrows():
        source = row["source"]
        raw_name = row["raw_country_name"]
        canonical = row["canonical_country_name"]
        if source not in normalizer:
            normalizer[source] = {}
        if raw_name in normalizer[source] and normalizer[source][raw_name] != canonical:
            raise ValueError(
                f"Conflicting country mapping for '{raw_name}' in source "
                f"'{source}': '{normalizer[source][raw_name]}' vs '{canonical}'. "
                "Resolve the duplicate row in country_code_map.csv."
            )
        normalizer[source][raw_name] = canonical
    return normalizer


def normalize_country_name(country_name: str, source: str, normalizer: dict) -> str:
    """Normalize a single country name using the mapping.

    If country name is not in mapping, raises ValueError (no silent normalization).
    """
    if source not in normalizer:
        raise ValueError(
            f"Source '{source}' not found in country code mapping. "
            f"Available sources: {list(normalizer.keys())}"
        )

    if country_name not in normalizer[source]:
        raise ValueError(
            f"Country '{country_name}' from source '{source}' not found in mapping. "
            f"Available countries for {source}: {list(normalizer[source].keys())}"
        )

    return normalizer[source][country_name]


def get_country_code(country_name: str, code_type: str = "fifa_code", mapping_df: pd.DataFrame = None) -> str:
    """Get a specific country code (fifa_code, iso2, iso3, world_bank_code) for a country."""
    if mapping_df is None:
        mapping_df = load_country_code_map()

    if code_type == "iso3" and "iso3" not in mapping_df.columns and "country_code_iso3" in mapping_df.columns:
        code_type = "country_code_iso3"

    if code_type not in mapping_df.columns:
        raise ValueError(f"Code type '{code_type}' not found. Available: {list(mapping_df.columns)}")

    canonical_col = "canonical_country_name" if "canonical_country_name" in mapping_df.columns else "canonical_team"
    matches = mapping_df[mapping_df[canonical_col] == country_name]
    if matches.empty:
        raise ValueError(f"Country '{country_name}' not found in mapping.")

    code_value = matches.iloc[0][code_type]
    if pd.isna(code_value):
        return None
    return code_value
