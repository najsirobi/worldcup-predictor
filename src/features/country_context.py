"""WC2026 country-context feature helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDICATOR_COLUMN_MAP = {
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita_current_usd": "NY.GDP.PCAP.CD",
    "population_total": "SP.POP.TOTL",
    "education_spend_pct_gdp": "SE.XPD.TOTL.GD.ZS",
    "rd_spend_pct_gdp": "GB.XPD.RSDV.GD.ZS",
    "urbanisation_pct": "SP.URB.TOTL.IN.ZS",
    "life_expectancy": "SP.DYN.LE00.IN",
}


def latest_value_before_year(
    wb_frame: pd.DataFrame,
    world_bank_code: str | None,
    indicator_code: str,
    cutoff_year: int,
) -> tuple[float | None, int | None]:
    """Return the latest non-null indicator value strictly before a cutoff year."""
    if not world_bank_code or indicator_code not in wb_frame.columns:
        return None, None

    subset = wb_frame[
        wb_frame["country_code"].eq(world_bank_code)
        & wb_frame["year"].lt(cutoff_year)
        & ~wb_frame["is_aggregate"].fillna(False)
    ][["year", indicator_code]].dropna(subset=[indicator_code])

    if subset.empty:
        return None, None

    latest = subset.sort_values("year").iloc[-1]
    return float(latest[indicator_code]), int(latest["year"])


def _safe_log10(value: float | None) -> float | None:
    if value is None or pd.isna(value) or value <= 0:
        return np.nan
    return float(np.log10(value))


def build_country_context_features(
    team_frame: pd.DataFrame,
    mapping_frame: pd.DataFrame,
    wb_frame: pd.DataFrame,
    *,
    tournament_year: int = 2026,
) -> pd.DataFrame:
    """Build one WC2026 country-context feature row per team."""
    mapping_cols = [
        "canonical_team",
        "fifa_code",
        "world_bank_code",
        "world_bank_country_name",
        "is_proxy_mapping",
    ]
    merged = team_frame.merge(
        mapping_frame[mapping_cols].drop_duplicates("canonical_team"),
        left_on="team",
        right_on="canonical_team",
        how="left",
    )
    merged["tournament_year"] = tournament_year

    for feature_name, indicator_code in INDICATOR_COLUMN_MAP.items():
        values = []
        years = []
        for code in merged["world_bank_code"]:
            value, year = latest_value_before_year(wb_frame, code, indicator_code, tournament_year)
            values.append(value)
            years.append(year)
        merged[feature_name] = values
        merged[f"{feature_name}_value_year"] = years
        merged[f"{feature_name}_missing"] = merged[feature_name].isna()

    merged["log_gdp"] = merged["gdp_current_usd"].map(_safe_log10)
    merged["log_population"] = merged["population_total"].map(_safe_log10)
    merged["log_gdp_per_capita"] = merged["gdp_per_capita_current_usd"].map(_safe_log10)
    merged["has_country_context_mapping"] = merged["world_bank_code"].notna()

    return merged[
        [
            "team",
            "group",
            "fifa_code",
            "tournament_year",
            "world_bank_code",
            "world_bank_country_name",
            "is_proxy_mapping",
            "log_gdp",
            "log_population",
            "log_gdp_per_capita",
            "education_spend_pct_gdp",
            "rd_spend_pct_gdp",
            "urbanisation_pct",
            "life_expectancy",
            "gdp_current_usd_value_year",
            "gdp_per_capita_current_usd_value_year",
            "population_total_value_year",
            "education_spend_pct_gdp_value_year",
            "rd_spend_pct_gdp_value_year",
            "urbanisation_pct_value_year",
            "life_expectancy_value_year",
            "gdp_current_usd_missing",
            "gdp_per_capita_current_usd_missing",
            "population_total_missing",
            "education_spend_pct_gdp_missing",
            "rd_spend_pct_gdp_missing",
            "urbanisation_pct_missing",
            "life_expectancy_missing",
            "has_country_context_mapping",
        ]
    ].rename(
        columns={
            "gdp_current_usd_value_year": "gdp_value_year",
            "gdp_per_capita_current_usd_value_year": "gdp_per_capita_value_year",
            "population_total_value_year": "population_value_year",
            "urbanisation_pct_value_year": "urbanisation_value_year",
            "life_expectancy_value_year": "life_expectancy_value_year",
        }
    )


def load_country_context_features(path: str | Path) -> pd.DataFrame:
    """Load a previously built country-context feature parquet."""
    return pd.read_parquet(Path(path))
