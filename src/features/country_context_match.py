"""Match-level, leakage-safe World Bank country-context features.

Builds per-side macro country-context values for each historical match and the
home-away difference features used by the country-context backtests.

Leakage rule (identical to the WC2026 readiness builder):
- For a match in calendar year Y, use the latest World Bank value strictly
  before Y (``year < Y``). Post-tournament / same-year values are never used.
- Missing values are left null and exposed through explicit missing flags. They
  are never filled with zero.

Proxy handling:
- ``country_code_map.csv`` marks England and Scotland as ``GBR`` proxies. Two
  variants are produced:
    1. ``all_with_proxy``  — England/Scotland use the GBR sovereign values and
       carry ``is_proxy_mapping = True``.
    2. ``direct_only_proxy_missing`` — England/Scotland are treated as missing
       for every country-context feature (their proxy values are dropped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.country_context import latest_value_before_year

# Feature name -> World Bank indicator code.
INDICATOR_FOR_FEATURE = {
    "log_gdp": "NY.GDP.MKTP.CD",
    "log_gdp_per_capita": "NY.GDP.PCAP.CD",
    "log_population": "SP.POP.TOTL",
    "urbanisation_pct": "SP.URB.TOTL.IN.ZS",
    "life_expectancy": "SP.DYN.LE00.IN",
    "education_spend_pct_gdp": "SE.XPD.TOTL.GD.ZS",
    "rd_spend_pct_gdp": "GB.XPD.RSDV.GD.ZS",
}

# Features stored as base-10 logs of the raw World Bank value.
LOG_FEATURES = {"log_gdp", "log_gdp_per_capita", "log_population"}

PRIMARY_FEATURES = [
    "log_gdp",
    "log_gdp_per_capita",
    "log_population",
    "urbanisation_pct",
    "life_expectancy",
]
SECONDARY_FEATURES = ["education_spend_pct_gdp", "rd_spend_pct_gdp"]
ALL_FEATURES = PRIMARY_FEATURES + SECONDARY_FEATURES

# Diff column name per feature (matches Task A naming for the core set).
DIFF_NAME = {
    "log_gdp": "log_gdp_diff",
    "log_gdp_per_capita": "log_gdp_per_capita_diff",
    "log_population": "log_population_diff",
    "urbanisation_pct": "urbanisation_diff",
    "life_expectancy": "life_expectancy_diff",
    "education_spend_pct_gdp": "education_spend_pct_gdp_diff",
    "rd_spend_pct_gdp": "rd_spend_pct_gdp_diff",
}

PROXY_TEAMS = {"England", "Scotland"}

IDENTITY_COLUMNS = ["date", "match_year", "tournament", "home_team", "away_team"]


def _safe_log10(value: float | None) -> float:
    if value is None or pd.isna(value) or value <= 0:
        return np.nan
    return float(np.log10(value))


def _team_code_map(mapping: pd.DataFrame) -> pd.DataFrame:
    cols = ["canonical_team", "world_bank_code", "is_proxy_mapping"]
    m = mapping[cols].drop_duplicates("canonical_team").copy()
    m["is_proxy_mapping"] = m["is_proxy_mapping"].fillna(False).astype(bool)
    return m


def _wb_clean(wb_frame: pd.DataFrame) -> pd.DataFrame:
    wb = wb_frame.copy()
    if "is_aggregate" not in wb.columns:
        wb["is_aggregate"] = False
    return wb


def _value_lookup(
    wb_frame: pd.DataFrame,
    code_year_pairs: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, tuple[float | None, int | None]]]:
    """Compute the latest-before value for every (code, cutoff_year) pair once."""
    cache: dict[tuple[str, int], dict[str, tuple[float | None, int | None]]] = {}
    for code, year in code_year_pairs:
        per_indicator: dict[str, tuple[float | None, int | None]] = {}
        for feature, indicator in INDICATOR_FOR_FEATURE.items():
            per_indicator[feature] = latest_value_before_year(
                wb_frame, code, indicator, year
            )
        cache[(code, year)] = per_indicator
    return cache


def _side_features(
    teams: pd.Series,
    years: pd.Series,
    team_codes: pd.DataFrame,
    cache: dict,
) -> pd.DataFrame:
    """Build the per-side country-context frame (all_with_proxy variant)."""
    side = pd.DataFrame({"team": teams.values, "match_year": years.values})
    side = side.merge(team_codes, left_on="team", right_on="canonical_team", how="left")
    side["is_proxy_mapping"] = side["is_proxy_mapping"].fillna(False).astype(bool)
    side["has_context"] = side["world_bank_code"].notna()

    feature_values: dict[str, list] = {f: [] for f in ALL_FEATURES}
    year_values: dict[str, list] = {f: [] for f in ALL_FEATURES}
    for code, year in zip(side["world_bank_code"], side["match_year"]):
        if pd.isna(code) or pd.isna(year):
            for f in ALL_FEATURES:
                feature_values[f].append(np.nan)
                year_values[f].append(np.nan)
            continue
        per_indicator = cache[(code, int(year))]
        for f in ALL_FEATURES:
            value, value_year = per_indicator[f]
            feature_values[f].append(
                _safe_log10(value) if f in LOG_FEATURES else (np.nan if value is None else float(value))
            )
            year_values[f].append(np.nan if value_year is None else int(value_year))

    for f in ALL_FEATURES:
        side[f] = feature_values[f]
        side[f"{f}_value_year"] = year_values[f]
        side[f"{f}_missing"] = side[f].isna()
    return side


def add_country_context_features(
    matches: pd.DataFrame,
    mapping: pd.DataFrame,
    wb_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return identity columns + per-side context + diff features for both variants."""
    required = {"home_team", "away_team", "match_year"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns: {sorted(missing)}")

    wb = _wb_clean(wb_frame)
    team_codes = _team_code_map(mapping)

    # Collect every (code, cutoff_year) pair actually needed, then resolve once.
    pairs: set[tuple[str, int]] = set()
    code_by_team = team_codes.set_index("canonical_team")["world_bank_code"].to_dict()
    for side_col in ["home_team", "away_team"]:
        sub = matches[[side_col, "match_year"]].dropna()
        for team, year in zip(sub[side_col], sub["match_year"]):
            code = code_by_team.get(team)
            if code is not None and not pd.isna(code):
                pairs.add((code, int(year)))
    cache = _value_lookup(wb, pairs)

    home = _side_features(matches["home_team"], matches["match_year"], team_codes, cache)
    away = _side_features(matches["away_team"], matches["match_year"], team_codes, cache)

    out = matches[[c for c in IDENTITY_COLUMNS if c in matches.columns]].reset_index(drop=True).copy()

    for side_name, frame in [("home", home), ("away", away)]:
        out[f"{side_name}_cc_world_bank_code"] = frame["world_bank_code"].values
        out[f"{side_name}_cc_is_proxy"] = frame["is_proxy_mapping"].values
        out[f"{side_name}_cc_has_context"] = frame["has_context"].values
        for f in ALL_FEATURES:
            out[f"{side_name}_cc_{f}"] = frame[f].values
            out[f"{side_name}_cc_{f}_missing"] = frame[f"{f}_missing"].values

    # --- all_with_proxy diffs ---
    out["any_proxy_mapping_in_match"] = (
        out["home_cc_is_proxy"].astype(int) + out["away_cc_is_proxy"].astype(int)
    ).clip(upper=1)
    out["proxy_mapping_flag_diff"] = (
        out["home_cc_is_proxy"].astype(int) - out["away_cc_is_proxy"].astype(int)
    )
    out["has_country_context_features"] = (
        out["home_cc_has_context"] & out["away_cc_has_context"]
    )
    for f in ALL_FEATURES:
        out[DIFF_NAME[f]] = out[f"home_cc_{f}"] - out[f"away_cc_{f}"]

    # --- direct_only_proxy_missing variant: drop proxy-team values ---
    home_proxy = out["home_cc_is_proxy"].to_numpy()
    away_proxy = out["away_cc_is_proxy"].to_numpy()
    out["home_cc_has_context_direct"] = out["home_cc_has_context"] & ~out["home_cc_is_proxy"]
    out["away_cc_has_context_direct"] = out["away_cc_has_context"] & ~out["away_cc_is_proxy"]
    out["has_country_context_features_direct"] = (
        out["home_cc_has_context_direct"] & out["away_cc_has_context_direct"]
    )
    for f in ALL_FEATURES:
        home_val = out[f"home_cc_{f}"].where(~out["home_cc_is_proxy"])
        away_val = out[f"away_cc_{f}"].where(~out["away_cc_is_proxy"])
        out[f"direct_{DIFF_NAME[f]}"] = home_val - away_val

    return out


def core_diff_features(variant: str = "all") -> list[str]:
    """Core (primary) diff feature names for a variant ('all' or 'direct')."""
    prefix = "" if variant == "all" else "direct_"
    return [f"{prefix}{DIFF_NAME[f]}" for f in PRIMARY_FEATURES]


def secondary_diff_features(variant: str = "all") -> list[str]:
    prefix = "" if variant == "all" else "direct_"
    return [f"{prefix}{DIFF_NAME[f]}" for f in SECONDARY_FEATURES]


def feature_columns() -> list[str]:
    """All country-context columns appended to the model matrix (excludes identity)."""
    cols: list[str] = []
    for side in ["home", "away"]:
        cols.append(f"{side}_cc_world_bank_code")
        cols.append(f"{side}_cc_is_proxy")
        cols.append(f"{side}_cc_has_context")
        cols.append(f"{side}_cc_has_context_direct")
        for f in ALL_FEATURES:
            cols.append(f"{side}_cc_{f}")
            cols.append(f"{side}_cc_{f}_missing")
    cols += [
        "any_proxy_mapping_in_match",
        "proxy_mapping_flag_diff",
        "has_country_context_features",
        "has_country_context_features_direct",
    ]
    cols += [DIFF_NAME[f] for f in ALL_FEATURES]
    cols += [f"direct_{DIFF_NAME[f]}" for f in ALL_FEATURES]
    return cols


def load_country_context_matrix(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
