"""Country context feature engineering (skeleton).

Features for macro country support capacity and football culture.
Not integrated into match model yet - scaffolding only.
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_log_population(df: pd.DataFrame) -> pd.Series:
    """Compute log population from World Bank data.

    Input: World Bank dataframe with column 'SP.POP.TOTL' (population).
    Output: log10(population).
    """
    if "SP.POP.TOTL" not in df.columns:
        logger.warning("Population column not found")
        return pd.Series(index=df.index, dtype=float)

    pop = df["SP.POP.TOTL"]
    pop = pd.to_numeric(pop, errors="coerce")
    return np.log10(pop)


def compute_log_gdp_per_capita_ppp(df: pd.DataFrame) -> pd.Series:
    """Compute log GDP per capita (PPP) from World Bank data.

    Input: World Bank dataframe with column 'NY.GDP.PCAP.PP.CD'.
    Output: log10(GDP per capita PPP).
    """
    if "NY.GDP.PCAP.PP.CD" not in df.columns:
        logger.warning("GDP per capita column not found")
        return pd.Series(index=df.index, dtype=float)

    gdp = df["NY.GDP.PCAP.PP.CD"]
    gdp = pd.to_numeric(gdp, errors="coerce")
    return np.log10(gdp)


def compute_log_total_gdp_ppp(df: pd.DataFrame) -> pd.Series:
    """Compute log total GDP (PPP) from World Bank data.

    Input: World Bank dataframe with column 'NY.GDP.MKTP.PP.CD' or fallback 'NY.GDP.MKTP.CD'.
    Output: log10(total GDP PPP).
    """
    if "NY.GDP.MKTP.PP.CD" in df.columns:
        gdp = df["NY.GDP.MKTP.PP.CD"]
    elif "NY.GDP.MKTP.CD" in df.columns:
        gdp = df["NY.GDP.MKTP.CD"]
    else:
        logger.warning("Total GDP columns not found")
        return pd.Series(index=df.index, dtype=float)

    gdp = pd.to_numeric(gdp, errors="coerce")
    return np.log10(gdp)


def compute_urbanization_rate(df: pd.DataFrame) -> pd.Series:
    """Compute urban population share from World Bank data.

    Input: World Bank dataframe with column 'SP.URB.TOTL.IN.ZS' (% urban population).
    Output: urbanization rate as fraction (0-1), or NaN if missing.
    """
    if "SP.URB.TOTL.IN.ZS" not in df.columns:
        logger.warning("Urbanization column not found")
        return pd.Series(index=df.index, dtype=float)

    urbanization = df["SP.URB.TOTL.IN.ZS"]
    urbanization = pd.to_numeric(urbanization, errors="coerce")
    return urbanization / 100.0


def compute_education_expenditure_share(df: pd.DataFrame) -> pd.Series:
    """Compute government education expenditure as % of GDP.

    Input: World Bank dataframe with column 'SE.XPD.TOTL.GD.ZS'.
    Output: education expenditure share (0-100 range).
    """
    if "SE.XPD.TOTL.GD.ZS" not in df.columns:
        logger.warning("Education expenditure column not found")
        return pd.Series(index=df.index, dtype=float)

    edu_exp = df["SE.XPD.TOTL.GD.ZS"]
    edu_exp = pd.to_numeric(edu_exp, errors="coerce")
    return edu_exp


def compute_rd_expenditure_share(df: pd.DataFrame) -> pd.Series:
    """Compute R&D expenditure as % of GDP (proxy for innovation/development).

    Input: World Bank dataframe with column 'GB.XPD.RSDV.GD.ZS'.
    Output: R&D expenditure share (0-100 range).
    """
    if "GB.XPD.RSDV.GD.ZS" not in df.columns:
        logger.warning("R&D expenditure column not found")
        return pd.Series(index=df.index, dtype=float)

    rd_exp = df["GB.XPD.RSDV.GD.ZS"]
    rd_exp = pd.to_numeric(rd_exp, errors="coerce")
    return rd_exp


def compute_pro_players_per_million_population(
    fifa_landscape_df: pd.DataFrame,
    wb_context_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute professional players per million population.

    Input:
      - fifa_landscape_df: FIFA professional landscape data with 'professional_players', 'population'
      - wb_context_df: World Bank data with 'SP.POP.TOTL'

    Output: dataframe with country and pro_players_per_million_population
    """
    if fifa_landscape_df.empty:
        logger.warning("FIFA landscape data is empty")
        return pd.DataFrame()

    result = fifa_landscape_df.copy()

    # Use FIFA landscape population if available, else join with World Bank
    if "population" not in result.columns and not wb_context_df.empty:
        if "country_code" in result.columns and "country_code" in wb_context_df.columns:
            result = result.merge(
                wb_context_df[["country_code", "SP.POP.TOTL"]],
                on="country_code",
                how="left"
            )
            result.rename(columns={"SP.POP.TOTL": "population"}, inplace=True)

    if "population" in result.columns and "professional_players" in result.columns:
        pop = pd.to_numeric(result["population"], errors="coerce")
        pro_players = pd.to_numeric(result["professional_players"], errors="coerce")
        result["pro_players_per_million"] = (pro_players / pop) * 1e6
    else:
        logger.warning("Cannot compute pro_players_per_million: missing data")

    return result


def compute_pro_clubs_per_million_population(
    fifa_landscape_df: pd.DataFrame,
    wb_context_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute professional clubs per million population."""
    if fifa_landscape_df.empty:
        logger.warning("FIFA landscape data is empty")
        return pd.DataFrame()

    result = fifa_landscape_df.copy()

    if "population" not in result.columns and not wb_context_df.empty:
        if "country_code" in result.columns and "country_code" in wb_context_df.columns:
            result = result.merge(
                wb_context_df[["country_code", "SP.POP.TOTL"]],
                on="country_code",
                how="left"
            )
            result.rename(columns={"SP.POP.TOTL": "population"}, inplace=True)

    if "population" in result.columns and "professional_clubs" in result.columns:
        pop = pd.to_numeric(result["population"], errors="coerce")
        pro_clubs = pd.to_numeric(result["professional_clubs"], errors="coerce")
        result["pro_clubs_per_million"] = (pro_clubs / pop) * 1e6
    else:
        logger.warning("Cannot compute pro_clubs_per_million: missing data")

    return result


def compute_registered_players_per_million_population(
    fifa_landscape_df: pd.DataFrame,
    wb_context_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute registered players per million population (all age/gender)."""
    if fifa_landscape_df.empty:
        logger.warning("FIFA landscape data is empty")
        return pd.DataFrame()

    result = fifa_landscape_df.copy()

    if "population" not in result.columns and not wb_context_df.empty:
        if "country_code" in result.columns and "country_code" in wb_context_df.columns:
            result = result.merge(
                wb_context_df[["country_code", "SP.POP.TOTL"]],
                on="country_code",
                how="left"
            )
            result.rename(columns={"SP.POP.TOTL": "population"}, inplace=True)

    if "population" in result.columns and "registered_players" in result.columns:
        pop = pd.to_numeric(result["population"], errors="coerce")
        reg_players = pd.to_numeric(result["registered_players"], errors="coerce")
        result["registered_players_per_million"] = (reg_players / pop) * 1e6
    else:
        logger.warning("Cannot compute registered_players_per_million: missing data")

    return result


def compute_country_context_features(
    country_code: str,
    year: int,
    wb_df: pd.DataFrame = None,
    trends_df: pd.DataFrame = None,
    landscape_df: pd.DataFrame = None,
    forward_df: pd.DataFrame = None
) -> dict:
    """Compute full country context feature set for a country-year.

    This is a placeholder that scaffolds the feature computation logic.
    Actual integration into match models happens in a separate phase.

    Args:
        country_code: ISO country code
        year: year (for temporal filtering)
        wb_df: World Bank country context dataframe
        trends_df: Google Trends dataframe
        landscape_df: FIFA professional landscape dataframe
        forward_df: FIFA Forward funding dataframe

    Returns:
        dict with feature keys and values
    """
    features = {
        "country_code": country_code,
        "year": year,
        "has_world_bank_context": wb_df is not None and not wb_df.empty,
        "has_google_trends_context": trends_df is not None and not trends_df.empty,
        "has_fifa_professional_landscape_context": landscape_df is not None and not landscape_df.empty,
        "has_fifa_forward_context": forward_df is not None and not forward_df.empty,
    }

    # World Bank features (if available)
    if wb_df is not None and not wb_df.empty:
        country_wb = wb_df[wb_df.get("country_code", "") == country_code]
        if not country_wb.empty:
            row = country_wb.iloc[0]
            features["log_population"] = np.log10(pd.to_numeric(row.get("SP.POP.TOTL"), errors="coerce"))
            features["log_gdp_per_capita_ppp"] = np.log10(pd.to_numeric(row.get("NY.GDP.PCAP.PP.CD"), errors="coerce"))
            features["urbanization_rate"] = pd.to_numeric(row.get("SP.URB.TOTL.IN.ZS"), errors="coerce") / 100.0

    # Google Trends features (if available)
    if trends_df is not None and not trends_df.empty:
        country_trends = trends_df[trends_df.get("country_code", "") == country_code]
        if not country_trends.empty:
            row = country_trends.iloc[0]
            features["football_culture_interest_index"] = pd.to_numeric(row.get("football_topic_interest"), errors="coerce")

    # FIFA landscape features (if available)
    if landscape_df is not None and not landscape_df.empty:
        country_landscape = landscape_df[landscape_df.get("country_code", "") == country_code]
        if not country_landscape.empty:
            row = country_landscape.iloc[0]
            # These are placeholders; actual computation happens in separate feature functions
            features["pro_players_ratio_proxy"] = pd.to_numeric(row.get("professional_players"), errors="coerce")

    # FIFA Forward features (if available)
    if forward_df is not None and not forward_df.empty:
        country_forward = forward_df[forward_df.get("country_code", "") == country_code]
        if not country_forward.empty:
            row = country_forward.iloc[0]
            features["has_fifa_forward_funding"] = pd.to_numeric(row.get("funding_approved_usd"), errors="coerce") > 0

    return features
