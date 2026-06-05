#!/usr/bin/env python3
"""Fetch World Bank country metadata and country-context indicators."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_WB_DIR = DATA_DIR / "raw" / "world_bank" / "country_context_raw"
RAW_WB_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = DATA_DIR / "raw" / "world_bank" / "country_metadata.json"
INTERIM_PATH = DATA_DIR / "interim" / "world_bank_country_context.csv"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "world_bank_country_context_report.md"

WB_COUNTRY_URL = "https://api.worldbank.org/v2/country"
WB_ALL_COUNTRIES = "https://api.worldbank.org/v2/country/all"
TOURNAMENT_YEAR = 2026
FETCH_WINDOW = "2000:2025"

INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "SP.POP.TOTL": "Population, total",
    "SE.XPD.TOTL.GD.ZS": "Government expenditure on education, total (% of GDP)",
    "GB.XPD.RSDV.GD.ZS": "Research and development expenditure (% of GDP)",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total population)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
}

API_PARAMS = {
    "format": "json",
    "per_page": "20000",
    "date": FETCH_WINDOW,
}

COUNTRY_METADATA_PARAMS = {
    "format": "json",
    "per_page": "400",
}

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_BACKOFF = 3
REQUEST_DELAY = 0.5


def _request_json(url: str, params: dict[str, str]) -> list[Any] | None:
    """Fetch a World Bank API page with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - exercised by integration run
            if attempt >= MAX_RETRIES:
                logger.error(f"  ✗ request failed after {MAX_RETRIES} attempts: {exc}")
                return None
            wait = RETRY_BACKOFF * attempt
            logger.warning(f"  … request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    return None


def fetch_country_metadata() -> list[dict[str, Any]]:
    """Fetch the full World Bank country metadata catalog."""
    records: list[dict[str, Any]] = []
    page = 1
    while True:
        logger.info(f"Fetching World Bank country metadata (page {page})...")
        payload = _request_json(WB_COUNTRY_URL, {**COUNTRY_METADATA_PARAMS, "page": str(page)})
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            break
        meta, page_records = payload[0], payload[1]
        records.extend(page_records)
        if page >= int(meta.get("pages", 1) or 1):
            break
        page += 1
        time.sleep(REQUEST_DELAY)
    logger.info(f"  ✓ Retrieved {len(records)} country metadata records")
    return records


def metadata_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten World Bank country metadata records."""
    rows = []
    for record in records:
        region = record.get("region") or {}
        income = record.get("incomeLevel") or {}
        lending = record.get("lendingType") or {}
        rows.append(
            {
                "world_bank_code": record.get("id"),
                "iso2_code": record.get("iso2Code"),
                "world_bank_country_name": record.get("name"),
                "region_id": region.get("id"),
                "region_name": region.get("value"),
                "income_level_id": income.get("id"),
                "income_level_name": income.get("value"),
                "lending_type_id": lending.get("id"),
                "lending_type_name": lending.get("value"),
                "capital_city": record.get("capitalCity"),
                "longitude": record.get("longitude"),
                "latitude": record.get("latitude"),
                "is_aggregate": region.get("value") == "Aggregates",
            }
        )
    return pd.DataFrame(rows).sort_values("world_bank_code").reset_index(drop=True)


def save_country_metadata(records: list[dict[str, Any]]) -> None:
    """Write raw country metadata to disk."""
    payload = {
        "source_url": WB_COUNTRY_URL,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info(f"  ✓ Saved metadata to {METADATA_PATH}")


def fetch_indicator(indicator_code: str) -> list[dict[str, Any]] | None:
    """Fetch all pages of a World Bank indicator for all countries."""
    url = f"{WB_ALL_COUNTRIES}/indicator/{indicator_code}"
    all_records: list[dict[str, Any]] = []
    page = 1
    while True:
        logger.info(f"Fetching {indicator_code} (page {page})...")
        payload = _request_json(url, {**API_PARAMS, "page": str(page)})
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            if page == 1:
                logger.warning(f"  ✗ No data returned for {indicator_code}")
            break
        meta, page_records = payload[0], payload[1]
        all_records.extend(page_records)
        if page >= int(meta.get("pages", 1) or 1):
            break
        page += 1
        time.sleep(REQUEST_DELAY)
    if all_records:
        logger.info(f"  ✓ Retrieved {len(all_records)} records for {indicator_code}")
        return all_records
    return None


def save_raw_response(indicator_code: str, records: list[dict[str, Any]]) -> Path:
    """Save flattened indicator records to JSON."""
    payload = {
        "indicator_code": indicator_code,
        "source_url": f"{WB_ALL_COUNTRIES}/indicator/{indicator_code}",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    }
    path = RAW_WB_DIR / f"{indicator_code}.json"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info(f"  ✓ Saved to {path}")
    return path


def convert_records_to_dataframe(all_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Convert World Bank indicator records into a single flat dataframe."""
    indicator_codes = list(all_records.keys())
    index: dict[tuple[str, str], dict[str, Any]] = {}

    for indicator_code, records in all_records.items():
        for record in records:
            country = record.get("countryiso3code") or ""
            year = record.get("date") or ""
            if not country or not year:
                continue
            key = (country, year)
            row = index.get(key)
            if row is None:
                row = {
                    "country_code": country,
                    "country_name": (record.get("country") or {}).get("value", ""),
                    "year": int(year),
                }
                for code in indicator_codes:
                    row[code] = None
                index[key] = row
            row[indicator_code] = record.get("value")

    if not index:
        return pd.DataFrame()

    frame = pd.DataFrame(list(index.values()))
    return frame.sort_values(["country_code", "year"]).reset_index(drop=True)


def build_report(df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """Write the World Bank data refresh report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    real_country_mask = ~metadata_df["is_aggregate"].fillna(False)
    with open(REPORT_PATH, "w") as handle:
        handle.write("# World Bank Country Context Data Report\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- Records retrieved: {len(df)}\n")
        handle.write(f"- Columns: {len(df.columns)}\n")
        handle.write(f"- Date range: {int(df['year'].min())} to {int(df['year'].max())}\n")
        handle.write(f"- Unique country codes: {df['country_code'].nunique()}\n")
        handle.write(
            f"- Metadata rows: {len(metadata_df)} ({int(real_country_mask.sum())} real countries, "
            f"{int(metadata_df['is_aggregate'].sum())} aggregates)\n\n"
        )

        handle.write("## Indicators Fetched\n\n")
        for indicator_code, label in INDICATORS.items():
            non_null = int(df[indicator_code].notna().sum())
            handle.write(f"- **{indicator_code}**: {label}\n")
            handle.write(f"  - Non-null values: {non_null}/{len(df)}\n")
            handle.write(f"  - Coverage: {100 * non_null / len(df):.1f}%\n\n")

        handle.write("## Data Quality Notes\n\n")
        handle.write("- Country metadata is fetched from the World Bank `/country` endpoint with pagination.\n")
        handle.write("- Aggregates are identified from metadata where `region.value == \"Aggregates\"` and remain flagged.\n")
        handle.write(f"- Indicator window requested: `{FETCH_WINDOW}`. Use latest value strictly before {TOURNAMENT_YEAR} for WC2026 context.\n")
        handle.write("- Missing indicator values are preserved as nulls; no zero fill is used.\n")
        handle.write("- These variables are macro/development proxies, not direct football-spending measures.\n\n")

        handle.write("## Files Generated\n\n")
        handle.write("- Raw metadata: `data/raw/world_bank/country_metadata.json`\n")
        handle.write("- Raw indicator cache: `data/raw/world_bank/country_context_raw/*.json`\n")
        handle.write("- Interim CSV: `data/interim/world_bank_country_context.csv`\n")
    logger.info(f"✓ Report written to {REPORT_PATH}")


def main() -> None:
    """Fetch metadata and indicator data from the World Bank API."""
    logger.info("Fetching World Bank country metadata and country-context indicators\n")

    metadata_records = fetch_country_metadata()
    if not metadata_records:
        raise SystemExit("No country metadata returned from World Bank API.")
    save_country_metadata(metadata_records)
    metadata_df = metadata_to_dataframe(metadata_records)

    all_records: dict[str, list[dict[str, Any]]] = {}
    for indicator_code in INDICATORS:
        records = fetch_indicator(indicator_code)
        if records:
            all_records[indicator_code] = records
            save_raw_response(indicator_code, records)
        time.sleep(REQUEST_DELAY)

    if not all_records:
        raise SystemExit("No indicators fetched successfully.")

    logger.info("\nConverting World Bank data to dataframe...")
    df = convert_records_to_dataframe(all_records)
    if df.empty:
        raise SystemExit("Indicator fetch succeeded but conversion produced no rows.")

    df = df.merge(
        metadata_df[["world_bank_code", "is_aggregate"]],
        left_on="country_code",
        right_on="world_bank_code",
        how="left",
    ).drop(columns=["world_bank_code"])

    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_PATH, index=False)
    logger.info(f"  ✓ Saved interim data to {INTERIM_PATH}")

    build_report(df, metadata_df)


if __name__ == "__main__":
    main()
