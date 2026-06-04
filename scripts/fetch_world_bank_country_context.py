#!/usr/bin/env python3
"""Fetch World Bank country context indicators."""
import logging
import json
import time
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_WB_DIR = DATA_DIR / "raw" / "world_bank" / "country_context_raw"
INTERIM_DIR = DATA_DIR / "interim"

WB_BASE_URL = "https://api.worldbank.org/v2/country"
WB_COUNTRY_LIST_URL = "https://api.worldbank.org/v2/country"

INDICATORS = {
    "SP.POP.TOTL": "Population",
    "NY.GDP.PCAP.PP.CD": "GDP per capita (PPP, current international $)",
    "NY.GDP.MKTP.PP.CD": "GDP (PPP, current international $)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "SE.XPD.TOTL.GD.ZS": "Government expenditure on education (% of GDP)",
    "GB.XPD.RSDV.GD.ZS": "Research and development expenditure (% of GDP)",
}

# Request params for World Bank API
API_PARAMS = {
    "format": "json",
    "per_page": "500",
    "date": "2000:2023",  # Fetch data from 2000 onwards
}

# Network resilience: the public API intermittently times out or returns
# transient 4xx/5xx under load, so retry each page with backoff and pace
# requests politely rather than dropping a whole indicator on one bad page.
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_BACKOFF = 3  # seconds, multiplied by attempt number
REQUEST_DELAY = 0.5  # polite pause between successful page requests


def _get_page(url: str, page: int) -> Optional[list]:
    """GET one page with retries; returns parsed JSON or None after exhaustion."""
    params = {**API_PARAMS, "page": str(page)}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning(f"  … page {page} attempt {attempt} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"  ✗ page {page} failed after {MAX_RETRIES} attempts: {e}")
    return None


def fetch_aggregate_codes() -> Optional[set]:
    """Return the set of ISO3 codes the World Bank classifies as aggregates.

    Sourced from the API's own country metadata (``region.value == 'Aggregates'``)
    so we flag aggregates without hardcoding or inventing a list. Returns None if
    the metadata cannot be fetched, in which case aggregates are left unflagged.
    """
    all_records: list[dict] = []
    page = 1
    while True:
        params = {"format": "json", "per_page": "400", "page": str(page)}
        try:
            response = requests.get(WB_COUNTRY_LIST_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"  ⚠ Could not fetch country metadata for aggregate flagging: {e}")
            return None

        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            break
        meta, records = data[0], data[1]
        all_records.extend(records)
        if page >= int(meta.get("pages", 1) or 1):
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    aggregates = {
        r.get("id")
        for r in all_records
        if (r.get("region") or {}).get("value") == "Aggregates"
    }
    logger.info(f"  ✓ Identified {len(aggregates)} aggregate codes from country metadata")
    return aggregates


def fetch_indicator(indicator_code: str) -> Optional[list[dict]]:
    """Fetch all pages of a World Bank indicator for every country.

    The API paginates results (page 1 alone is only region aggregates, no real
    countries), so we must follow the ``pages`` count in the response metadata
    instead of reading a single page.
    """
    url = f"{WB_BASE_URL}/all/indicator/{indicator_code}"
    all_records: list[dict] = []
    page = 1
    pages_fetched = 0

    while True:
        logger.info(f"Fetching {indicator_code} (page {page})...")
        data = _get_page(url, page)

        if data is None:
            # Page exhausted its retries. Keep whatever we already have rather
            # than discarding the indicator entirely.
            logger.warning(f"  ⚠ stopping {indicator_code} at page {page} with {len(all_records)} record(s) so far")
            break

        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            if page == 1:
                logger.warning("  ✗ No data returned")
            break

        meta, records = data[0], data[1]
        all_records.extend(records)
        pages_fetched += 1

        total_pages = int(meta.get("pages", 1) or 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    if all_records:
        logger.info(f"  ✓ Retrieved {len(all_records)} records across {pages_fetched} page(s)")
        return all_records
    return None


def save_raw_response(indicator_code: str, data: list) -> Path:
    """Save raw API response to JSON."""
    RAW_WB_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_WB_DIR / f"{indicator_code}.json"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"  ✓ Saved to {filepath}")
    return filepath


def convert_records_to_dataframe(all_records: dict) -> pd.DataFrame:
    """Convert World Bank API records to a flat dataframe.

    Input: dict with indicator codes as keys and lists of records as values
    Output: dataframe with columns: country_code, country_name, year, and one
    column per indicator.

    A single pass builds a (country_code, year) -> row index keyed across the
    union of all indicators. This avoids the previous O(n^2) nested rescan and
    fixes a bug where only country-years present in the *first* indicator were
    kept, silently dropping rows that other indicators covered.
    """
    indicator_codes = list(all_records.keys())
    index: dict[tuple[str, str], dict] = {}

    for indicator_code in indicator_codes:
        for record in all_records[indicator_code]:
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

    df = pd.DataFrame(list(index.values()))
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    return df


def main():
    """Fetch World Bank country context data."""
    logger.info("Fetching World Bank country context indicators\n")

    all_records = {}
    for indicator_code, indicator_name in INDICATORS.items():
        records = fetch_indicator(indicator_code)
        if records:
            all_records[indicator_code] = records
            save_raw_response(indicator_code, records)

    logger.info("")

    if not all_records:
        logger.error("No indicators fetched successfully")
        exit(1)

    # Convert to dataframe
    logger.info("Converting World Bank data to dataframe...")
    df = convert_records_to_dataframe(all_records)
    logger.info(f"  ✓ Created dataframe: {len(df)} rows, {len(df.columns)} columns")

    # Flag World Bank region/income aggregates (e.g. WLD, EUU, AFE) so downstream
    # code can exclude them from country-level joins without guessing.
    logger.info("Flagging region aggregates from country metadata...")
    aggregate_codes = fetch_aggregate_codes()
    if aggregate_codes is None:
        df["is_aggregate"] = pd.NA  # metadata unavailable; do not guess
        n_aggregates = None
    else:
        df["is_aggregate"] = df["country_code"].isin(aggregate_codes)
        n_aggregates = int(df["is_aggregate"].sum())

    # Save interim CSV
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    interim_path = INTERIM_DIR / "world_bank_country_context.csv"
    df.to_csv(interim_path, index=False)
    logger.info(f"  ✓ Saved to {interim_path}\n")

    # Generate report
    report_dir = REPO_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "world_bank_country_context_report.md"

    with open(report_path, "w") as f:
        f.write("# World Bank Country Context Data Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Records retrieved: {len(df)}\n")
        f.write(f"- Columns: {len(df.columns)}\n")
        f.write(f"- Date range: {df['year'].min()} to {df['year'].max()}\n")
        f.write(f"- Unique country codes: {df['country_code'].nunique()}\n")
        if n_aggregates is None:
            f.write("- Aggregates: not flagged (country metadata unavailable this run)\n\n")
        else:
            n_real_codes = df.loc[~df["is_aggregate"].fillna(False), "country_code"].nunique()
            f.write(
                f"- Region/income aggregates flagged via `is_aggregate`: "
                f"{n_aggregates} aggregate rows; {n_real_codes} real country codes\n\n"
            )

        f.write("## Indicators Fetched\n\n")
        for indicator_code, indicator_name in INDICATORS.items():
            if indicator_code in df.columns:
                non_null = df[indicator_code].notna().sum()
                f.write(f"- **{indicator_code}**: {indicator_name}\n")
                f.write(f"  - Non-null values: {non_null}/{len(df)}\n")
                f.write(f"  - Coverage: {100*non_null/len(df):.1f}%\n\n")

        f.write("## Data Quality Notes\n\n")
        f.write("- World Bank data is aggregated annually\n")
        f.write("- Some indicators may have missing values for certain countries/years\n")
        f.write("- Rows where `is_aggregate` is True are region/income groupings "
                "(e.g. WLD, EUU, AFE), not countries; exclude them from country joins\n")
        f.write("- For historical match analysis, use latest available value before match year\n")
        f.write("- For current tournament predictions, use most recent available data\n\n")

        f.write("## Files Generated\n\n")
        f.write(f"- Interim CSV: `data/interim/world_bank_country_context.csv`\n")
        f.write(f"- Raw JSON responses: `data/raw/world_bank/country_context_raw/`\n")

    logger.info(f"✓ Report written to {report_path}")


if __name__ == "__main__":
    main()
