"""Tests for the expanded WC2026 country-code mapping."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
MAPPING = ROOT / "data" / "reference" / "country_code_map.csv"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
METADATA = ROOT / "data" / "raw" / "world_bank" / "country_metadata.json"


def _wc2026_teams() -> set[str]:
    template = pd.read_csv(TEMPLATE)
    return set(template["team_a"]) | set(template["team_b"])


def _mapping() -> pd.DataFrame:
    return pd.read_csv(MAPPING)


def _metadata() -> pd.DataFrame:
    with open(METADATA) as handle:
        payload = json.load(handle)
    rows = []
    for record in payload["records"]:
        region = record.get("region") or {}
        rows.append(
            {
                "world_bank_code": record["id"],
                "name": record["name"],
                "is_aggregate": region.get("value") == "Aggregates",
            }
        )
    return pd.DataFrame(rows)


def test_all_48_wc2026_teams_appear_in_country_code_map():
    mapping = _mapping()
    assert mapping["canonical_team"].nunique() == 48
    assert set(mapping["canonical_team"]) == _wc2026_teams()


def test_all_non_null_world_bank_codes_exist_in_country_metadata():
    mapping = _mapping()
    metadata = _metadata()
    merged = mapping.merge(metadata, on="world_bank_code", how="left")
    assert merged["name"].notna().all()


def test_region_aggregates_are_not_accepted_as_team_countries():
    mapping = _mapping()
    metadata = _metadata()
    merged = mapping.merge(metadata[["world_bank_code", "is_aggregate"]], on="world_bank_code", how="left")
    assert not merged["is_aggregate"].fillna(False).any()


def test_special_cases_map_correctly():
    mapping = _mapping().set_index("canonical_team")

    assert mapping.loc["Türkiye", "world_bank_code"] == "TUR"
    assert mapping.loc["Türkiye", "world_bank_country_name"] == "Turkiye"
    assert mapping.loc["Czechia", "world_bank_code"] == "CZE"
    assert mapping.loc["IR Iran", "world_bank_code"] == "IRN"
    assert mapping.loc["IR Iran", "world_bank_country_name"] == "Iran, Islamic Rep."
    assert mapping.loc["Côte d'Ivoire", "world_bank_code"] == "CIV"
    assert mapping.loc["Congo DR", "world_bank_code"] == "COD"
    assert mapping.loc["Congo DR", "world_bank_country_name"] == "Congo, Dem. Rep."
    assert mapping.loc["Cabo Verde", "world_bank_code"] == "CPV"
    assert mapping.loc["Bosnia and Herzegovina", "world_bank_code"] == "BIH"
    assert mapping.loc["Curaçao", "world_bank_code"] == "CUW"


def test_scotland_is_explicit_proxy_mapping():
    mapping = _mapping().set_index("canonical_team")
    assert mapping.loc["Scotland", "world_bank_code"] == "GBR"
    assert bool(mapping.loc["Scotland", "is_proxy_mapping"]) is True


def test_england_is_explicit_proxy_mapping():
    mapping = _mapping().set_index("canonical_team")
    assert mapping.loc["England", "world_bank_code"] == "GBR"
    assert bool(mapping.loc["England", "is_proxy_mapping"]) is True
