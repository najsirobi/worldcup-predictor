"""Test World Bank record flattening (convert_records_to_dataframe)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# The function lives in a standalone script (not a package), so load it by path.
_SCRIPT = Path(__file__).parent.parent / "scripts" / "fetch_world_bank_country_context.py"
_spec = importlib.util.spec_from_file_location("fetch_wb", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
convert_records_to_dataframe = _mod.convert_records_to_dataframe


def _rec(iso3, year, value, name="Country"):
    return {
        "countryiso3code": iso3,
        "date": year,
        "value": value,
        "country": {"value": name},
    }


def test_union_of_country_years_across_indicators():
    """A country-year present only in a later indicator must not be dropped."""
    all_records = {
        "SP.POP.TOTL": [_rec("FRA", "2020", 67000000)],
        # GDP has an extra country-year that population lacks.
        "NY.GDP.MKTP.CD": [_rec("FRA", "2020", 2.6e12), _rec("DEU", "2020", 3.8e12)],
    }
    df = convert_records_to_dataframe(all_records)

    assert set(zip(df["country_code"], df["year"])) == {("FRA", 2020), ("DEU", 2020)}
    deu = df[df["country_code"] == "DEU"].iloc[0]
    # DEU has no population record -> stays missing, GDP populated.
    assert pd.isna(deu["SP.POP.TOTL"])
    assert deu["NY.GDP.MKTP.CD"] == 3.8e12


def test_missing_value_is_none():
    all_records = {
        "SP.POP.TOTL": [_rec("FRA", "2020", None)],
    }
    df = convert_records_to_dataframe(all_records)
    assert pd.isna(df.loc[0, "SP.POP.TOTL"])


def test_blank_country_code_skipped():
    all_records = {
        "SP.POP.TOTL": [_rec("", "2020", 1), _rec("FRA", "2020", 67000000)],
    }
    df = convert_records_to_dataframe(all_records)
    assert list(df["country_code"]) == ["FRA"]


def test_year_is_integer_and_sorted():
    all_records = {
        "SP.POP.TOTL": [_rec("FRA", "2021", 1), _rec("FRA", "2019", 1), _rec("DEU", "2020", 1)],
    }
    df = convert_records_to_dataframe(all_records)
    assert df["year"].tolist() == [2020, 2019, 2021]  # sorted by (code, year): DEU then FRA
    assert df["year"].dtype.kind == "i"


def test_empty_input_returns_empty_frame():
    assert convert_records_to_dataframe({"SP.POP.TOTL": []}).empty
