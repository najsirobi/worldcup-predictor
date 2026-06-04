"""Test team-name canonicalization helpers (identity passthrough + whitespace)."""
import pandas as pd

from src.ingest.team_names import canonicalize_team_series, normalize_team_whitespace


def _map(rows):
    cols = ["source", "raw_name", "canonical_team_name", "country_code", "notes"]
    return pd.DataFrame(rows, columns=cols)


def test_normalize_team_whitespace_nbsp():
    s = pd.Series(["South\xa0Africa", "  France ", "Costa  Rica"])
    out = normalize_team_whitespace(s)
    assert list(out) == ["South Africa", "France", "Costa Rica"]


def test_canonicalize_identity_passthrough():
    # no map rows -> identity (NOT a guess, keeps literal name); aliases stay distinct
    s = pd.Series(["Cape Verde", "Brazil"])
    canon, used = canonicalize_team_series(s, "international_results", _map([]))
    assert list(canon) == ["Cape Verde", "Brazil"]
    assert list(used) == [False, False]


def test_canonicalize_uses_explicit_map():
    mp = _map([{"source": "fifa_ranking", "raw_name": "Korea Republic",
                "canonical_team_name": "South Korea", "country_code": "KOR", "notes": ""}])
    canon, used = canonicalize_team_series(pd.Series(["Korea Republic", "Brazil"]), "fifa_ranking", mp)
    assert list(canon) == ["South Korea", "Brazil"]   # mapped, then identity
    assert list(used) == [True, False]


def test_canonicalize_does_not_merge_unmapped_alias():
    # "Cabo Verde" without a map row must NOT become "Cape Verde"
    canon, _ = canonicalize_team_series(pd.Series(["Cabo Verde"]), "fifa_ranking", _map([]))
    assert canon.iloc[0] == "Cabo Verde"
