"""Tests for the clean fill-only group score export (presentation clarity)."""

import re
from pathlib import Path

import pandas as pd

from scripts import build_fill_only_scores
from src.live.active_candidate import load_active_candidate

FILL_ONLY_NAME = "final_group_score_predictions_fill_only.csv"
FILL_ONLY_PATHS = [
    Path("outputs/final_candidate_v4_recent_rollforward") / FILL_ONLY_NAME,
    Path("outputs/predictions") / FILL_ONLY_NAME,
]
EXPECTED_COLUMNS = [
    "match_number",
    "group",
    "team_a",
    "team_b",
    "score_to_fill_in",
    "copy_text",
]
SCORE_RE = re.compile(r"^\d+-\d+$")


def test_build_main_writes_both_fill_only_files():
    build_fill_only_scores.main()
    for path in FILL_ONLY_PATHS:
        assert path.exists(), f"missing fill-only file: {path}"


def test_fill_only_files_have_72_clean_unique_rows():
    build_fill_only_scores.main()
    for path in FILL_ONLY_PATHS:
        frame = pd.read_csv(path)
        assert list(frame.columns) == EXPECTED_COLUMNS
        assert len(frame) == 72
        assert not frame["score_to_fill_in"].isna().any()
        assert frame["score_to_fill_in"].astype(str).str.match(SCORE_RE).all()
        assert sorted(frame["match_number"].tolist()) == list(range(1, 73))
        assert frame["match_number"].is_unique


def test_score_to_fill_in_equals_final_recommended_score():
    build_fill_only_scores.main()
    candidate = load_active_candidate()
    source = candidate.load_score_predictions().sort_values("match_number")
    frame = pd.read_csv(FILL_ONLY_PATHS[0]).sort_values("match_number")
    assert (
        frame["score_to_fill_in"].astype(str).tolist()
        == source["final_recommended_score"].astype(str).tolist()
    )


def test_copy_text_matches_expected_format():
    build_fill_only_scores.main()
    frame = pd.read_csv(FILL_ONLY_PATHS[0])
    row = frame.loc[frame["match_number"] == 1].iloc[0]
    assert row["copy_text"] == "1. Mexico 1-0 South Africa"
    # Every copy_text follows "<n>. <team_a> <score> <team_b>".
    for r in frame.itertuples(index=False):
        assert r.copy_text == f"{r.match_number}. {r.team_a} {r.score_to_fill_in} {r.team_b}"
