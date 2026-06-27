#!/usr/bin/env python3
"""Build round-by-round knockout exact-score predictions (Travel Mode, live).

Generates the model's predicted exact score, advancing team, and shoot-out call
for every knockout match (73-104). Teams that are not yet decided come from the
frozen group-stage projection (the up-front gamble); once actual results are in,
the Round of 32 and later-round participants are pinned from real standings and
played knockout matches, and the current recommendation refreshes accordingly.

This is a live, derived output. It never reads or modifies the submitted
group-stage prediction files, and it does not retrain anything.

Outputs:
    outputs/live/knockout_predictions.json
    outputs/live/knockout_predictions.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.live.active_candidate import load_active_candidate
from src.live.knockout_predictions import build_knockout_predictions
from src.live.scores_override import OVERRIDE_PATH, load_override
from src.live.submission_guard import guard_frozen_submission
from src.live.tournament_state import compute_group_tables

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
REFERENCE_DIR = ROOT / "data" / "reference"
R32_MAPPING = REFERENCE_DIR / "round_of_32_mapping.csv"
PROGRESSION = REFERENCE_DIR / "knockout_round_progression.csv"
ANNEX = REFERENCE_DIR / "third_place_assignment_annex_c.csv"

CSV_COLUMNS = [
    "match_number",
    "round",
    "round_label",
    "projected_team_a",
    "projected_team_b",
    "projected_score",
    "projected_advancing_team",
    "projected_shootout",
    "current_team_a",
    "current_team_b",
    "current_score",
    "current_advancing_team",
    "current_shootout",
    "teams_source",
    "actual_score",
    "actual_advancing_team",
    "actual_shootout",
    "points_earned_estimate",
    "status",
    "copy_text",
]


def _build_group_view_from_active_candidate() -> pd.DataFrame:
    """Build group-view data from the active candidate's standing predictions.
    
    This replaces the dependency on outputs/predictions/group_submission_view.csv,
    which is not committed. We derive the necessary group-view structure from the
    active candidate's standing predictions instead.
    """
    candidate = load_active_candidate()
    standings = candidate.load_standing_predictions()
    
    # The standings file has rank_1, rank_2, rank_3, rank_4 per group.
    # Extract team names and build a synthetic group_view compatible with
    # the knockout builder. We use the standing order and probabilities
    # from the frozen group-stage projection as context.
    
    rows = []
    for _, row in standings.iterrows():
        group = row["group"]
        for standing in [1, 2, 3, 4]:
            team_col = f"rank_{standing}"
            if team_col in standings.columns:
                team = row[team_col]
                if pd.isna(team) or not team:
                    continue
                # Synthetic probabilities: rank 1 has highest p_top2, etc.
                # The exact values don't matter much since we use only the
                # standing order for bracket construction in deterministic mode.
                p_top2 = max(0.0, 1.0 - (standing - 1) * 0.25)
                p_top3 = max(0.0, 1.0 - (standing - 1) * 0.20)
                rows.append({
                    "group": group,
                    "team": team,
                    "suggested_group_standing": standing,
                    "p_top2": p_top2,
                    "p_top3": p_top3,
                    "p_advance_with_best_thirds": p_top2 + 0.05,
                    "p_finish_1st": 0.25 if standing == 1 else 0.0,
                    "p_finish_2nd": 0.25 if standing == 2 else 0.0,
                    "expected_points": 3.0 * (5 - standing),  # Rough heuristic
                    "expected_goal_difference": 0.0,
                    "confidence_level": "high" if standing <= 2 else "medium",
                    "group_flags": "none",
                    "likely_best_third_signal": 0.0,
                    "likely_best_third_team": False,
                })
    
    return pd.DataFrame(rows)


def build() -> dict:
    """Build knockout predictions from the active candidate's group-stage output."""
    # Use active candidate's standing predictions to derive group-view context.
    group_view = _build_group_view_from_active_candidate()
    
    r32_mapping = pd.read_csv(R32_MAPPING)
    progression = pd.read_csv(PROGRESSION)
    annex = pd.read_csv(ANNEX)

    scores = load_override(OVERRIDE_PATH) if OVERRIDE_PATH.exists() else None
    actual_group_tables = None
    if scores is not None and not scores.empty:
        try:
            actual_group_tables = compute_group_tables(scores)
        except Exception:
            actual_group_tables = None

    return build_knockout_predictions(
        group_view,
        r32_mapping,
        progression,
        annex,
        scores=scores,
        actual_group_tables=actual_group_tables,
    )


def main() -> None:
    with guard_frozen_submission("build_knockout_predictions.py"):
        payload = build()
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        (LIVE_DIR / "knockout_predictions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        frame = pd.DataFrame(payload["matches"], columns=CSV_COLUMNS)
        frame.to_csv(LIVE_DIR / "knockout_predictions.csv", index=False)

    played = sum(1 for m in payload["matches"] if m["status"] == "played")
    print(
        f"Built knockout predictions: {len(payload['matches'])} matches "
        f"({played} played, group stage complete={payload['group_stage_complete']}). "
        "Wrote JSON + CSV."
    )


if __name__ == "__main__":
    main()
