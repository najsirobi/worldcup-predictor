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

from src.live.knockout_predictions import build_knockout_predictions
from src.live.scores_override import OVERRIDE_PATH, load_override
from src.live.submission_guard import guard_frozen_submission
from src.live.tournament_state import compute_group_tables

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
REFERENCE_DIR = ROOT / "data" / "reference"
GROUP_VIEW = ROOT / "outputs" / "predictions" / "group_submission_view.csv"
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


def build() -> dict:
    group_view = pd.read_csv(GROUP_VIEW)
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
