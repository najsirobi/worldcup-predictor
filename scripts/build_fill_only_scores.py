#!/usr/bin/env python3
"""Build the clean "fill-only" group score files (presentation clarity).

The active candidate score CSV carries several score columns
(``final_recommended_score``, ``safe_score``, ``ev_score``,
``auto_consensus_score``) plus audit metadata. Only ``final_recommended_score``
is the score the operator should actually fill in; the rest are
audit/alternative columns. This script projects the active candidate down to a
single unambiguous score per match so there is exactly one number to copy.

Output columns:
    match_number, group, team_a, team_b, score_to_fill_in, copy_text

where ``score_to_fill_in = final_recommended_score`` and ``copy_text`` is a
copy-friendly line such as ``"1. Mexico 1-0 South Africa"``.

This script never retrains models or changes predictions; it is a pure
re-projection of the frozen active candidate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.live.active_candidate import load_active_candidate

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
FILL_ONLY_NAME = "final_group_score_predictions_fill_only.csv"

_SCORE_RE = re.compile(r"^\d+-\d+$")


def build_fill_only_frame() -> pd.DataFrame:
    """Project the active candidate score file to fill-only columns."""
    candidate = load_active_candidate()
    scores = candidate.load_score_predictions()

    required = {"match_number", "group", "team_a", "team_b", "final_recommended_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(
            f"Active candidate score file is missing column(s): {sorted(missing)}"
        )

    frame = scores.loc[
        :, ["match_number", "group", "team_a", "team_b", "final_recommended_score"]
    ].copy()
    frame = frame.rename(columns={"final_recommended_score": "score_to_fill_in"})
    frame["match_number"] = frame["match_number"].astype(int)
    frame = frame.sort_values("match_number").reset_index(drop=True)
    frame["copy_text"] = [
        f"{int(row.match_number)}. {row.team_a} {row.score_to_fill_in} {row.team_b}"
        for row in frame.itertuples(index=False)
    ]
    return frame


def validate(frame: pd.DataFrame) -> None:
    """Raise if the fill-only frame is not exactly 72 clean, unique rows."""
    if len(frame) != 72:
        raise ValueError(f"Expected exactly 72 rows, got {len(frame)}.")
    if frame["score_to_fill_in"].isna().any():
        raise ValueError("Found missing score_to_fill_in values.")
    bad = frame.loc[~frame["score_to_fill_in"].astype(str).str.match(_SCORE_RE)]
    if not bad.empty:
        raise ValueError(
            "score_to_fill_in values must parse as X-Y; offending match numbers: "
            f"{bad['match_number'].tolist()}"
        )
    numbers = frame["match_number"].tolist()
    if sorted(numbers) != list(range(1, 73)):
        raise ValueError("match_number must be the unique set 1..72.")


def main() -> None:
    frame = build_fill_only_frame()
    validate(frame)

    candidate = load_active_candidate()
    targets = [
        candidate.candidate_dir / FILL_ONLY_NAME,
        PREDICTIONS_DIR / FILL_ONLY_NAME,
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        print(f"Wrote {len(frame)} fill-only rows -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
