"""Tests for atomic batch score ingestion (Travel Mode, Task C)."""

import pytest

from src.live.batch_update import BatchValidationError, apply_batch
from src.live.scores_override import build_initial_override


def _rows(*specs):
    """specs: (match_number, a, b, status, notes) tuples -> list of dicts."""
    keys = ("match_number", "team_a_goals", "team_b_goals", "status", "notes")
    return [dict(zip(keys, (str(x) for x in s))) for s in specs]


def test_valid_batch_applies_all_rows():
    frame = build_initial_override()
    rows = _rows((1, 2, 1, "played", "a"), (2, 0, 0, "played", "b"), (5, 3, 0, "played", "c"))
    updated, applied = apply_batch(frame, rows, source="batch_update")
    assert len(applied) == 3
    played = updated[updated["status"] == "played"]
    assert sorted(played["match_number"]) == [1, 2, 5]
    assert (played["source"] == "batch_update").all()
    # Match 1 recorded correctly.
    m1 = updated[updated["match_number"] == 1].iloc[0]
    assert int(m1["team_a_goals"]) == 2 and int(m1["team_b_goals"]) == 1


def test_invalid_batch_applies_no_rows():
    """A single bad row rejects the whole batch; nothing is applied."""
    frame = build_initial_override()
    rows = _rows((3, 1, 0, "played", "ok"), (4, "", "", "played", "missing goals"))
    with pytest.raises(BatchValidationError):
        apply_batch(frame, rows, source="batch_update")
    # The original frame object is untouched (apply works on a copy).
    assert (frame["status"] == "scheduled").all()


def test_batch_preserves_unmentioned_rows():
    frame = build_initial_override()
    updated, _ = apply_batch(frame, _rows((1, 1, 0, "played", "")), source="batch_update")
    # Only match 1 changed; everything else stays scheduled.
    assert (updated.loc[updated["match_number"] != 1, "status"] == "scheduled").all()


def test_batch_invalid_match_number_fails():
    frame = build_initial_override()
    with pytest.raises(BatchValidationError, match="999"):
        apply_batch(frame, _rows((999, 1, 0, "played", "")), source="batch_update")


def test_batch_non_integer_goals_fail():
    frame = build_initial_override()
    with pytest.raises(BatchValidationError):
        apply_batch(frame, _rows((5, "x", 0, "played", "")), source="batch_update")


def test_batch_played_without_goals_fails():
    frame = build_initial_override()
    with pytest.raises(BatchValidationError):
        apply_batch(frame, _rows((5, "", "", "played", "")), source="batch_update")


def test_batch_duplicate_match_in_batch_fails():
    frame = build_initial_override()
    rows = _rows((1, 1, 0, "played", ""), (1, 2, 0, "played", ""))
    with pytest.raises(BatchValidationError, match="duplicate"):
        apply_batch(frame, rows, source="batch_update")
