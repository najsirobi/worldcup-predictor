"""Atomic multi-match score ingestion shared by batch CSV and issue comments.

Both the CSV batch updater (Task C) and the GitHub Issue comment updater
(Task D) funnel their parsed rows through :func:`apply_batch` so the rules are
identical and enforced in one place:

* every row is validated *before* anything is written;
* if a single row is invalid, the whole batch is rejected and the override file
  is left untouched (no partial application);
* matches not mentioned in the batch keep their existing values;
* the supplied ``source`` and a fresh ``updated_at`` timestamp are stamped on
  every changed row.

Atomicity is structural: we build the result on a copy and only the caller's
final ``write_override`` persists it, so a mid-batch failure never reaches disk.
"""

from __future__ import annotations

import pandas as pd

from src.live.scores_override import VALID_STATUSES, update_match

BATCH_FIELDS = ("match_number", "team_a_goals", "team_b_goals", "status", "notes")


class BatchValidationError(ValueError):
    """Raised when a batch is malformed; carries the per-row error list."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _coerce_goal_cell(value):
    """Empty cell -> None (leave unchanged); otherwise the raw value to validate."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("na", "nan", "none"):
        return None
    return text


def normalise_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Light structural validation that does not need the override frame.

    Returns ``(clean_rows, errors)``. Each clean row has int ``match_number``,
    string ``status``, raw goal cells (None when blank) and ``notes``.
    """
    errors: list[str] = []
    clean: list[dict] = []
    seen: set[int] = set()

    for i, row in enumerate(rows, start=1):
        missing = [f for f in BATCH_FIELDS if f not in row]
        if missing:
            errors.append(f"row {i}: missing field(s) {missing}")
            continue

        raw_mn = str(row["match_number"]).strip()
        try:
            mn = int(raw_mn)
        except (TypeError, ValueError):
            errors.append(f"row {i}: match_number {raw_mn!r} is not an integer")
            continue
        if mn in seen:
            errors.append(f"row {i}: duplicate match_number {mn} within the batch")
            continue
        seen.add(mn)

        status = str(row["status"]).strip().lower()
        if status not in VALID_STATUSES:
            errors.append(
                f"row {i} (match {mn}): invalid status {status!r}; allowed {VALID_STATUSES}"
            )
            continue

        clean.append(
            {
                "match_number": mn,
                "team_a_goals": _coerce_goal_cell(row["team_a_goals"]),
                "team_b_goals": _coerce_goal_cell(row["team_b_goals"]),
                "status": status,
                "notes": (str(row["notes"]).strip() if row["notes"] is not None else ""),
            }
        )

    return clean, errors


def apply_batch(
    frame: pd.DataFrame, rows: list[dict], source: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply a validated batch atomically; raise ``BatchValidationError`` on any error.

    On success returns ``(new_frame, applied)`` where ``applied`` is a list of
    ``{match_number, team_a, team_b, status, score, notes}`` summaries.
    """
    clean, errors = normalise_rows(rows)
    if errors:
        raise BatchValidationError(errors)
    if not clean:
        raise BatchValidationError(["batch contained no data rows"])

    working = frame.copy()
    applied: list[dict] = []
    for row in clean:
        try:
            working = update_match(
                working,
                match_number=row["match_number"],
                team_a_goals=row["team_a_goals"],
                team_b_goals=row["team_b_goals"],
                status=row["status"],
                notes=row["notes"] or None,
                source=source,
            )
        except ValueError as exc:
            raise BatchValidationError(
                [f"match {row['match_number']}: {exc}"]
            ) from exc
        out = working.loc[working["match_number"] == row["match_number"]].iloc[0]
        score = (
            f"{int(out['team_a_goals'])}-{int(out['team_b_goals'])}"
            if out["status"] == "played"
            else "(no result)"
        )
        applied.append(
            {
                "match_number": row["match_number"],
                "team_a": out["team_a"],
                "team_b": out["team_b"],
                "status": out["status"],
                "score": score,
                "notes": out["notes"],
            }
        )

    return working, applied
