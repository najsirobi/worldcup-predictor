"""Parse score updates out of a GitHub Issue comment (Travel Mode, Task D).

Expected comment block (anywhere in the comment body):

    /WK-SCORES
    match_number,team_a_goals,team_b_goals,status,notes
    1,2,1,played,Mexico v South Africa
    2,1,1,played,Korea Republic v Czechia
    /END-WK-SCORES

A knockout-aware 6-column header is also accepted (backward compatible); the
extra ``advanced_team`` column names the side that goes through when a knockout
match is level after extra time (a shoot-out)::

    /WK-SCORES
    match_number,team_a_goals,team_b_goals,status,notes,advanced_team
    73,1,1,played,R32 penalties,Canada
    74,2,0,played,R32,
    /END-WK-SCORES

Only the text strictly between the ``/WK-SCORES`` and ``/END-WK-SCORES`` markers
is parsed. The first non-empty line inside the block must be one of the accepted
CSV headers. Rows are returned as dicts ready for
:func:`src.live.batch_update.apply_batch`; this module does NOT touch the
override frame -- value validation (match exists, integer goals,
played-needs-goals, etc.) happens there so the rules stay in one place.
"""

from __future__ import annotations

import csv
import io

START_MARKER = "/WK-SCORES"
END_MARKER = "/END-WK-SCORES"
# The original group-stage header (5 columns) stays valid for backward
# compatibility; a 6th optional ``advanced_team`` column enables knockout entry.
REQUIRED_HEADER = ["match_number", "team_a_goals", "team_b_goals", "status", "notes"]
KNOCKOUT_HEADER = REQUIRED_HEADER + ["advanced_team"]
ACCEPTED_HEADERS = (REQUIRED_HEADER, KNOCKOUT_HEADER)


class CommentParseError(ValueError):
    """Raised when the comment block is missing or malformed."""


def has_score_block(text: str) -> bool:
    """Cheap check used to gate the workflow before doing real work."""
    return START_MARKER in (text or "")


def extract_block(text: str) -> str:
    """Return the raw text between the markers, or raise ``CommentParseError``."""
    if text is None:
        raise CommentParseError("Empty comment: no /WK-SCORES block found.")
    start = text.find(START_MARKER)
    if start == -1:
        raise CommentParseError(f"Missing start marker {START_MARKER!r}.")
    end = text.find(END_MARKER, start + len(START_MARKER))
    if end == -1:
        raise CommentParseError(f"Missing end marker {END_MARKER!r}.")
    return text[start + len(START_MARKER) : end]


def parse_comment(text: str) -> list[dict]:
    """Extract and shape the score rows from a comment body.

    Raises ``CommentParseError`` if the markers are missing, the header is wrong,
    or the block has no data rows. Per-value validation is deferred to
    ``apply_batch``.
    """
    block = extract_block(text)
    # Drop blank lines and stray indentation; keep field content intact.
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        raise CommentParseError("Score block is empty between the markers.")

    reader = csv.reader(io.StringIO("\n".join(lines)))
    records = list(reader)
    header = [h.strip() for h in records[0]]
    if header not in ACCEPTED_HEADERS:
        raise CommentParseError(
            f"Invalid header {header}; expected one of {REQUIRED_HEADER} "
            f"or {KNOCKOUT_HEADER}."
        )
    has_advanced = header == KNOCKOUT_HEADER
    n_cols = len(header)

    data_rows = records[1:]
    if not data_rows:
        raise CommentParseError("No data rows found after the header.")

    rows: list[dict] = []
    for i, rec in enumerate(data_rows, start=1):
        if len(rec) < n_cols:
            raise CommentParseError(
                f"Data row {i} has {len(rec)} field(s); expected "
                f"{n_cols} ({','.join(header)})."
            )
        if has_advanced:
            # match_number,team_a_goals,team_b_goals,status, <notes...>, advanced_team
            # advanced_team is the final field; notes may itself contain commas.
            fixed = rec[:4]
            advanced = rec[-1]
            notes = ",".join(rec[4:-1])
            values = fixed + [notes, advanced]
        else:
            # Allow extra commas in the free-text notes field by re-joining the tail.
            values = rec[: n_cols - 1] + [",".join(rec[n_cols - 1 :])]
        rows.append(dict(zip(header, (f.strip() for f in values))))
    return rows
