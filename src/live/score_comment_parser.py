"""Parse score updates out of a GitHub Issue comment (Travel Mode, Task D).

Expected comment block (anywhere in the comment body):

    /WK-SCORES
    match_number,team_a_goals,team_b_goals,status,notes
    1,2,1,played,Mexico v South Africa
    2,1,1,played,Korea Republic v Czechia
    /END-WK-SCORES

Only the text strictly between the ``/WK-SCORES`` and ``/END-WK-SCORES`` markers
is parsed. The first non-empty line inside the block must be the exact CSV
header. Rows are returned as dicts ready for :func:`src.live.batch_update.apply_batch`;
this module does NOT touch the override frame -- value validation (match exists,
integer goals, played-needs-goals, etc.) happens there so the rules stay in one
place.
"""

from __future__ import annotations

import csv
import io

START_MARKER = "/WK-SCORES"
END_MARKER = "/END-WK-SCORES"
REQUIRED_HEADER = ["match_number", "team_a_goals", "team_b_goals", "status", "notes"]


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
    if header != REQUIRED_HEADER:
        raise CommentParseError(
            f"Invalid header {header}; expected exactly {REQUIRED_HEADER}."
        )

    data_rows = records[1:]
    if not data_rows:
        raise CommentParseError("No data rows found after the header.")

    rows: list[dict] = []
    for i, rec in enumerate(data_rows, start=1):
        if len(rec) < len(REQUIRED_HEADER):
            raise CommentParseError(
                f"Data row {i} has {len(rec)} field(s); expected "
                f"{len(REQUIRED_HEADER)} ({','.join(REQUIRED_HEADER)})."
            )
        # Allow extra commas in the free-text notes field by re-joining the tail.
        fields = rec[: len(REQUIRED_HEADER) - 1] + [",".join(rec[len(REQUIRED_HEADER) - 1 :])]
        rows.append(dict(zip(REQUIRED_HEADER, (f.strip() for f in fields))))
    return rows
