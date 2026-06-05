"""Read/validate/update the manual live score override file (Travel Mode, Task A).

The override file is the single source of truth for manually entered group-stage
results while travelling. It is initialised from the frozen fixture template and
only ever edited through :func:`update_match` (or the thin CLI wrappers), so the
validation rules below are guaranteed to hold for every downstream consumer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
OVERRIDE_PATH = ROOT / "data" / "live" / "scores_override.csv"

BASE_COLUMNS = [
    "match_number",
    "group",
    "date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "status",
    "source",
    "updated_at",
    "notes",
]
# ``advanced_team`` is the knockout-only extension: it pins which team goes
# through when a knockout match is level after normal/extra time (a shoot-out).
# It is optional and defaults to "" so existing group-only files stay valid.
COLUMNS = BASE_COLUMNS + ["advanced_team"]
REQUIRED_COLUMNS = BASE_COLUMNS

VALID_STATUSES = ("scheduled", "played", "postponed", "void")

# Group stage is matches 1-72; the knockout bracket is 73-104 (R32 -> Final).
EXPECTED_MATCH_COUNT = 72  # group-stage matches built from the fixture template
GROUP_STAGE_MATCH_MAX = 72
KNOCKOUT_MATCH_NUMBERS = tuple(range(73, 105))
TOTAL_MATCH_COUNT = 104

# Round label per knockout match number (used for placeholder ``group`` values).
KNOCKOUT_ROUND_BY_MATCH: dict[int, str] = {}
for _n in range(73, 89):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "R32"
for _n in range(89, 97):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "R16"
for _n in range(97, 101):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "QF"
for _n in (101, 102):
    KNOCKOUT_ROUND_BY_MATCH[_n] = "SF"
KNOCKOUT_ROUND_BY_MATCH[103] = "Third-place"
KNOCKOUT_ROUND_BY_MATCH[104] = "Final"


def utc_now_iso() -> str:
    """Return the current UTC timestamp as a stable ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def knockout_placeholder_rows() -> pd.DataFrame:
    """Scheduled placeholder rows for the 32 knockout matches (73-104).

    Knockout participants are not known until the feeding matches are resolved,
    so team names start blank. The round label is stored in ``group`` (R32, R16,
    QF, SF, Third-place, Final) purely for display; group-table logic ignores any
    match number above 72.
    """
    numbers = list(KNOCKOUT_MATCH_NUMBERS)
    return pd.DataFrame(
        {
            "match_number": pd.Series(numbers, dtype=int),
            "group": [KNOCKOUT_ROUND_BY_MATCH[n] for n in numbers],
            "date": "",
            "team_a": "",
            "team_b": "",
            "team_a_goals": pd.Series([pd.NA] * len(numbers), dtype="Int64"),
            "team_b_goals": pd.Series([pd.NA] * len(numbers), dtype="Int64"),
            "status": "scheduled",
            "source": "template",
            "updated_at": "",
            "notes": "",
            "advanced_team": "",
        }
    )[COLUMNS]


def ensure_knockout_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with any missing knockout rows (73-104) appended.

    Idempotent: existing rows are kept untouched; only absent knockout match
    numbers are added as scheduled placeholders. This lets group-only override
    files created before knockout support gain the 73-104 rows transparently."""
    present = set(frame["match_number"].astype(int))
    missing = [n for n in KNOCKOUT_MATCH_NUMBERS if n not in present]
    if not missing:
        return frame
    extra = knockout_placeholder_rows()
    extra = extra[extra["match_number"].isin(missing)]
    combined = pd.concat([frame, extra], ignore_index=True)
    combined["team_a_goals"] = combined["team_a_goals"].astype("Int64")
    combined["team_b_goals"] = combined["team_b_goals"].astype("Int64")
    return combined[COLUMNS].sort_values("match_number").reset_index(drop=True)


def build_initial_override(template_path: Path = TEMPLATE_PATH) -> pd.DataFrame:
    """Build the initial override frame: 72 group matches + 32 knockout placeholders.

    Every match starts as ``scheduled`` with empty goals. Goal columns are kept
    as a nullable integer dtype so an un-entered score is a genuine missing value
    rather than ``0``. Knockout rows (73-104) start with blank team names that are
    only filled in once their feeders resolve.
    """
    template = pd.read_csv(template_path)
    group = pd.DataFrame(
        {
            "match_number": template["match_number"].astype(int),
            "group": template["group"].astype(str),
            "date": template["date"].astype(str),
            "team_a": template["team_a"].astype(str),
            "team_b": template["team_b"].astype(str),
            "team_a_goals": pd.Series([pd.NA] * len(template), dtype="Int64"),
            "team_b_goals": pd.Series([pd.NA] * len(template), dtype="Int64"),
            "status": "scheduled",
            "source": "template",
            "updated_at": "",
            "notes": "",
            "advanced_team": "",
        }
    )[COLUMNS]
    frame = pd.concat([group, knockout_placeholder_rows()], ignore_index=True)
    frame["team_a_goals"] = frame["team_a_goals"].astype("Int64")
    frame["team_b_goals"] = frame["team_b_goals"].astype("Int64")
    return frame[COLUMNS].sort_values("match_number").reset_index(drop=True)


def load_override(path: Path = OVERRIDE_PATH) -> pd.DataFrame:
    """Load the override file with the goal columns as nullable integers.

    Backward compatible: a file without the optional ``advanced_team`` column
    gets it added as empty, and missing knockout rows (73-104) are appended so
    knockout results can be entered on files created before knockout support.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Score override not found at {path}. Run scripts/init_scores_override.py first."
        )
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Override file missing columns: {missing}")
    if "advanced_team" not in frame.columns:
        frame["advanced_team"] = ""
    frame["match_number"] = frame["match_number"].astype(int)
    for col in ("team_a_goals", "team_b_goals"):
        frame[col] = frame[col].astype("Int64")
    for col in ("group", "team_a", "team_b", "status", "source", "updated_at", "notes", "advanced_team"):
        frame[col] = frame[col].fillna("").astype(str)
    frame = ensure_knockout_rows(frame[COLUMNS])
    return frame[COLUMNS]


def write_override(frame: pd.DataFrame, path: Path = OVERRIDE_PATH) -> None:
    """Validate then write the override frame to disk."""
    validate_override(frame)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].sort_values("match_number").to_csv(path, index=False)


def validate_override(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the override frame violates an invariant."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Override file missing columns: {missing}")
    has_advanced = "advanced_team" in frame.columns
    if frame["match_number"].duplicated().any():
        dupes = sorted(frame.loc[frame["match_number"].duplicated(), "match_number"].tolist())
        raise ValueError(f"Duplicate match rows detected for match_number(s): {dupes}")
    bad_status = sorted(set(frame["status"]) - set(VALID_STATUSES))
    if bad_status:
        raise ValueError(f"Invalid status value(s) {bad_status}; allowed: {VALID_STATUSES}")
    for _, row in frame.iterrows():
        match_number = int(row["match_number"])
        is_knockout = match_number > GROUP_STAGE_MATCH_MAX
        _validate_goals(row["team_a_goals"], "team_a_goals", match_number)
        _validate_goals(row["team_b_goals"], "team_b_goals", match_number)
        if row["status"] == "played":
            if pd.isna(row["team_a_goals"]) or pd.isna(row["team_b_goals"]):
                raise ValueError(
                    f"Match {match_number} is marked 'played' but is missing goals; "
                    "both team_a_goals and team_b_goals are required."
                )
        advanced = str(row["advanced_team"]).strip() if has_advanced else ""
        if advanced:
            if not is_knockout:
                raise ValueError(
                    f"Match {match_number}: advanced_team is only valid for knockout "
                    "matches (73-104)."
                )
            teams = {str(row["team_a"]).strip(), str(row["team_b"]).strip()} - {""}
            if teams and advanced not in teams:
                raise ValueError(
                    f"Match {match_number}: advanced_team {advanced!r} must be one of "
                    f"{sorted(teams)}."
                )
        # A played knockout level after normal/extra time must name who advanced
        # (shoot-out winner) so the next round can resolve.
        if (
            is_knockout
            and row["status"] == "played"
            and not pd.isna(row["team_a_goals"])
            and not pd.isna(row["team_b_goals"])
            and int(row["team_a_goals"]) == int(row["team_b_goals"])
            and not advanced
        ):
            raise ValueError(
                f"Knockout match {match_number} is level "
                f"{int(row['team_a_goals'])}-{int(row['team_b_goals'])}; set advanced_team "
                "to the shoot-out winner so the next round can be filled."
            )


def _validate_goals(value, label: str, match_number) -> None:
    if pd.isna(value):
        return
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Match {match_number}: {label}={value!r} is not an integer.")
    if as_int != float(value):
        raise ValueError(f"Match {match_number}: {label}={value!r} is not a whole number.")
    if as_int < 0:
        raise ValueError(f"Match {match_number}: {label}={value!r} must be non-negative.")


def _coerce_goal(value) -> "pd.NA | int":
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    as_int = int(value)
    if as_int < 0:
        raise ValueError(f"Goals must be non-negative, got {value!r}.")
    return as_int


def update_match(
    frame: pd.DataFrame,
    match_number: int,
    team_a_goals=None,
    team_b_goals=None,
    status: str = "played",
    notes: str | None = None,
    source: str = "manual",
    advanced_team: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with one match row updated and validated.

    A ``status`` of ``played`` requires both goals to be present (either already
    on the row or supplied in this call). For a level knockout match (73-104),
    ``advanced_team`` names the shoot-out winner so the next round can resolve.
    Raises ``ValueError`` if the match number does not exist or the resulting row
    is invalid.
    """
    frame = frame.copy()
    if "advanced_team" not in frame.columns:
        frame["advanced_team"] = ""
    match_number = int(match_number)
    mask = frame["match_number"] == match_number
    if not mask.any():
        valid_range = f"{frame['match_number'].min()}-{frame['match_number'].max()}"
        raise ValueError(
            f"Match number {match_number} does not exist (valid range {valid_range})."
        )
    if mask.sum() > 1:
        raise ValueError(f"Duplicate rows for match {match_number}; file is corrupt.")

    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; allowed: {VALID_STATUSES}")

    idx = frame.index[mask][0]
    if team_a_goals is not None:
        frame.at[idx, "team_a_goals"] = _coerce_goal(team_a_goals)
    if team_b_goals is not None:
        frame.at[idx, "team_b_goals"] = _coerce_goal(team_b_goals)
    if advanced_team is not None:
        frame.at[idx, "advanced_team"] = str(advanced_team).strip()

    if status == "played":
        ga = frame.at[idx, "team_a_goals"]
        gb = frame.at[idx, "team_b_goals"]
        if pd.isna(ga) or pd.isna(gb):
            raise ValueError(
                f"Match {match_number}: status 'played' requires both team_a_goals and "
                "team_b_goals."
            )
    elif status in ("scheduled", "void"):
        # Clearing a result: drop any goals and advancement so downstream treats
        # it as unplayed.
        frame.at[idx, "team_a_goals"] = pd.NA
        frame.at[idx, "team_b_goals"] = pd.NA
        frame.at[idx, "advanced_team"] = ""

    frame.at[idx, "status"] = status
    frame.at[idx, "source"] = source
    frame.at[idx, "updated_at"] = utc_now_iso()
    if notes is not None:
        frame.at[idx, "notes"] = notes

    frame["team_a_goals"] = frame["team_a_goals"].astype("Int64")
    frame["team_b_goals"] = frame["team_b_goals"].astype("Int64")
    validate_override(frame)
    return frame
