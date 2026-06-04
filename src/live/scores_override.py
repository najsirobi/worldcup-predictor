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

COLUMNS = [
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

VALID_STATUSES = ("scheduled", "played", "postponed", "void")
EXPECTED_MATCH_COUNT = 72


def utc_now_iso() -> str:
    """Return the current UTC timestamp as a stable ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_initial_override(template_path: Path = TEMPLATE_PATH) -> pd.DataFrame:
    """Build the initial override frame from the frozen fixture template.

    Every match starts as ``scheduled`` with empty goals. Goal columns are kept
    as a nullable integer dtype so an un-entered score is a genuine missing value
    rather than ``0``.
    """
    template = pd.read_csv(template_path)
    frame = pd.DataFrame(
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
        }
    )
    return frame[COLUMNS].sort_values("match_number").reset_index(drop=True)


def load_override(path: Path = OVERRIDE_PATH) -> pd.DataFrame:
    """Load the override file with the goal columns as nullable integers."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Score override not found at {path}. Run scripts/init_scores_override.py first."
        )
    frame = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Override file missing columns: {missing}")
    frame["match_number"] = frame["match_number"].astype(int)
    for col in ("team_a_goals", "team_b_goals"):
        frame[col] = frame[col].astype("Int64")
    for col in ("status", "source", "updated_at", "notes"):
        frame[col] = frame[col].fillna("").astype(str)
    return frame[COLUMNS]


def write_override(frame: pd.DataFrame, path: Path = OVERRIDE_PATH) -> None:
    """Validate then write the override frame to disk."""
    validate_override(frame)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].sort_values("match_number").to_csv(path, index=False)


def validate_override(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the override frame violates an invariant."""
    if list(frame.columns) != COLUMNS:
        # Allow column subset/superset as long as required columns exist.
        missing = [c for c in COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Override file missing columns: {missing}")
    if frame["match_number"].duplicated().any():
        dupes = sorted(frame.loc[frame["match_number"].duplicated(), "match_number"].tolist())
        raise ValueError(f"Duplicate match rows detected for match_number(s): {dupes}")
    bad_status = sorted(set(frame["status"]) - set(VALID_STATUSES))
    if bad_status:
        raise ValueError(f"Invalid status value(s) {bad_status}; allowed: {VALID_STATUSES}")
    for _, row in frame.iterrows():
        _validate_goals(row["team_a_goals"], "team_a_goals", row["match_number"])
        _validate_goals(row["team_b_goals"], "team_b_goals", row["match_number"])
        if row["status"] == "played":
            if pd.isna(row["team_a_goals"]) or pd.isna(row["team_b_goals"]):
                raise ValueError(
                    f"Match {row['match_number']} is marked 'played' but is missing goals; "
                    "both team_a_goals and team_b_goals are required."
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
) -> pd.DataFrame:
    """Return a copy of ``frame`` with one match row updated and validated.

    A ``status`` of ``played`` requires both goals to be present (either already
    on the row or supplied in this call). Raises ``ValueError`` if the match
    number does not exist or the resulting row is invalid.
    """
    frame = frame.copy()
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

    if status == "played":
        ga = frame.at[idx, "team_a_goals"]
        gb = frame.at[idx, "team_b_goals"]
        if pd.isna(ga) or pd.isna(gb):
            raise ValueError(
                f"Match {match_number}: status 'played' requires both team_a_goals and "
                "team_b_goals."
            )
    elif status in ("scheduled", "void"):
        # Clearing a result: drop any goals so downstream treats it as unplayed.
        frame.at[idx, "team_a_goals"] = pd.NA
        frame.at[idx, "team_b_goals"] = pd.NA

    frame.at[idx, "status"] = status
    frame.at[idx, "source"] = source
    frame.at[idx, "updated_at"] = utc_now_iso()
    if notes is not None:
        frame.at[idx, "notes"] = notes

    frame["team_a_goals"] = frame["team_a_goals"].astype("Int64")
    frame["team_b_goals"] = frame["team_b_goals"].astype("Int64")
    validate_override(frame)
    return frame
