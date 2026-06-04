"""Load / validate / parse the FIF8A 2026 group-stage fixture & odds template.

The canonical source for the 72 group fixtures and their template odds is
``Rules of the game/RULES_AND_SCORING.md`` (§6). ``parse_fif8a_md`` extracts the
fixtures from that file; the extracted table is persisted to
``data/reference/fif8a_group_stage_template.csv`` and read back by
``load_fif8a_group_template``.

Odds are template-derived (FIFA-ranking-point Poisson, odd = 1/probability) —
NOT bookmaker odds.
"""
import logging
import re
from pathlib import Path

import pandas as pd

from .common import get_data_dir, load_csv_file, validate_required_columns

logger = logging.getLogger(__name__)

TEMPLATE_COLUMNS = [
    "match_number", "group", "date",
    "team_a", "rate_a", "rate_draw", "rate_b", "team_b",
    "source", "source_date", "notes",
]

GROUPS = list("ABCDEFGHIJKL")
EXPECTED_MATCHES_PER_GROUP = 6
EXPECTED_TOTAL_MATCHES = 72

_GROUP_HEADER_RE = re.compile(r"^####\s+Group\s+([A-L])\s*$")


def parse_fif8a_md(md_path: Path = None) -> pd.DataFrame:
    """Parse the 72 group fixtures + odds from RULES_AND_SCORING.md §6.

    Returns a DataFrame with the canonical TEMPLATE_COLUMNS. Raises on a clearly
    malformed table row rather than guessing. Does NOT invent missing fixtures.
    """
    if md_path is None:
        md_path = get_data_dir().parent / "Rules of the game" / "RULES_AND_SCORING.md"
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"RULES_AND_SCORING.md not found at {md_path}")

    rows = []
    current_group = None
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = _GROUP_HEADER_RE.match(line)
        if m:
            current_group = m.group(1)
            continue
        if current_group is None or not line.startswith("|"):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        if cells[0] in ("#", ""):  # header row
            continue
        if set("".join(cells)) <= set("-: "):  # separator row
            continue

        num, date, team_a, rate_a, rate_draw, rate_b, team_b = cells
        try:
            match_number = int(num)
            ra, rd, rb = float(rate_a), float(rate_draw), float(rate_b)
        except ValueError as e:
            raise ValueError(
                f"Malformed fixture row in group {current_group}: {cells} ({e})"
            )
        rows.append({
            "match_number": match_number,
            "group": current_group,
            "date": date,
            "team_a": team_a,
            "rate_a": ra,
            "rate_draw": rd,
            "rate_b": rb,
            "team_b": team_b,
            "source": "Rules of the game/RULES_AND_SCORING.md §6",
            "source_date": "2026-06-03",
            "notes": "odds rounded to 2dp as published in RULES_AND_SCORING.md; underlying FIFA rankings dated 2026-05-21",
        })

    df = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)
    df = df.sort_values("match_number").reset_index(drop=True)
    return df


def load_fif8a_group_template() -> pd.DataFrame:
    """Load the persisted FIF8A group-stage template CSV."""
    path = get_data_dir("reference") / "fif8a_group_stage_template.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"FIF8A group template not found at {path}. "
            "Generate it with scripts/build_fif8a_template.py."
        )
    df = load_csv_file(path)
    return df


def validate_fif8a_group_template(df: pd.DataFrame, require_full: bool = True) -> None:
    """Validate the FIF8A group template.

    Always checked: required columns, non-null teams/odds, positive odds,
    unique match numbers, groups within A-L.
    If ``require_full``: exactly 12 groups, 6 matches each, 72 total.

    Raises ValueError listing every problem found.
    """
    validate_required_columns(df, TEMPLATE_COLUMNS, name="fif8a_group_stage_template")
    problems = []

    for col in ("team_a", "team_b"):
        if df[col].isna().any():
            problems.append(f"'{col}' contains null team names")

    for col in ("rate_a", "rate_draw", "rate_b"):
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            problems.append(f"'{col}' contains non-numeric or null odds")
        elif (vals <= 0).any():
            problems.append(f"'{col}' contains non-positive odds")

    if df["match_number"].duplicated().any():
        dups = sorted(df.loc[df["match_number"].duplicated(keep=False), "match_number"].unique())
        problems.append(f"duplicate match_number values: {dups}")

    bad_groups = sorted(set(df["group"].dropna()) - set(GROUPS))
    if bad_groups:
        problems.append(f"unexpected group labels (not A-L): {bad_groups}")

    if require_full:
        if len(df) != EXPECTED_TOTAL_MATCHES:
            problems.append(f"expected {EXPECTED_TOTAL_MATCHES} matches, found {len(df)}")
        present_groups = sorted(set(df["group"].dropna()))
        if present_groups != GROUPS:
            problems.append(f"expected groups {GROUPS}, found {present_groups}")
        counts = df.groupby("group").size()
        wrong = {g: int(n) for g, n in counts.items() if n != EXPECTED_MATCHES_PER_GROUP}
        if wrong:
            problems.append(f"groups without exactly {EXPECTED_MATCHES_PER_GROUP} matches: {wrong}")

    if problems:
        raise ValueError(
            "Invalid FIF8A group template:\n  - " + "\n  - ".join(problems)
        )
