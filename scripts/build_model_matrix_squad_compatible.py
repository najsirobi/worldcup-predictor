#!/usr/bin/env python3
"""Build the squad-compatible model matrix (Phase 5C, Task F).

Preserves every Phase 4.5 baseline row and feature, and left-joins ONLY the
historically-comparable squad features (age + position mix) and leak-free coach
features. Squad/coach features attach exclusively to World Cup rows where they
are historically valid; all other rows keep null features and a False flag.
WC2026-only squad features (height, club-country shares) are never added here.
Missing values stay null — never filled with fake zeros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.historical_squad_features import COMPARABLE_FEATURE_COLUMNS

ROOT = Path(__file__).parent.parent
BASELINE = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
SQUAD = ROOT / "data" / "interim" / "historical_squad_features.parquet"
COACH = ROOT / "data" / "interim" / "historical_coach_features.parquet"
OUT = ROOT / "data" / "processed" / "model_matrix_squad_compatible.parquet"
REPORT = ROOT / "outputs" / "reports" / "model_matrix_squad_compatible_report.md"

# Squad features carried as home-away differences (comparable + symmetric).
SQUAD_DIFF_COLS = COMPARABLE_FEATURE_COLUMNS
COACH_DIFF_COLS = [
    "coach_tenure_days_at_tournament_start",
    "coach_matches_before_match",
    "coach_winrate_before_match",
]
TARGET_COLUMNS = ["home_score", "away_score", "result_label", "home_goals", "away_goals"]


def _tournament_name(row: pd.Series) -> str:
    if row["tournament"] == "FIFA World Cup":
        return f"{int(row['match_year'])} FIFA World Cup"
    return ""


def _join_side(base: pd.DataFrame, feats: pd.DataFrame, side: str, keys: list[str],
               value_cols: list[str], prefix: str) -> pd.DataFrame:
    right = feats[keys + ["team"] + value_cols].copy()
    right = right.rename(columns={c: f"{prefix}{c}" for c in value_cols})
    right = right.rename(columns={"team": f"{side}_team"})
    return base.merge(right, on=keys + [f"{side}_team"], how="left")


def main() -> None:
    baseline = pd.read_parquet(BASELINE)
    squad = pd.read_parquet(SQUAD)
    coach = pd.read_parquet(COACH)

    out = baseline.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["tournament_name"] = out.apply(_tournament_name, axis=1)
    out["match_date"] = out["date"]

    # --- squad join (tournament_name + team) ---
    squad_cols = SQUAD_DIFF_COLS + ["has_historical_squad_features"]
    out = _join_side(out, squad, "home", ["tournament_name"], squad_cols, "home_squad_")
    out = _join_side(out, squad, "away", ["tournament_name"], squad_cols, "away_squad_")

    # --- coach join (tournament_name + match_date + team) ---
    coach = coach.copy()
    coach["match_date"] = pd.to_datetime(coach["match_date"])
    coach_cols = COACH_DIFF_COLS + ["has_historical_coach_features"]
    out = _join_side(out, coach, "home", ["tournament_name", "match_date"], coach_cols, "home_coach_")
    out = _join_side(out, coach, "away", ["tournament_name", "match_date"], coach_cols, "away_coach_")

    # --- presence flags (missing-data flags, default False) ---
    out["has_squad_features"] = (
        out["home_squad_has_historical_squad_features"].fillna(False).astype(bool)
        & out["away_squad_has_historical_squad_features"].fillna(False).astype(bool)
    )
    out["has_coach_features"] = (
        out["home_coach_has_historical_coach_features"].fillna(False).astype(bool)
        & out["away_coach_has_historical_coach_features"].fillna(False).astype(bool)
    )

    # --- difference features (NaN-preserving: null where a side is missing) ---
    for col in SQUAD_DIFF_COLS:
        out[f"{col}_diff"] = out[f"home_squad_{col}"] - out[f"away_squad_{col}"]
    for col in COACH_DIFF_COLS:
        out[f"{col}_diff"] = out[f"home_coach_{col}"] - out[f"away_coach_{col}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    # Leakage guard: no WC2026-only columns present.
    forbidden = [c for c in out.columns if "height" in c or "club" in c or "domestic" in c or "foreign" in c]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("# Model Matrix Squad-Compatible Report\n\n")
        fh.write(f"- Baseline rows: **{len(baseline)}**, columns: **{len(baseline.columns)}**\n")
        fh.write(f"- Squad-compatible rows: **{len(out)}**, columns: **{len(out.columns)}**\n")
        fh.write(f"- Row count preserved: **{len(out) == len(baseline)}**\n")
        for col in TARGET_COLUMNS:
            fh.write(f"- Preserves target `{col}`: **{col in out.columns}**\n")
        fh.write("\n## Feature scope\n\n")
        fh.write("- Squad diff features (comparable): " + ", ".join(f"`{c}_diff`" for c in SQUAD_DIFF_COLS) + "\n")
        fh.write("- Coach diff features (leak-free): " + ", ".join(f"`{c}_diff`" for c in COACH_DIFF_COLS) + "\n")
        fh.write(f"- WC2026-only / non-comparable columns present (must be 0): **{len(forbidden)}**\n\n")
        fh.write("## Coverage (rows with both teams' features)\n\n")
        fh.write("| Flag | Rows true | Share | WC rows true |\n|---|--:|--:|--:|\n")
        wc = out["tournament"].eq("FIFA World Cup")
        for col in ["has_squad_features", "has_coach_features"]:
            fh.write(f"| {col} | {int(out[col].sum())} | {out[col].mean():.4f} | {int((out[col] & wc).sum())} |\n")
        fh.write("\n## Squad coverage by World Cup year\n\n")
        fh.write("| Year | WC rows | rows w/ squad features |\n|---|--:|--:|\n")
        for yr in sorted(out.loc[wc, "match_year"].unique()):
            sub = out[wc & out["match_year"].eq(yr)]
            fh.write(f"| {int(yr)} | {len(sub)} | {int(sub['has_squad_features'].sum())} |\n")
        fh.write("\n- Squad/coach features attach only to World Cup rows; all other rows keep null features and `has_*_features=False`.\n")
        fh.write("- Missing squad/coach values are null, not zero. Difference features are null whenever either side is missing.\n")

    print(f"Wrote {OUT} ({out.shape}) and {REPORT}; forbidden cols={len(forbidden)}")


if __name__ == "__main__":
    main()
