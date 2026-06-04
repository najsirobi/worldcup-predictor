#!/usr/bin/env python3
"""Build model_matrix_plus by left-joining controlled squad/coach features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
BASELINE = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
SQUAD = ROOT / "data" / "interim" / "squad_features.parquet"
COACH = ROOT / "data" / "interim" / "coach_features.parquet"
OUT = ROOT / "data" / "processed" / "model_matrix_plus.parquet"
REPORT = ROOT / "outputs" / "reports" / "model_matrix_plus_report.md"

SQUAD_DIFF_COLS = [
    "squad_player_count",
    "squad_total_value",
    "squad_top_11_value",
    "squad_top_15_value",
    "top_3_attacker_value",
    "attacker_depth_value",
    "squad_avg_age",
    "squad_median_age",
    "squad_age_std",
    "players_with_position",
    "players_with_age",
    "attackers_identified",
]
COACH_DIFF_COLS = [
    "coach_tenure_days",
    "coach_matches_before_match",
    "coach_winrate_before_match",
    "coach_goal_diff_per_match_before_match",
]
TARGET_COLUMNS = ["home_score", "away_score", "result_label", "home_goals", "away_goals"]


def tournament_name(row: pd.Series) -> str:
    if row["tournament"] == "FIFA World Cup":
        return f"{int(row['match_year'])} FIFA World Cup"
    return ""


def add_team_features(
    baseline: pd.DataFrame,
    features: pd.DataFrame,
    *,
    side: str,
    team_col: str,
    prefix: str,
    join_cols: list[str],
) -> pd.DataFrame:
    renamed = features.rename(columns={team_col: f"{side}_team"})
    feature_cols = [column for column in renamed.columns if column not in join_cols + [f"{side}_team"]]
    renamed = renamed[join_cols + [f"{side}_team"] + feature_cols].add_prefix(prefix)
    left = baseline.copy()
    left_keys = {f"{prefix}{column}": column for column in join_cols}
    left = left.rename(columns={value: key for key, value in left_keys.items()})
    left = left.merge(
        renamed,
        left_on=[f"{prefix}{column}" for column in join_cols] + [f"{side}_team"],
        right_on=[f"{prefix}{column}" for column in join_cols] + [f"{prefix}{side}_team"],
        how="left",
    )
    left = left.rename(columns=left_keys)
    extra_team_col = f"{prefix}{side}_team"
    if extra_team_col in left:
        left = left.drop(columns=[extra_team_col])
    return left


def build_plus_matrix(baseline: pd.DataFrame, squad: pd.DataFrame, coach: pd.DataFrame) -> pd.DataFrame:
    out = baseline.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["tournament_name"] = out.apply(tournament_name, axis=1)

    squad_features = squad.rename(columns={"team": "team"})
    out = add_team_features(out, squad_features, side="home", team_col="team", prefix="home_squad_", join_cols=["tournament_name"])
    out = add_team_features(out, squad_features, side="away", team_col="team", prefix="away_squad_", join_cols=["tournament_name"])

    coach_features = coach.rename(columns={"team": "team"})
    coach_features["match_date"] = pd.to_datetime(coach_features["match_date"])
    out["match_date"] = out["date"]
    out = add_team_features(out, coach_features, side="home", team_col="team", prefix="home_coach_", join_cols=["tournament_name", "match_date"])
    out = add_team_features(out, coach_features, side="away", team_col="team", prefix="away_coach_", join_cols=["tournament_name", "match_date"])

    out["has_squad_features"] = out["home_squad_has_squad_features"].fillna(False) & out["away_squad_has_squad_features"].fillna(False)
    out["has_attacker_features"] = out["home_squad_has_attacker_features"].fillna(False) & out["away_squad_has_attacker_features"].fillna(False)
    out["has_wc2026_squad_features"] = out["home_squad_has_wc2026_squad_features"].fillna(False) & out["away_squad_has_wc2026_squad_features"].fillna(False)
    out["has_coach_features"] = out["home_coach_has_coach_features"].fillna(False) & out["away_coach_has_coach_features"].fillna(False)

    for col in SQUAD_DIFF_COLS:
        home = f"home_squad_{col}"
        away = f"away_squad_{col}"
        if home in out and away in out:
            out[f"{col}_diff"] = out[home] - out[away]
    for col in COACH_DIFF_COLS:
        home = f"home_coach_{col}"
        away = f"away_coach_{col}"
        if home in out and away in out:
            out[f"{col}_diff"] = out[home] - out[away]
    return out


def main() -> None:
    baseline = pd.read_parquet(BASELINE)
    squad = pd.read_parquet(SQUAD) if SQUAD.exists() else pd.DataFrame()
    coach = pd.read_parquet(COACH) if COACH.exists() else pd.DataFrame()
    if squad.empty or coach.empty:
        raise FileNotFoundError("Run build_squad_features.py and build_coach_features.py first.")

    plus = build_plus_matrix(baseline, squad, coach)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plus.to_parquet(OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as handle:
        handle.write("# Model Matrix Plus Report\n\n")
        handle.write(f"- Baseline rows: **{len(baseline)}**\n")
        handle.write(f"- Plus rows: **{len(plus)}**\n")
        handle.write(f"- Baseline columns: **{len(baseline.columns)}**\n")
        handle.write(f"- Plus columns: **{len(plus.columns)}**\n")
        for col in TARGET_COLUMNS:
            handle.write(f"- Preserves `{col}`: **{col in plus.columns}**\n")
        handle.write("\n## Coverage\n\n")
        handle.write("| Feature flag | Rows true | Share |\n|---|--:|--:|\n")
        for col in ["has_squad_features", "has_attacker_features", "has_wc2026_squad_features", "has_coach_features"]:
            handle.write(f"| {col} | {int(plus[col].sum())} | {plus[col].mean():.3f} |\n")
        handle.write("\nPlayer/coach features are sparse by design and remain missing outside supported World Cup rows.\n")


if __name__ == "__main__":
    main()
