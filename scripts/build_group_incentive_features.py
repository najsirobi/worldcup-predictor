#!/usr/bin/env python3
"""Build historical World Cup group-stage incentive features.

The output is an interim, reproducible feature table. Raw Kaggle files are read
but never edited. Team names are not normalized silently; model-matrix join
misses are reported for review.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.group_incentives import build_incentive_features_for_matches
from src.models.baselines import POISSON_AWAY_FEATURES, POISSON_HOME_FEATURES


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "raw" / "kaggle" / "world_cup_history" / "matches_1930_2022.csv"
MODEL_MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
TEAM_NAME_MAP = ROOT / "data" / "reference" / "team_name_map.csv"
OUT = ROOT / "data" / "interim" / "group_incentive_features.parquet"
REPORT = ROOT / "outputs" / "reports" / "group_incentive_feature_report.md"

# Same 32-team, 8-groups-of-4 group-stage format as the controlled 2018/2022
# backtests. Keeping this scope avoids mixing old best-third formats into the
# primary feature audit.
YEARS = (2010, 2014, 2018, 2022)


def df_to_md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _components(nodes: set[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    adj: dict[str, set[str]] = {node: set() for node in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    comps: list[set[str]] = []
    for node in sorted(nodes):
        if node in seen:
            continue
        stack = [node]
        comp: set[str] = set()
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def infer_group_labels(matches: pd.DataFrame) -> pd.DataFrame:
    """Infer group labels from round-robin connected components by year."""

    labelled = []
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for year, sub in matches.groupby("year", sort=True):
        teams = set(sub["team_a"]) | set(sub["team_b"])
        edges = list(zip(sub["team_a"], sub["team_b"]))
        comps = _components(teams, edges)

        comp_rows = []
        for comp in comps:
            mask = sub["team_a"].isin(comp) & sub["team_b"].isin(comp)
            comp_match_rows = sub[mask]
            comp_rows.append(
                {
                    "teams": comp,
                    "min_date": comp_match_rows["date"].min(),
                    "min_source_order": int(comp_match_rows["_source_order"].min()),
                    "n_teams": len(comp),
                    "n_matches": len(comp_match_rows),
                }
            )
        comp_rows = sorted(comp_rows, key=lambda row: (row["min_date"], row["min_source_order"]))
        team_to_group: dict[str, str] = {}
        for idx, comp in enumerate(comp_rows):
            group = labels[idx]
            for team in comp["teams"]:
                team_to_group[team] = group

        out = sub.copy()
        out["group"] = out["team_a"].map(team_to_group)
        labelled.append(out)
    return pd.concat(labelled, ignore_index=True)


def load_historical_group_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(HISTORY)
    raw = raw.reset_index(names="_source_order")
    raw["date"] = pd.to_datetime(raw["Date"])
    group = raw[
        raw["Year"].isin(YEARS)
        & raw["Round"].astype(str).str.casefold().eq("group stage")
    ].copy()
    group = group.rename(
        columns={
            "Year": "year",
            "home_team": "team_a",
            "away_team": "team_b",
            "home_score": "team_a_goals",
            "away_score": "team_b_goals",
        }
    )
    mapping = pd.read_csv(TEAM_NAME_MAP)
    history_map = mapping[mapping["source"].eq("world_cup_history")].set_index("raw_name")[
        "canonical_team_name"
    ].to_dict()
    group["team_a_source_name"] = group["team_a"]
    group["team_b_source_name"] = group["team_b"]
    group["team_a"] = group["team_a"].map(history_map).fillna(group["team_a"])
    group["team_b"] = group["team_b"].map(history_map).fillna(group["team_b"])
    group["tournament_id"] = "WC-" + group["year"].astype(str)
    group["tournament_name"] = group["year"].astype(str) + " FIFA World Cup"
    group = infer_group_labels(group)
    group = group.sort_values(["year", "date", "_source_order"]).reset_index(drop=True)
    group["match_number"] = group.groupby("year").cumcount() + 1
    return group[
        [
            "year",
            "tournament_id",
            "tournament_name",
            "match_number",
            "group",
            "date",
            "team_a",
            "team_b",
            "team_a_source_name",
            "team_b_source_name",
            "team_a_goals",
            "team_b_goals",
            "_source_order",
        ]
    ], raw


def merge_model_matrix(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mm = pd.read_parquet(MODEL_MATRIX)
    mm["date"] = pd.to_datetime(mm["date"])
    mm["year"] = mm["date"].dt.year
    mm = mm[
        (mm["tournament"].eq("FIFA World Cup"))
        & (mm["year"].isin(YEARS))
    ].copy()
    feature_cols = sorted(
        set(
            [
                "year",
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "home_goals",
                "away_goals",
                "result_label",
                "home_points",
                "away_points",
                "goal_diff",
                "total_goals",
                "home_elo",
                "away_elo",
                "elo_diff",
                "home_fifa_rank",
                "away_fifa_rank",
                "home_fifa_points",
                "away_fifa_points",
                "fifa_rank_diff",
                "fifa_points_diff",
                "match_year",
                "tournament_category",
            ]
            + POISSON_HOME_FEATURES
            + POISSON_AWAY_FEATURES
        )
    )
    feature_cols = [col for col in feature_cols if col in mm.columns]
    mm_join = mm[feature_cols].rename(
        columns={
            "home_team": "team_a",
            "away_team": "team_b",
            "home_score": "model_matrix_team_a_goals",
            "away_score": "model_matrix_team_b_goals",
            "home_goals": "model_matrix_home_goals",
            "away_goals": "model_matrix_away_goals",
        }
    )

    lookup: dict[tuple[int, pd.Timestamp, frozenset[str]], pd.Series] = {}
    for _, row in mm_join.iterrows():
        key = (
            int(row["year"]),
            pd.Timestamp(row["date"]),
            frozenset([str(row["team_a"]), str(row["team_b"])]),
        )
        lookup[key] = row

    records = []
    missing_rows = []
    for _, row in matches.iterrows():
        key = (
            int(row["year"]),
            pd.Timestamp(row["date"]),
            frozenset([str(row["team_a"]), str(row["team_b"])]),
        )
        base = row.to_dict()
        mm_row = lookup.get(key)
        if mm_row is None:
            missing_rows.append(
                {
                    "year": row["year"],
                    "date": row["date"],
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "team_a_source_name": row.get("team_a_source_name"),
                    "team_b_source_name": row.get("team_b_source_name"),
                    "team_a_goals": row["team_a_goals"],
                    "team_b_goals": row["team_b_goals"],
                }
            )
            records.append(base)
            continue

        # Use the model-matrix orientation for model features and lambdas. The
        # group label/match number still come from the historical group graph.
        for col, value in mm_row.items():
            base[col] = value
        base["team_a"] = mm_row["team_a"]
        base["team_b"] = mm_row["team_b"]
        base["team_a_goals"] = int(mm_row["model_matrix_team_a_goals"])
        base["team_b_goals"] = int(mm_row["model_matrix_team_b_goals"])
        records.append(base)

    merged = pd.DataFrame.from_records(records)
    missing = pd.DataFrame.from_records(missing_rows)
    return merged, missing


def write_report(features: pd.DataFrame, missing: pd.DataFrame, raw_rows: int) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    by_year = (
        features.groupby("year")
        .agg(
            matches=("match_number", "count"),
            groups=("group", "nunique"),
            final_group_matches=("final_group_match_flag", "sum"),
            team_a_low=("team_a_low_incentive_flag", "sum"),
            team_b_low=("team_b_low_incentive_flag", "sum"),
            team_a_high=("team_a_high_incentive_flag", "sum"),
            team_b_high=("team_b_high_incentive_flag", "sum"),
            favourite_low=("favourite_low_incentive", "sum"),
            underdog_high=("underdog_high_incentive", "sum"),
        )
        .reset_index()
    )
    final = features[features["final_group_match_flag"]].copy()
    segment_counts = pd.DataFrame(
        [
            {
                "segment": "favourite already clinched first",
                "n_matches": int(
                    (
                        (final["favourite_side"].eq("team_a") & final["team_a_has_clinched_1st"])
                        | (final["favourite_side"].eq("team_b") & final["team_b_has_clinched_1st"])
                    ).sum()
                ),
            },
            {
                "segment": "favourite already qualified",
                "n_matches": int(
                    (
                        (final["favourite_side"].eq("team_a") & final["team_a_has_clinched_top2"])
                        | (final["favourite_side"].eq("team_b") & final["team_b_has_clinched_top2"])
                    ).sum()
                ),
            },
            {
                "segment": "one team eliminated",
                "n_matches": int((final["team_a_is_eliminated"] | final["team_b_is_eliminated"]).sum()),
            },
            {
                "segment": "one team must win",
                "n_matches": int((final["team_a_must_win_for_top2"] | final["team_b_must_win_for_top2"]).sum()),
            },
            {
                "segment": "both teams low incentive",
                "n_matches": int(final["both_low_incentive"].sum()),
            },
            {
                "segment": "one high incentive vs opponent low",
                "n_matches": int(
                    (
                        (final["team_a_high_incentive_flag"] & final["team_b_low_incentive_flag"])
                        | (final["team_b_high_incentive_flag"] & final["team_a_low_incentive_flag"])
                    ).sum()
                ),
            },
        ]
    )

    lines = [
        "# Group Incentive Feature Report",
        "",
        "## Source and Scope",
        "",
        f"- Raw source: `{HISTORY.relative_to(ROOT)}`",
        f"- Raw rows read: **{raw_rows}**",
        f"- Scope: FIFA World Cup group-stage matches for **{', '.join(map(str, YEARS))}**.",
        "- Group labels are inferred from the round-robin match graph per year; no raw data is edited.",
        "- This scope matches the modern 32-team top-two group format used by the WC2018/WC2022 controlled backtests.",
        "",
        "## Leakage Boundary",
        "",
        "- For each labelled match, team table stats use only group matches with dates strictly before that match date.",
        "- Same-date final group fixtures are not treated as prior information for each other.",
        "- Possibility flags enumerate unknown W/D/L outcomes for the current and remaining group fixtures; actual current/later results are not read.",
        "- Tie-break handling is conservative at the points level: possible qualification gives the team the benefit of tied points; clinched qualification assumes tied teams could pass it.",
        "",
        "## Outputs",
        "",
        f"- Feature parquet: `{OUT.relative_to(ROOT)}`",
        f"- Rows: **{len(features)}**",
        f"- Columns: **{len(features.columns)}**",
        "",
        "## Counts by Year",
        "",
        df_to_md(by_year),
        "## Final-Match Segment Sample Sizes",
        "",
        df_to_md(segment_counts),
        "## Model-Matrix Join Coverage",
        "",
        f"- Rows missing an explicit-map, unordered-pair join to `model_matrix_baseline`: **{len(missing)} / {len(features)}**",
    ]
    if not missing.empty:
        lines.extend(
            [
                "",
                "These team names were left as-is and should be reviewed through `data/reference/team_name_map.csv` before any normalization is introduced:",
                "",
                df_to_md(missing.head(40)),
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `has_clinched_advancement_if_best_thirds_supported` is only meaningful when best-thirds are enabled. The historical scope here is top-two only, so it remains false except where a future/live best-thirds context uses the same helper.",
            "- `low_incentive_flag` includes final-match teams that have clinched first/top-two or are already eliminated.",
            "- `high_incentive_flag` marks final-match teams still alive for top two and not already qualified, with `must_win_for_top2` separated explicitly.",
            "- No production model was retrained and no frozen candidate file was read or modified by this feature build.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    matches, raw = load_historical_group_matches()
    merged, missing = merge_model_matrix(matches)
    features = build_incentive_features_for_matches(merged, best_thirds_supported=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT, index=False)
    write_report(features, missing, raw_rows=len(raw))
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(features)} rows)")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    if len(missing):
        print(f"Model-matrix join misses: {len(missing)}")


if __name__ == "__main__":
    main()
