"""Full-tournament simulation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.group_stage import _sample_scores
from src.simulation.knockout_bracket import assign_official_r32_matches, assign_r32_teams
from src.simulation.match_probability import knockout_match_probability


def source_positions_from_group_view(group_view: pd.DataFrame) -> dict[str, str]:
    """Build A1/A2/A3... source positions from an existing group submission view."""
    required = {"group", "team", "suggested_group_standing", "likely_best_third_signal", "p_top3"}
    missing = required - set(group_view.columns)
    if missing:
        raise ValueError(f"Group view missing required columns: {sorted(missing)}")
    source: dict[str, str] = {}
    for _, row in group_view.iterrows():
        group = str(row["group"])
        standing = int(row["suggested_group_standing"])
        source[f"{group}{standing}"] = str(row["team"])
    third = group_view[group_view["suggested_group_standing"].eq(3)].sort_values(
        ["likely_best_third_signal", "p_top3"], ascending=False
    )
    for idx, (_, row) in enumerate(third.head(8).iterrows(), start=1):
        source[f"BT{idx}"] = str(row["team"])
    return source


def _simulate_round(
    teams: list[str],
    group_view: pd.DataFrame,
    rng: np.random.Generator,
) -> list[str]:
    if len(teams) % 2 != 0:
        raise ValueError("Knockout round must contain an even number of teams.")
    winners = []
    for idx in range(0, len(teams), 2):
        team_a = teams[idx]
        team_b = teams[idx + 1]
        probability = knockout_match_probability(team_a, team_b, group_view)
        winners.append(team_a if rng.random() < probability.p_team_a_advance else team_b)
    if len(winners) != len(set(winners)):
        raise ValueError(f"Duplicate team in simulated knockout round: {winners}")
    return winners


def simulate_full_tournament_from_mapping(
    mapping: pd.DataFrame,
    group_view: pd.DataFrame,
    *,
    n_sims: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate knockout progression from explicit R32 mapping and group-view source positions."""
    rng = np.random.default_rng(seed)
    source = source_positions_from_group_view(group_view)
    r32 = assign_r32_teams(mapping, source).sort_values("slot")
    r32_teams = r32["team"].tolist()
    if len(r32_teams) != 32 or len(set(r32_teams)) != 32:
        raise ValueError("R32 bracket must contain 32 unique teams.")

    teams = sorted(group_view["team"].unique())
    counts = {
        team: {
            "reach_r32": 0,
            "reach_r16": 0,
            "reach_qf": 0,
            "reach_sf": 0,
            "reach_final": 0,
            "win_world_cup": 0,
        }
        for team in teams
    }
    for team in r32_teams:
        counts[team]["reach_r32"] = n_sims

    for _ in range(n_sims):
        r16 = _simulate_round(r32_teams, group_view, rng)
        qf = _simulate_round(r16, group_view, rng)
        sf = _simulate_round(qf, group_view, rng)
        final = _simulate_round(sf, group_view, rng)
        winner = _simulate_round(final, group_view, rng)[0]
        for team in r16:
            counts[team]["reach_r16"] += 1
        for team in qf:
            counts[team]["reach_qf"] += 1
        for team in sf:
            counts[team]["reach_sf"] += 1
        for team in final:
            counts[team]["reach_final"] += 1
        counts[winner]["win_world_cup"] += 1

    group_lookup = group_view.drop_duplicates("team").set_index("team")["group"].to_dict()
    p_top2 = group_view.drop_duplicates("team").set_index("team")["p_top2"].to_dict()
    p_advance = group_view.drop_duplicates("team").set_index("team").get("p_advance_with_best_thirds")
    rows = []
    for team in teams:
        row_counts = counts[team]
        rows.append(
            {
                "team": team,
                "group": group_lookup.get(team, ""),
                "p_win_group": float(
                    group_view.loc[group_view["team"].eq(team), "p_finish_1st"].iloc[0]
                    if "p_finish_1st" in group_view.columns
                    else 0.0
                ),
                "p_finish_top2": float(p_top2.get(team, 0.0)),
                "p_advance_group": float(p_advance.get(team, p_top2.get(team, 0.0))) if p_advance is not None else float(p_top2.get(team, 0.0)),
                "p_reach_r32": row_counts["reach_r32"] / n_sims,
                "p_reach_r16": row_counts["reach_r16"] / n_sims,
                "p_reach_qf": row_counts["reach_qf"] / n_sims,
                "p_reach_sf": row_counts["reach_sf"] / n_sims,
                "p_reach_final": row_counts["reach_final"] / n_sims,
                "p_win_world_cup": row_counts["win_world_cup"] / n_sims,
                "expected_tournament_points_proxy": (
                    row_counts["reach_qf"] / n_sims * 20
                    + row_counts["reach_sf"] / n_sims * 40
                    + row_counts["reach_final"] / n_sims * 60
                    + row_counts["win_world_cup"] / n_sims * 100
                ),
                "notes": "Knockout simulation from explicit bracket mapping and fallback match_probability.",
            }
        )
    return pd.DataFrame(rows)


def validate_no_duplicate_round(teams: list[str]) -> None:
    if len(teams) != len(set(teams)):
        raise ValueError(f"Duplicate team in knockout round: {teams}")


def _group_standings_for_sim(
    group_matches: pd.DataFrame,
    sampled_scores: dict[int, tuple[np.ndarray, np.ndarray]],
    sim_index: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    rows = []
    source_positions: dict[str, str] = {}
    third_rows = []

    for group_name in sorted(group_matches["group"].unique()):
        matches = group_matches[group_matches["group"].eq(group_name)]
        teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
        table = {
            team: {
                "group": group_name,
                "team": team,
                "points": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "tie_break": float(rng.random()),
            }
            for team in teams
        }
        for _, match in matches.iterrows():
            goals_a, goals_b = sampled_scores[int(match["match_number"])]
            score_a = int(goals_a[sim_index])
            score_b = int(goals_b[sim_index])
            team_a = str(match["team_a"])
            team_b = str(match["team_b"])
            table[team_a]["goals_for"] += score_a
            table[team_a]["goals_against"] += score_b
            table[team_b]["goals_for"] += score_b
            table[team_b]["goals_against"] += score_a
            if score_a > score_b:
                table[team_a]["points"] += 3
            elif score_b > score_a:
                table[team_b]["points"] += 3
            else:
                table[team_a]["points"] += 1
                table[team_b]["points"] += 1

        for team_row in table.values():
            team_row["goal_difference"] = team_row["goals_for"] - team_row["goals_against"]
        ordered = sorted(
            table.values(),
            key=lambda item: (
                item["points"],
                item["goal_difference"],
                item["goals_for"],
                item["tie_break"],
            ),
            reverse=True,
        )
        for rank, team_row in enumerate(ordered, start=1):
            team_row["rank"] = rank
            source_positions[f"{group_name}{rank}"] = str(team_row["team"])
            rows.append(team_row)
            if rank == 3:
                third_rows.append(team_row)

    third_ordered = sorted(
        third_rows,
        key=lambda item: (
            item["points"],
            item["goal_difference"],
            item["goals_for"],
            item["tie_break"],
        ),
        reverse=True,
    )
    qualified_third_groups = [str(row["group"]) for row in third_ordered[:8]]
    return pd.DataFrame(rows), source_positions, qualified_third_groups


def _resolve_source(source: str, winners: dict[int, str], losers: dict[int, str]) -> str:
    value = str(source)
    if value.startswith("W"):
        return winners[int(value[1:])]
    if value.startswith("L"):
        return losers[int(value[1:])]
    raise ValueError(f"Unsupported knockout source: {source}")


def _simulate_match_winner(
    team_a: str,
    team_b: str,
    group_view: pd.DataFrame,
    rng: np.random.Generator,
    probability_cache: dict[tuple[str, str], float] | None = None,
) -> tuple[str, str]:
    if probability_cache is not None:
        cache_key = (team_a, team_b)
        if cache_key not in probability_cache:
            probability_cache[cache_key] = knockout_match_probability(team_a, team_b, group_view).p_team_a_advance
        p_team_a_advance = probability_cache[cache_key]
    else:
        p_team_a_advance = knockout_match_probability(team_a, team_b, group_view).p_team_a_advance
    if rng.random() < p_team_a_advance:
        return team_a, team_b
    return team_b, team_a


def simulate_full_tournament_official(
    group_matches: pd.DataFrame,
    matrices: dict[int, np.ndarray],
    r32_mapping: pd.DataFrame,
    progression: pd.DataFrame,
    annex: pd.DataFrame,
    group_view: pd.DataFrame,
    *,
    n_sims: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate group stage, Annexe C R32 assignment, and knockouts through final."""
    required_matches = {"group", "match_number", "team_a", "team_b"}
    missing_matches = required_matches - set(group_matches.columns)
    if missing_matches:
        raise ValueError(f"Group matches missing columns: {sorted(missing_matches)}")

    rng = np.random.default_rng(seed)
    teams = sorted(set(group_matches["team_a"]) | set(group_matches["team_b"]))
    group_lookup = (
        group_matches[["group", "team_a"]]
        .rename(columns={"team_a": "team"})
        .drop_duplicates()
        .set_index("team")["group"]
        .to_dict()
    )
    group_lookup.update(
        group_matches[["group", "team_b"]]
        .rename(columns={"team_b": "team"})
        .drop_duplicates()
        .set_index("team")["group"]
        .to_dict()
    )

    counts = {
        team: {
            "win_group": 0,
            "finish_top2": 0,
            "advance_group": 0,
            "reach_r32": 0,
            "reach_r16": 0,
            "reach_qf": 0,
            "reach_sf": 0,
            "reach_final": 0,
            "win_world_cup": 0,
        }
        for team in teams
    }

    sampled_scores = {
        int(match_number): _sample_scores(matrix, n_sims, rng)
        for match_number, matrix in matrices.items()
    }
    probability_cache: dict[tuple[str, str], float] = {}
    knockout_rounds = progression.sort_values("match_number")
    for sim_index in range(n_sims):
        standings, source_positions, qualified_third_groups = _group_standings_for_sim(
            group_matches,
            sampled_scores,
            sim_index,
            rng,
        )
        for _, row in standings.iterrows():
            team = str(row["team"])
            rank = int(row["rank"])
            if rank == 1:
                counts[team]["win_group"] += 1
            if rank <= 2:
                counts[team]["finish_top2"] += 1
                counts[team]["advance_group"] += 1
            elif rank == 3 and str(row["group"]) in qualified_third_groups:
                counts[team]["advance_group"] += 1

        r32 = assign_official_r32_matches(r32_mapping, source_positions, annex, qualified_third_groups)
        winners: dict[int, str] = {}
        losers: dict[int, str] = {}
        for _, row in r32.iterrows():
            team_a = str(row["team_a"])
            team_b = str(row["team_b"])
            counts[team_a]["reach_r32"] += 1
            counts[team_b]["reach_r32"] += 1
            winner, loser = _simulate_match_winner(team_a, team_b, group_view, rng, probability_cache)
            winners[int(row["match_number"])] = winner
            losers[int(row["match_number"])] = loser

        for _, row in knockout_rounds.iterrows():
            round_name = str(row["round"])
            match_number = int(row["match_number"])
            if round_name == "Third-place":
                team_a = _resolve_source(row["team_a_source"], winners, losers)
                team_b = _resolve_source(row["team_b_source"], winners, losers)
                winner, loser = _simulate_match_winner(team_a, team_b, group_view, rng, probability_cache)
                winners[match_number] = winner
                losers[match_number] = loser
                continue

            team_a = _resolve_source(row["team_a_source"], winners, losers)
            team_b = _resolve_source(row["team_b_source"], winners, losers)
            if round_name == "R16":
                counts[team_a]["reach_r16"] += 1
                counts[team_b]["reach_r16"] += 1
            elif round_name == "QF":
                counts[team_a]["reach_qf"] += 1
                counts[team_b]["reach_qf"] += 1
            elif round_name == "SF":
                counts[team_a]["reach_sf"] += 1
                counts[team_b]["reach_sf"] += 1
            elif round_name == "Final":
                counts[team_a]["reach_final"] += 1
                counts[team_b]["reach_final"] += 1

            winner, loser = _simulate_match_winner(team_a, team_b, group_view, rng, probability_cache)
            winners[match_number] = winner
            losers[match_number] = loser
            if str(row["winner_to_match"]) == "Winner":
                counts[winner]["win_world_cup"] += 1

    rows = []
    for team in teams:
        row_counts = counts[team]
        rows.append(
            {
                "team": team,
                "group": group_lookup.get(team, ""),
                "p_win_group": row_counts["win_group"] / n_sims,
                "p_finish_top2": row_counts["finish_top2"] / n_sims,
                "p_advance_group": row_counts["advance_group"] / n_sims,
                "p_reach_r32": row_counts["reach_r32"] / n_sims,
                "p_reach_r16": row_counts["reach_r16"] / n_sims,
                "p_reach_qf": row_counts["reach_qf"] / n_sims,
                "p_reach_sf": row_counts["reach_sf"] / n_sims,
                "p_reach_final": row_counts["reach_final"] / n_sims,
                "p_win_world_cup": row_counts["win_world_cup"] / n_sims,
                "expected_tournament_points_proxy": (
                    row_counts["reach_qf"] / n_sims * 20
                    + row_counts["reach_sf"] / n_sims * 40
                    + row_counts["reach_final"] / n_sims * 60
                    + row_counts["win_world_cup"] / n_sims * 100
                ),
                "notes": "Official FIFA bracket and Annexe C; group stage sampled from Phase 4.5 Poisson scoreline matrices.",
            }
        )
    return pd.DataFrame(rows)
