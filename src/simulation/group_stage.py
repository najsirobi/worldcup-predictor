"""Monte Carlo group-stage simulation from Poisson scoreline matrices (Phase 4, Task E).

Samples scorelines for all 72 group matches, builds the 12 group tables, applies
table sorting (points, goal difference, goals for, random tie-break), and computes
finish probabilities including the 8 best third-placed teams that advance.

Tie-break limitation: real FIFA tie-breakers also use head-to-head and fair-play;
here we use points -> GD -> GF -> random, which is documented as an approximation.
"""
import numpy as np
import pandas as pd

from src.features.group_incentives import (
    IncentiveAdjustmentConfig,
    adjust_score_matrix_for_incentives,
    compute_team_state_from_table,
    expected_goals_from_matrix,
)

MAX_GOALS_DEFAULT = 10


def _sample_scores(M, n, rng):
    """Sample n (home_goals, away_goals) pairs from scoreline matrix M."""
    k = M.shape[0]
    flat = (M / M.sum()).ravel()
    idx = rng.choice(k * k, size=n, p=flat)
    return idx // k, idx % k  # home_goals, away_goals


def _empty_path_table(teams):
    return {
        team: {"points": 0, "goals_for": 0, "goals_against": 0, "goal_difference": 0, "played": 0}
        for team in teams
    }


def _apply_path_score(table, team_a, team_b, goals_a, goals_b):
    a = table[team_a]
    b = table[team_b]
    a["played"] += 1
    b["played"] += 1
    a["goals_for"] += int(goals_a)
    a["goals_against"] += int(goals_b)
    b["goals_for"] += int(goals_b)
    b["goals_against"] += int(goals_a)
    a["goal_difference"] = a["goals_for"] - a["goals_against"]
    b["goal_difference"] = b["goals_for"] - b["goals_against"]
    if goals_a > goals_b:
        a["points"] += 3
    elif goals_a < goals_b:
        b["points"] += 3
    else:
        a["points"] += 1
        b["points"] += 1


def _path_rank_table(table, rng):
    teams = list(table)
    key = {
        team: (
            table[team]["points"] * 1e6
            + table[team]["goal_difference"] * 1e3
            + table[team]["goals_for"]
            + float(rng.random()) * 1e-3
        )
        for team in teams
    }
    ordered = sorted(teams, key=lambda team: key[team], reverse=True)
    return {team: rank for rank, team in enumerate(ordered, start=1)}, key


def _sort_matches_for_path(group_matches):
    cols = [col for col in ("date", "match_number") if col in group_matches.columns]
    if cols:
        out = group_matches.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
        return out.sort_values(cols).reset_index(drop=True)
    return group_matches.sort_values("match_number").reset_index(drop=True)


def simulate_groups_with_path_incentives(
    group_matches,
    matrices,
    lambdas=None,
    n_sims=20000,
    seed=42,
    incentive_adjustment_config=None,
    return_diagnostics=False,
):
    """Path-dependent group simulation with optional final-match incentives.

    This is intentionally separate from the vectorized baseline path because
    incentives depend on the simulated table before each final group fixture.
    """

    rng = np.random.default_rng(seed)
    config = incentive_adjustment_config or IncentiveAdjustmentConfig()
    groups = sorted(group_matches["group"].unique())
    per_group_rank_records = {
        group: {
            "teams": sorted(
                set(group_matches[group_matches["group"] == group]["team_a"])
                | set(group_matches[group_matches["group"] == group]["team_b"])
            ),
            "pts": [],
            "gd": [],
            "gf": [],
            "ranks": [],
            "keys": [],
        }
        for group in groups
    }
    diagnostics = {
        "adjustment_applications": 0,
        "low_incentive_team_counts": {},
        "adjusted_match_numbers": {},
    }

    third_keys = np.zeros((n_sims, len(groups)))
    for sim_idx in range(n_sims):
        for gi, group in enumerate(groups):
            gm = _sort_matches_for_path(group_matches[group_matches["group"] == group])
            teams = sorted(set(gm["team_a"]) | set(gm["team_b"]))
            table = _empty_path_table(teams)

            for row_idx, row in gm.iterrows():
                match_number = int(row["match_number"])
                matrix = matrices[match_number]
                team_a = row["team_a"]
                team_b = row["team_b"]
                final_group_match = table[team_a]["played"] >= 2 and table[team_b]["played"] >= 2
                if incentive_adjustment_config is not None and final_group_match:
                    remaining_rows = gm.iloc[row_idx:]
                    remaining_fixtures = [
                        {
                            "match_number": int(fixture["match_number"]),
                            "team_a": fixture["team_a"],
                            "team_b": fixture["team_b"],
                        }
                        for _, fixture in remaining_rows.iterrows()
                    ]
                    current_fixture = {
                        "match_number": match_number,
                        "team_a": team_a,
                        "team_b": team_b,
                    }
                    state_a = compute_team_state_from_table(
                        table,
                        remaining_fixtures,
                        current_fixture,
                        team_a,
                        best_thirds_supported=False,
                        final_group_match=True,
                    )
                    state_b = compute_team_state_from_table(
                        table,
                        remaining_fixtures,
                        current_fixture,
                        team_b,
                        best_thirds_supported=False,
                        final_group_match=True,
                    )
                    lambda_a, lambda_b = (
                        lambdas[match_number]
                        if lambdas and match_number in lambdas
                        else expected_goals_from_matrix(matrix)
                    )
                    adjusted, meta = adjust_score_matrix_for_incentives(
                        matrix,
                        lambda_a,
                        lambda_b,
                        state_a,
                        state_b,
                        final_group_match=True,
                        config=config,
                    )
                    if meta["applied"]:
                        diagnostics["adjustment_applications"] += 1
                        diagnostics["adjusted_match_numbers"][match_number] = (
                            diagnostics["adjusted_match_numbers"].get(match_number, 0) + 1
                        )
                        for team, state in ((team_a, state_a), (team_b, state_b)):
                            if state["low_incentive_flag"]:
                                diagnostics["low_incentive_team_counts"][team] = (
                                    diagnostics["low_incentive_team_counts"].get(team, 0) + 1
                                )
                    matrix = adjusted
                flat = (matrix / matrix.sum()).ravel()
                idx = rng.choice(matrix.shape[0] * matrix.shape[1], p=flat)
                goals_a, goals_b = int(idx // matrix.shape[1]), int(idx % matrix.shape[1])
                _apply_path_score(table, team_a, team_b, goals_a, goals_b)

            ranks, key = _path_rank_table(table, rng)
            info = per_group_rank_records[group]
            info["pts"].append([table[team]["points"] for team in info["teams"]])
            info["gd"].append([table[team]["goal_difference"] for team in info["teams"]])
            info["gf"].append([table[team]["goals_for"] for team in info["teams"]])
            info["ranks"].append([ranks[team] for team in info["teams"]])
            info["keys"].append([key[team] for team in info["teams"]])
            third_team = next(team for team in teams if ranks[team] == 3)
            third_keys[sim_idx, gi] = key[third_team]

    third_better = np.zeros((n_sims, len(groups)), dtype=int)
    for gi in range(len(groups)):
        third_better[:, gi] = (third_keys > third_keys[:, [gi]]).sum(axis=1)
    third_top8 = third_better < 8

    rows = []
    for gi, group in enumerate(groups):
        info = per_group_rank_records[group]
        pts = np.asarray(info["pts"], dtype=float)
        gd = np.asarray(info["gd"], dtype=float)
        ranks = np.asarray(info["ranks"], dtype=int)
        for team_idx, team in enumerate(info["teams"]):
            r = ranks[:, team_idx]
            p1 = np.mean(r == 1); p2 = np.mean(r == 2)
            p3 = np.mean(r == 3); p4 = np.mean(r == 4)
            advance_third = (r == 3) & third_top8[:, gi]
            rows.append({
                "group": group, "team": team,
                "expected_points": round(float(pts[:, team_idx].mean()), 3),
                "expected_goal_difference": round(float(gd[:, team_idx].mean()), 3),
                "p_finish_1st": round(float(p1), 4), "p_finish_2nd": round(float(p2), 4),
                "p_finish_3rd": round(float(p3), 4), "p_finish_4th": round(float(p4), 4),
                "p_top2": round(float(p1 + p2), 4), "p_top3": round(float(p1 + p2 + p3), 4),
                "p_advance_assuming_top2": round(float(p1 + p2), 4),
                "p_advance_with_best_thirds": round(float(p1 + p2 + advance_third.mean()), 4),
                "most_likely_group_rank": int(np.argmax([p1, p2, p3, p4]) + 1),
            })
    summary = pd.DataFrame(rows)
    if return_diagnostics:
        return summary, diagnostics
    return summary


def simulate_groups(
    group_matches,
    matrices,
    n_sims=20000,
    seed=42,
    incentive_adjustment_config=None,
    lambdas=None,
    return_diagnostics=False,
):
    """Simulate all groups jointly.

    group_matches: DataFrame with columns group, match_number, team_a, team_b
                   (canonical/display names) for the 72 fixtures.
    matrices: dict match_number -> scoreline matrix (home=team_a, away=team_b).
    Returns a per-team summary DataFrame.
    """
    if incentive_adjustment_config is not None:
        return simulate_groups_with_path_incentives(
            group_matches,
            matrices,
            lambdas=lambdas,
            n_sims=n_sims,
            seed=seed,
            incentive_adjustment_config=incentive_adjustment_config,
            return_diagnostics=return_diagnostics,
        )
    rng = np.random.default_rng(seed)
    groups = sorted(group_matches["group"].unique())

    # Pass 1: per-group team points/gd/gf arrays + ranks + third-place key
    per_group = {}
    third_keys = np.zeros((n_sims, len(groups)))
    for gi, g in enumerate(groups):
        gm = group_matches[group_matches["group"] == g]
        teams = sorted(set(gm["team_a"]) | set(gm["team_b"]))
        tidx = {t: i for i, t in enumerate(teams)}
        pts = np.zeros((n_sims, len(teams)))
        gf = np.zeros((n_sims, len(teams)))
        ga = np.zeros((n_sims, len(teams)))
        for _, mrow in gm.iterrows():
            hg, ag = _sample_scores(matrices[int(mrow["match_number"])], n_sims, rng)
            a, b = tidx[mrow["team_a"]], tidx[mrow["team_b"]]
            gf[:, a] += hg; ga[:, a] += ag
            gf[:, b] += ag; ga[:, b] += hg
            pts[:, a] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            pts[:, b] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gd = gf - ga
        # lexicographic key with tiny random tie-break (ensures unique ranks)
        key = pts * 1e6 + gd * 1e3 + gf + rng.random((n_sims, len(teams))) * 1e-3
        ranks = np.empty_like(key, dtype=int)
        for t in range(len(teams)):
            ranks[:, t] = (key > key[:, [t]]).sum(axis=1) + 1
        per_group[g] = {"teams": teams, "pts": pts, "gd": gd, "gf": gf,
                        "ranks": ranks, "key": key}
        # key of whichever team finished 3rd in this group, per sim
        third_keys[:, gi] = (key * (ranks == 3)).sum(axis=1)

    # Pass 2: best-thirds — top 8 of the 12 third-placed teams advance
    third_better = np.zeros((n_sims, len(groups)), dtype=int)
    for gi in range(len(groups)):
        third_better[:, gi] = (third_keys > third_keys[:, [gi]]).sum(axis=1)
    third_top8 = third_better < 8  # (n_sims, n_groups)

    rows = []
    for gi, g in enumerate(groups):
        d = per_group[g]
        for t, team in enumerate(d["teams"]):
            r = d["ranks"][:, t]
            p1 = np.mean(r == 1); p2 = np.mean(r == 2)
            p3 = np.mean(r == 3); p4 = np.mean(r == 4)
            advance_third = (r == 3) & third_top8[:, gi]
            rows.append({
                "group": g, "team": team,
                "expected_points": round(float(d["pts"][:, t].mean()), 3),
                "expected_goal_difference": round(float(d["gd"][:, t].mean()), 3),
                "p_finish_1st": round(float(p1), 4), "p_finish_2nd": round(float(p2), 4),
                "p_finish_3rd": round(float(p3), 4), "p_finish_4th": round(float(p4), 4),
                "p_top2": round(float(p1 + p2), 4), "p_top3": round(float(p1 + p2 + p3), 4),
                "p_advance_assuming_top2": round(float(p1 + p2), 4),
                "p_advance_with_best_thirds": round(float(p1 + p2 + advance_third.mean()), 4),
                "most_likely_group_rank": int(np.argmax([p1, p2, p3, p4]) + 1),
            })
    return pd.DataFrame(rows)
