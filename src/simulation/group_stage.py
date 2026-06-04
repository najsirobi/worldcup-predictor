"""Monte Carlo group-stage simulation from Poisson scoreline matrices (Phase 4, Task E).

Samples scorelines for all 72 group matches, builds the 12 group tables, applies
table sorting (points, goal difference, goals for, random tie-break), and computes
finish probabilities including the 8 best third-placed teams that advance.

Tie-break limitation: real FIFA tie-breakers also use head-to-head and fair-play;
here we use points -> GD -> GF -> random, which is documented as an approximation.
"""
import numpy as np
import pandas as pd

MAX_GOALS_DEFAULT = 10


def _sample_scores(M, n, rng):
    """Sample n (home_goals, away_goals) pairs from scoreline matrix M."""
    k = M.shape[0]
    flat = (M / M.sum()).ravel()
    idx = rng.choice(k * k, size=n, p=flat)
    return idx // k, idx % k  # home_goals, away_goals


def simulate_groups(group_matches, matrices, n_sims=20000, seed=42):
    """Simulate all groups jointly.

    group_matches: DataFrame with columns group, match_number, team_a, team_b
                   (canonical/display names) for the 72 fixtures.
    matrices: dict match_number -> scoreline matrix (home=team_a, away=team_b).
    Returns a per-team summary DataFrame.
    """
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
