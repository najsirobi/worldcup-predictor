"""Live group-stage Monte Carlo on top of frozen predictions (Travel Mode, Task C).

For each group we simulate ``n_sims`` completions of the group stage:

* **Played** matches are pinned to their actual entered scoreline.
* **Unplayed** matches are sampled from independent Poisson distributions whose
  means are read off the published ``final_recommended_score`` for that fixture.

This deliberately re-uses the existing baseline scorelines as the fallback
distribution -- it does NOT retrain anything and does NOT alter the baseline
predictions. When no matches have been entered, the simulation is effectively a
Monte Carlo expansion of the frozen final-candidate picks.

The FIFA 2026 format advances 32 of 48 teams: the top two from each of the 12
groups plus the 8 best third-placed teams.

Tie-break limitation: ranking within a group uses points -> goal difference ->
goals for -> random draw (to break exact ties uniformly across sims). The 8 best
thirds are chosen by points -> GD -> GF -> random. Head-to-head and fair-play
tie-breakers are not modelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Poisson mean for an unplayed match is the predicted goals, floored so that even
# a 0-goal prediction keeps some variance. This is a transparent smoothing of the
# published scoreline, not a new model.
GOAL_FLOOR = 0.25
N_ADVANCING_THIRDS = 8

SUMMARY_COLUMNS = [
    "group",
    "team",
    "matches_played",
    "current_points",
    "p_rank1",
    "p_rank2",
    "p_rank3",
    "p_rank4",
    "p_top2",
    "p_advance",
    "expected_points",
    "expected_goal_difference",
    "status",
]


def _parse_score(text: str) -> tuple[float, float]:
    a, b = str(text).split("-")
    return float(a), float(b)


def build_match_lambdas(predictions: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Map match_number -> (lambda_a, lambda_b) from recommended scorelines."""
    lambdas: dict[int, tuple[float, float]] = {}
    for _, row in predictions.iterrows():
        ga, gb = _parse_score(row["final_recommended_score"])
        lambdas[int(row["match_number"])] = (max(ga, GOAL_FLOOR), max(gb, GOAL_FLOOR))
    return lambdas


def simulate_live(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
    n_sims: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the live group-stage Monte Carlo and return a per-team summary frame."""
    rng = np.random.default_rng(seed)
    lambdas = build_match_lambdas(predictions)

    # Exclude void matches entirely; they award no points and never get played.
    active = scores[scores["status"] != "void"].copy()
    groups = sorted(active["group"].unique())

    per_group: dict[str, dict] = {}
    third_keys = np.zeros((n_sims, len(groups)))

    for gi, group in enumerate(groups):
        gm = active[active["group"] == group]
        teams = sorted(set(gm["team_a"]) | set(gm["team_b"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        pts = np.zeros((n_sims, n))
        gf = np.zeros((n_sims, n))
        ga_arr = np.zeros((n_sims, n))
        played_count = np.zeros(n, dtype=int)
        cur_points = np.zeros(n, dtype=int)

        for _, m in gm.iterrows():
            a, b = tidx[m["team_a"]], tidx[m["team_b"]]
            if m["status"] == "played":
                hg = np.full(n_sims, int(m["team_a_goals"]))
                ag = np.full(n_sims, int(m["team_b_goals"]))
                played_count[a] += 1
                played_count[b] += 1
                # Track the deterministic current points for reporting.
                if hg[0] > ag[0]:
                    cur_points[a] += 3
                elif hg[0] < ag[0]:
                    cur_points[b] += 3
                else:
                    cur_points[a] += 1
                    cur_points[b] += 1
            else:
                lam_a, lam_b = lambdas[int(m["match_number"])]
                hg = rng.poisson(lam_a, size=n_sims)
                ag = rng.poisson(lam_b, size=n_sims)

            gf[:, a] += hg
            ga_arr[:, a] += ag
            gf[:, b] += ag
            ga_arr[:, b] += hg
            pts[:, a] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            pts[:, b] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

        gd = gf - ga_arr
        key = pts * 1e6 + gd * 1e3 + gf + rng.random((n_sims, n)) * 1e-3
        ranks = np.empty_like(key, dtype=int)
        for t in range(n):
            ranks[:, t] = (key > key[:, [t]]).sum(axis=1) + 1

        # Key of whichever team finished 3rd in this group, per sim.
        third_mask = ranks == 3
        third_keys[:, gi] = (key * third_mask).sum(axis=1)

        per_group[group] = {
            "teams": teams,
            "pts": pts,
            "gd": gd,
            "ranks": ranks,
            "played_count": played_count,
            "cur_points": cur_points,
        }

    # Which groups' third-placed team is among the 8 best thirds, per sim.
    # argsort descending; the top N_ADVANCING_THIRDS columns advance.
    order = np.argsort(-third_keys, axis=1)
    advancing_third_group = order[:, :N_ADVANCING_THIRDS]
    third_advances = np.zeros((n_sims, len(groups)), dtype=bool)
    rows = np.arange(n_sims)[:, None]
    third_advances[rows, advancing_third_group] = True

    records = []
    for gi, group in enumerate(groups):
        info = per_group[group]
        teams = info["teams"]
        ranks = info["ranks"]
        n = len(teams)
        all_played = int(info["played_count"].max(initial=0)) == 3 if n else False
        for t, team in enumerate(teams):
            r = ranks[:, t]
            p_rank1 = float((r == 1).mean())
            p_rank2 = float((r == 2).mean())
            p_rank3 = float((r == 3).mean())
            p_rank4 = float((r == 4).mean())
            p_top2 = p_rank1 + p_rank2
            advanced = (r <= 2) | ((r == 3) & third_advances[:, gi])
            p_advance = float(advanced.mean())
            exp_points = float(info["pts"][:, t].mean())
            exp_gd = float(info["gd"][:, t].mean())
            matches_played = int(info["played_count"][t])

            if matches_played == 3:
                status = "advanced" if p_advance > 0.5 else "eliminated"
            elif p_advance >= 0.995:
                status = "advanced (clinched)"
            elif p_advance <= 0.005:
                status = "eliminated"
            else:
                status = "active"

            records.append(
                {
                    "group": group,
                    "team": team,
                    "matches_played": matches_played,
                    "current_points": int(info["cur_points"][t]),
                    "p_rank1": round(p_rank1, 4),
                    "p_rank2": round(p_rank2, 4),
                    "p_rank3": round(p_rank3, 4),
                    "p_rank4": round(p_rank4, 4),
                    "p_top2": round(p_top2, 4),
                    "p_advance": round(p_advance, 4),
                    "expected_points": round(exp_points, 3),
                    "expected_goal_difference": round(exp_gd, 3),
                    "status": status,
                }
            )

    summary = pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["group", "p_advance", "expected_points"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return summary


def load_predictions(path: Path) -> pd.DataFrame:
    preds = pd.read_csv(path)
    required = {"match_number", "final_recommended_score"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    return preds
