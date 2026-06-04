"""Time-aware backtesting helpers (Phase 4, Task C).

No random splits: always train on the past, test on the future.
"""
import numpy as np
import pandas as pd

CLASSES = ["home_win", "draw", "away_win"]


def time_split(df, train_end_date, test_mask):
    """Train = rows on/before train_end_date; test = rows matching test_mask."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= pd.Timestamp(train_end_date)]
    test = df[test_mask]
    return train, test


def scoreline_metrics(matrices, actual_a, actual_b, eps=1e-12):
    """Metrics for a list of scoreline matrices vs actual scores.

    matrices[i] is M[a_goals, b_goals]. Returns dict with scoreline log loss,
    exact-score / goal-difference / outcome hit rates.
    """
    n = len(matrices)
    ll = 0.0
    exact = gd = outc = 0
    for M, aa, ab in zip(matrices, actual_a, actual_b):
        k = M.shape[0] - 1
        ca, cb = min(int(aa), k), min(int(ab), k)
        ll += -np.log(max(M[ca, cb], eps))
        i, j = np.unravel_index(np.argmax(M), M.shape)  # most probable score
        if (i, j) == (ca, cb):
            exact += 1
        if (i - j) == (ca - cb):
            gd += 1
        po = "home" if i > j else ("draw" if i == j else "away")
        ao = "home" if aa > ab else ("draw" if aa == ab else "away")
        if po == ao:
            outc += 1
    return {
        "scoreline_log_loss": float(round(ll / n, 4)),
        "exact_score_hit_rate": float(round(exact / n, 4)),
        "goal_diff_hit_rate": float(round(gd / n, 4)),
        "outcome_hit_rate": float(round(outc / n, 4)),
    }


# canonical scoreline used to turn a W/D/L pick into a concrete predicted score
CANONICAL_SCORE = {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}


def realized_points_odds1(pred_scores, actual_a, actual_b, rules):
    """Average realized FIF8A group-match points with template odds approximated
    as 1.0 (historical odds unavailable). Comparable across models."""
    base = rules["group_match_correct_outcome_base_points"]
    gd_b = rules["group_match_exact_goal_difference_bonus"]
    ex_b = rules["group_match_exact_score_bonus"]
    total = 0.0
    for (pa, pb), aa, ab in zip(pred_scores, actual_a, actual_b):
        po = "home" if pa > pb else ("draw" if pa == pb else "away")
        ao = "home" if aa > ab else ("draw" if aa == ab else "away")
        if po != ao:
            continue
        pts = base * 1.0
        if (pa - pb) == (aa - ab):
            pts += gd_b
        if pa == aa and pb == ab:
            pts += ex_b
        total += pts
    return round(total / len(pred_scores), 4)


def most_probable_scores(matrices):
    out = []
    for M in matrices:
        i, j = np.unravel_index(np.argmax(M), M.shape)
        out.append((int(i), int(j)))
    return out
