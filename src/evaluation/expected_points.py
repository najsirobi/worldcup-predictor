"""Expected FIF8A points for group-stage predictions (Phase 4, Task D).

Combines a model's scoreline probability matrix with the FIF8A scoring rules and
the template odds to compute the expected points of any candidate prediction,
and to select the most-probable and expected-points-maximising predictions.

Scoring (group match), from RULES_AND_SCORING.md / scoring_rules.yml:
  correct outcome -> base(6) * template_odd_of_predicted_outcome
  exact goal difference -> +2 (flat, only if outcome correct)
  exact score -> +3 (flat, only if outcome correct)
"""
import numpy as np

OUTCOMES = ["a_win", "draw", "b_win"]  # 'a' = home/Team A, 'b' = away/Team B


def outcome_probs_from_matrix(M):
    """[p_a_win, p_draw, p_b_win] from scoreline matrix M[a_goals, b_goals]."""
    a = np.tril(M, -1).sum()   # a_goals > b_goals
    d = np.trace(M)
    b = np.triu(M, 1).sum()
    return np.array([a, d, b])


def _outcome_of(a, b):
    return "a_win" if a > b else ("draw" if a == b else "b_win")


def expected_points_for_score(pred_a, pred_b, M, odds, rules):
    """Expected FIF8A points for predicting scoreline (pred_a, pred_b).

    odds: dict {'a_win','draw','b_win'} template odds.
    rules: scoring-rules dict (needs the three group-match point fields).
    """
    base = rules["group_match_correct_outcome_base_points"]
    gd_bonus = rules["group_match_exact_goal_difference_bonus"]
    exact_bonus = rules["group_match_exact_score_bonus"]

    n = M.shape[0]
    pred_outcome = _outcome_of(pred_a, pred_b)
    pred_gd = pred_a - pred_b

    ev = 0.0
    for ag in range(n):
        for bg in range(n):
            p = M[ag, bg]
            if p <= 0:
                continue
            if _outcome_of(ag, bg) != pred_outcome:
                continue  # bonuses only stack on correct outcome
            pts = base * odds[pred_outcome]
            if (ag - bg) == pred_gd:
                pts += gd_bonus
            if ag == pred_a and bg == pred_b:
                pts += exact_bonus
            ev += p * pts
    return ev


def most_probable_score(M):
    i, j = np.unravel_index(np.argmax(M), M.shape)
    return int(i), int(j)


def ev_max_score(M, odds, rules, max_goals=6):
    """Scoreline maximising expected points (search small scoreline grid)."""
    best, best_ev = None, -1.0
    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            ev = expected_points_for_score(a, b, M, odds, rules)
            if ev > best_ev:
                best, best_ev = (a, b), ev
    return best, best_ev


def expected_points_for_outcome(outcome, M, odds, rules):
    """Expected points if you only commit to an outcome (no exact-score/gd claim).

    Picks the most likely scoreline consistent with that outcome as the entry,
    so gd/exact bonuses are still possible. Returns (ev, chosen_score)."""
    n = M.shape[0]
    mask = np.zeros_like(M, dtype=bool)
    for a in range(n):
        for b in range(n):
            if _outcome_of(a, b) == outcome:
                mask[a, b] = True
    masked = np.where(mask, M, -1)
    a, b = np.unravel_index(np.argmax(masked), M.shape)
    return expected_points_for_score(int(a), int(b), M, odds, rules), (int(a), int(b))
