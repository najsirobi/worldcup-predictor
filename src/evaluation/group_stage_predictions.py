"""Helpers for Team A / Team B oriented group-stage prediction outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_KEYS = ("a_win", "draw", "b_win")


def score_to_string(score: tuple[int, int]) -> str:
    return f"{int(score[0])}-{int(score[1])}"


def parse_score(score_text: str) -> tuple[int, int]:
    left, right = str(score_text).split("-", 1)
    return int(left), int(right)


def outcome_key_from_score(score: tuple[int, int]) -> str:
    goals_a, goals_b = score
    if goals_a > goals_b:
        return "a_win"
    if goals_a < goals_b:
        return "b_win"
    return "draw"


def named_outcome_from_key(outcome_key: str, team_a: str, team_b: str) -> str:
    if outcome_key == "a_win":
        return team_a
    if outcome_key == "b_win":
        return team_b
    return "Draw"


def score_display(team_a: str, team_b: str, score: tuple[int, int]) -> str:
    goals_a, goals_b = score
    return f"{team_a} vs {team_b}: {goals_a}-{goals_b}"


def most_probable_score_for_outcome(M: np.ndarray, outcome_key: str) -> tuple[int, int]:
    mask = np.zeros_like(M, dtype=bool)
    for goals_a in range(M.shape[0]):
        for goals_b in range(M.shape[1]):
            if outcome_key_from_score((goals_a, goals_b)) == outcome_key:
                mask[goals_a, goals_b] = True
    masked = np.where(mask, M, -1.0)
    goals_a, goals_b = np.unravel_index(np.argmax(masked), M.shape)
    return int(goals_a), int(goals_b)


def add_score_columns(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()

    safe_scores = out["recommended_score_safe"].map(parse_score)
    ev_scores = out["recommended_score_ev"].map(parse_score)

    out["recommended_team_a_goals_safe"] = [score[0] for score in safe_scores]
    out["recommended_team_b_goals_safe"] = [score[1] for score in safe_scores]
    out["recommended_team_a_goals_ev"] = [score[0] for score in ev_scores]
    out["recommended_team_b_goals_ev"] = [score[1] for score in ev_scores]
    out["recommended_score_safe_display"] = [
        score_display(team_a, team_b, score)
        for team_a, team_b, score in zip(out["team_a"], out["team_b"], safe_scores)
    ]
    out["recommended_score_ev_display"] = [
        score_display(team_a, team_b, score)
        for team_a, team_b, score in zip(out["team_a"], out["team_b"], ev_scores)
    ]
    return out


def probability_sum_error(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return (frame[columns].sum(axis=1) - 1.0).abs()

