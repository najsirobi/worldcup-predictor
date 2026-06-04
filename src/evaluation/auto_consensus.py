"""Deterministic score consensus helpers for final auto-science candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
import re
from typing import Iterable

import pandas as pd


SCORE_RE = re.compile(r"^\d{1,2}-\d{1,2}$")
OUTCOME_PROB_COLUMNS = {
    "a_win": "model_p_a_win",
    "draw": "model_p_draw",
    "b_win": "model_p_b_win",
}


@dataclass(frozen=True)
class AutoPolicyConfig:
    min_ev_uplift_to_override_safe: float = 0.25
    max_allowed_variance_flag_for_ev: bool = False
    contrarian_ev_allowed_by_default: bool = False


def parse_score(score: object) -> tuple[int, int]:
    text = str(score)
    if not SCORE_RE.match(text):
        raise ValueError(f"Invalid score `{score}`; expected X-Y with integer goals.")
    left, right = text.split("-", maxsplit=1)
    return int(left), int(right)


def score_outcome(score: object) -> str:
    goals_a, goals_b = parse_score(score)
    if goals_a > goals_b:
        return "a_win"
    if goals_a < goals_b:
        return "b_win"
    return "draw"


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def score_model_confidence(row: pd.Series, score: object) -> float | None:
    try:
        outcome = score_outcome(score)
    except ValueError:
        return None
    column = OUTCOME_PROB_COLUMNS[outcome]
    if column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def average_expected_points(candidates: pd.DataFrame, score: str) -> float | None:
    values = pd.to_numeric(
        candidates.loc[candidates["candidate_score"].eq(score), "expected_points"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return None
    return float(values.mean())


def _expected_points_for_score(base_row: pd.Series, score: str) -> float | None:
    safe = str(base_row.get("recommended_score_safe", ""))
    ev = str(base_row.get("recommended_score_ev", ""))
    if score == safe and "expected_points_safe" in base_row and not pd.isna(base_row["expected_points_safe"]):
        return float(base_row["expected_points_safe"])
    if score == ev and "expected_points_ev" in base_row and not pd.isna(base_row["expected_points_ev"]):
        return float(base_row["expected_points_ev"])
    return None


def _candidate_row(
    match_row: pd.Series,
    source: str,
    score: str,
    expected_points: float | None,
    is_ev: bool = False,
    is_safe: bool = False,
    notes: str = "",
) -> dict[str, object]:
    return {
        "match_number": int(match_row["match_number"]),
        "group": match_row["group"],
        "team_a": match_row["team_a"],
        "team_b": match_row["team_b"],
        "candidate_source": source,
        "candidate_score": score,
        "expected_points": expected_points,
        "model_confidence": score_model_confidence(match_row, score),
        "is_ev": is_ev,
        "is_safe": is_safe,
        "is_contrarian": "CONTRARIAN" in str(match_row.get("notes", "")).upper() if is_ev else False,
        "notes": notes,
    }


def collect_candidate_scores(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame | None = None,
    ensemble: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect available scientific candidate score recommendations."""

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    decision_by_match = decisions.set_index("match_number") if decisions is not None else pd.DataFrame()
    ensemble_by_match = ensemble.set_index("match_number") if ensemble is not None else pd.DataFrame()

    for _, match_row in predictions.sort_values("match_number").iterrows():
        match_number = int(match_row["match_number"])
        safe_score = str(match_row["recommended_score_safe"])
        ev_score = str(match_row["recommended_score_ev"])

        rows.append(
            _candidate_row(
                match_row,
                "safe_score",
                safe_score,
                float(match_row["expected_points_safe"]),
                is_safe=True,
                notes="Primary deterministic safe recommendation.",
            )
        )
        rows.append(
            _candidate_row(
                match_row,
                "ev_score",
                ev_score,
                float(match_row["expected_points_ev"]),
                is_ev=True,
                notes=str(match_row.get("notes", "")),
            )
        )

        if "most_probable_score" in match_row and pd.notna(match_row["most_probable_score"]):
            mp_score = str(match_row["most_probable_score"])
            rows.append(
                _candidate_row(
                    match_row,
                    "most_probable_score",
                    mp_score,
                    _expected_points_for_score(match_row, mp_score),
                    notes="Primary most probable scoreline.",
                )
            )
        else:
            skipped.append(f"match {match_number}: most_probable_score unavailable")

        if "ev_max_score" in match_row and pd.notna(match_row["ev_max_score"]):
            ev_max_score = str(match_row["ev_max_score"])
            rows.append(
                _candidate_row(
                    match_row,
                    "expected_points_max_score",
                    ev_max_score,
                    _expected_points_for_score(match_row, ev_max_score),
                    is_ev=ev_max_score == ev_score,
                    notes="Primary expected-points-max scoreline.",
                )
            )
        else:
            skipped.append(f"match {match_number}: ev_max_score unavailable")

        if not ensemble_by_match.empty and match_number in ensemble_by_match.index:
            ensemble_row = ensemble_by_match.loc[match_number]
            ensemble_score = str(ensemble_row["recommended_score_safe"])
            rows.append(
                {
                    **_candidate_row(
                        match_row,
                        "ensemble_score",
                        ensemble_score,
                        float(ensemble_row["expected_points_safe"])
                        if "expected_points_safe" in ensemble_row and not pd.isna(ensemble_row["expected_points_safe"])
                        else _expected_points_for_score(match_row, ensemble_score),
                        notes="Equal-weight ensemble safe recommendation.",
                    ),
                    "model_confidence": score_model_confidence(ensemble_row, ensemble_score),
                }
            )
        else:
            skipped.append(f"match {match_number}: ensemble score unavailable")

        if not decision_by_match.empty and match_number not in decision_by_match.index:
            skipped.append(f"match {match_number}: decision-table row unavailable")

    candidates = pd.DataFrame(rows)
    candidates = candidates[
        [
            "match_number",
            "group",
            "team_a",
            "team_b",
            "candidate_source",
            "candidate_score",
            "expected_points",
            "model_confidence",
            "is_ev",
            "is_safe",
            "is_contrarian",
            "notes",
        ]
    ]
    return candidates, skipped


def modal_score(
    candidates: pd.DataFrame,
    seed_scores: Iterable[str] | None = None,
    safe_score: str | None = None,
) -> tuple[str, int, str]:
    """Return modal score with expected-points and safe-score tie breaks."""

    scores = candidates["candidate_score"].astype(str).tolist()
    if seed_scores is not None:
        scores.extend(str(score) for score in seed_scores)
    counts = pd.Series(scores).value_counts()
    max_count = int(counts.max())
    tied = sorted(counts[counts.eq(max_count)].index.tolist())
    if len(tied) == 1:
        return tied[0], max_count, "modal_score"

    ep_by_score = {score: average_expected_points(candidates, score) for score in tied}
    best_ep = max(value for value in ep_by_score.values() if value is not None) if any(
        value is not None for value in ep_by_score.values()
    ) else None
    if best_ep is not None:
        ep_tied = sorted(score for score, value in ep_by_score.items() if value == best_ep)
        if len(ep_tied) == 1:
            return ep_tied[0], max_count, "tie_highest_average_expected_points"
        tied = ep_tied

    if safe_score is not None and safe_score in tied:
        return safe_score, max_count, "tie_final_fallback_safe_score"
    return tied[0], max_count, "tie_deterministic_lexical_fallback"


def select_final_scores(
    predictions: pd.DataFrame,
    final_v1: pd.DataFrame,
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    seed_scores_by_match: dict[int, list[str]] | None = None,
    config: AutoPolicyConfig = AutoPolicyConfig(),
) -> pd.DataFrame:
    """Apply the deterministic automatic final score policy."""

    v1_by_match = final_v1.set_index("match_number")
    decision_by_match = decisions.set_index("match_number")
    rows: list[dict[str, object]] = []
    for _, match_row in predictions.sort_values("match_number").iterrows():
        match_number = int(match_row["match_number"])
        match_candidates = candidates[candidates["match_number"].eq(match_number)]
        safe_score = str(match_row["recommended_score_safe"])
        ev_score = str(match_row["recommended_score_ev"])
        seed_scores = (seed_scores_by_match or {}).get(match_number, [])

        if match_candidates["candidate_score"].nunique() == 1:
            consensus_score = str(match_candidates["candidate_score"].iloc[0])
            support = len(match_candidates) + len(seed_scores)
            policy_reason = "all_scientific_sources_agree"
        else:
            consensus_score, support, policy_reason = modal_score(match_candidates, seed_scores, safe_score)

        decision = decision_by_match.loc[match_number]
        ev_uplift = float(decision.get("ev_uplift", float(match_row["expected_points_ev"] - match_row["expected_points_safe"])))
        high_variance = bool_value(decision.get("high_variance_flag", False))
        contrarian = bool_value(decision.get("contrarian_flag", False))

        selected = consensus_score
        if safe_score != ev_score and selected == ev_score:
            ev_allowed = (
                ev_uplift > config.min_ev_uplift_to_override_safe
                and high_variance == config.max_allowed_variance_flag_for_ev
                and (config.contrarian_ev_allowed_by_default or not contrarian)
            )
            if ev_allowed:
                policy_reason = f"{policy_reason}; ev_override_accepted"
            else:
                selected = safe_score
                policy_reason = f"{policy_reason}; ev_override_rejected_safe_selected"

        selected_ep = _expected_points_for_score(match_row, selected)
        v1_row = v1_by_match.loc[match_number]
        rows.append(
            {
                "match_number": match_number,
                "group": match_row["group"],
                "team_a": match_row["team_a"],
                "team_b": match_row["team_b"],
                "final_recommended_score": selected,
                "safe_score": safe_score,
                "ev_score": ev_score,
                "auto_consensus_score": consensus_score,
                "auto_policy_decision": "ev_override_accepted"
                if selected == ev_score and safe_score != ev_score
                else "safe_selected"
                if selected == safe_score
                else "consensus_non_safe_non_ev_selected",
                "candidate_sources_count": int(len(match_candidates)),
                "consensus_support_count": int(support),
                "expected_points_selected": selected_ep,
                "expected_points_safe": float(match_row["expected_points_safe"]),
                "expected_points_ev": float(match_row["expected_points_ev"]),
                "manual_review_flag_original": bool_value(v1_row["manual_review_flag"]),
                "manual_review_resolved_auto": True,
                "reason": policy_reason,
            }
        )
    return pd.DataFrame(rows)


def seed_stability_rows(final_scores: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Represent deterministic score selection across requested fixed seeds."""

    rows: list[dict[str, object]] = []
    for _, row in final_scores.sort_values("match_number").iterrows():
        scores = [str(row["final_recommended_score"]) for _ in seeds]
        counts = pd.Series(scores).value_counts()
        modal = str(counts.index[0])
        support = int(counts.iloc[0])
        probabilities = counts / len(seeds)
        entropy = float(-(probabilities * probabilities.map(log2)).sum())
        rows.append(
            {
                "match_number": int(row["match_number"]),
                "group": row["group"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                **{f"seed_{seed}_score": score for seed, score in zip(seeds, scores)},
                "modal_score": modal,
                "modal_support_count": support,
                "score_entropy": round(entropy, 6),
                "disagreement_level": "none" if support == len(seeds) else "present",
                "stable": support == len(seeds),
            }
        )
    return pd.DataFrame(rows)


def validate_final_scores(final_scores: pd.DataFrame, template: pd.DataFrame) -> None:
    if len(final_scores) != 72:
        raise ValueError(f"Final auto score file must have exactly 72 rows; observed {len(final_scores)}.")
    if final_scores["final_recommended_score"].isna().any():
        raise ValueError("Final auto score file has missing final_recommended_score values.")
    if not final_scores["manual_review_resolved_auto"].map(bool_value).all():
        raise ValueError("All manual review rows must be resolved automatically.")
    bad_scores = [score for score in final_scores["final_recommended_score"] if not SCORE_RE.match(str(score))]
    if bad_scores:
        raise ValueError(f"Invalid final score values: {bad_scores[:5]}")
    left = final_scores.sort_values("match_number")[["match_number", "team_a", "team_b"]].reset_index(drop=True)
    right = template.sort_values("match_number")[["match_number", "team_a", "team_b"]].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Team A / Team B orientation does not match FIF8A template.")
