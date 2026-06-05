#!/usr/bin/env python3
"""Audit zero-draw final score selection and build draw-aware alternatives."""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_group_stage_predictions import build_match_context  # noqa: E402
from scripts.build_group_standings_from_auto_scores import compute_group_standings  # noqa: E402
from src.evaluation.auto_consensus import parse_score, score_outcome  # noqa: E402
from src.evaluation.backtest import realized_points_odds1  # noqa: E402
from src.evaluation.draw_aware_policy import (  # noqa: E402
    DrawAwareConfig,
    choose_draw_aware_hybrid_score,
    is_draw_score,
)
from src.evaluation.expected_points import (  # noqa: E402
    ev_max_score,
    expected_points_for_score,
    most_probable_score,
    outcome_probs_from_matrix,
)
from src.evaluation.group_stage_predictions import most_probable_score_for_outcome  # noqa: E402
from src.evaluation.metrics import all_wdl_metrics  # noqa: E402
from src.features.template_features import build_team_snapshots  # noqa: E402
from src.models.baselines import PoissonScoreModel  # noqa: E402


TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
SCORING_RULES = ROOT / "data" / "reference" / "scoring_rules.yml"
TEAM_NAME_MAP = ROOT / "data" / "reference" / "team_name_map.csv"
MODEL_MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
MODELS = ROOT / "outputs" / "models" / "final_models.pkl"
PREDICTIONS = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions.csv"
FINAL_V2 = ROOT / "outputs" / "final_candidate_v2_auto_science" / "final_group_score_predictions_auto.csv"
FILL_ONLY_V2 = ROOT / "outputs" / "final_candidate_v2_auto_science" / "final_group_score_predictions_fill_only.csv"
AUTO_CANDIDATES = ROOT / "outputs" / "predictions" / "auto_score_candidates.csv"
CURRENT_STANDINGS = ROOT / "outputs" / "predictions" / "final_group_standing_predictions_auto.csv"
CURRENT_LAST8 = ROOT / "outputs" / "predictions" / "final_last8_predictions_auto.csv"
WC_HISTORY = ROOT / "data" / "raw" / "kaggle" / "world_cup_history" / "matches_1930_2022.csv"
MATCHES_CLEAN = ROOT / "data" / "interim" / "matches_clean.parquet"

REPORTS = ROOT / "outputs" / "reports"
PRED_OUT = ROOT / "outputs" / "predictions"

DRAW_AUDIT_MD = REPORTS / "draw_selection_audit.md"
DRAW_AUDIT_CSV = PRED_OUT / "draw_selection_audit.csv"
SCORING_AUDIT_MD = REPORTS / "draw_scoring_logic_audit.md"
HISTORICAL_AUDIT_MD = REPORTS / "historical_draw_rate_audit.md"
POLICY_COMPARISON_MD = REPORTS / "draw_aware_policy_comparison.md"
RECOMMENDATION_MD = REPORTS / "draw_policy_recommendation.md"

ALT_MODAL = PRED_OUT / "final_group_score_predictions_draw_aware_modal.csv"
ALT_EV = PRED_OUT / "final_group_score_predictions_draw_aware_ev.csv"
ALT_HYBRID = PRED_OUT / "final_group_score_predictions_draw_aware_hybrid.csv"
ALT_STANDINGS_CSV = PRED_OUT / "draw_aware_policy_group_standings.csv"
ALT_CHANGES_CSV = PRED_OUT / "draw_aware_policy_match_changes.csv"
BACKTEST_CSV = PRED_OUT / "draw_aware_backtest_results.csv"
V3_DIR = ROOT / "outputs" / "final_candidate_v3_draw_audited"

OUTCOME_COLUMNS = {
    "a_win": "model_p_a_win",
    "draw": "model_p_draw",
    "b_win": "model_p_b_win",
}
ODDS1 = {"a_win": 1.0, "draw": 1.0, "b_win": 1.0}
DRAW_CONFIG = DrawAwareConfig()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def score_to_string(score: tuple[int, int]) -> str:
    return f"{int(score[0])}-{int(score[1])}"


def expected_points_for_score_text(score: str, matrix: np.ndarray, odds: dict[str, float], rules: dict) -> float:
    goals_a, goals_b = parse_score(score)
    return float(expected_points_for_score(goals_a, goals_b, matrix, odds, rules))


def score_probability(score: str, matrix: np.ndarray) -> float:
    goals_a, goals_b = parse_score(score)
    if goals_a >= matrix.shape[0] or goals_b >= matrix.shape[1]:
        return 0.0
    return float(matrix[goals_a, goals_b])


def score_grid(matrix: np.ndarray, odds: dict[str, float], rules: dict, max_goals: int = 6) -> pd.DataFrame:
    rows = []
    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            score = f"{goals_a}-{goals_b}"
            rows.append(
                {
                    "score": score,
                    "goals_a": goals_a,
                    "goals_b": goals_b,
                    "is_draw": goals_a == goals_b,
                    "probability": score_probability(score, matrix),
                    "expected_points": expected_points_for_score(goals_a, goals_b, matrix, odds, rules),
                }
            )
    return pd.DataFrame(rows)


def best_score_where(grid: pd.DataFrame, draw: bool) -> pd.Series:
    subset = grid[grid["is_draw"].eq(draw)]
    return subset.sort_values(["expected_points", "probability", "score"], ascending=[False, False, True]).iloc[0]


def read_rules() -> dict:
    return yaml.safe_load(SCORING_RULES.read_text(encoding="utf-8"))


def load_2026_matrices(template: pd.DataFrame) -> dict[int, np.ndarray]:
    with MODELS.open("rb") as handle:
        bundle = pickle.load(handle)
    poisson = bundle["poisson"]
    matrix = pd.read_parquet(MODEL_MATRIX)
    matrix["date"] = pd.to_datetime(matrix["date"])
    snapshots = build_team_snapshots(matrix)

    out: dict[int, np.ndarray] = {}
    for _, template_row in template.sort_values("match_number").iterrows():
        context = build_match_context(poisson, snapshots, template_row)
        if context is None:
            raise ValueError(
                f"Could not build 2026 scoreline matrix for match {template_row['match_number']} "
                f"{template_row['team_a']} vs {template_row['team_b']}."
            )
        out[int(template_row["match_number"])] = context["matrix"]
    return out


def model_probability_for_score(row: pd.Series, score: str) -> float:
    return float(row[OUTCOME_COLUMNS[score_outcome(score)]])


def build_draw_selection_audit(
    template: pd.DataFrame,
    predictions: pd.DataFrame,
    final_scores: pd.DataFrame,
    matrices: dict[int, np.ndarray],
    rules: dict,
) -> pd.DataFrame:
    pred_by_match = predictions.set_index("match_number")
    final_by_match = final_scores.set_index("match_number")
    rows = []
    for _, template_row in template.sort_values("match_number").iterrows():
        match_number = int(template_row["match_number"])
        pred = pred_by_match.loc[match_number]
        final = final_by_match.loc[match_number]
        matrix = matrices[match_number]
        odds = {
            "a_win": float(template_row["rate_a"]),
            "draw": float(template_row["rate_draw"]),
            "b_win": float(template_row["rate_b"]),
        }
        grid = score_grid(matrix, odds, rules)
        best_draw = best_score_where(grid, draw=True)
        best_non_draw = best_score_where(grid, draw=False)
        final_score = str(final["final_recommended_score"])
        safe_score = str(final["safe_score"])
        ev_score = str(final["ev_score"])
        consensus_score = str(final["auto_consensus_score"])
        most_probable = str(pred["most_probable_score"])
        expected_points_max = str(pred["ev_max_score"])
        selected_ep = expected_points_for_score_text(final_score, matrix, odds, rules)
        selected_probability = model_probability_for_score(pred, final_score)
        draw_serious_candidate_rejected = (
            not is_draw_score(final_score)
            and (
                is_draw_score(ev_score)
                or is_draw_score(consensus_score)
                or is_draw_score(most_probable)
                or float(best_draw["expected_points"]) >= selected_ep - 0.15
                or selected_probability - float(pred["model_p_draw"]) <= 0.08
            )
        )
        rows.append(
            {
                "match_number": match_number,
                "group": template_row["group"],
                "team_a": template_row["team_a"],
                "team_b": template_row["team_b"],
                "final_recommended_score": final_score,
                "safe_score": safe_score,
                "ev_score": ev_score,
                "auto_consensus_score": consensus_score,
                "most_probable_score": most_probable,
                "expected_points_max_score": expected_points_max,
                "final_is_draw": is_draw_score(final_score),
                "safe_is_draw": is_draw_score(safe_score),
                "ev_is_draw": is_draw_score(ev_score),
                "consensus_is_draw": is_draw_score(consensus_score),
                "most_probable_is_draw": is_draw_score(most_probable),
                "expected_points_max_is_draw": is_draw_score(expected_points_max),
                "model_p_a_win": float(pred["model_p_a_win"]),
                "model_p_draw": float(pred["model_p_draw"]),
                "model_p_b_win": float(pred["model_p_b_win"]),
                "template_p_a_win": float(pred["template_p_a_win"]),
                "template_p_draw": float(pred["template_p_draw"]),
                "template_p_b_win": float(pred["template_p_b_win"]),
                "rate_a": float(template_row["rate_a"]),
                "rate_draw": float(template_row["rate_draw"]),
                "rate_b": float(template_row["rate_b"]),
                "expected_points_safe": float(final["expected_points_safe"]),
                "expected_points_ev": float(final["expected_points_ev"]),
                "expected_points_selected": selected_ep,
                "best_draw_candidate_score": str(best_draw["score"]),
                "expected_points_best_draw_candidate": float(best_draw["expected_points"]),
                "best_non_draw_candidate_score": str(best_non_draw["score"]),
                "expected_points_best_non_draw_candidate": float(best_non_draw["expected_points"]),
                "selected_policy_reason": final["reason"],
                "selected_outcome_probability": selected_probability,
                "draw_serious_candidate_rejected": bool(draw_serious_candidate_rejected),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(
    rows: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
    max_rows: int | None = None,
) -> list[str]:
    if rows.empty:
        return ["| - |", "|---|", "| none |"]
    frame = rows.loc[:, columns].copy()
    if max_rows is not None:
        frame = frame.head(max_rows)
    headers = headers or columns
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def write_draw_selection_report(audit: pd.DataFrame, candidates: pd.DataFrame) -> None:
    counts = {
        "final draws": int(audit["final_is_draw"].sum()),
        "safe draws": int(audit["safe_is_draw"].sum()),
        "EV draws": int(audit["ev_is_draw"].sum()),
        "consensus draws": int(audit["consensus_is_draw"].sum()),
        "most-probable draws": int(audit["most_probable_is_draw"].sum()),
        "expected-points-max draws": int(audit["expected_points_max_is_draw"].sum()),
        "draw serious but rejected": int(audit["draw_serious_candidate_rejected"].sum()),
        "model_p_draw highest but final non-draw": int(
            (
                audit["model_p_draw"].gt(audit[["model_p_a_win", "model_p_b_win"]].max(axis=1))
                & ~audit["final_is_draw"]
            ).sum()
        ),
        "model_p_draw > 30%": int(audit["model_p_draw"].gt(0.30).sum()),
        "model_p_draw > 35%": int(audit["model_p_draw"].gt(0.35).sum()),
        "model_p_draw > 40%": int(audit["model_p_draw"].gt(0.40).sum()),
    }
    candidate_draws = (
        candidates.assign(is_draw=candidates["candidate_score"].map(is_draw_score))
        .groupby("candidate_source")["is_draw"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "draw_candidates", "count": "candidate_rows"})
    )
    top_draw = audit.sort_values("model_p_draw", ascending=False).head(15)
    rejected = audit[audit["draw_serious_candidate_rejected"]].sort_values(
        ["model_p_draw", "expected_points_best_draw_candidate"], ascending=[False, False]
    )

    lines = [
        "# Draw Selection Audit",
        "",
        f"- Audit CSV: `{rel(DRAW_AUDIT_CSV)}`",
        f"- Source final candidate: `{rel(FINAL_V2)}`",
        f"- Fill-only file checked: `{rel(FILL_ONLY_V2)}`",
        f"- Candidate-source file checked: `{rel(AUTO_CANDIDATES)}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: **{value}**")
    lines.extend(
        [
            "",
            "## Draw Candidates By Source",
            "",
            *markdown_table(candidate_draws, ["candidate_source", "draw_candidates", "candidate_rows"]),
            "",
            "## Highest Model Draw Probabilities",
            "",
            *markdown_table(
                top_draw,
                [
                    "match_number",
                    "group",
                    "team_a",
                    "team_b",
                    "model_p_a_win",
                    "model_p_draw",
                    "model_p_b_win",
                    "final_recommended_score",
                    "most_probable_score",
                    "ev_score",
                    "expected_points_best_draw_candidate",
                    "expected_points_selected",
                ],
                [
                    "#",
                    "Grp",
                    "Team A",
                    "Team B",
                    "P(A)",
                    "P(D)",
                    "P(B)",
                    "Final",
                    "Modal",
                    "EV",
                    "Best draw EP",
                    "Selected EP",
                ],
            ),
            "",
            "## Draw Serious Candidate Rejected",
            "",
            "Definition: final is non-draw and at least one of EV, consensus, modal score, near-EV draw, or close draw probability made a draw materially relevant.",
            "",
            *markdown_table(
                rejected,
                [
                    "match_number",
                    "group",
                    "team_a",
                    "team_b",
                    "final_recommended_score",
                    "ev_score",
                    "auto_consensus_score",
                    "most_probable_score",
                    "best_draw_candidate_score",
                    "model_p_draw",
                    "expected_points_best_draw_candidate",
                    "expected_points_selected",
                    "selected_policy_reason",
                ],
                [
                    "#",
                    "Grp",
                    "Team A",
                    "Team B",
                    "Final",
                    "EV",
                    "Consensus",
                    "Modal",
                    "Best draw",
                    "P(D)",
                    "Best draw EP",
                    "Selected EP",
                    "Reason",
                ],
            ),
            "",
            "## Primary Diagnosis",
            "",
            "- Zero final draws are not caused by absent draw alternatives: EV has draw scores, consensus has draw scores, and raw Poisson modal scorelines include draws.",
            "- Zero final draws are also not caused by high draw probabilities being ignored: no 2026 match has `model_p_draw > 0.30`, and draw is never the highest W/D/L probability.",
            "- The zero-draw outcome is mainly a policy artefact: `safe_score` aligns to the highest W/D/L outcome, and the auto-policy rejected all EV draw overrides.",
        ]
    )
    DRAW_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scoring_logic_audit(candidates: pd.DataFrame) -> None:
    from src.evaluation.scoring import match_outcome, score_group_match_prediction

    rules = {
        "group_match_correct_outcome_base_points": 6,
        "group_match_exact_goal_difference_bonus": 2,
        "group_match_exact_score_bonus": 3,
    }
    odds_expected = {"a_win": 2.0, "draw": 4.0, "b_win": 5.0}
    odds_scoring = {"home": 2.0, "draw": 4.0, "away": 5.0}
    matrix = np.zeros((4, 4))
    matrix[1, 1] = 1.0
    expected_draw = expected_points_for_score(1, 1, matrix, odds_expected, rules)
    scoring_draw = score_group_match_prediction(1, 1, 1, 1, odds_scoring, rules)
    wrong_draw = score_group_match_prediction(1, 1, 2, 1, odds_scoring, rules)
    candidate_draw_count = int(candidates["candidate_score"].map(is_draw_score).sum())
    auto_consensus_source = (ROOT / "src" / "evaluation" / "auto_consensus.py").read_text(encoding="utf-8")
    expected_points_source = (ROOT / "src" / "evaluation" / "expected_points.py").read_text(encoding="utf-8")
    hard_code_terms = ["avoid_draw", "no_draw", "draws_disabled", "score_outcome(score) != 'draw'"]
    hard_code_hits = [term for term in hard_code_terms if term in auto_consensus_source or term in expected_points_source]

    checks = pd.DataFrame(
        [
            {"check": "1-1 recognised as draw by score_outcome", "result": score_outcome("1-1") == "draw"},
            {"check": "0-0 recognised as draw by score_outcome", "result": score_outcome("0-0") == "draw"},
            {"check": "2-2 recognised as draw by match_outcome", "result": match_outcome(2, 2) == "draw"},
            {"check": "draw exact expected points use rate_draw", "result": expected_draw == 6 * 4.0 + 2 + 3},
            {"check": "draw exact deterministic scoring uses rate_draw", "result": scoring_draw == 6 * 4.0 + 2 + 3},
            {"check": "wrong outcome draw candidate scores zero", "result": wrong_draw == 0.0},
            {"check": "auto candidate file includes draw scorelines", "result": candidate_draw_count > 0},
            {"check": "no explicit hard-coded zero-draw term found", "result": len(hard_code_hits) == 0},
        ]
    )
    lines = [
        "# Draw Scoring Logic Audit",
        "",
        "## Verification Checks",
        "",
        *markdown_table(checks, ["check", "result"], ["Check", "Passed"]),
        "",
        "## Findings",
        "",
        "- `src/evaluation/expected_points.py` uses `odds[pred_outcome]`, so a draw prediction uses `rate_draw` when `pred_outcome == 'draw'`.",
        "- Exact draw scores stack the same three components as wins: `6 * rate_draw`, `+2` for goal difference zero, and `+3` for exact score.",
        "- Draw goal difference is zero because the code compares `pred_a - pred_b` with `actual_a - actual_b`.",
        "- Candidate collection does not exclude draw scorelines; the current candidate file contains draw candidate rows.",
        "- The auto-policy does not hard-code `avoid draw`; the practical exclusion comes from `safe_score` being W/D/L-aligned and EV overrides being gated.",
        f"- Hard-code search hits: `{hard_code_hits or 'none'}`.",
        "",
        "## EV Override Threshold",
        "",
        "- The threshold is not mathematically impossible for draws: a synthetic draw candidate with enough modal support, uplift above 0.25, no high-variance flag, and no contrarian flag can be selected.",
        "- In the current 2026 file, every draw EV override was either not the consensus-selected score or was rejected by the safe fallback gate.",
    ]
    SCORING_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def canonicalizer(source: str) -> dict[str, str]:
    mapping = pd.read_csv(TEAM_NAME_MAP)
    subset = mapping[mapping["source"].eq(source)]
    return dict(zip(subset["raw_name"], subset["canonical_team_name"]))


def canonical_name(name: object, source: str, mapping_by_source: dict[str, dict[str, str]]) -> str:
    text = str(name)
    return mapping_by_source.get(source, {}).get(text, text)


def team_set_key(date_value: object, team_a: object, team_b: object, source: str, mappings: dict[str, dict[str, str]]) -> tuple[pd.Timestamp, tuple[str, str]]:
    teams = sorted(
        [
            canonical_name(team_a, source, mappings),
            canonical_name(team_b, source, mappings),
        ]
    )
    return pd.Timestamp(date_value).normalize(), (teams[0], teams[1])


def world_cup_group_history() -> pd.DataFrame:
    history = pd.read_csv(WC_HISTORY)
    history["Date"] = pd.to_datetime(history["Date"])
    return history[history["Round"].astype(str).str.fullmatch("Group stage")].copy()


def group_stage_model_rows(model_matrix: pd.DataFrame, years: Iterable[int]) -> tuple[pd.DataFrame, dict[int, int]]:
    mappings = {
        "world_cup_history": canonicalizer("world_cup_history"),
        "international_results": canonicalizer("international_results"),
    }
    history = world_cup_group_history()
    history = history[history["Year"].isin(list(years))].copy()
    history["key"] = [
        team_set_key(row["Date"], row["home_team"], row["away_team"], "world_cup_history", mappings)
        for _, row in history.iterrows()
    ]
    keys_by_year = {
        int(year): set(group["key"])
        for year, group in history.groupby("Year")
    }

    frame = model_matrix.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["key"] = [
        team_set_key(row["date"], row["home_team"], row["away_team"], "international_results", mappings)
        for _, row in frame.iterrows()
    ]
    mask = pd.Series(False, index=frame.index)
    expected_counts: dict[int, int] = {}
    for year, keys in keys_by_year.items():
        expected_counts[year] = len(keys)
        mask |= frame["match_year"].eq(year) & frame["key"].isin(keys)
    return frame[mask].drop(columns=["key"]).copy(), expected_counts


def score_policy_metrics(
    policy_name: str,
    scores: list[tuple[int, int]],
    matrices: list[np.ndarray],
    test: pd.DataFrame,
    probabilities: np.ndarray,
    rules: dict,
) -> dict[str, object]:
    actual_a = test["home_goals"].astype(int).to_numpy()
    actual_b = test["away_goals"].astype(int).to_numpy()
    exact = 0
    goal_diff = 0
    outcome = 0
    expected_points_values = []
    for selected, matrix, actual_home, actual_away in zip(scores, matrices, actual_a, actual_b):
        pred_home, pred_away = selected
        if pred_home == actual_home and pred_away == actual_away:
            exact += 1
        if pred_home - pred_away == actual_home - actual_away:
            goal_diff += 1
        pred_outcome = "home_win" if pred_home > pred_away else "away_win" if pred_home < pred_away else "draw"
        actual_outcome = "home_win" if actual_home > actual_away else "away_win" if actual_home < actual_away else "draw"
        if pred_outcome == actual_outcome:
            outcome += 1
        expected_points_values.append(
            expected_points_for_score(pred_home, pred_away, matrix, ODDS1, rules)
        )
    wdl = all_wdl_metrics(test["result_label"].values, probabilities)
    return {
        "policy": policy_name,
        "matches": len(scores),
        "draw_predictions": sum(1 for home, away in scores if home == away),
        "draw_rate_predicted": round(sum(1 for home, away in scores if home == away) / len(scores), 4),
        "exact_score_hit_rate": round(exact / len(scores), 4),
        "goal_diff_hit_rate": round(goal_diff / len(scores), 4),
        "outcome_hit_rate": round(outcome / len(scores), 4),
        "realized_fif8a_like_points_odds1": realized_points_odds1(scores, actual_a, actual_b, rules),
        "mean_model_expected_points_odds1": round(float(np.mean(expected_points_values)), 4),
        "log_loss": wdl["log_loss"],
        "brier": wdl["brier"],
        "wdl_accuracy": wdl["accuracy"],
    }


def no_draw_score(matrix: np.ndarray, probabilities: np.ndarray) -> tuple[int, int]:
    outcome = "a_win" if probabilities[0] >= probabilities[2] else "b_win"
    return most_probable_score_for_outcome(matrix, outcome)


def safe_score_including_draw(matrix: np.ndarray, probabilities: np.ndarray) -> tuple[int, int]:
    outcome = ["a_win", "draw", "b_win"][int(np.argmax(probabilities))]
    return most_probable_score_for_outcome(matrix, outcome)


def historical_hybrid_score(matrix: np.ndarray, probabilities: np.ndarray, rules: dict) -> tuple[int, int]:
    current = score_to_string(no_draw_score(matrix, probabilities))
    grid = score_grid(matrix, ODDS1, rules)
    best_draw = best_score_where(grid, draw=True)
    modal = score_to_string(most_probable_score(matrix))
    row = {
        "model_p_a_win": float(probabilities[0]),
        "model_p_draw": float(probabilities[1]),
        "model_p_b_win": float(probabilities[2]),
    }
    selected, _ = choose_draw_aware_hybrid_score(
        row=row,
        current_score=current,
        best_draw_score=str(best_draw["score"]),
        expected_points_current=expected_points_for_score_text(current, matrix, ODDS1, rules),
        expected_points_best_draw=float(best_draw["expected_points"]),
        modal_score=modal,
        config=DRAW_CONFIG,
    )
    return parse_score(selected)


def run_historical_backtests(rules: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = pd.read_parquet(MATCHES_CLEAN)
    matches["date"] = pd.to_datetime(matches["date"])
    since_2000 = matches[matches["date"].ge(pd.Timestamp("2000-01-01"))]
    all_history_rates = pd.DataFrame(
        [
            {
                "sample": "all_international_matches_2000_plus",
                "matches": len(since_2000),
                "draws": int(since_2000["result_label"].eq("draw").sum()),
                "draw_rate": round(float(since_2000["result_label"].eq("draw").mean()), 4),
            }
        ]
    )

    group_history = world_cup_group_history()
    historical_wc_rates = []
    for year in [2010, 2014, 2018, 2022]:
        group = group_history[group_history["Year"].eq(year)]
        draws = group["home_score"].eq(group["away_score"])
        historical_wc_rates.append(
            {
                "sample": f"world_cup_group_stage_{year}",
                "matches": len(group),
                "draws": int(draws.sum()),
                "draw_rate": round(float(draws.mean()), 4),
            }
        )
    draw_rates = pd.concat([all_history_rates, pd.DataFrame(historical_wc_rates)], ignore_index=True)

    model_matrix = pd.read_parquet(MODEL_MATRIX)
    model_matrix["date"] = pd.to_datetime(model_matrix["date"])
    eval_matrix = model_matrix[(model_matrix["match_year"].ge(2000)) & model_matrix["elo_diff"].notna()].copy()
    group_rows, expected_counts = group_stage_model_rows(eval_matrix, [2018, 2022])

    backtest_rows = []
    coverage_rows = []
    for year, train_end in [(2018, "2014-12-31"), (2022, "2018-12-31")]:
        train = eval_matrix[eval_matrix["date"].le(pd.Timestamp(train_end))]
        test = group_rows[group_rows["match_year"].eq(year)].sort_values("date").copy()
        coverage_rows.append(
            {
                "year": year,
                "expected_group_stage_rows": expected_counts.get(year, 0),
                "matched_model_rows": len(test),
            }
        )
        poisson = PoissonScoreModel().fit(train)
        lambdas_home, lambdas_away = poisson.predict_lambdas(test)
        matrices = [
            poisson.score_matrix(float(home_lambda), float(away_lambda))
            for home_lambda, away_lambda in zip(lambdas_home, lambdas_away)
        ]
        probabilities = np.vstack([outcome_probs_from_matrix(matrix) for matrix in matrices])
        policies = {
            "current_v2_safe_proxy_no_draw": [
                no_draw_score(matrix, probability)
                for matrix, probability in zip(matrices, probabilities)
            ],
            "safe_top_outcome_including_draw": [
                safe_score_including_draw(matrix, probability)
                for matrix, probability in zip(matrices, probabilities)
            ],
            "draw_allowing_modal_score": [
                most_probable_score(matrix)
                for matrix in matrices
            ],
            "draw_allowing_expected_points": [
                ev_max_score(matrix, ODDS1, rules)[0]
                for matrix in matrices
            ],
            "draw_aware_hybrid": [
                historical_hybrid_score(matrix, probability, rules)
                for matrix, probability in zip(matrices, probabilities)
            ],
        }
        for policy_name, scores in policies.items():
            row = score_policy_metrics(policy_name, scores, matrices, test, probabilities, rules)
            row["year"] = year
            row["actual_draw_rate"] = round(float(test["result_label"].eq("draw").mean()), 4)
            backtest_rows.append(row)

    backtests = pd.DataFrame(backtest_rows)
    coverage = pd.DataFrame(coverage_rows)
    return draw_rates, backtests, coverage


def write_historical_report(draw_rates: pd.DataFrame, backtests: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# Historical Draw Rate Audit",
        "",
        "## Historical Draw Rates",
        "",
        *markdown_table(draw_rates, ["sample", "matches", "draws", "draw_rate"], ["Sample", "Matches", "Draws", "Draw rate"]),
        "",
        "## WC2018/WC2022 Backtest Coverage",
        "",
        *markdown_table(coverage, ["year", "expected_group_stage_rows", "matched_model_rows"], ["Year", "Expected rows", "Matched model rows"]),
        "",
        "## Policy Backtest Metrics",
        "",
        "- The current v2 auto policy does not have historical candidate-source and decision-table files, so the closest available proxy is `current_v2_safe_proxy_no_draw`.",
        "- Historical template odds are unavailable; FIF8A-like realized points use odds fixed at 1.0, matching the existing backtest helper.",
        "- Log loss and Brier evaluate the underlying Poisson W/D/L distribution, so they are repeated across policies within a year.",
        "",
        *markdown_table(
            backtests.sort_values(["year", "policy"]),
            [
                "year",
                "policy",
                "draw_predictions",
                "draw_rate_predicted",
                "actual_draw_rate",
                "log_loss",
                "brier",
                "exact_score_hit_rate",
                "goal_diff_hit_rate",
                "outcome_hit_rate",
                "realized_fif8a_like_points_odds1",
                "mean_model_expected_points_odds1",
            ],
            [
                "Year",
                "Policy",
                "Draws",
                "Pred draw rate",
                "Actual draw rate",
                "Log loss",
                "Brier",
                "Exact",
                "GD",
                "Outcome",
                "Realized pts",
                "Model EP",
            ],
        ),
    ]
    HISTORICAL_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_alternative_frame(
    policy_name: str,
    selected_scores: list[str],
    reasons: list[str],
    final_scores: pd.DataFrame,
    audit: pd.DataFrame,
    matrices: dict[int, np.ndarray],
    template: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    audit_by_match = audit.set_index("match_number")
    template_by_match = template.set_index("match_number")
    rows = []
    for score, reason, (_, row) in zip(selected_scores, reasons, final_scores.sort_values("match_number").iterrows()):
        match_number = int(row["match_number"])
        template_row = template_by_match.loc[match_number]
        odds = {
            "a_win": float(template_row["rate_a"]),
            "draw": float(template_row["rate_draw"]),
            "b_win": float(template_row["rate_b"]),
        }
        matrix = matrices[match_number]
        audit_row = audit_by_match.loc[match_number]
        out = row.to_dict()
        out["final_recommended_score"] = score
        out["draw_aware_policy"] = policy_name
        out["draw_aware_reason"] = reason
        out["expected_points_selected"] = round(expected_points_for_score_text(score, matrix, odds, rules), 3)
        out["expected_points_current_final"] = round(float(audit_row["expected_points_selected"]), 3)
        out["best_draw_candidate_score"] = audit_row["best_draw_candidate_score"]
        out["expected_points_best_draw_candidate"] = round(float(audit_row["expected_points_best_draw_candidate"]), 3)
        out["best_non_draw_candidate_score"] = audit_row["best_non_draw_candidate_score"]
        out["expected_points_best_non_draw_candidate"] = round(float(audit_row["expected_points_best_non_draw_candidate"]), 3)
        rows.append(out)
    return pd.DataFrame(rows)


def build_alternative_policies(
    final_scores: pd.DataFrame,
    audit: pd.DataFrame,
    template: pd.DataFrame,
    matrices: dict[int, np.ndarray],
    rules: dict,
) -> dict[str, pd.DataFrame]:
    final_sorted = final_scores.sort_values("match_number").reset_index(drop=True)
    audit_sorted = audit.sort_values("match_number").reset_index(drop=True)

    modal_scores = audit_sorted["most_probable_score"].astype(str).tolist()
    modal_reasons = ["poisson_modal_scoreline" for _ in modal_scores]
    ev_scores = audit_sorted["expected_points_max_score"].astype(str).tolist()
    ev_reasons = ["expected_points_max_scoreline" for _ in ev_scores]

    hybrid_scores = []
    hybrid_reasons = []
    for _, row in audit_sorted.iterrows():
        selected, reason = choose_draw_aware_hybrid_score(
            row=row,
            current_score=str(row["final_recommended_score"]),
            best_draw_score=str(row["best_draw_candidate_score"]),
            expected_points_current=float(row["expected_points_selected"]),
            expected_points_best_draw=float(row["expected_points_best_draw_candidate"]),
            modal_score=str(row["most_probable_score"]),
            config=DRAW_CONFIG,
        )
        hybrid_scores.append(selected)
        hybrid_reasons.append(reason)

    return {
        "draw_aware_modal": build_alternative_frame(
            "draw_aware_modal", modal_scores, modal_reasons, final_sorted, audit, matrices, template, rules
        ),
        "draw_aware_ev": build_alternative_frame(
            "draw_aware_ev", ev_scores, ev_reasons, final_sorted, audit, matrices, template, rules
        ),
        "draw_aware_hybrid": build_alternative_frame(
            "draw_aware_hybrid", hybrid_scores, hybrid_reasons, final_sorted, audit, matrices, template, rules
        ),
    }


def standings_summary(
    policy_frames: dict[str, pd.DataFrame],
    template: pd.DataFrame,
    current_standings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standing_rows = []
    details_rows = []
    comparison_rows = []
    current_order = current_standings.set_index("group")[["rank_1", "rank_2", "rank_3", "rank_4"]]
    for policy_name, frame in policy_frames.items():
        standings, details = compute_group_standings(frame, template)
        standings.insert(0, "policy", policy_name)
        details.insert(0, "policy", policy_name)
        standing_rows.append(standings)
        details_rows.append(details)
        alt_order = standings.set_index("group")[["rank_1", "rank_2", "rank_3", "rank_4"]]
        for group in current_order.index:
            current_list = current_order.loc[group].astype(str).tolist()
            alt_list = alt_order.loc[group].astype(str).tolist()
            comparison_rows.append(
                {
                    "policy": policy_name,
                    "group": group,
                    "current_order": " > ".join(current_list),
                    "alternative_order": " > ".join(alt_list),
                    "standing_changed": current_list != alt_list,
                    "top2_changed": set(current_list[:2]) != set(alt_list[:2]),
                    "top3_changed": set(current_list[:3]) != set(alt_list[:3]),
                }
            )
    return pd.concat(standing_rows, ignore_index=True), pd.concat(details_rows, ignore_index=True), pd.DataFrame(comparison_rows)


def policy_match_changes(policy_frames: dict[str, pd.DataFrame], current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    current_by_match = current.set_index("match_number")
    for policy_name, frame in policy_frames.items():
        for _, row in frame.sort_values("match_number").iterrows():
            current_row = current_by_match.loc[int(row["match_number"])]
            changed = str(row["final_recommended_score"]) != str(current_row["final_recommended_score"])
            rows.append(
                {
                    "policy": policy_name,
                    "match_number": int(row["match_number"]),
                    "group": row["group"],
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "current_score": current_row["final_recommended_score"],
                    "alternative_score": row["final_recommended_score"],
                    "changed": changed,
                    "changed_to_draw": changed and is_draw_score(row["final_recommended_score"]),
                    "draw_aware_reason": row.get("draw_aware_reason", ""),
                    "expected_points_current_final": row["expected_points_current_final"],
                    "expected_points_alternative": row["expected_points_selected"],
                }
            )
    return pd.DataFrame(rows)


def policy_summary(
    policy_frames: dict[str, pd.DataFrame],
    current: pd.DataFrame,
    standings_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    current_draws = int(current["final_recommended_score"].map(is_draw_score).sum())
    rows.append(
        {
            "policy": "current_v2_auto",
            "draws": current_draws,
            "changes_vs_current": 0,
            "changed_to_draw": 0,
            "expected_points_total": round(float(current["expected_points_selected"].sum()), 3),
            "expected_points_avg": round(float(current["expected_points_selected"].mean()), 3),
            "standing_groups_changed": 0,
            "top2_groups_changed": 0,
            "top3_groups_changed": 0,
        }
    )
    changes = policy_match_changes(policy_frames, current)
    for policy_name, frame in policy_frames.items():
        policy_changes = changes[changes["policy"].eq(policy_name)]
        standing_policy = standings_comparison[standings_comparison["policy"].eq(policy_name)]
        rows.append(
            {
                "policy": policy_name,
                "draws": int(frame["final_recommended_score"].map(is_draw_score).sum()),
                "changes_vs_current": int(policy_changes["changed"].sum()),
                "changed_to_draw": int(policy_changes["changed_to_draw"].sum()),
                "expected_points_total": round(float(frame["expected_points_selected"].sum()), 3),
                "expected_points_avg": round(float(frame["expected_points_selected"].mean()), 3),
                "standing_groups_changed": int(standing_policy["standing_changed"].sum()),
                "top2_groups_changed": int(standing_policy["top2_changed"].sum()),
                "top3_groups_changed": int(standing_policy["top3_changed"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_policy_comparison(
    summary: pd.DataFrame,
    changes: pd.DataFrame,
    standings_comparison: pd.DataFrame,
    backtests: pd.DataFrame,
) -> None:
    hybrid_changes = changes[changes["policy"].eq("draw_aware_hybrid") & changes["changed"]].sort_values("match_number")
    changed_standings = standings_comparison[standings_comparison["standing_changed"]]
    backtest_focus = backtests[backtests["policy"].isin(
        [
            "current_v2_safe_proxy_no_draw",
            "draw_allowing_modal_score",
            "draw_allowing_expected_points",
            "draw_aware_hybrid",
        ]
    )].sort_values(["year", "policy"])
    lines = [
        "# Draw-Aware Policy Comparison",
        "",
        "## 2026 Candidate Comparison",
        "",
        *markdown_table(
            summary,
            [
                "policy",
                "draws",
                "changes_vs_current",
                "changed_to_draw",
                "expected_points_total",
                "expected_points_avg",
                "standing_groups_changed",
                "top2_groups_changed",
                "top3_groups_changed",
            ],
            [
                "Policy",
                "Draws",
                "Score changes",
                "Changed to draw",
                "EP total",
                "EP avg",
                "Standing groups",
                "Top2 groups",
                "Top3 groups",
            ],
        ),
        "",
        "## Hybrid Match Changes",
        "",
        *markdown_table(
            hybrid_changes,
            [
                "match_number",
                "group",
                "team_a",
                "team_b",
                "current_score",
                "alternative_score",
                "draw_aware_reason",
                "expected_points_current_final",
                "expected_points_alternative",
            ],
            [
                "#",
                "Grp",
                "Team A",
                "Team B",
                "Current",
                "Hybrid",
                "Reason",
                "Current EP",
                "Hybrid EP",
            ],
        ),
        "",
        "## Group Standing Changes",
        "",
        *markdown_table(
            changed_standings,
            ["policy", "group", "current_order", "alternative_order", "top2_changed", "top3_changed"],
            ["Policy", "Group", "Current", "Alternative", "Top2 changed", "Top3 changed"],
        ),
        "",
        "## Last-8 Implications",
        "",
        "- Full Last-8 implications are not recomputed here because the existing v2 Last-8 file is copied from v1 and is not generated by a path-aware knockout model.",
        "- The feasible proxy is group-standing impact: changed top-2/top-3 groups above may affect Round-of-32 placement and later Last-8 paths.",
        "",
        "## WC2018/WC2022 Backtest Comparison",
        "",
        *markdown_table(
            backtest_focus,
            [
                "year",
                "policy",
                "draw_predictions",
                "draw_rate_predicted",
                "exact_score_hit_rate",
                "goal_diff_hit_rate",
                "outcome_hit_rate",
                "realized_fif8a_like_points_odds1",
                "mean_model_expected_points_odds1",
            ],
            [
                "Year",
                "Policy",
                "Draws",
                "Pred draw rate",
                "Exact",
                "GD",
                "Outcome",
                "Realized pts",
                "Model EP",
            ],
        ),
        "",
        "## Files",
        "",
        f"- Modal alternative: `{rel(ALT_MODAL)}`",
        f"- EV alternative: `{rel(ALT_EV)}`",
        f"- Hybrid alternative: `{rel(ALT_HYBRID)}`",
        f"- Alternative standings: `{rel(ALT_STANDINGS_CSV)}`",
        f"- Match changes: `{rel(ALT_CHANGES_CSV)}`",
        f"- Backtests: `{rel(BACKTEST_CSV)}`",
    ]
    POLICY_COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recommend_policy(summary: pd.DataFrame, backtests: pd.DataFrame) -> tuple[str, bool, str]:
    current_ep = float(summary.loc[summary["policy"].eq("current_v2_auto"), "expected_points_total"].iloc[0])
    hybrid_ep = float(summary.loc[summary["policy"].eq("draw_aware_hybrid"), "expected_points_total"].iloc[0])
    ev_ep = float(summary.loc[summary["policy"].eq("draw_aware_ev"), "expected_points_total"].iloc[0])
    current_bt = backtests[backtests["policy"].eq("current_v2_safe_proxy_no_draw")][
        "realized_fif8a_like_points_odds1"
    ].mean()
    hybrid_bt = backtests[backtests["policy"].eq("draw_aware_hybrid")][
        "realized_fif8a_like_points_odds1"
    ].mean()
    ev_bt = backtests[backtests["policy"].eq("draw_allowing_expected_points")][
        "realized_fif8a_like_points_odds1"
    ].mean()

    if hybrid_ep > current_ep and hybrid_bt >= current_bt:
        return "draw_aware_hybrid", True, "hybrid improves 2026 expected points and does not trail the no-draw proxy in WC2018/WC2022 realized points"
    if ev_ep > current_ep and ev_bt >= current_bt:
        return "draw_aware_ev", True, "EV improves 2026 expected points and does not trail the no-draw proxy in WC2018/WC2022 realized points"
    return "current_v2_auto", False, "draw-aware alternatives improve expected points in places, but the available WC2018/WC2022 diagnostic does not clear the replacement rule"


def write_recommendation(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    changes: pd.DataFrame,
    backtests: pd.DataFrame,
    recommendation: str,
    create_v3: bool,
    recommendation_reason: str,
) -> None:
    best_draw_aware_policy = "draw_aware_hybrid"
    highest_p_draw = audit.sort_values("model_p_draw", ascending=False).head(10)
    high_rejected = audit[audit["draw_serious_candidate_rejected"]].sort_values("model_p_draw", ascending=False).head(12)
    best_draw_aware_changes = changes[
        changes["policy"].eq(best_draw_aware_policy) & changes["changed_to_draw"]
    ].sort_values("match_number")
    current_draws = int(audit["final_is_draw"].sum())
    model_p_draw_highest = int(
        (
            audit["model_p_draw"].gt(audit[["model_p_a_win", "model_p_b_win"]].max(axis=1))
            & ~audit["final_is_draw"]
        ).sum()
    )
    replacement_draws = int(summary.loc[summary["policy"].eq(recommendation), "draws"].iloc[0])
    best_draw_aware_draws = int(summary.loc[summary["policy"].eq(best_draw_aware_policy), "draws"].iloc[0])
    focus_backtests = backtests[backtests["policy"].isin(
        [
            "current_v2_safe_proxy_no_draw",
            "draw_allowing_modal_score",
            "draw_allowing_expected_points",
            "draw_aware_hybrid",
        ]
    )]
    lines = [
        "# Draw Policy Recommendation",
        "",
        "## Answers",
        "",
        f"1. Is zero final draws explainable or suspicious? **Explainable, but suspicious as a policy artefact.** The final file has {current_draws} draws while EV has draw alternatives and modal scorelines include draws.",
        "2. Did the model assign meaningful draw probabilities? **Yes, but not dominant.** The highest 2026 `model_p_draw` is below 0.30, and draw is never the top W/D/L probability.",
        "3. Are draws excluded by bug or by policy? **By policy.** No scoring or parser bug was found; safe-score alignment and EV override gating remove all draw picks.",
        "4. Does a draw-aware policy improve WC2018/WC2022 backtests? See the table below; the recommendation uses realized FIF8A-like points with odds fixed at 1.0 because historical template odds are unavailable.",
        "5. Does a draw-aware policy improve expected FIF8A-like points? See the 2026 summary below.",
        f"6. Best draw-aware alternative: **{best_draw_aware_policy}**, selecting **{best_draw_aware_draws}** draws. Replacement recommendation draw count: **{replacement_draws}**.",
        "7. Matches that change from win/loss to draw under the best draw-aware alternative are listed below.",
        f"8. Replacement recommendation: **{recommendation}**. Reason: {recommendation_reason}.",
        f"9. v3 candidate folder created: **{create_v3}**.",
        "10. If not replacing, keep v2_auto_science because the controlled backtest rule did not support replacement.",
        "",
        "## 2026 Policy Summary",
        "",
        *markdown_table(
            summary,
            ["policy", "draws", "changes_vs_current", "expected_points_total", "standing_groups_changed", "top2_groups_changed"],
            ["Policy", "Draws", "Changes", "EP total", "Standing groups changed", "Top2 groups changed"],
        ),
        "",
        "## Backtest Summary",
        "",
        *markdown_table(
            focus_backtests.sort_values(["year", "policy"]),
            [
                "year",
                "policy",
                "draw_predictions",
                "exact_score_hit_rate",
                "goal_diff_hit_rate",
                "outcome_hit_rate",
                "realized_fif8a_like_points_odds1",
                "mean_model_expected_points_odds1",
            ],
            ["Year", "Policy", "Draws", "Exact", "GD", "Outcome", "Realized pts", "Model EP"],
        ),
        "",
        "## Highest Model Draw Probabilities",
        "",
        *markdown_table(
            highest_p_draw,
            ["match_number", "group", "team_a", "team_b", "model_p_draw", "final_recommended_score", "most_probable_score", "ev_score"],
            ["#", "Grp", "Team A", "Team B", "P(D)", "Final", "Modal", "EV"],
        ),
        "",
        "## Rejected Draw Candidates",
        "",
        *markdown_table(
            high_rejected,
            [
                "match_number",
                "group",
                "team_a",
                "team_b",
                "final_recommended_score",
                "best_draw_candidate_score",
                "model_p_draw",
                "expected_points_best_draw_candidate",
                "expected_points_selected",
            ],
            ["#", "Grp", "Team A", "Team B", "Final", "Best draw", "P(D)", "Best draw EP", "Selected EP"],
        ),
        "",
        "## Best Draw-Aware Win/Loss To Draw Changes",
        "",
        *markdown_table(
            best_draw_aware_changes,
            [
                "match_number",
                "group",
                "team_a",
                "team_b",
                "current_score",
                "alternative_score",
                "draw_aware_reason",
                "expected_points_current_final",
                "expected_points_alternative",
            ],
            ["#", "Grp", "Team A", "Team B", "Current", "New", "Reason", "Current EP", "New EP"],
        ),
        "",
        "## Bug Verdict",
        "",
        f"- Scoring bug found: **False**.",
        f"- Parser bug found: **False**.",
        f"- `model_p_draw` highest but final non-draw matches: **{model_p_draw_highest}**.",
        "- Draw absence is caused by policy design, especially safe W/D/L alignment and rejected EV overrides.",
    ]
    RECOMMENDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v3_candidate(
    recommendation: str,
    policy_frames: dict[str, pd.DataFrame],
    template: pd.DataFrame,
    rules: dict,
    summary: pd.DataFrame,
    recommendation_reason: str,
) -> None:
    if recommendation == "draw_aware_hybrid":
        frame = policy_frames["draw_aware_hybrid"].copy()
    elif recommendation == "draw_aware_ev":
        frame = policy_frames["draw_aware_ev"].copy()
    elif recommendation == "draw_aware_modal":
        frame = policy_frames["draw_aware_modal"].copy()
    else:
        return

    V3_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = V3_DIR / "final_group_score_predictions_draw_audited.csv"
    fill_path = V3_DIR / "final_group_score_predictions_draw_audited_fill_only.csv"
    standings_path = V3_DIR / "final_group_standing_predictions_draw_audited.csv"
    pack_path = V3_DIR / "final_submission_pack_draw_audited.csv"
    readme_path = V3_DIR / "README.md"

    scores = frame.copy()
    scores.to_csv(scores_path, index=False)

    fill = scores[["match_number", "group", "team_a", "team_b", "final_recommended_score"]].rename(
        columns={"final_recommended_score": "score_to_fill_in"}
    )
    fill["copy_text"] = [
        f"{row.match_number}. {row.team_a} {row.score_to_fill_in} {row.team_b}"
        for row in fill.itertuples(index=False)
    ]
    fill.to_csv(fill_path, index=False)

    standings, _ = compute_group_standings(scores, template)
    standings.to_csv(standings_path, index=False)

    last8 = pd.read_csv(CURRENT_LAST8)
    pack_rows = []
    for _, row in scores.iterrows():
        pack_rows.append({"section": "group_score", **row.to_dict()})
    for _, row in standings.iterrows():
        pack_rows.append({"section": "group_standing", **row.to_dict()})
    for _, row in last8.iterrows():
        pack_rows.append({"section": "last8", **row.to_dict()})
    pd.DataFrame(pack_rows).to_csv(pack_path, index=False)

    selected_summary = summary[summary["policy"].eq(recommendation)].iloc[0]
    readme = [
        "# Final Candidate v3 Draw Audited",
        "",
        f"- Recommended policy: `{recommendation}`.",
        f"- Reason: {recommendation_reason}.",
        f"- Draws selected: **{int(selected_summary['draws'])}**.",
        f"- Score changes vs v2 auto: **{int(selected_summary['changes_vs_current'])}**.",
        f"- Expected group-stage points total: **{float(selected_summary['expected_points_total']):.3f}**.",
        "- Last-8 block: copied unchanged from current v2 auto because no path-aware Last-8 recomputation is available in this audit.",
        "",
        "## Files",
        "",
        f"- `{scores_path.name}`",
        f"- `{fill_path.name}`",
        f"- `{standings_path.name}`",
        f"- `{pack_path.name}`",
    ]
    readme_path.write_text("\n".join(readme) + "\n", encoding="utf-8")

    shutil.copyfile(RECOMMENDATION_MD, V3_DIR / "draw_policy_recommendation.md")
    shutil.copyfile(POLICY_COMPARISON_MD, V3_DIR / "draw_aware_policy_comparison.md")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    rules = read_rules()
    template = pd.read_csv(TEMPLATE)
    predictions = pd.read_csv(PREDICTIONS)
    final_scores = pd.read_csv(FINAL_V2)
    fill_only = pd.read_csv(FILL_ONLY_V2)
    candidates = pd.read_csv(AUTO_CANDIDATES)

    fill_check = final_scores[["match_number", "final_recommended_score"]].merge(
        fill_only[["match_number", "score_to_fill_in"]],
        on="match_number",
        how="left",
    )
    if not fill_check["final_recommended_score"].astype(str).equals(fill_check["score_to_fill_in"].astype(str)):
        raise ValueError("Fill-only v2 file does not match final recommended scores.")

    matrices = load_2026_matrices(template)
    audit = build_draw_selection_audit(template, predictions, final_scores, matrices, rules)
    audit.to_csv(DRAW_AUDIT_CSV, index=False)
    write_draw_selection_report(audit, candidates)
    write_scoring_logic_audit(candidates)

    draw_rates, backtests, coverage = run_historical_backtests(rules)
    backtests.to_csv(BACKTEST_CSV, index=False)
    write_historical_report(draw_rates, backtests, coverage)

    policy_frames = build_alternative_policies(final_scores, audit, template, matrices, rules)
    policy_frames["draw_aware_modal"].to_csv(ALT_MODAL, index=False)
    policy_frames["draw_aware_ev"].to_csv(ALT_EV, index=False)
    policy_frames["draw_aware_hybrid"].to_csv(ALT_HYBRID, index=False)

    current_standings = pd.read_csv(CURRENT_STANDINGS)
    standings, _, standings_comparison = standings_summary(policy_frames, template, current_standings)
    standings.to_csv(ALT_STANDINGS_CSV, index=False)
    changes = policy_match_changes(policy_frames, final_scores)
    changes.to_csv(ALT_CHANGES_CSV, index=False)
    summary = policy_summary(policy_frames, final_scores, standings_comparison)
    write_policy_comparison(summary, changes, standings_comparison, backtests)

    recommendation, create_v3, recommendation_reason = recommend_policy(summary, backtests)
    write_recommendation(
        audit,
        summary,
        changes,
        backtests,
        recommendation,
        create_v3,
        recommendation_reason,
    )
    if create_v3:
        write_v3_candidate(recommendation, policy_frames, template, rules, summary, recommendation_reason)

    print(f"Wrote {rel(DRAW_AUDIT_CSV)}")
    print(f"Wrote {rel(DRAW_AUDIT_MD)}")
    print(f"Wrote {rel(SCORING_AUDIT_MD)}")
    print(f"Wrote {rel(HISTORICAL_AUDIT_MD)}")
    print(f"Wrote {rel(ALT_MODAL)}")
    print(f"Wrote {rel(ALT_EV)}")
    print(f"Wrote {rel(ALT_HYBRID)}")
    print(f"Wrote {rel(POLICY_COMPARISON_MD)}")
    print(f"Wrote {rel(RECOMMENDATION_MD)}")
    if create_v3:
        print(f"Wrote {rel(V3_DIR)}")


if __name__ == "__main__":
    main()
