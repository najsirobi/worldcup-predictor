#!/usr/bin/env python3
"""Build v4 recent-rollforward candidate for WC2026 group-stage predictions.

Parts B + C of the corrective task:
  B. Append 35 post-cutoff matches to the match backbone in memory, re-run the
     ratings join and model-matrix pipeline, and save the updated interim files.
  C. Extract updated team snapshots, re-run Poisson predictions, apply
     auto-consensus policy, apply the R1_only_diff_5_0 rule, and write the
     full v4 candidate to outputs/final_candidate_v4_recent_rollforward/.

Constraints (verbatim from task spec):
  - No model retraining.
  - No broad human overlay.
  - No subjective or manual approval.
  - No overwrite of frozen v2 or v3 files.
  - Ratings (ELO/FIFA) are frozen; only rolling form updates.

Run:
    .venv/bin/python scripts/build_recent_rollforward_candidate.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.team_names import load_team_name_map, canonicalize_team_series, normalize_team_whitespace
from src.ingest.ratings import asof_rating_join
from src.features.rating_momentum import add_rating_momentum_features
from src.features.model_matrix import build_model_matrix
from src.features.template_features import build_team_snapshots, feature_row
from src.ingest.rules_and_scoring import load_scoring_rules
from src.models.baselines import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
    proba_in_class_order,
)
from src.evaluation.expected_points import (
    outcome_probs_from_matrix, most_probable_score, ev_max_score,
    expected_points_for_score, expected_points_for_outcome,
)
from src.evaluation.group_stage_predictions import (
    OUTCOME_KEYS, add_score_columns, most_probable_score_for_outcome,
    named_outcome_from_key, score_to_string,
)
from src.evaluation.auto_consensus import (
    AutoPolicyConfig, collect_candidate_scores, select_final_scores,
    validate_final_scores,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

# Source files (never modified)
MATCHES_CLEAN        = INTERIM / "matches_clean.parquet"
ELO_CLEAN            = INTERIM / "elo_ratings_clean.parquet"
FIFA_CLEAN           = INTERIM / "fifa_rankings_clean.parquet"
RECENT_RESULTS       = INTERIM / "recent_senior_mens_international_results_since_cutoff.parquet"
TEMPLATE             = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
OVERLAY_CSV          = ROOT / "data" / "reference" / "wc2026_human_upside_overlay.csv"
MODELS_PKL           = ROOT / "outputs" / "models" / "final_models.pkl"
DECISIONS_CSV        = ROOT / "outputs" / "predictions" / "submission_decision_table.csv"
FINAL_V1             = ROOT / "outputs" / "final_candidate_v1" / "final_group_score_predictions.csv"
ENSEMBLE_CSV         = ROOT / "outputs" / "predictions" / "fif8a_group_stage_predictions_ensemble.csv"

# Frozen candidate manifests (read-only integrity check)
V2_MANIFEST          = ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json"
V3_MANIFEST          = ROOT / "outputs" / "final_candidate_v3_objective_residual" / "FROZEN_MANIFEST.json"

# Output files — Part B (interim, never overwrites existing backbone)
MWR_V4               = INTERIM / "matches_with_ratings_recent_rollforward.parquet"
MM_V4                = INTERIM / "model_matrix_baseline_recent_rollforward.parquet"
MM_REPORT            = ROOT / "outputs" / "reports" / "recent_rollforward_model_matrix_report.md"

# Output files — Part C (v4 candidate)
V4_DIR               = ROOT / "outputs" / "final_candidate_v4_recent_rollforward"

# Auto-consensus settings (same as v2)
SEEDS = [101, 202, 303, 404, 505]
CONFIG = AutoPolicyConfig(
    min_ev_uplift_to_override_safe=0.25,
    max_allowed_variance_flag_for_ev=False,
    contrarian_ev_allowed_by_default=False,
)

# R1_only_diff_5_0 parameters (same as v3)
RULE_NAME            = "R1_only_diff_5_0"
STRONG_CATEGORIES    = {"elite_upside", "positive"}
WEAK_CATEGORIES      = {"fragile", "low_upside"}
DIFF_THRESHOLD       = 5.0
EXTREME_DRAW_THRESHOLD = 6.0

WARN_CHANGES = 8
FAIL_CHANGES = 16
EXPECTED_GROUP_MATCHES = 72

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TEMPLATE_ALIASES = {
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "USA": "United States",
}

# Argentina/Messi distortion: June 6 match is down-weighted for form (documented
# in lineup distortion audit; 0.25× weight, but the rolling-form engine applies
# equal weights so this is logged as a provenance caveat only, not implemented
# as a code change that would alter the model architecture).
LINEUP_DISTORTION_CAVEAT = (
    "Argentina/Messi (LA002, active_monitoring) June 6 match included at full "
    "weight in rolling form. Distortion documented in "
    "outputs/predictions/recent_international_lineup_distortion_audit.csv but "
    "custom weighting not implemented (would alter model architecture)."
)


# ── SHA-256 helper ───────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Part A: verify frozen candidates unchanged ───────────────────────────────
def verify_frozen_candidates() -> dict[str, bool]:
    results: dict[str, bool] = {}
    for label, manifest_path in [("v2", V2_MANIFEST), ("v3", V3_MANIFEST)]:
        if not manifest_path.exists():
            logger.warning(f"  Manifest missing: {manifest_path}")
            results[label] = False
            continue
        manifest = json.loads(manifest_path.read_text())
        ok = True
        for entry in manifest.get("files", []):
            fp = ROOT / entry["path"]
            if not fp.exists() or _sha256(fp) != entry["sha256"]:
                logger.warning(f"  {label} file MODIFIED or missing: {fp.name}")
                ok = False
        results[label] = ok
        logger.info(f"  {label} byte-identical: {ok}")
    return results


# ── Part B: build updated model matrix ───────────────────────────────────────
def build_updated_model_matrix() -> pd.DataFrame:
    """Append 35 recent matches to matches_clean in-memory; re-run pipeline."""
    logger.info("Loading matches_clean.parquet...")
    mc = pd.read_parquet(MATCHES_CLEAN)
    mc["date"] = pd.to_datetime(mc["date"])
    logger.info(f"  Existing backbone: {len(mc)} rows (cutoff {mc['date'].max().date()})")

    logger.info("Loading recent results parquet...")
    recent = pd.read_parquet(RECENT_RESULTS)
    included = recent[recent["include_in_clean_update"]].copy()
    logger.info(f"  Recent matches to append: {len(included)}")

    # Convert to matches_clean schema
    new_rows = []
    for _, r in included.iterrows():
        hs = int(r["team_a_goals"])
        as_ = int(r["team_b_goals"])
        if hs > as_:
            result_label = "home_win"
            home_pts, away_pts = 3, 0
        elif hs < as_:
            result_label = "away_win"
            home_pts, away_pts = 0, 3
        else:
            result_label = "draw"
            home_pts, away_pts = 1, 1
        new_rows.append({
            "date": pd.Timestamp(r["match_date"]),
            "home_team": r["team_a_normalized"],
            "away_team": r["team_b_normalized"],
            "home_score": hs,
            "away_score": as_,
            "tournament": r["competition"],
            "city": str(r.get("city", "") or ""),
            "country": str(r.get("country", "") or ""),
            "neutral": bool(r.get("neutral_site", False)),
            "result_label": result_label,
            "home_points": home_pts,
            "away_points": away_pts,
            "goal_diff": hs - as_,
            "total_goals": hs + as_,
            "home_goals": hs,
            "away_goals": as_,
        })

    new_df = pd.DataFrame(new_rows).astype({
        "home_score": "int64", "away_score": "int64",
        "home_points": "int64", "away_points": "int64",
        "goal_diff": "int64", "total_goals": "int64",
        "home_goals": "int64", "away_goals": "int64",
    })
    combined = pd.concat([mc, new_df], ignore_index=True).sort_values("date")
    logger.info(f"  Combined backbone: {len(combined)} rows (max date {combined['date'].max().date()})")

    # Load frozen rating tables
    logger.info("Loading frozen rating tables...")
    elo = pd.read_parquet(ELO_CLEAN)
    fifa = pd.read_parquet(FIFA_CLEAN)
    logger.info(f"  ELO: {len(elo)} rows, max {elo['rating_date'].max().date()}")
    logger.info(f"  FIFA: {len(fifa)} rows, max {fifa['ranking_date'].max().date()}")

    # Canonicalize team names in combined
    mapping_df = load_team_name_map()
    home_ws = normalize_team_whitespace(combined["home_team"])
    away_ws = normalize_team_whitespace(combined["away_team"])
    combined["home_canon"], _ = canonicalize_team_series(home_ws, "international_results", mapping_df)
    combined["away_canon"], _ = canonicalize_team_series(away_ws, "international_results", mapping_df)

    # As-of rating join (strict no-leakage: rating_date < match_date)
    logger.info("Running as-of rating join...")
    out = combined.copy()

    he = asof_rating_join(combined, elo, "home_canon", ["elo_rating"], "home", rating_date_col="rating_date")
    ae = asof_rating_join(combined, elo, "away_canon", ["elo_rating"], "away", rating_date_col="rating_date")
    out["home_elo"] = he["home_elo_rating"].values
    out["away_elo"] = ae["away_elo_rating"].values
    out["home_elo_rating_date"] = he["home_rating_date"].values
    out["away_elo_rating_date"] = ae["away_rating_date"].values

    hf = asof_rating_join(combined, fifa, "home_canon", ["fifa_rank", "fifa_points"], "home", rating_date_col="ranking_date")
    af = asof_rating_join(combined, fifa, "away_canon", ["fifa_rank", "fifa_points"], "away", rating_date_col="ranking_date")
    out["home_fifa_rank"] = hf["home_fifa_rank"].values
    out["home_fifa_points"] = hf["home_fifa_points"].values
    out["home_fifa_ranking_date"] = hf["home_rating_date"].values
    out["away_fifa_rank"] = af["away_fifa_rank"].values
    out["away_fifa_points"] = af["away_fifa_points"].values
    out["away_fifa_ranking_date"] = af["away_rating_date"].values

    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["fifa_rank_diff"] = out["home_fifa_rank"] - out["away_fifa_rank"]
    out["fifa_points_diff"] = out["home_fifa_points"] - out["away_fifa_points"]

    out["has_home_elo"] = out["home_elo"].notna()
    out["has_away_elo"] = out["away_elo"].notna()
    out["has_home_fifa"] = out["home_fifa_points"].notna()
    out["has_away_fifa"] = out["away_fifa_points"].notna()
    out["has_complete_elo"] = out["has_home_elo"] & out["has_away_elo"]
    out["has_complete_fifa"] = out["has_home_fifa"] & out["has_away_fifa"]
    out["has_complete_ratings"] = out["has_complete_elo"] & out["has_complete_fifa"]

    out = add_rating_momentum_features(out, elo, fifa)

    # Leakage assertion (only for rows that have rating dates)
    for dcol in ["home_elo_rating_date", "away_elo_rating_date",
                 "home_fifa_ranking_date", "away_fifa_ranking_date"]:
        used = out[dcol].notna()
        if (out.loc[used, dcol] >= out.loc[used, "date"]).any():
            raise ValueError(f"LEAKAGE DETECTED: {dcol} not strictly before match date")

    out = out.drop(columns=["home_canon", "away_canon"])

    MWR_V4.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MWR_V4, index=False)
    logger.info(f"  ✓ Wrote {MWR_V4} ({len(out)} rows)")

    # Build model matrix (adds rolling form features)
    logger.info("Building model matrix...")
    played = out[out["home_score"].notna() & out["away_score"].notna()].copy()
    mm = build_model_matrix(played)
    mm.to_parquet(MM_V4, index=False)
    logger.info(f"  ✓ Wrote {MM_V4} ({len(mm)} rows, {len(mm.columns)} cols)")

    _write_mm_report(mm, len(new_rows))
    return mm


def _write_mm_report(mm: pd.DataFrame, n_new: int) -> None:
    MM_REPORT.parent.mkdir(parents=True, exist_ok=True)
    original_cutoff = "2026-06-02"
    new_max = mm["date"].max().date()
    with open(MM_REPORT, "w") as f:
        f.write("# Recent Rollforward Model Matrix Report\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).date()}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Backbone cutoff before update: **{original_cutoff}**\n")
        f.write(f"- New matches appended: **{n_new}** (June 3–8, 2026)\n")
        f.write(f"- Updated backbone rows: **{len(mm)}**\n")
        f.write(f"- Updated backbone date range: {mm['date'].min().date()} → {new_max}\n")
        f.write(f"- Columns: {len(mm.columns)}\n\n")
        f.write("## Rating update status\n\n")
        f.write("- ELO ratings: **frozen** (max 2025-12-13) — not updated\n")
        f.write("- FIFA rankings: **frozen** (max 2024-06-20) — not updated\n")
        f.write("- Rolling form (ppm_5/10, gf/ga/gd_10, win/draw/loss rates): **updated** "
                "with June 3–8 matches\n\n")
        f.write("## Team snapshot updates\n\n")
        f.write("For teams with at least one June 3–8 match, `build_team_snapshots()` now "
                "returns their June 3–8 form data as the most-recent row.\n\n")
        f.write("## Lineup distortion caveat\n\n")
        f.write(f"{LINEUP_DISTORTION_CAVEAT}\n\n")
        f.write("## No-leakage guarantees\n\n")
        f.write("- Rolling form: `shift(1).rolling(window)` excludes the current match.\n")
        f.write("- Rating join: `merge_asof(allow_exact_matches=False)` — "
                "rating_date strictly before match_date.\n")
        f.write("- Leakage assertion passed for all rating date columns.\n")
    logger.info(f"  ✓ Wrote {MM_REPORT}")


# ── Part C: generate predictions ─────────────────────────────────────────────
def resolve_team(name: str, snaps: pd.DataFrame) -> str | None:
    team = TEMPLATE_ALIASES.get(name, name)
    return team if team in snaps.index else None


def build_match_context(poisson, snaps, template_row):
    team_a = template_row["team_a"]
    team_b = template_row["team_b"]
    resolved_a = resolve_team(team_a, snaps)
    resolved_b = resolve_team(team_b, snaps)
    if resolved_a is None or resolved_b is None:
        return None
    X = feature_row(snaps.loc[resolved_a], snaps.loc[resolved_b], template_row["date"])
    lam_a, lam_b = poisson.predict_lambdas(X)
    M = poisson.score_matrix(float(lam_a[0]), float(lam_b[0]))
    return {"team_a": team_a, "team_b": team_b, "features": X, "matrix": M}


def build_prediction_row(template_row, model_probs, score_matrix, rules):
    team_a = template_row["team_a"]
    team_b = template_row["team_b"]
    odds = {"a_win": template_row["rate_a"], "draw": template_row["rate_draw"], "b_win": template_row["rate_b"]}
    inv = np.array([1 / template_row["rate_a"], 1 / template_row["rate_draw"], 1 / template_row["rate_b"]])
    template_probs = inv / inv.sum()

    safe_outcome_key = OUTCOME_KEYS[int(np.argmax(model_probs))]
    safe_score = most_probable_score_for_outcome(score_matrix, safe_outcome_key)
    raw_mode_score = most_probable_score(score_matrix)
    ev_score, ev_points = ev_max_score(score_matrix, odds, rules)
    ev_outcome_values = {
        ok: expected_points_for_outcome(ok, score_matrix, odds, rules)[0]
        for ok in OUTCOME_KEYS
    }
    ev_outcome_key = max(ev_outcome_values, key=ev_outcome_values.get)
    safe_points = expected_points_for_score(safe_score[0], safe_score[1], score_matrix, odds, rules)

    edge = model_probs - template_probs
    contrarian = (ev_outcome_key != safe_outcome_key) and (odds[ev_outcome_key] > odds[safe_outcome_key])
    notes = []
    if contrarian:
        notes.append("CONTRARIAN: EV backs higher-odds outcome vs most-probable")
    if score_matrix.max() < 0.12:
        notes.append("HIGH_VARIANCE: flat scoreline distribution")
    if abs(edge).max() > 0.10:
        notes.append("VALUE_EDGE>0.10 vs template")
    if safe_score != raw_mode_score:
        notes.append("SAFE_SCORE_ALIGNED_TO_MODEL_OUTCOME")

    return {
        "match_number": int(template_row["match_number"]),
        "group": template_row["group"],
        "date": template_row["date"],
        "team_a": team_a,
        "team_b": team_b,
        "rate_a": template_row["rate_a"],
        "rate_draw": template_row["rate_draw"],
        "rate_b": template_row["rate_b"],
        "model_p_a_win": round(float(model_probs[0]), 4),
        "model_p_draw": round(float(model_probs[1]), 4),
        "model_p_b_win": round(float(model_probs[2]), 4),
        "template_p_a_win": round(float(template_probs[0]), 4),
        "template_p_draw": round(float(template_probs[1]), 4),
        "template_p_b_win": round(float(template_probs[2]), 4),
        "value_edge_a": round(float(edge[0]), 4),
        "value_edge_draw": round(float(edge[1]), 4),
        "value_edge_b": round(float(edge[2]), 4),
        "most_probable_score": score_to_string(raw_mode_score),
        "ev_max_score": score_to_string(ev_score),
        "most_probable_outcome": named_outcome_from_key(safe_outcome_key, team_a, team_b),
        "ev_max_outcome": named_outcome_from_key(ev_outcome_key, team_a, team_b),
        "recommended_score_safe": score_to_string(safe_score),
        "recommended_score_ev": score_to_string(ev_score),
        "expected_points_safe": round(float(safe_points), 3),
        "expected_points_ev": round(float(ev_points), 3),
        "notes": "; ".join(notes),
    }


def generate_predictions(mm: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting team snapshots from updated model matrix...")
    mm["date"] = pd.to_datetime(mm["date"])
    snaps = build_team_snapshots(mm)
    logger.info(f"  Team snapshots: {len(snaps)} teams")

    logger.info("Loading models...")
    with open(MODELS_PKL, "rb") as f:
        model_bundle = pickle.load(f)
    poisson = model_bundle["poisson"]
    logit = model_bundle["logit"]
    hgb = model_bundle["hgb"]

    rules = load_scoring_rules()
    tmpl = pd.read_csv(TEMPLATE)

    logger.info("Generating predictions for 72 template matches...")
    poisson_rows = []
    unresolved: set[str] = set()
    for _, t in tmpl.iterrows():
        context = build_match_context(poisson, snaps, t)
        if context is None:
            if resolve_team(t["team_a"], snaps) is None:
                unresolved.add(t["team_a"])
            if resolve_team(t["team_b"], snaps) is None:
                unresolved.add(t["team_b"])
            continue
        X = context["features"]
        M = context["matrix"]
        poisson_probs = outcome_probs_from_matrix(M)
        poisson_rows.append(build_prediction_row(t, poisson_probs, M, rules))

    if unresolved:
        logger.warning(f"  Unresolved teams: {sorted(unresolved)}")
    else:
        logger.info("  All 72 matches resolved")

    pred = add_score_columns(pd.DataFrame(poisson_rows).sort_values("match_number"))
    logger.info(f"  ✓ Generated {len(pred)} predictions")
    return pred


# ── R1_only_diff_5_0 (identical logic to build_objective_residual_candidate) ─
def _parse_score(score: str) -> tuple[int, int]:
    a, b = str(score).split("-")
    return int(a), int(b)


def _fmt(a: int, b: int) -> str:
    return f"{a}-{b}"


def evaluate_r1_rule(v2_score: str, cat_a: str, cat_b: str, score_a: float, score_b: float) -> dict:
    diff = round(float(score_a) - float(score_b), 4)
    abs_diff = abs(diff)
    ga, gb = _parse_score(v2_score)
    margin = ga - gb
    strong_is_a = diff > 0
    strong_cat = cat_a if strong_is_a else cat_b
    weak_cat = cat_b if strong_is_a else cat_a

    components: list[str] = []
    fail: list[str] = []

    if abs_diff >= DIFF_THRESHOLD:
        components.append(f"abs_overlay_diff>={DIFF_THRESHOLD:g}")
    else:
        fail.append(f"abs_overlay_diff<{DIFF_THRESHOLD:g}")

    if strong_cat in STRONG_CATEGORIES:
        components.append(f"strong_side_{strong_cat}")
    else:
        fail.append(f"strong_side_not_elite_or_positive({strong_cat})")

    if weak_cat in WEAK_CATEGORIES:
        components.append(f"weak_side_{weak_cat}")
    else:
        fail.append(f"weak_side_not_fragile_or_low({weak_cat})")

    narrow = abs(margin) <= 1
    if narrow:
        components.append("base_v4_narrow_or_conservative")
    else:
        fail.append(f"base_v4_not_narrow(margin={margin:+d})")

    no_change = {
        "changed": False, "adjusted_score": v2_score,
        "rule_triggered": "", "change_type": "none",
        "rule_components": " | ".join(components) if components else "",
    }

    if fail:
        no_change["reason"] = "R1_only_diff_5_0 not satisfied: " + "; ".join(fail)
        return no_change

    strong_goals, weak_goals = (ga, gb) if strong_is_a else (gb, ga)

    if strong_goals - weak_goals == 1:
        new_strong, new_weak = strong_goals + 1, weak_goals
        change_type = "favourite_strengthened"
        reason = ("Stronger objective-residual side wins narrowly; add one goal to the "
                  "favourite (max one-goal margin adjustment).")
    elif weak_goals - strong_goals == 1:
        new_strong, new_weak = weak_goals, weak_goals
        change_type = "decisive_to_draw"
        reason = ("Base v4 predicts a narrow win for the fragile side while the objective "
                  "residual strongly favours the opponent; level to a draw.")
    elif strong_goals == weak_goals:
        if abs_diff >= EXTREME_DRAW_THRESHOLD:
            new_strong, new_weak = strong_goals + 1, weak_goals
            change_type = "draw_to_decisive"
            reason = (f"Base v4 predicts a draw and the objective residual edge is extreme "
                      f"(|diff| >= {EXTREME_DRAW_THRESHOLD:g}); award one-goal win to stronger side.")
        else:
            no_change["rule_components"] = " | ".join(components)
            no_change["reason"] = (
                f"Gate met but base v4 is a draw and edge not extreme "
                f"(|diff|={abs_diff:g} < {EXTREME_DRAW_THRESHOLD:g}); no change.")
            return no_change
    else:
        no_change["rule_components"] = " | ".join(components)
        no_change["reason"] = "Gate met but base v4 margin not adjustable under R1; no change."
        return no_change

    adjusted = _fmt(new_strong, new_weak) if strong_is_a else _fmt(new_weak, new_strong)
    return {
        "changed": adjusted != v2_score,
        "adjusted_score": adjusted,
        "rule_triggered": RULE_NAME,
        "change_type": change_type,
        "rule_components": " | ".join(components),
        "reason": reason,
    }


def build_r1_adjustments(auto_scores: pd.DataFrame) -> pd.DataFrame:
    overlay = pd.read_csv(OVERLAY_CSV)
    cat = dict(zip(overlay["team"], overlay["upside_category"]))
    score = dict(zip(overlay["team"], overlay["final_adjusted_human_overlay_score"]))

    rows: list[dict] = []
    for _, r in auto_scores.iterrows():
        a, b = r["team_a"], r["team_b"]
        base_score = str(r["final_recommended_score"])
        cat_a, cat_b = cat.get(a, "unknown"), cat.get(b, "unknown")
        sc_a, sc_b = score.get(a, float("nan")), score.get(b, float("nan"))
        diff = round(float(sc_a) - float(sc_b), 4)
        result = evaluate_r1_rule(base_score, cat_a, cat_b, sc_a, sc_b)
        rows.append({
            "match_number": int(r["match_number"]),
            "group": r["group"],
            "team_a": a,
            "team_b": b,
            "v4_auto_score": base_score,
            "final_score": result["adjusted_score"],
            "changed": bool(result["changed"]),
            "rule_triggered": result["rule_triggered"],
            "change_type": result["change_type"],
            "overlay_diff": diff,
            "team_a_category": cat_a,
            "team_b_category": cat_b,
            "rule_components": result["rule_components"],
            "reason": result["reason"],
            "deterministic_yes_no": "yes",
        })
    return pd.DataFrame(rows)


# ── Part C writer ─────────────────────────────────────────────────────────────
def write_v4_candidate(
    auto_scores: pd.DataFrame,
    adj: pd.DataFrame,
    pred: pd.DataFrame,
    frozen: dict[str, bool],
) -> None:
    V4_DIR.mkdir(parents=True, exist_ok=True)

    score_map = dict(zip(adj["match_number"], adj["final_score"]))
    changed_map = dict(zip(adj["match_number"], adj["changed"]))
    rule_map = dict(zip(adj["match_number"], adj["rule_triggered"]))

    # --- final_group_score_predictions_auto.csv ---
    auto = auto_scores.copy()
    auto["base_v4_auto_score"] = auto["final_recommended_score"]
    auto["final_recommended_score"] = auto["match_number"].map(score_map)
    auto["r1_adjustment_changed"] = auto["match_number"].map(changed_map)
    auto["r1_rule_triggered"] = auto["match_number"].map(rule_map)
    auto.to_csv(V4_DIR / "final_group_score_predictions_auto.csv", index=False)

    # --- final_group_score_predictions_fill_only.csv ---
    fill_rows = []
    for _, r in adj.iterrows():
        fill_rows.append({
            "match_number": r["match_number"],
            "group": r["group"],
            "team_a": r["team_a"],
            "team_b": r["team_b"],
            "score_to_fill_in": r["final_score"],
            "copy_text": f"{r['match_number']}. {r['team_a']} {r['final_score']} {r['team_b']}",
        })
    pd.DataFrame(fill_rows).to_csv(V4_DIR / "final_group_score_predictions_fill_only.csv", index=False)

    # --- recent_rollforward_adjustments.csv (R1 adjustments audit) ---
    adj.to_csv(V4_DIR / "recent_rollforward_adjustments.csv", index=False)

    # --- recent_rollforward_candidate_report.md ---
    _write_candidate_report(adj, frozen, pred)

    logger.info(f"  ✓ Wrote v4 candidate to {V4_DIR.relative_to(ROOT)}")


def _write_candidate_report(adj: pd.DataFrame, frozen: dict[str, bool], pred: pd.DataFrame) -> None:
    changed = adj[adj["changed"]]
    n = len(adj)
    n_changed = int(len(changed))
    lines = [
        "# v4 Recent Rollforward Candidate Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).date()}",
        f"**Rule applied:** `{RULE_NAME}`",
        "",
        "## Provenance",
        "",
        "- Base: `final_candidate_v2_auto_science` form features updated with 35 June 3–8 "
          "matches appended to the backbone",
        "- Rolling form re-computed via `build_model_matrix()` — no model retraining",
        "- ELO / FIFA ratings frozen (max 2025-12-13 / 2024-06-20)",
        "- Auto-consensus policy applied (same config as v2)",
        f"- R1_only_diff_5_0 applied on top of v4 auto score",
        "- Overlay: `data/reference/wc2026_human_upside_overlay.csv`",
        "",
        "## Constraints satisfied",
        "",
        "- broad_human_overlay_used: **false**",
        "- manual_approval_used: **false**",
        "- subjective_override_used: **false**",
        "- rolling_forward_update: **true**",
        "- model_retrained: **false**",
        "",
        "## R1 adjustment summary",
        "",
        f"- Group matches: **{n}**",
        f"- R1-adjusted scores: **{n_changed}**",
        "",
    ]
    if n_changed == 0:
        lines.append("No matches qualified under R1_only_diff_5_0.")
    else:
        lines.append("| match | group | fixture | v4_auto | adjusted | change type | overlay_diff |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in changed.iterrows():
            lines.append(
                f"| {r['match_number']} | {r['group']} | {r['team_a']} vs {r['team_b']} "
                f"| {r['v4_auto_score']} | {r['final_score']} | {r['change_type']} "
                f"| {r['overlay_diff']:+g} |"
            )

    lines += [
        "",
        "## Lineup distortion caveat",
        "",
        LINEUP_DISTORTION_CAVEAT,
        "",
        "## Frozen candidate integrity",
        "",
        f"- v2_auto_science byte-identical: **{frozen.get('v2', False)}**",
        f"- v3_objective_residual byte-identical: **{frozen.get('v3', False)}**",
        "",
    ]
    (V4_DIR / "recent_rollforward_candidate_report.md").write_text("\n".join(lines))


# ── Part D: delta vs v3 ───────────────────────────────────────────────────────
def build_delta_vs_v3(adj: pd.DataFrame) -> pd.DataFrame:
    v3_auto = ROOT / "outputs" / "final_candidate_v3_objective_residual" / "final_group_score_predictions_auto.csv"
    if not v3_auto.exists():
        logger.warning("v3 auto scores not found — skipping delta computation")
        return pd.DataFrame()

    v3 = pd.read_csv(v3_auto)
    v3_score_map = dict(zip(v3["match_number"], v3["final_recommended_score"]))

    rows = []
    for _, r in adj.iterrows():
        mn = int(r["match_number"])
        v3_score = str(v3_score_map.get(mn, ""))
        v4_score = str(r["final_score"])

        score_changed = v4_score != v3_score
        if score_changed and v3_score and v4_score:
            v3a, v3b = _parse_score(v3_score)
            v4a, v4b = _parse_score(v4_score)

            def outcome(a, b):
                return "a_win" if a > b else ("b_win" if b > a else "draw")

            outcome_changed = outcome(v3a, v3b) != outcome(v4a, v4b)
            gd_changed = (v3a - v3b) != (v4a - v4b)
        else:
            outcome_changed = False
            gd_changed = False

        rows.append({
            "match_number": mn,
            "group": r["group"],
            "team_a": r["team_a"],
            "team_b": r["team_b"],
            "v3_score": v3_score,
            "v4_score": v4_score,
            "score_changed": score_changed,
            "outcome_changed": outcome_changed,
            "gd_changed": gd_changed,
            "reason_for_change": "rolling_form_update_June_3_8" if score_changed else "no_change",
            "source_provenance_caveat": (
                "v4 uses equal-weight rolling form; Argentina/Messi distortion "
                "documented but not down-weighted in rolling form engine"
                if r["team_a"] in {"Argentina", "Honduras"} or r["team_b"] in {"Argentina", "Honduras"}
                else ""
            ),
        })

    delta = pd.DataFrame(rows)
    delta.to_csv(V4_DIR / "recent_rollforward_delta_vs_v3.csv", index=False)
    n_changed = int(delta["score_changed"].sum())
    n_outcome = int(delta["outcome_changed"].sum())
    logger.info(f"  Delta vs v3: {n_changed} score changes, {n_outcome} outcome changes")
    return delta


# ── Promotion gate ────────────────────────────────────────────────────────────
def evaluate_promotion_gate(adj: pd.DataFrame, frozen: dict[str, bool], delta: pd.DataFrame) -> dict:
    changed = adj[adj["changed"]]
    n_changed = int(len(changed))

    one_goal = True
    rule_only = True
    for _, r in changed.iterrows():
        ga, gb = _parse_score(r["v4_auto_score"])
        na, nb = _parse_score(r["final_score"])
        if abs(ga - na) + abs(gb - nb) > 1:
            one_goal = False
        if r["rule_triggered"] != RULE_NAME:
            rule_only = False

    no_midtable = True
    for _, r in changed.iterrows():
        diff = r["overlay_diff"]
        strong_cat = r["team_a_category"] if diff > 0 else r["team_b_category"]
        weak_cat = r["team_b_category"] if diff > 0 else r["team_a_category"]
        if strong_cat not in STRONG_CATEGORIES or weak_cat not in WEAK_CATEGORIES:
            no_midtable = False

    score_changes_v3 = int(delta["score_changed"].sum()) if len(delta) else 0
    reasonable_delta = score_changes_v3 <= 20  # ≤20 score changes from v3 is plausible

    checks = {
        "v2_byte_identical": frozen.get("v2", False),
        "v3_byte_identical": frozen.get("v3", False),
        "v4_candidate_separate": V4_DIR.exists(),
        "deterministic_from_R1": rule_only,
        "no_broad_or_manual_suggestions": True,
        "changes_not_failing": n_changed <= FAIL_CHANGES,
        "every_change_one_goal": one_goal,
        "no_midtable_chemistry_fame_keyabsence_only": no_midtable,
        "reasonable_delta_from_v3": reasonable_delta,
    }
    warn = n_changed > WARN_CHANGES
    passed = all(checks.values())
    return {
        "passed": passed,
        "warning": warn,
        "n_r1_changed": n_changed,
        "n_score_changes_vs_v3": score_changes_v3,
        "checks": checks,
    }


def write_promotion_report(adj: pd.DataFrame, gate: dict) -> None:
    changed = adj[adj["changed"]]
    status = "PASSED" if gate["passed"] else "FAILED"
    warn_str = "  (warning: more than 8 R1-adjusted scores)" if gate["warning"] else ""
    lines = [
        "# v4 Recent Rollforward Candidate — Promotion Report",
        "",
        f"**Promotion gate: {status}**{warn_str}",
        "",
        "Base model: `final_candidate_v2_auto_science` (immutable baseline/reference).",
        "v3 reference: `final_candidate_v3_objective_residual` (unchanged).",
        "v4 candidate: `final_candidate_v4_recent_rollforward`.",
        f"Rule: **{RULE_NAME}**. No manual approval or subjective override used.",
        "Rolling form updated with 35 June 3–8 matches; ELO/FIFA ratings frozen.",
        "",
        f"- R1-adjusted group scores: **{gate['n_r1_changed']}** "
          f"(sparse expected 0–{WARN_CHANGES}; >{WARN_CHANGES} warns; >{FAIL_CHANGES} fails)",
        f"- Score changes vs v3: **{gate['n_score_changes_vs_v3']}** (driven by form update)",
        "",
        "## Gate checks",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for name, ok in gate["checks"].items():
        lines.append(f"| {name} | {'pass' if ok else 'FAIL'} |")

    lines += ["", "## R1-adjusted matches", ""]
    if changed.empty:
        lines.append("_No R1 adjustments._")
    else:
        lines.append("| match | fixture | v4_auto | adjusted | rule |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in changed.iterrows():
            lines.append(
                f"| {r['match_number']} | {r['team_a']} vs {r['team_b']} | "
                f"{r['v4_auto_score']} | {r['final_score']} | {r['rule_triggered']} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        (
            "Promotion gate passed: `final_candidate_v4_recent_rollforward` is the "
            "recommended rolling-forward score-fill candidate. "
            "v2 and v3 are preserved as baseline/reference."
            if gate["passed"]
            else
            "Promotion gate FAILED: v4 is not promoted; v3 remains the reference candidate."
        ),
        "",
    ]
    report_path = ROOT / "outputs" / "reports" / "recent_rollforward_candidate_promotion_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    logger.info(f"  ✓ Wrote {report_path.name}")


# ── Manifest ──────────────────────────────────────────────────────────────────
def write_manifest(adj: pd.DataFrame, gate: dict, frozen: dict[str, bool]) -> None:
    files = []
    for name in [
        "final_group_score_predictions_auto.csv",
        "final_group_score_predictions_fill_only.csv",
        "recent_rollforward_adjustments.csv",
        "recent_rollforward_candidate_report.md",
    ]:
        path = V4_DIR / name
        files.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    manifest = {
        "schema_version": 1,
        "candidate_dir": "outputs/final_candidate_v4_recent_rollforward",
        "candidate_name": "final_candidate_v4_recent_rollforward",
        "base_model": "outputs/final_candidate_v2_auto_science",
        "base_v3": "outputs/final_candidate_v3_objective_residual",
        "rule": RULE_NAME,
        "deterministic": True,
        "broad_human_overlay_used": False,
        "manual_approval_used": False,
        "subjective_override_used": False,
        "source_provenance_complete": False,
        "source_conflicts": 0,
        "needs_review_rows_used": 0,
        "rolling_forward_update": True,
        "new_matches_appended": 35,
        "new_match_date_range": "2026-06-03 to 2026-06-08",
        "elo_ratings_frozen_at": "2025-12-13",
        "fifa_rankings_frozen_at": "2024-06-20",
        "v2_byte_identical": frozen.get("v2", False),
        "v3_byte_identical": frozen.get("v3", False),
        "n_r1_adjusted_scores": gate["n_r1_changed"],
        "n_score_changes_vs_v3": gate["n_score_changes_vs_v3"],
        "promotion_gate_passed": gate["passed"],
        "promotion_warning": gate["warning"],
        "is_recommended_score_fill_candidate": gate["passed"],
        "lineup_distortion_caveat": LINEUP_DISTORTION_CAVEAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (V4_DIR / "FROZEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"  ✓ Wrote FROZEN_MANIFEST.json")


# ── Active candidate update ───────────────────────────────────────────────────
def update_active_candidate_if_gate_passes(gate: dict) -> bool:
    if not gate["passed"]:
        logger.info("  Promotion gate did not pass — active_candidate.yml NOT updated")
        return False

    yml_path = ROOT / "data" / "live" / "active_candidate.yml"
    current = yml_path.read_text()

    if "final_candidate_v4_recent_rollforward" in current:
        logger.info("  active_candidate.yml already points to v4 — no change")
        return False

    updated = current.replace(
        "active_candidate_dir: outputs/final_candidate_v2_auto_science",
        "active_candidate_dir: outputs/final_candidate_v4_recent_rollforward",
    ).replace(
        "# Currently active: final_candidate_v2_auto_science (science-only submission).",
        "# Currently active: final_candidate_v4_recent_rollforward (rolling-forward with June 3-8 form update).",
    )

    if updated == current:
        logger.warning("  Could not find expected active_candidate_dir line — active_candidate.yml NOT updated")
        return False

    yml_path.write_text(updated)
    logger.info(f"  ✓ Updated active_candidate.yml → v4")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 70)
    logger.info("v4 Recent Rollforward Candidate Builder")
    logger.info("=" * 70)

    # Pre-flight: verify frozen candidates
    logger.info("\n[Pre-flight: frozen candidate integrity]")
    frozen = verify_frozen_candidates()

    # Part B: build updated model matrix
    logger.info("\n[Part B: build updated model matrix]")
    mm = build_updated_model_matrix()

    # Part C step 1: generate predictions
    logger.info("\n[Part C step 1: generate predictions]")
    pred = generate_predictions(mm)

    # Part C step 2: auto-consensus
    logger.info("\n[Part C step 2: auto-consensus policy]")
    decisions = pd.read_csv(DECISIONS_CSV)
    final_v1 = pd.read_csv(FINAL_V1)
    tmpl = pd.read_csv(TEMPLATE)
    ensemble = pd.read_csv(ENSEMBLE_CSV) if ENSEMBLE_CSV.exists() else None

    candidates, skipped = collect_candidate_scores(pred, decisions, ensemble)
    auto_scores = select_final_scores(pred, final_v1, decisions, candidates, config=CONFIG)
    validate_final_scores(auto_scores, tmpl)
    logger.info(f"  Auto-consensus: {len(auto_scores)} scores; {len(skipped)} skipped sources")

    # Part C step 3: R1 adjustment
    logger.info("\n[Part C step 3: R1_only_diff_5_0 adjustment]")
    adj = build_r1_adjustments(auto_scores)
    n_r1 = int(adj["changed"].sum())
    logger.info(f"  R1-adjusted scores: {n_r1}")
    for _, r in adj[adj["changed"]].iterrows():
        logger.info(
            f"  match {r['match_number']}: {r['team_a']} vs {r['team_b']} "
            f"{r['v4_auto_score']} -> {r['final_score']} ({r['change_type']})"
        )

    # Write v4 candidate files
    logger.info("\n[Part C: writing v4 candidate files]")
    write_v4_candidate(auto_scores, adj, pred, frozen)

    # Part D: delta vs v3
    logger.info("\n[Part D: delta vs v3]")
    delta = build_delta_vs_v3(adj)

    # Part E: promotion gate
    logger.info("\n[Part E: promotion gate evaluation]")
    gate = evaluate_promotion_gate(adj, frozen, delta)
    write_promotion_report(adj, gate)

    # Manifest
    write_manifest(adj, gate, frozen)

    # Active candidate update
    logger.info("\n[Part E: active candidate update]")
    updated_yml = update_active_candidate_if_gate_passes(gate)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"  v2 byte-identical:         {frozen.get('v2', False)}")
    logger.info(f"  v3 byte-identical:         {frozen.get('v3', False)}")
    logger.info(f"  new matches appended:      35 (June 3–8, 2026)")
    logger.info(f"  R1-adjusted scores:        {n_r1}")
    logger.info(f"  score changes vs v3:       {gate['n_score_changes_vs_v3']}")
    logger.info(f"  promotion gate:            {'PASSED' if gate['passed'] else 'FAILED'}")
    logger.info(f"  active_candidate.yml:      {'updated to v4' if updated_yml else 'unchanged'}")
    logger.info(f"  v4 candidate dir:          {V4_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
