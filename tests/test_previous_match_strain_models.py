"""Tests for the previous-match strain backtest / model layer (Task F).

These exercise the controlled-backtest helpers and guard the experiment's
invariants — most importantly that opponent-strength and previous-score feature
families stay separate and that the frozen v2 candidate is never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_previous_match_strain_models import (
    BASELINE_FEATURES,
    OPP_STRENGTH_ONLY,
    RESULT_CONTROLS_ONLY,
    STRAIN_SCORE_ONLY,
    _canonical_scores,
    _conservative_residual_adjust,
    _metrics_block,
    make_logit_pipeline_dynamic,
)
from src.ingest.rules_and_scoring import load_scoring_rules

ROOT = Path(__file__).parent.parent
V2_MANIFEST = ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json"
CLASSES = ["home_win", "draw", "away_win"]


# ---- feature-family separation (modelling discipline) ---------------------
def test_opponent_strength_and_result_families_are_disjoint():
    opp = set(OPP_STRENGTH_ONLY)
    res = set(RESULT_CONTROLS_ONLY)
    assert opp.isdisjoint(res), opp & res
    # Opponent-strength family must reference opponent ratings, not scorelines.
    assert all("strength" in c or "opponent" in c for c in OPP_STRENGTH_ONLY)
    # Result family must reference goals/result, never opponent ratings.
    assert not any("opponent_strength" in c for c in RESULT_CONTROLS_ONLY)
    # The composite strain score is its own family.
    assert "previous_match_strain_score_diff" in STRAIN_SCORE_ONLY


def test_strain_features_are_additive_to_baseline():
    # Variants are baseline + extras; baseline features are never dropped.
    for extra in (OPP_STRENGTH_ONLY, RESULT_CONTROLS_ONLY, STRAIN_SCORE_ONLY):
        assert not set(extra) & set(BASELINE_FEATURES)


# ---- helper correctness ----------------------------------------------------
def test_canonical_scores_map_argmax_to_scoreline():
    proba = np.array([[0.7, 0.2, 0.1],   # home_win -> 1-0
                      [0.2, 0.6, 0.2],   # draw     -> 0-0
                      [0.1, 0.2, 0.7]])  # away_win -> 0-1
    assert _canonical_scores(proba) == [(1, 0), (0, 0), (0, 1)]


def test_conservative_residual_adjust_returns_valid_distribution():
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({
        "result_label": rng.choice(CLASSES, n),
        "previous_match_strain_score_diff": rng.normal(size=n),
    })
    test = pd.DataFrame({
        "previous_match_strain_score_diff": rng.normal(size=120),
    })
    p_train = rng.dirichlet(np.ones(3), size=n)
    p_test = rng.dirichlet(np.ones(3), size=120)
    out = _conservative_residual_adjust(train, p_train, test, p_test,
                                        "previous_match_strain_score_diff")
    assert out.shape == p_test.shape
    # Still a valid probability distribution.
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all()


def test_conservative_adjust_is_noop_without_spread():
    # All strain values identical -> cannot bucket -> returns input unchanged.
    train = pd.DataFrame({
        "result_label": ["home_win", "draw", "away_win"] * 10,
        "previous_match_strain_score_diff": [0.0] * 30,
    })
    test = pd.DataFrame({"previous_match_strain_score_diff": [0.0] * 5})
    p_train = np.tile([0.4, 0.3, 0.3], (30, 1))
    p_test = np.tile([0.4, 0.3, 0.3], (5, 1))
    out = _conservative_residual_adjust(train, p_train, test, p_test,
                                        "previous_match_strain_score_diff")
    assert np.allclose(out, p_test)


def test_metrics_block_has_expected_keys():
    rules = load_scoring_rules()
    y = pd.Series(["home_win", "draw", "away_win", "home_win"])
    proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2],
                      [0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])
    aa = pd.Series([1, 0, 0, 2])
    ab = pd.Series([0, 0, 1, 1])
    m = _metrics_block(y, proba, aa, ab, rules)
    for key in ["log_loss", "brier", "accuracy", "calib_err", "high_conf_err",
                "exact_score_hit_canon", "goal_diff_hit_canon", "exp_points_canon"]:
        assert key in m
    assert 0.0 <= m["accuracy"] <= 1.0


# ---- dynamic pipeline trains and predicts ---------------------------------
def _synthetic_matrix(n=300, seed=1):
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 120, n)
    p_home = 1 / (1 + np.exp(-elo_diff / 150))
    labels = []
    for ph in p_home:
        labels.append(rng.choice(CLASSES, p=[ph * 0.8, 0.2, (1 - ph) * 0.8]))
    df = pd.DataFrame({
        "elo_diff": elo_diff,
        "fifa_points_diff": rng.normal(0, 100, n),
        "fifa_rank_diff": rng.normal(0, 20, n),
        "ppm_10_diff": rng.normal(0, 1, n),
        "gd_10_diff": rng.normal(0, 1, n),
        "ppm_5_diff": rng.normal(0, 1, n),
        "win_rate_10_diff": rng.normal(0, 0.3, n),
        "gf_10_diff": rng.normal(0, 1, n),
        "ga_10_diff": rng.normal(0, 1, n),
        "days_since_2000": rng.integers(0, 9000, n),
        "overperf_elo_10_diff": rng.normal(0, 0.3, n),
        "tournament_category": rng.choice(["world_cup", "continental_championship"], n),
        "result_label": labels,
    })
    for b in ["neutral", "is_world_cup", "is_friendly", "is_world_cup_qualifier",
              "is_continental_championship", "nonneutral", "home_adv_friendly",
              "home_adv_world_cup_qualifier", "home_adv_continental",
              "home_adv_world_cup", "neutral_world_cup_context"]:
        df[b] = rng.integers(0, 2, n)
    # strain extras
    for c in OPP_STRENGTH_ONLY + RESULT_CONTROLS_ONLY + STRAIN_SCORE_ONLY:
        df[c] = rng.normal(0, 1, n)
    return df


def test_dynamic_pipeline_with_strain_features_trains_and_predicts():
    df = _synthetic_matrix()
    feats = BASELINE_FEATURES + STRAIN_SCORE_ONLY
    pipe = make_logit_pipeline_dynamic(feats).fit(df[feats], df["result_label"])
    proba = pipe.predict_proba(df[feats])
    assert proba.shape == (len(df), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---- 7. final_candidate_v2_auto_science not modified ----------------------
def test_final_candidate_v2_auto_science_unchanged_by_models():
    assert V2_MANIFEST.exists()
    manifest = json.loads(V2_MANIFEST.read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.exists(), f"v2 file missing: {entry['path']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"v2 file modified: {entry['path']}"


def test_no_model_driven_v3_candidate_created():
    # The strain-features experiment must not spawn a *model-driven* v3 candidate.
    # The only sanctioned v3 is the deterministic objective-residual layer
    # (final_candidate_v3_objective_residual), which retrains nothing and is produced
    # solely by the validated R1_only_diff_5_0 rule.
    v3 = list(ROOT.glob("outputs/final_candidate_v3*"))
    for path in v3:
        assert path.name == "final_candidate_v3_objective_residual", (
            f"unexpected model-driven v3 candidate(s): {path}"
        )
        manifest = json.loads((path / "FROZEN_MANIFEST.json").read_text())
        assert manifest["rule"] == "R1_only_diff_5_0"
        assert manifest["deterministic"] is True
        assert manifest["manual_approval_used"] is False
        assert manifest["base_model"] == "outputs/final_candidate_v2_auto_science"
