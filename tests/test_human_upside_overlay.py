"""Tests for the Human Upside + Current-State + Chemistry analyst overlay.

The overlay is advisory only. These tests assert it stays internally consistent,
is validated against the official squad, applies the analyst-selected replacement
overrides, respects the final objective residual policy, and
never modifies the frozen ``final_candidate_v2_auto_science`` submission.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts import build_human_upside_overlay as build
from scripts import validate_human_upside_overlay as validate
from src.overlay.human_upside_overlay import (
    KEY_ABSENCE_COLUMNS,
    OVERLAY_LABEL,
    V2_SCORES_PATH,
    compute_final_overlay,
    compute_net_human_upside,
    load_official_squads,
    load_seed,
    resolve_candidate,
    team_key,
)

SCORE_0_5_COLUMNS = [
    "star_upside_score_0_5", "star_risk_score_0_5", "talent_breakout_score_0_5",
    "talent_risk_score_0_5", "star_current_state_score_0_5", "talent_current_state_score_0_5",
    "team_environment_score_0_5", "availability_fragility_score_0_5",
    "star_talent_chemistry_score_0_5", "team_core_cohesion_score_0_5",
]

# Amendment 2 — analyst-selected replacements for genuine squad absences.
REPLACEMENT_TEAMS = {
    "Ghana": ("Antoine Semenyo", "Ernest Nuamah"),
    "Morocco": ("Achraf Hakimi", "Bilal El Khannouss"),
    "Netherlands": ("Virgil van Dijk", "Jorrel Hato"),
    "IR Iran": ("Mehdi Taremi", "Aria Yousefi"),
    "Uruguay": ("Federico Valverde", "Facundo Pellistri"),
    "Congo DR": ("Yoane Wissa", "Ngal'ayel Mukau"),
    "Colombia": ("Luis Díaz", "Jhon Arias"),
}


@pytest.fixture(scope="module", autouse=True)
def built_artifacts():
    validate.main()
    build.main()


@pytest.fixture(scope="module")
def seed():
    return load_seed()


@pytest.fixture(scope="module")
def overlay():
    return pd.read_csv(build.OVERLAY_CSV)


@pytest.fixture(scope="module")
def match_overlay():
    return pd.read_csv(build.MATCH_OVERLAY_CSV)


@pytest.fixture(scope="module")
def suggestions():
    return pd.read_csv(build.OBJECTIVE_CANDIDATES_CSV)


# --- core overlay invariants -------------------------------------------------------------

def _team_row(frame: pd.DataFrame, team: str) -> pd.Series:
    return frame[frame["team"] == team].iloc[0]


def test_seed_has_exactly_48_teams(seed):
    assert seed["team"].nunique() == 48 and len(seed) == 48


def test_final_overlay_has_exactly_48_teams(overlay):
    assert overlay["team"].nunique() == 48 and len(overlay) == 48


def test_match_overlay_covers_72_group_matches(match_overlay):
    assert len(match_overlay) == 72 and match_overlay["match_number"].nunique() == 72


def test_all_score_columns_in_valid_ranges(overlay):
    for col in SCORE_0_5_COLUMNS:
        assert overlay[col].between(0, 5).all(), f"{col} out of 0..5"


def test_net_human_upside_formula_correct(overlay):
    assert (overlay["net_human_upside_score"] == overlay.apply(compute_net_human_upside, axis=1)).all()


def test_final_overlay_formula_correct(overlay):
    expected = overlay.apply(compute_final_overlay, axis=1)
    assert ((overlay["final_adjusted_human_overlay_score"] - expected).abs() < 1e-9).all()


def test_final_overlay_formula_includes_key_absence_residual(overlay):
    expected = (
        overlay["net_human_upside_score"]
        + overlay["current_state_adjustment"]
        + overlay["chemistry_adjustment"]
        + overlay["key_absence_residual_penalty"]
    ).round(4)
    assert ((overlay["final_adjusted_human_overlay_score"] - expected).abs() < 1e-9).all()


def test_current_state_and_chemistry_adjustment_ranges(overlay):
    assert overlay["current_state_adjustment"].between(-2.0, 2.0).all()
    assert overlay["chemistry_adjustment"].between(-1.0, 1.0).all()
    assert overlay["star_talent_chemistry_score_0_5"].between(0, 5).all()
    assert overlay["team_core_cohesion_score_0_5"].between(0, 5).all()


def test_key_absence_columns_exist(seed, overlay, match_overlay):
    for frame in (seed, overlay, match_overlay):
        missing = [col for col in KEY_ABSENCE_COLUMNS if col not in frame.columns]
        assert not missing


def test_key_absence_residual_penalty_range(overlay):
    assert overlay["key_absence_residual_penalty"].between(-1.25, 0.0).all()


def test_ghana_kudus_absence_audited(overlay):
    row = _team_row(overlay, "Ghana")
    assert row["key_absent_player"] == "Mohammed Kudus"
    assert row["key_absence_residual_penalty"] == pytest.approx(-0.50)
    assert "creative-ceiling loss" in row["key_absence_rationale"]


def test_netherlands_xavi_simons_absence_audited(overlay):
    row = _team_row(overlay, "Netherlands")
    assert row["key_absent_player"] == "Xavi Simons"
    assert row["key_absence_residual_penalty"] in {-0.50, -0.25}
    if row["key_absence_residual_penalty"] == -0.25:
        assert "cap" in str(row["key_absence_rationale"]).lower()


def test_scotland_gilmour_absence_audited(overlay):
    row = _team_row(overlay, "Scotland")
    assert row["key_absent_player"] == "Billy Gilmour"
    assert row["key_absence_timing_category"] == "late_absence"
    assert row["key_absence_residual_penalty"] == pytest.approx(-0.50)


def test_sweden_kulusevski_absence_discounted(overlay):
    row = _team_row(overlay, "Sweden")
    assert row["key_absent_player"] == "Dejan Kulusevski"
    assert row["key_absence_raw_penalty"] < 0
    assert row["key_absence_already_reflected_score_0_5"] == 5
    assert row["key_absence_residual_penalty"] == pytest.approx(0.0)


def test_senegal_mane_not_penalised_because_in_squad(overlay):
    squads = load_official_squads()
    squad = squads[squads["team_key"] == team_key("Senegal")]
    assert resolve_candidate(squad, "Sadio Mané", role="absence").found
    row = _team_row(overlay, "Senegal")
    assert row["star_player"] == "Sadio Mané"
    assert row["key_absence_residual_penalty"] == pytest.approx(0.0)
    assert "Mané" not in str(row["key_absent_player"])


def test_selection_only_omissions_have_no_penalty_without_explicit_rupture(overlay):
    selection = overlay[overlay["key_absence_timing_category"] == "selection_omission"]
    if selection.empty:
        return
    allowed = selection["key_absence_rationale"].astype(str).str.lower().str.contains(
        "tactical rupture|morale|unexpected major loss|explicit"
    )
    assert (
        (selection["key_absence_residual_penalty"] == 0.0) | allowed
    ).all()


def test_long_known_absences_are_discounted(overlay):
    long_known = overlay[overlay["key_absence_timing_category"].astype(str).str.contains("long", case=False, na=False)]
    if long_known.empty:
        return
    assert (long_known["key_absence_already_reflected_score_0_5"] >= 3).all()


def test_late_material_absence_candidate_thresholds(suggestions, overlay):
    by_team = overlay.set_index("team")
    late_rows = suggestions[suggestions["late_absence_component"].astype(bool)]
    for _, r in late_rows.iterrows():
        team = r["opponent_team"]
        team_row = by_team.loc[team]
        assert team_row["key_absence_role_importance_0_5"] >= 4
        assert team_row["key_absence_already_reflected_score_0_5"] <= 2
        assert team_row["key_absence_residual_penalty"] <= -0.50


# --- amendment 2: replacements + squad validity ------------------------------------------

def test_all_seven_replacements_validate_or_are_reported(overlay):
    squads = load_official_squads()
    issues = pd.read_csv(validate.ISSUES_PATH)
    for team, (star, talent) in REPLACEMENT_TEAMS.items():
        squad = squads[squads["team_key"] == team_key(team)]
        row = overlay[overlay["team"] == team].iloc[0]
        for who, name in (("star", star), ("talent", talent)):
            res = resolve_candidate(squad, name, role=who)
            reported = ((issues["team"] == team) & (issues["role"] == who)).any() if len(issues) else False
            assert res.found or bool(row["needs_review"]) or reported, (
                f"{team} {who} '{name}' neither validated nor reported"
            )
        # The override player must actually be the one carried into the overlay.
        assert row["star_player"] == star and row["biggest_talent"] == talent


def test_no_selected_player_absent_unless_flagged(overlay):
    squads = load_official_squads()
    for _, row in overlay.iterrows():
        squad = squads[squads["team_key"] == team_key(row["team"])]
        for who, value in (("star", row["star_player"]), ("talent", row["biggest_talent"])):
            if str(value).strip().lower() == "none_clear":
                continue
            res = resolve_candidate(squad, value, role=who)
            assert res.found or bool(row["needs_review"]), (
                f"{row['team']} {who} '{value}' absent and not flagged needs_review"
            )


# --- final policy: extreme-mismatch review shortlist --------------------------------------

def test_suggestions_do_not_overwrite_final_candidate(suggestions):
    final_dir = V2_SCORES_PATH.parent.resolve()
    assert build.OBJECTIVE_CANDIDATES_CSV.resolve().parent != final_dir
    assert final_dir not in build.OBJECTIVE_CANDIDATES_CSV.resolve().parents
    if not suggestions.empty:
        assert "final_recommended_score" not in suggestions.columns
        assert "suggested_review_score" in suggestions.columns


def test_group_objective_residual_candidates_exist():
    assert build.OBJECTIVE_CANDIDATES_CSV.exists()
    assert build.OBJECTIVE_CANDIDATES_MD.exists()


def test_wc2026_objective_candidates_at_most_eight_primary_rows(suggestions):
    primary = suggestions[suggestions["review_scope"].eq("primary_shortlist")]
    assert len(primary) <= 8


def test_no_midtable_vs_midtable_match_in_primary_shortlist(suggestions):
    mid = {"useful_context", "high_variance"}
    primary = suggestions[suggestions["review_scope"].eq("primary_shortlist")]
    assert not (
        primary["advantaged_category"].isin(mid)
        & primary["opponent_category"].isin(mid)
    ).any()


def test_no_suggestion_is_based_only_on_chemistry(suggestions):
    for _, r in suggestions.iterrows():
        criteria = set(str(r["qualifying_criteria"]).split(" | "))
        assert "chemistry_supporting_tiebreaker" not in criteria or len(criteria - {"chemistry_supporting_tiebreaker"}) >= 4
        assert not bool(r["chemistry_support_only"])
        assert "strong_vs_fragile_or_low" in criteria
        assert any(c.startswith("overlay_diff>=") for c in criteria)


def test_no_suggestion_is_based_only_on_key_absence(suggestions):
    for _, r in suggestions.iterrows():
        criteria = set(str(r["qualifying_criteria"]).split(" | "))
        assert "key_absence_supporting_tiebreaker" not in criteria or len(criteria - {"key_absence_supporting_tiebreaker"}) >= 4
        assert "strong_vs_fragile_or_low" in criteria
        assert any(c.startswith("overlay_diff>=") for c in criteria)


def test_fame_only_signal_cannot_trigger_candidate(suggestions):
    assert not suggestions["objective_rule_triggered"].astype(str).str.contains("fame", case=False).any()


def test_every_suggestion_changes_at_most_one_goal(suggestions):
    for _, r in suggestions.iterrows():
        if pd.isna(r["suggested_review_score"]) or not str(r["suggested_review_score"]).strip():
            continue
        (a1, b1) = map(int, str(r["v2_score"]).split("-"))
        (a2, b2) = map(int, str(r["suggested_review_score"]).split("-"))
        assert abs(a1 - a2) + abs(b1 - b2) == 1, f"match {r['match_number']} changes more than one goal"


def test_every_suggestion_has_final_policy_gates(suggestions):
    for _, r in suggestions.iterrows():
        criteria = set(str(r["qualifying_criteria"]).split(" | "))
        assert _team_row(pd.read_csv(build.OVERLAY_CSV), r["advantaged_team"])["upside_category"] in {"elite_upside", "positive"}
        assert _team_row(pd.read_csv(build.OVERLAY_CSV), r["opponent_team"])["upside_category"] in {"fragile", "low_upside"}
        assert abs(float(r["final_human_overlay_diff"])) >= 5.0
        assert {"strong_vs_fragile_or_low", "narrow_or_conservative_v2_score"}.issubset(criteria)
        assert any(c.startswith("overlay_diff>=") for c in criteria)
        assert {"advantaged_attack_heavy", "opponent_materially_fragile"} & criteria


def test_no_automatic_overwrite_of_v2_group_scores(suggestions):
    v2 = pd.read_csv(V2_SCORES_PATH)
    assert "suggested_review_score" not in v2.columns
    assert len(v2) == 72


def test_policy_report_does_not_recommend_broad_manual_approval():
    text = build.FINAL_POLICY_MD.read_text(encoding="utf-8").lower()
    assert "manual approval" not in text
    assert "manually approved" not in text
    assert "objective residual" in text


def test_dashboard_labels_suggestions_as_review_only():
    import scripts.build_mobile_dashboard as dash

    data = {"human_upside_overlay": dash._load_human_upside_overlay()}
    rendered = dash._render_human_upside(data)
    assert "Context only — not used in final prediction." in rendered
    assert "Objective residual review — not applied." in rendered
    assert "Broad human-overlay suggestions (audit-only, hidden by default)" in rendered
    assert "Mohammed Kudus" in rendered
    assert "already reflected" in rendered
    assert dash.SECTION_DEFAULT_OPEN.get("human-upside-overlay") is False


# --- objective residual rule alignment (feeds v3) ----------------------------------------

def test_overlay_objective_residual_matches_deterministic_v3_rule():
    """The overlay's objective-residual review shortlist must equal the deterministic
    R1_only_diff_5_0 changes used to build the v3 candidate (no broad overlay leakage)."""
    from pathlib import Path

    from scripts import build_objective_residual_candidate as v3

    build.main()  # refresh overlay artifacts
    candidates = pd.read_csv(
        Path("outputs/predictions/human_upside_objective_residual_review_candidates.csv")
    )
    overlay_matches = {
        int(r["match_number"]): (str(r["v2_score"]), str(r["suggested_review_score"]))
        for _, r in candidates.iterrows()
    }
    adj = v3.build_adjustments()
    changed = adj[adj["changed"]]
    v3_matches = {
        int(r["match_number"]): (str(r["v2_score"]), str(r["objective_residual_score"]))
        for _, r in changed.iterrows()
    }
    # The deterministic v3 rule selects exactly the overlay's objective shortlist.
    assert v3_matches == overlay_matches
    # All driven by the single validated rule; the broad overlay is not used.
    assert (changed["rule_triggered"] == "R1_only_diff_5_0").all()


def test_broad_overlay_not_used_for_score_fill():
    """Broad human-overlay suggestions remain audit-only and never feed score_to_fill_in."""
    from pathlib import Path

    from scripts import build_objective_residual_candidate as v3

    adj = v3.build_adjustments()
    changed = adj[adj["changed"]]
    # Only fragile/low opponents against elite/positive favourites — never mid-table.
    for _, r in changed.iterrows():
        strong = r["team_a_category"] if r["overlay_diff"] > 0 else r["team_b_category"]
        weak = r["team_b_category"] if r["overlay_diff"] > 0 else r["team_a_category"]
        assert strong in {"elite_upside", "positive"}
        assert weak in {"fragile", "low_upside"}


# --- frozen-submission guarantee ---------------------------------------------------------

def test_final_candidate_v2_not_modified():
    def digest() -> str:
        return hashlib.sha256(V2_SCORES_PATH.read_bytes()).hexdigest()

    before = digest()
    build.main()
    validate.main()
    assert digest() == before, "final_candidate_v2_auto_science score file was modified"
